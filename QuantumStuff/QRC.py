import numpy as np
from tqdm import tqdm
from scipy.sparse import csc_matrix, csc_array
from .Evolution import Super_H, Super_D, Lindblad_Propagator
from .States import zero, random_qubit, one
from .Metrics import trace_distance
from .utils import is_herm, is_state, dag
from .Operators import measure, local_measurements, two_qubits_measurements, sigmax, sigmay, sigmaz
import sklearn.linear_model as LM
from scipy.stats import pearsonr
from typing import Union, Dict, Any

MatrixOrSparse = Union[np.ndarray, csc_matrix, csc_array]
MatrixLike = Union[np.ndarray, list]

sx = sigmax()
sy = sigmay()
sz = sigmaz()
sqo = [sx, sy, sz]
tqo = [np.kron(sx, sx), np.kron(sy, sy), np.kron(sz, sz)]
operators = []
meas_ind = None

DEFAULTS: Dict[str, Dict[str, Any]] = {
    "local & correlations": {"sqo": sqo, "tqo": tqo},
    "probabilities": {"shots": 0},
    "other": {"operators": operators, "meas_ind": meas_ind}
}

# =============================================================================
# CONTINOUS DISSIPATION QRC FUNCTIONS
#           FORECASTING TASK
# =============================================================================



def CD_evolution(sk: np.ndarray | list, H_enc: np.ndarray | csc_matrix | csc_array, H0: np.ndarray | csc_matrix | csc_array, c_ops: list, δt: float,  steps: int, state = None, disable_progress_bar = False, ignore = False):
    """
    Evolution of a quantum state under the continuous dissipation encoding as implemented in 
    https://doi.org/10.22331/q-2024-03-20-1291.
    The encoding is done directly in the Hamiltonian, such that the Hamiltonian is given by:
    H = (1 + sk) * H_enc + H0
    The evolution is performed considering an update rule of the form:
    ρ(t + δt) = e^L(sk)δt ρ(t)
    where L(sk) is the Liouvillian superoperator defined by the Hamiltonian and the collapse operators.

    L = -i(H⊗I - I⊗H.T) + ∑_i γ_i(c_i ⊗ c_i* - 1/2(c_i†c_i ⊗ I + I ⊗ c_i†c_i)
    Args:
        sk (np.ndarray): classical time series encoded in the hamiltonian
        H_enc (np.ndarray | csc_matrix | csc_array): Encoding term of the hamiltonian.
        H0 (np.ndarray | csc_matrix | csc_array): Free Hamiltonian for the system.
        c_ops (list): List of collapse operators.
        δt (float): Time step for the evolution.
        steps (int): Number of time steps to evolve.
        state (np.ndarray, optional): Initial density matrix. Defaults to None, which initializes to the zero state.
        disable_progress_bar (bool, optional): If True, disables the progress bar. Defaults to False.
    Returns:
        np.ndarray: Time-evolved density matrix at each step.
    """

    if not isinstance(sk, (np.ndarray, list)):
        raise TypeError("sk must be a numpy array or a list")
    sk = np.asarray(sk, dtype = float)
    if not (is_herm(H_enc) and is_herm(H0)):
       raise TypeError("H0 and H_enc must be Hermitian matrices")
    if not isinstance(c_ops, list):
        raise TypeError("c_ops must be a list of numpy arrays or csc_matrix")
    if not isinstance(δt, (int, float)):
        raise TypeError("δt must be an integer or a float")
    if (not isinstance(steps, int)) or steps <= 0 or sk.shape[0] < steps:
        raise ValueError("Steps must be a positive integer, whose length is less than or equal to the length of sk")
    if not all(isinstance(c, (np.ndarray, csc_matrix, csc_array)) for c in c_ops):
        raise TypeError("All collapse operators in c_ops must be numpy arrays, csc_matrix, or csc_array")

    if not isinstance(H_enc, list):
        H_enc = [H_enc]
    Nq = int(np.log2(H0.shape[0]))
    if len(c_ops) != 0:
        superd = csc_array(Super_D(c_ops), dtype = complex)
    else:
        superd = csc_array(np.zeros((4**Nq, 4**Nq), dtype = complex))
    if state is None:
        state = zero(dm = True, N = Nq)
    state_t = np.zeros((steps, 2**Nq, 2**Nq), dtype = complex)
    
    for i in tqdm(range(steps), disable = disable_progress_bar):
        H = H0
        for k in range(len(H_enc)):
            H += (1+sk[i][k])*H_enc[k]
        H = csc_array(H, dtype = complex)
        superh = csc_array(Super_H(H), dtype = complex)
        state = Lindblad_Propagator(superh, superd, δt, state, ignore = ignore)

        state_t[i] = state

    return state_t

def CD_reservoir(sk: np.ndarray | list, H_enc: MatrixOrSparse, H0: MatrixOrSparse, c_ops: list, dt: float, mode: str = 'local & correlations', wo: int = 1000, train_size: int = 1000, test_size: int = 100, windows: int = 10, rho: np.ndarray | None = None, disable_progress_bar: bool = False, **kwargs):
    """ 
    Returns the outcomes of a reservori using the Continous Dissipation approach (CD) used by 
    Sannia et Al. in https://doi.org/10.22331/q-2024-03-20-1291 . After the evolution of the system
    a set of measurements is performed which depends on the selected mode.
    Thus:
    X_i = Tr{rho(t)O_i}
    
    Args:
        sk (np.ndarray | list): The input signal to be encoded in the Hamiltonian.
        y_target (np.ndarray | list): The target output for training.
        H_enc (np.ndarray | sp.csc_matrix | sp.csc_array): The encoding Hamiltonian.
        H0 (np.ndarray | sp.csc_matrix | sp.csc_array): The initial Hamiltonian.
        c_ops (list): List of collapse operators for the system.
        dt (float): Time step for the evolution.
        mode (str): Measurement mode. Can be 'local & correlations', 'probabilities', or 'other'. Default is 'local & correlations'. 
                    If 'local & correlations' is used the function measures as default the local operators [σx, σy, σz] and the two qubit correlations [σx⊗σx, σy⊗σy, σz⊗σz]. 
                    To change it local measurements and two qubit correlations operators use the keyword arguments 'sqo' and 'tqo' respectively.
                    If 'probabilities' is used the function measures the state probabilities in the computational basis at infinite statistics. 
                    'shots' parameter changes the statistics. Default is 0 (infinite statistics).
                    If 'other' is used the function measures custom operators provided in the keyword arguments 'operators' and 'meas_ind'. Default is empty list of operators and None for meas_ind 
                    (in this case whatever operator list the user inserts, the function measures all the possible qubits and combination of qubits for those operator).
                    meas_ind parameter should be a list of lists, where each inner list contains the indices of qubits where the corresponding operator should be measured.
        wo (int, optional): Wash out time for the reservoir. Defaults to 1000.
        train_size (int, optional): Size of the training set. Defaults to 1000.
        test_size (int, optional): Size of the test. Needed to test later on more testing windows
        windows (int, optional): Number of testing windows. Default to 10. Needed at least 1.
        rho (np.ndarray, optional): Initial state of the system. If Default the system is initialized to the Zero state.
        disable_progress_bar (bool, optional): If True, disables the progress bar. Defaults to False.
        kwargs: Additional keyword arguments for measurement modes.
    
    Returns:
        x_train (np.ndarray): The training data after measurement.
        rhot (list): The final state of the system after evolution.
    """
    if mode not in DEFAULTS:
        raise ValueError("invalid mode")
    params = {**DEFAULTS[mode], **kwargs}  # user kwargs override defaults

    Nq = int(np.log2(H0.shape[0]))
    rhot = CD_evolution(sk, H_enc, H0, c_ops, dt, wo + train_size + int(windows * test_size), rho, disable_progress_bar)
    
    if mode == "local & correlations":
        sqo = params.get("sqo")
        tqo = params.get("tqo")
        x_train = np.real(np.hstack((local_measurements(rhot[wo: wo + train_size], sqo), two_qubits_measurements(rhot[wo: wo + train_size], tqo))))
    
    elif mode == "probabilities":
        shots = params.get("shots")
        x_train = np.real(np.linalg.diagonal(rhot[wo : wo + train_size]))
        if shots != 0:
            x_train = np.array([np.random.multinomial(shots, x_train[i]) for i in range(len(x_train))]) / shots
    
    elif mode == "other":
        operators = params.get("operators")
        meas_ind = params.get("meas_ind")
        if len(meas_ind) == 0:
            meas_ind = [[] for i in range(len(operators))]
        x_train = np.real(measure(rhot[wo:wo +train_size], operators, meas_ind))

    return x_train, [rhot[wo + train_size -1 + int(j*test_size)] for j in range(windows)]

def CD_training(x_train, y_target: np.ndarray | list):
    """
    Train a Ridge regression model using the provided training data and target output.
    y_target_j = ∑_iW^{i,j}X_i + b_j
    where y_target_j is j-th feature of the target output, W^{i,j} is the weight for the i-th measurement 
    operator and j-th target feature, and b_j is the bias for the j-th target feature.
    Args:
        x_train (np.ndarray): The training data.
        y_target (np.ndarray | list): The target output.

    Returns:
        ridge (LM.RidgeCV): The trained Ridge regression model.
    """

    alpha = np.logspace(-9,3,1000)
    ridge = LM.RidgeCV(alphas = alpha.tolist())
    #For forecasting problems y_target = sk[wo+1:wo+train_size+1]
    ridge.fit((x_train), y_target)
    return ridge

def CD_forecast_test(ridge: LM.Ridge, sk: np.ndarray, rhof: np.ndarray | list, H_enc: MatrixOrSparse, H0: MatrixOrSparse, c_ops: list, dt: float, mode: str = 'local & correlations', wo: int = 1000, train_size: int = 1000, test_size: int = 100, windows: int = 10, delay = 1, **kwargs):
    """
    Tests a trained QRC (Quantum Reservoir Computer) using the Continous Dissipation approach (CD) used by Sannia et Al. in https://doi.org/10.22331/q-2024-03-20-1291.
    After the evolution of the system a set of measurements is performed and the results are used to predict the next value of the input signal. The test is performed on multiple windows
    of size `test_size`. 
    Args:
        ridge (LM.Ridge): The trained Ridge regression model.
        sk (np.ndarray): The input signal to be encoded in the Hamiltonian.
        rhof (np.ndarray | list): The final state of the system after evolution for each window.
        H_enc (np.ndarray | sp.csc_matrix | sp.csc_array): The encoding Hamiltonian.
        H0 (np.ndarray | sp.csc_matrix | sp.csc_array): The initial Hamiltonian.
        c_ops (list): List of collapse operators for the system.
        dt (float): Time step for the evolution.
        mode (str): Measurement mode. Can be 'local & correlations', 'probabilities', or 'other'. Default is 'local & correlations'. 
                    If 'local & correlations' is used the function measures as default the local operators [σx, σy, σz] and the two qubit correlations [σx⊗σx, σy⊗σy, σz⊗σz]. 
                    To change it local measurements and two qubit correlations operators use the keyword arguments 'sqo' and 'tqo' respectively.
                    If 'probabilities' is used the function measures the state probabilities in the computational basis at infinite statistics. 
                    'shots' parameter changes the statistics. Default is 0 (infinite statistics).
                    If 'other' is used the function measures custom operators provided in the keyword arguments 'operators' and 'meas_ind'. Default is empty list of operators and None for meas_ind 
                    (in this case whatever operator list the user inserts, the function measures all the possible qubits and combination of qubits for those operator).
                    meas_ind parameter should be a list of lists, where each inner list contains the indices of qubits where the corresponding operator should be measured.
        wo (int, optional): Wash out time for the reservoir. Defaults to 1000.
        train_size (int, optional): Size of the training set. Defaults to 1000.
        test_size (int, optional): Size of the test. Needed to test later on more testing windows. Defaults to 100.
        windows (int, optional): Number of testing windows. Default to 10. Needed at least 1.
    Returns:
        y_pred (np.ndarray): The predicted output for the test set.
    """

    if mode not in DEFAULTS:
        raise ValueError("invalid mode")
    params = {**DEFAULTS[mode], **kwargs}  # user kwargs override defaults

    Nq = int(np.log2(H0.shape[0]))
    superd = (Super_D(c_ops))
    sk = np.array(sk)
    shape = (windows, test_size, sk.shape[1])
    y_pred = np.zeros(shape)
    if not isinstance(H_enc, list):
        H_enc = [H_enc]
    for j in range(windows):
        rho = rhof[j]
        for t in range(delay):
            y_pred[j][t] = sk[wo + train_size + j * test_size + t]
        
        for i in tqdm(range(test_size-delay)):
            H = H0.copy()
            for k in range(len(H_enc)):
                H += ((1+y_pred[j][i][k])*H_enc[k])
            H = csc_array(H, dtype = complex)
            superh = Super_H(H)
            superh = csc_array(superh, dtype = complex)
            rho = Lindblad_Propagator(superh, superd, dt, rho)
            
            #Measurements
            if mode == "local & correlations":
                sqo = params.get("sqo")
                tqo = params.get("tqo")
                x_test = np.real(np.hstack((local_measurements(rho, sqo, batchmode = False), two_qubits_measurements(rho, tqo))))
            
            elif mode == "probabilities":
                shots = params.get("shots")
                x_test = np.real(np.linalg.diagonal(rho))
                if shots != 0:
                    x_test = np.random.multinomial(shots, x_test) / shots
            
            elif mode == "other":
                operators = params.get("operators")
                meas_ind = params.get("meas_ind")
                if len(meas_ind) == 0:
                    meas_ind = [[] for i in range(len(operators))]
                x_test = np.real(measure(rho, operators, meas_ind))
            
            y_pred[j][i+delay] = ridge.predict(x_test[0].reshape(1, -1))
            for k in range(y_pred.shape[2]):
                if y_pred[j][i+delay][k] < np.min(sk[:,k]):
                    y_pred[j][i+delay][k] = np.min(sk[:,k])
                if y_pred[j][i+delay][k] > np.max(sk[:,k]):
                    y_pred[j][i+delay][k] = np.max(sk[:,k])
    return y_pred

def CD_Forecasting_RC(sk: np.ndarray | list, H_enc: MatrixOrSparse, H0: MatrixOrSparse, c_ops: list, dt: float, mode: str = 'local & correlations', wo: int = 1000, train_size: int = 1000, test_size: int = 100, windows: int = 10, rho: np.ndarray | None = None, disable_progress_bar: bool = False, delay: int = 1, **kwargs):
    """
    Wrapper function that trains and tests a Quantum Reservoir Computer (QRC) using the Continous Dissipation approach (CD) as implemented in 
    https://doi.org/10.22331/q-2024-03-20-1291 for forecasting tasks.
    The function first trains the QRC using the provided training data and target output, and then tests the trained model on the test data.
    
    Args:
        sk (np.ndarray | list): The input signal to be encoded in the Hamiltonian.
        y_target (np.ndarray | list): The target output for training.
        H_enc (np.ndarray | sp.csc_matrix | sp.csc_array): The encoding Hamiltonian.
        H0 (np.ndarray | sp.csc_matrix | sp.csc_array): The initial Hamiltonian.
        c_ops (list): List of collapse operators for the system.
        dt (float): Time step for the evolution.
        mode (str): Measurement mode. Can be 'local & correlations', 'probabilities', or 'other'. Default is 'local & correlations'. 
                    If 'local & correlations' is used the function measures as default the local operators [σx, σy, σz] and the two qubit correlations [σx⊗σx, σy⊗σy, σz⊗σz]. 
                    To change it local measurements and two qubit correlations operators use the keyword arguments 'sqo' and 'tqo' respectively.
                    If 'probabilities' is used the function measures the state probabilities in the computational basis at infinite statistics. 
                    'shots' parameter changes the statistics. Default is 0 (infinite statistics).
                    If 'other' is used the function measures custom operators provided in the keyword arguments 'operators' and 'meas_ind'. Default is empty list of operators and None for meas_ind 
                    (in this case whatever operator list the user inserts, the function measures all the possible qubits and combination of qubits for those operator).
                    meas_ind parameter should be a list of lists, where each inner list contains the indices of qubits where the corresponding operator should be measured.
        wo (int, optional): Wash out time for the reservoir. Defaults to 1000.
        train_size (int, optional): Size of the training set. Defaults to 1000.
        test_size (int, optional): Size of the test. Needed to test later on more testing windows
        windows (int, optional): Number of testing windows. Default to 10. Needed at least 1.
        rho (np.ndarray, optional): Initial state of the system. If Default the system is initialized to the Zero state.
        disable_progress_bar (bool, optional): If True, disables the progress bar. Defaults to False.
        kwargs: Additional keyword arguments for measurement modes.
    
    Returns:
        ypred (np.ndarray): The predicted output for the test set.
    """

    Nq = int(np.log2(H0.shape[0]))
    if len(np.shape(sk)) == 1:
        sk = sk.reshape((len(sk),1))
    
    print('Evolution...')
    xtrain, rhofs = CD_reservoir(sk, H_enc, H0, c_ops, dt, mode, wo, train_size, test_size, windows, rho, disable_progress_bar, **kwargs)

    print('Training and testing...')
    ypred = np.zeros((delay, windows, test_size, sk.shape[1]))
    for i in range(1, delay + 1):
        ytarget = sk[wo + i : wo + train_size + i]
        ridge = CD_training(xtrain, ytarget)
        ypred[i - 1] = CD_forecast_test(ridge, sk, rhofs, H_enc, H0, c_ops, dt, mode, wo, train_size, test_size, windows, delay, **kwargs)
    print('Done.')
    return ypred

# =============================================================================
#           TESTING TASKS
# =============================================================================

def CD_cooldown(rho: np.ndarray, H0: MatrixOrSparse, c_ops: list, cool: int, dt: float):
    '''Cools down the reservoir state by letting it evolve under free evolution for a certain number of steps.
     Args:
         rho (np.ndarray): Initial density matrix.
         H0 (np.ndarray | csc_matrix): Free Hamiltonian of the quantum reservoir.
         c_ops (list): List of collapse operators for the quantum reservoir.
         cool (int): Number of cooling steps.
         δt (float): Time step for the evolution.
     Returns:
         np.ndarray: Cooled density matrix.
     '''
    Nq = int(np.log2(H0.shape[0]))
    superd = csc_matrix(Super_D(c_ops), dtype = complex)
    superh = csc_matrix(Super_H(H0), dtype = complex)
    for _ in range(cool):
        rho = Lindblad_Propagator(superh, superd, dt, rho)
    return rho

def CD_consistency_test(sk: MatrixLike, H1: MatrixOrSparse, H0: MatrixOrSparse, cops: list, dt: float, wo: int, train_size: int, cool: int):
    """Tests the consistency of the quantum reservoir computer by comparing the measurements before and after cooling.
    Consistency is defined as the pearson correlation coefficient squared between the measurements before and after cooling. After cooling the reservoir is
    re-evolved for the same time period with the same input sequence.
    Args:
        sk (np.ndarray): Input signal sequence.
        H1 (np.ndarray | csc_matrix | csc_array): Encoding Hamiltonian of the quantum reservoir.
        H0 (np.ndarray | csc_matrix | csc_array): Free Hamiltonian of the quantum reservoir.
        cops (list): List of collapse operators for the quantum reservoir.
        dt (float): Time step for the evolution.
        wo (int): Washout period to discard initial transient states.
        train_size (int): Size of the training dataset.
        cool (int): Number of cooling steps to apply.
    """
    
    sk = np.array(sk)
    rhot = CD_evolution(sk, H1, H0, cops, dt, wo + train_size + 100)
    x1 = np.hstack((local_measurements(rhot[wo + train_size : wo + train_size + 100], sqo), two_qubits_measurements(rhot[wo + train_size : wo + train_size + 100], tqo)))
    rho = CD_cooldown(rhot[-1], H0, cops, cool, dt)
    rhot = CD_evolution(sk, H1, H0, cops, dt, wo + train_size + 100, rho)
    x2 = np.hstack((local_measurements(rhot[wo + train_size : wo + train_size + 100], sqo), two_qubits_measurements(rhot[wo + train_size : wo + train_size + 100], tqo)))
    consistency = pearsonr(np.real(x1), np.real(x2))[0]**2
    return consistency

def CD_ShortTermMemory(max_tau: int, H_enc: MatrixOrSparse, H0: MatrixOrSparse, c_ops: list, dt: float, mode: str = 'local & correlations', wo: int = 1000, train_size: int = 1000, test_size: int = 300, rho: np.ndarray | None = None, disable_progress_bar: bool = False, **kwargs):
    '''Computes the short term memory capacity of a quantum reservoir computer using the CD evolution method.
    The short term memory task consists in testing the capacity of the reservoir to remember inputs after a certain delay tau.
    The function trains a ridge regression model to predict the input signal at time t - tau using the reservoir states at time t.
    The memory capacity is evaluated as the squared correlation coefficient between the predicted and target signals.
    Args:
        max_tau (int): Maximum delay time to evaluate the memory capacity.
        H_enc (np.ndarray | csc_matrix | csc_array): Encoding Hamiltonian of the quantum reservoir.
        H0 (np.ndarray | csc_matrix | csc_array): Free Hamiltonian of the quantum reservoir.
        c_ops (list): List of collapse operators for the quantum reservoir.
        dt (float): Time step for the evolution.
        mode (str): Measurement mode. Can be 'local & correlations', 'probabilities', or 'other'. Default is 'local & correlations'. 
                    If 'local & correlations' is used the function measures as default the local operators [σx, σy, σz] and the two qubit correlations [σx⊗σx, σy⊗σy, σz⊗σz]. 
                    To change it local measurements and two qubit correlations operators use the keyword arguments 'sqo' and 'tqo' respectively.
                    If 'probabilities' is used the function measures the state probabilities in the computational basis at infinite statistics. 
                    'shots' parameter changes the statistics. Default is 0 (infinite statistics).
                    If 'other' is used the function measures custom operators provided in the keyword arguments 'operators' and 'meas_ind'. Default is empty list of operators and None for meas_ind 
                    (in this case whatever operator list the user inserts, the function measures all the possible qubits and combination of qubits for those operator).
                    meas_ind parameter should be a list of lists, where each inner list contains the indices of qubits where the corresponding operator should be measured.
        wo (int, optional): Washout period to discard initial transient states. Defaults to 1000.
        train_size (int, optional): Size of the training dataset. Defaults to 1000
        test_size (int, optional): Size of the testing dataset. Defaults to 300.
        rho (np.ndarray, optional): Initial state of the system. If Default the system is initialized to the Zero state.
        disable_progress_bar (bool, optional): If True the progress bar is disabled. Defaults to False.
        kwargs: Additional keyword arguments for measurement modes.
    Returns:
        np.ndarray: Array of memory capacities for delays from 0 to max_tau - 1.
    '''
    np.random.seed(seed = 42)
    sk = np.random.random((wo + train_size + test_size,1))
    windows = 1
    x, _ = CD_reservoir(sk, H_enc, H0, c_ops, dt, mode, wo, train_size, test_size, windows, rho, disable_progress_bar, **kwargs)

    x_train = x[:train_size]
    x_test = x[train_size: train_size + test_size]
    ypred = np.zeros((max_tau, test_size))
    r = np.zeros((max_tau))
    for tau in range(max_tau):
        y_target = sk[wo - tau : wo + train_size - tau]
        ridge = CD_training(x_train, y_target)
        ypred[tau] = ridge.predict(x_test).flatten()

        corr, _ = pearsonr(ypred[tau], sk[wo + train_size - tau : wo + train_size + test_size - tau,0])
        r[tau] = corr**2
    return r

def CD_echo_state_property(sk: np.ndarray, H_enc: MatrixOrSparse, H0: MatrixOrSparse, cops: list, dt: int, wo: int, disable_progress_bar = False):
    """
    Verifies the washout time of the reservoir in the Continous Dissipation model.

    Args:
        sk : np.ndarray
            The input data.
        H1 : np.ndarray | sp.csc_array | sp.csc_matrix
            Hamiltonian encoding the input
        H0 : np.ndarray | sp.csc_array | sp.csc_matrix
            Free Hamiltonian
        cops : list
            The list of collapse operators.
        δt : int
            The time step.
        wo : int
            Wash out time test.

    Returns
        td : np.ndarray
            The trace distance between two random initialization of the reservoir in time.
    """
    Nq = int(np.log2(H0.shape[0]))
    rho1 = random_qubit(Nq, pure = True, dm = True)
    rho2 = random_qubit(Nq, pure = True, dm = True)
    drho = rho1 - rho2
    drhot = CD_evolution(sk, H_enc, H0, cops, dt, wo + 100, drho, disable_progress_bar, ignore = True)
    td = trace_distance(drhot[-100:])
    return td

