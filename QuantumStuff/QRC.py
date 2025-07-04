import numpy as np
from tqdm import tqdm
from scipy.sparse import csc_matrix, csc_array
from .Evolution import Super_H, Super_D, Lindblad_Propagator
from .States import zero
from .utils import is_herm, is_state, dag
from .Operators import measure
import sklearn.linear_model as LM

def CD_evolution(sk: np.ndarray | list, H_enc: np.ndarray | csc_matrix | csc_array, H0: np.ndarray | csc_matrix | csc_array, c_ops: list, δt: float,  steps: int, state = None, disable_progress_bar = False):
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
    if (not isinstance(steps, int)) or steps <= 0 or len(sk) < steps:
        raise ValueError("Steps must be a positive integer, whose length is less than or equal to the length of sk")
    if not all(isinstance(c, (np.ndarray, csc_matrix, csc_array)) for c in c_ops):
        raise TypeError("All collapse operators in c_ops must be numpy arrays, csc_matrix, or csc_array")
    
    Nq = int(np.log2(H0.shape[0]))
    if len(c_ops) != 0:
        superd = csc_array(Super_D(c_ops), dtype = complex)
    else:
        superd = csc_array(np.zeros((4**Nq, 4**Nq), dtype = complex))
    if state is None:
        state = zero(dm = True, N = Nq)
    state_t = np.zeros((steps, 2**Nq, 2**Nq), dtype = complex)
    for i in tqdm(range(steps), disable = disable_progress_bar):
        superh = csc_array(Super_H(H0 + (sk[i] + 1)*H_enc), dtype = complex)
        state_t[i] = Lindblad_Propagator(superh, superd, δt, state).reshape(2**Nq, 2**Nq)
        state = state_t[i]

    return state_t

def CD_training(sk: np.ndarray | list, y_target: np.ndarray | list, H_enc: np.ndarray | csc_matrix | csc_array, H0: np.ndarray | csc_matrix | csc_array, c_ops: list, δt: float, operators: list, meas_ind: list, wo: int = 1000, train_size: int = 1000, rho = None, disable_progress_bar = False):
    """ 
    Trains a QRC (Quantum Reservoir Computer) using the Continous Dissipation approach (CD) used by 
    Sannia et Al. in https://doi.org/10.22331/q-2024-03-20-1291 . After the evolution of the system
    a set of measurements is performed and the results are used to train a Ridge regression model.
    Thus:
    X_i = Tr{rho(t)O_i}
    and we train a linear model such that:
    y_target_j = ∑_iW^{i,j}X_i + b_j
    where y_target_j is j-th feature of the target output, W^{i,j} is the weight for the i-th measurement 
    operator and j-th target feature, and b_j is the bias for the j-th target feature.

    Args:
        sk (np.ndarray | list): The input signal to be encoded in the Hamiltonian.
        y_target (np.ndarray | list): The target output for training.
        H_enc (np.ndarray | sp.csc_matrix | sp.csc_array): The encoding Hamiltonian.
        H0 (np.ndarray | sp.csc_matrix | sp.csc_array): The initial Hamiltonian.
        c_ops (list): List of collapse operators for the system.
        δt (float): Time step for the evolution.
        operators (list): List of measurement operators.
        meas_ind (list): Where the operators must be measured.
        wo (int, optional): Wash out time for the reservoir. Defaults to 1000.
        train_size (int, optional): Size of the training set. Defaults to 1000.
        rho (np.ndarray, optional): Initial state of the system. If Default the system is initialized to the Zero state.
    
    Returns:
        ridge (LM.RidgeCV): The trained Ridge regression model.
        x_train (np.ndarray): The training data after measurement.
        ρt (np.ndarray): The final state of the system after evolution.
    """

    Nq = int(np.log2(H0.shape[0]))
    if rho is None:
        rho = zero(dm = True, N = Nq)
    else:
        if not is_state(rho)[1]:
            raise ValueError("The provided initial state is not a valid density matrix.")
    if not (is_herm(H0) or is_herm(H_enc)):
        raise ValueError("H0 and H1 must be a Hermitian matrix.")
    
    y_target = np.array(y_target)
    sk = np.array(sk)
    rho = np.array(rho)

    rhot = CD_evolution(sk, H_enc, H0, c_ops, δt, wo + train_size, rho, disable_progress_bar)
    x_train = measure(rhot[wo:], operators, meas_ind)

    alpha = np.logspace(-9,3,1000)
    ridge = LM.RidgeCV(alphas = alpha.tolist())
    #For forecasting problems y_target = sk[wo+1:wo+train_size+1]
    ridge.fit((x_train), y_target)

    return ridge, x_train, rhot[-1]