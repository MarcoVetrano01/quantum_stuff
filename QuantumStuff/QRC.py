import numpy as np
from tqdm import tqdm
from scipy.sparse import csc_matrix, csc_array
from .Evolution import Super_H, Super_D, Lindblad_Propagator
from .States import zero
from .utils import is_herm, is_state, dag

def CD_evolution(sk: np.ndarray | list, H1: np.ndarray | csc_matrix | csc_array, H0: np.ndarray | csc_matrix | csc_array, c_ops: list, δt: float,  steps: int, state = None, disable_progress_bar = False):
    """
    Evolution of a quantum state under the continuous dissipation encoding as implemented in https://doi.org/10.22331/q-2024-03-20-1291.
    Args:
        sk (np.ndarray): classical time series encoded in the hamiltonian
        H1 (np.ndarray | csc_matrix | csc_array): Encoding term of the hamiltonian.
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
    if not (is_herm(H1) and is_herm(H0)):
        raise TypeError("H0 and H1 must be Hermitian matrices")
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
        superd = None
    if state is None:
        state = zero(dm = True, N = Nq)
    state_t = np.zeros((steps, 2**Nq, 2**Nq), dtype = complex)
    for i in tqdm(range(steps), disable = disable_progress_bar):
        superh = csc_array(Super_H(H0 + (sk[i] + 1)*H1), dtype = complex)
        state_t[i] = Lindblad_Propagator(superh, superd, δt, state).reshape(2**Nq, 2**Nq)
        state = state_t[i]

    return state_t

