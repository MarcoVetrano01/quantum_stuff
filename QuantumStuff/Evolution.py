from scipy.integrate import solve_ivp
import numpy as np
from .utils import is_state, is_herm, dag, ket_to_dm
from .Operators import anticommutator, commutator
from .States import zero
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import csc_matrix, csc_array, kron
from scipy.linalg import expm
from tqdm import tqdm

def dissipator(state: np.ndarray, L: np.ndarray | list):
    """
    Dissipator for a Lindblad operator L acting on the state ρ.
    Args:
        state (np.ndarray): The density matrix ρ.
        L (np.ndarray): The jump operator L.
    Returns:
        np.ndarray: The dissipator term for the Lindblad equation.
    """
    if not is_state(state):
        raise ValueError("Input must be a valid quantum state (density matrix).")
    if not isinstance(L, (np.ndarray, list)):
        raise ValueError("L must be a numpy array or a list of numpy arrays.")
    L = np.asarray(L)
    
    LL = dag(L) @ L
    return (L @ state @ dag(L) - 0.5 * anticommutator(LL, state))

def evolve_lindblad(state0: np.ndarray, H: np.ndarray, t: np.ndarray, c_ops:list = []):
    """
    Evolve a quantum state under the Lindblad equation.
    Args:
        ρ0 (np.ndarray): Initial density matrix.
        H (np.ndarray): Hamiltonian operator.
        t (np.ndarray): Time points for evolution.
        c_ops (list, optional): List of jump operators. Defaults to [].
    Returns:
        np.ndarray: Time-evolved density matrix.
    """
    
    state0 = state0.reshape(int(state0.shape[0]**2))
    t0 = t[0]
    tf = t[-1]
    return solve_ivp(Liouvillian, [t0, tf], state0, t_eval = t, args = (H, c_ops)).y

def evolve_unitary(U: np.ndarray, state: np.ndarray | list):
    """
    Evolve a quantum state using a unitary operator.
    Args:
        U (np.ndarray): Unitary operator.
        state (np.ndarray | list): Quantum state to evolve.
    Returns:
        np.ndarray: Evolved quantum state.
    """
    
    if not isinstance(state, (np.ndarray, list)):
        raise TypeError("The state must be a numpy array or a list")
    state = np.asarray(state, dtype = complex)
    isdm = is_state(state)
    if not isdm[1]:
        raise ValueError("The state must be a valid quantum state")
    if state.ndim == 1 or (state.shape[0] != state.shape[1] and state.ndim == 2):
        state = ket_to_dm(state)
    return U @ state @ dag(U)

def interaction(op: np.ndarray | list, J: np.ndarray | list):
    """
    Create an interaction Hamiltonian for a system with local operators `op` and coupling matrix `J`.
    Args:
        op (np.ndarray | list): List of local operators.
        J (np.ndarray | list): Coupling matrix or list of coupling strengths.
    Returns:
        np.ndarray: Interaction Hamiltonian as a dense matrix.
    """
    if not (isinstance(op, (np.ndarray, list)) and isinstance(J, (np.ndarray, list))):
        raise TypeError("op and J must be numpy arrays or lists")
    
    sites = len(op) < 2
    if sites:
        raise ValueError("The system must have at least two sites")
    
    op = np.asarray(op, dtype = complex)
    J = np.asarray(J)
    ax = len(J.shape)
    result = np.tensordot(J, np.matmul(op[:,None], op[None,:]),axes = ([ax - 2, ax - 1], [0,1]))
    return result

def Lindblad_Propagator(SH: np.ndarray | csc_matrix, SD: np.ndarray | csc_matrix | None, dt: float, rho: np.ndarray, ignore = False):
    """
    Lindblad propagator for Lindblad equation
    Args:
        SH (np.ndarray | csc_matrix): Super Hamiltonian operator.
        SD (np.ndarray | csc_matrix | None): Super dissipator operator, if None it is set to zero.
        dt (float): Time step for the evolution.
        ρ (np.ndarray): Initial density matrix.
    Returns:
        np.ndarray: Time-evolved density matrix.
    """
    if ignore:
        pass
    elif not is_state(rho)[1]:
        raise ValueError("Input must be a valid quantum state.")
    if SD == None:
        SD = csc_array(np.zeros_like(SH.toarray()))
    L = SH + SD
    is_sparse = type(L) == csc_matrix or csc_array
    if rho.ndim != 1:
        rho = rho.flatten()
    if is_sparse:
        return expm_multiply(L, rho, start = 0 , stop = dt, num = 2)[-1]
    else:
        return expm(L * dt) @ rho

def Liouvillian(t: float, state: np.ndarray, H: np.ndarray, c_ops: list |np.ndarray):
    """
    Liouvillian for the Lindblad equation.
    Args:
        t (float): Time variable (not used in this case).
        state (np.ndarray): The density matrix ρ.
        H (np.ndarray): The Hamiltonian operator.
        c_ops (list): List of jump operators.
    Returns:
        np.ndarray: The time derivative of the density matrix ρ.
    """
    if not is_state(state):
        raise ValueError("Input must be a valid quantum state (density matrix).")
    if not is_herm(H):
        raise ValueError("Hamiltonian must be a Hermitian operator.")
    if not isinstance(c_ops, (list, np.ndarray)):
        raise ValueError("c_ops must be a list or numpy array of jump operators.")
    
    if len(state.shape) != 2:
        state = state.reshape([int(np.sqrt(state.shape[0]))]*2)
    F = -1j * commutator(H, state)
    for i in range(len(c_ops)):
        F += dissipator(state, c_ops[i])
    return F.ravel()

def random_coupling(Js: float, sites: int):
    """
    Generate a random coupling matrix for a system with `sites` number of sites.
    The matrix is symmetric and has zero diagonal elements.
    Args:
        Js (float): Maximum coupling strength.
        sites (int): Number of sites in the system.
    Returns:
        np.ndarray: Random coupling matrix of shape (sites, sites).
    """

    if not isinstance(Js, (int, float)):
        raise TypeError("Js must be a number (int or float)")
    if (not isinstance(sites, int)) or sites <= 0:
        raise ValueError("sites must be a positive integer")
    
    J = np.random.uniform(-Js, Js, size = (sites, sites))
    J = np.triu(J) - np.diag(np.diag(J))
    J = J + J.T
    return J

def Super_D(c_ops = []):
    """
    Super operator for Lindblad equation
    :c_ops: list of collapse operators multiplied by their decay rates
    :return: super dissipator
    """
    if len(c_ops) == 0:
        return None
    N = np.shape(c_ops[0])[1]
    is_sparse = isinstance(c_ops[0], (csc_matrix, csc_array))
    if is_sparse:
        SI = csc_array(np.eye(N))
        N2 = N*N
        superd = csc_array((N2, N2), dtype=complex)
        for c in c_ops:
            LL = dag(c).dot(c)
            superd += (kron(c.conj(), c) - 0.5 * (kron(SI, LL) + kron(LL.T, SI)))
            superd = csc_array(superd)
    else:
        SI = np.eye(N)
        superd = 0
        
        for c in c_ops:
            LL = dag(c)@c
            superd += (np.kron(c.conj(), c)-0.5*(np.kron(SI,LL) + np.kron(LL, SI)))
    return superd

def Super_H(H: np.ndarray | csc_matrix | csc_array):
    """
    Super operator for Hamiltonian
    :H: Hamiltonian
    :return: super hamiltonian
    """
    is_sparse = isinstance(H, (csc_matrix, csc_array))
    N = np.shape(H)[0]
    if is_sparse:
        SI = csc_matrix(np.eye(N))
        superh = -1j * (kron(SI, H) - kron(H.T, SI))
        superh = csc_matrix(superh)
    else:
        H = np.array(H, dtype = complex)
        SI = np.eye(N)
        superh = 1j * (np.kron(SI, H.T) - np.kron(H, SI))
    return superh
