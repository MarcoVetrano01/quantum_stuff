from scipy.integrate import solve_ivp
import numpy as np
from .utils import is_state, is_herm, dag, ket_to_dm, operator2vector, vector2operator, nqubit, MatrixLike, MatrixOrSparse, SparseLike
from .Operators import anticommutator, commutator
from .States import zero
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import csc_matrix, csc_array, kron
from scipy.linalg import expm
from tqdm import tqdm


def dissipator(state: MatrixLike, L: MatrixLike):
    """
    Dissipator for a Lindblad operator L acting on the state ρ.
    Args:
        state (np.ndarray): The density matrix ρ.
        L (np.ndarray): The jump operator L.
    Returns:
        np.ndarray: The dissipator term for the Lindblad equation.
    """
    if len(L) == 0:
        return 0
    if len(np.shape(state)) == 3:
        raise ValueError("Batch mode is not supported")
    if not isinstance(L, (np.ndarray, list, csc_array, csc_matrix)):
        raise ValueError("L must be a numpy array or a list of numpy arrays.")
    L = np.asarray(L, dtype=complex)
        
    state = np.asarray(state, dtype=complex)
    state = ket_to_dm(state, batchmode = False)
    if L.ndim == 2:
        LL = dag(L) @ L
        return (L @ state @ dag(L) - 0.5 * anticommutator(LL, state))
    else:
        LL = dag(L) @ L
        return np.sum(L @ state @ dag(L)- 0.5 *(LL @ state + (state @ LL.transpose((1,2,0))).transpose(2,1,0)),0)

def evolve_lindblad(state0: MatrixLike, H: MatrixLike, t: MatrixLike, c_ops:list = []):
    """
    Evolve a quantum state under the Lindblad equation with improved numerical stability.
    Args:
        state0 (MatrixLike): Initial quantum state (density matrix or ket).
        H (MatrixLike): Hamiltonian operator.
        t (MatrixLike): Array of time points for the evolution.
        c_ops (list): List of jump operators.
    Returns:
        np.ndarray: Time-evolved quantum states at each time point in t.
    """
    state0 = ket_to_dm(state0, batchmode = False)
    n = nqubit(state0)
    state0 = np.array(state0, dtype = complex).ravel('F')
    t0 = t[0]
    tf = t[-1]
    
    # Use much stricter tolerances
    solution = solve_ivp(Liouvillian, [t0,tf], state0, t_eval = t, args = (H, c_ops),
                        rtol=1e-8, atol=1e-10, method='DOP853')
    
    return solution.y.reshape((2**n,2**n, len(t)), order='F').transpose((2,0,1))

def evolve_unitary(U: MatrixOrSparse, state: MatrixOrSparse, batchmode: bool):
    """
    Evolve a quantum state using a unitary operator.
    Args:
        U (MatrixOrSparse): Unitary operator.
        state (MatrixOrSparse): Quantum state to evolve.
        batchmode (bool): If True, process a batch of states.
    Returns:
        np.ndarray: Evolved quantum state.
    """
    
    if not isinstance(state, (np.ndarray, list)):
        raise TypeError("The state must be a numpy array or a list")
    if isinstance(state, list):
        state = np.asarray(state, dtype = complex)
    if not isinstance(U, (np.ndarray, csc_array, csc_matrix)):
        raise TypeError("The unitary operator must be a numpy array, csc_array or csc_matrix")
    if isinstance(U, (csc_array, csc_matrix)) and (state.ndim == 1 or state.ndim == 2 and batchmode):
        U = U.toarray()
    if isinstance(U, (csc_array, csc_matrix)) or isinstance(state, (csc_array, csc_matrix)):
        U = csc_array(U, dtype = complex)
        state = csc_array(state, dtype = complex)
    
    isdm = is_state(state, batchmode)
    if isdm[0] == 1:
        return U @ state
    elif isdm[0] == 2:
        return (U @ state.T).T
    else:
        return U @ state @ dag(U)

def interaction(op: MatrixLike, J: MatrixLike):
    """
    Create an interaction Hamiltonian for a system with local operators `op` and coupling matrix `J`.
    Args:
        op (MatrixLike): List of local operators.
        J (MatrixLike): Coupling matrix or list of coupling strengths.
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

def Lindblad_Propagator(SH: MatrixOrSparse, SD: MatrixOrSparse | None, dt: float, rho: MatrixLike, ignore = False):
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
    rho = np.asarray(rho, dtype = complex)
    if not ignore:
        # No need for explicit is_state check - ket_to_dm will validate and convert
        rho = ket_to_dm(rho, False) if rho.ndim != 1 else operator2vector(ket_to_dm(vector2operator(rho), False))
    if SD == None:
        SD = csc_array(np.zeros_like(SH.toarray()))
    L = SH + SD
    is_sparse = isinstance(L, (csc_matrix, csc_array))
    if rho.ndim != 1:
        rho = operator2vector(rho)
    if is_sparse:
        return vector2operator(expm_multiply(L, rho, start = 0 , stop = dt, num = 2)[-1])
    else:
        return expm(L * dt) @ rho

def Liouvillian(t: float, state: MatrixLike, H: MatrixLike, c_ops: list):
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
    state = np.array(state, dtype = complex)
    if state.ndim == 2:
        n = nqubit(state)
    if state.ndim == 1:
        n = int(np.log2(len(state)/2))
        state = state.reshape((2**n,2**n), order = 'F')
    if not is_herm(H):
        raise ValueError("Hamiltonian must be a Hermitian operator.")
    if not isinstance(c_ops, (list, np.ndarray)):
        raise ValueError("c_ops must be a list or numpy array of jump operators.")
    
    F = -1j * commutator(H, state)
    
    F += dissipator(state, c_ops)
    return F.ravel('F')

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

def Super_D(c_ops: list = []):
    """
    Super operator for Lindblad equation
    SD = ∑ (L*⊗L - 1/2(I⊗L†L + (LᵀL*)⊗I))
    Args:
        c_ops (list): list of collapse operators multiplied by their decay rates
    Returns:
        MatrixOrSparse: super dissipator
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
            superd += (np.kron(c.conj(), c) - 0.5 * (np.kron(SI, LL) + np.kron(LL.T, SI)))
    return superd

def Super_H(H: MatrixOrSparse):
    """
    Super operator for Hamiltonian
    SH = -1j(I⊗H - Hᵀ⊗I)
    Args:
        H (MatrixOrSparse): Hamiltonian operator.
    Returns:
        MatrixOrSparse: super hamiltonian
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
        superh = -1j * (kron(SI, H) - kron(H.T, SI))
    return superh
