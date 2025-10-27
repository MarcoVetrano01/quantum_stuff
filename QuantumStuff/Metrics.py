import numpy as np
from .utils import nqubit, dag, is_state, is_herm, is_norm, tensor_product, ket_to_dm, ptrace, MatrixLike
import warnings
from scipy.linalg import sqrtm

warnings.filterwarnings(
    action='ignore',
    category=RuntimeWarning,
    message=r'.*invalid value encountered in log2.*'
)

def fidelity(state1: MatrixLike, state2: MatrixLike):
    """
    Calculate the fidelity between two quantum states.
    The fidelity is defined as:
    F(ρ, σ) = Tr(sqrt(sqrt(ρ) σ sqrt(ρ)))
    Args:
        state1 (MatrixLike): First quantum state or list of quantum states.
        state2 (MatrixLike): Second quantum state or list of quantum states.
    Returns:
        float: Fidelity between the two quantum states.
    """
    # ket_to_dm will validate the states, so no need for redundant check
    state1 = np.asarray(state1, dtype = complex)
    state2 = np.asarray(state2, dtype = complex)

    state1 = ket_to_dm(state1, batchmode = False)
    state2 = ket_to_dm(state2, batchmode = False)
    
    eigs = np.linalg.eigvals(state1 @ state2)
    return np.sum(np.sqrt(eigs))**2

def Holevo_Info(states: MatrixLike, probabilities: MatrixLike):

    """
    Compute the Holevo information for an ensemble of quantum states.
    The Holevo information quantifies the amount of classical information that can be extracted from a quantum system.
    Given a set of quantum states and their corresponding probabilities, it measures the difference between the average entropy of the states and the entropy of the average state.
    η = ∑ p_i * ρ_i
    χ(η) = S(η) - ∑ p_i * S(ρ_i)
    where S(ρ) is the von Neumann entropy of the state ρ, p_i is the probability of state ρ_i, and η is the average state over the ensemble.
    Args:
        states (MatrixLike): A list or array of quantum states (density matrices).
        probabilities (MatrixLike): A list or array of probabilities corresponding to each state.
    Returns:
        float: The Holevo information of the ensemble.
    """
    check = is_state(states, batchmode = True)
    if np.False_ in check:
        raise ValueError("states must be a list or array of quantum states.")
    states = ket_to_dm(states, batchmode = True)
    if not isinstance(probabilities, (np.ndarray, list)):
        raise ValueError("Probabilities must be a list or array of probabilities.")
    if np.sum(probabilities) != 1:
        raise ValueError("Probabilities must sum to 1.")
    
    eta = np.average(states, weights= probabilities, axis = 0)
    return von_neumann_entropy(eta) - np.average(von_neumann_entropy(states), weights = probabilities, axis = 0)


def mutual_info(state_total: MatrixLike, indices: list):
    """
    Calculate the mutual information between two subsystems A and B given a total state.
    The mutual information is defined as:
    I(A:B) = S(A) + S(B) - S(A:B)

    where S(X) is the von Neumann entropy of subsystem X.
    Args:
        state_total (MatrixLike): The total quantum state, can be a density matrix or a ket.
        indices (list): indices of the qubits to leave untraced in the bipartition (e.g. format [[0],[1]])
    Returns:
        float: The mutual information I(A:B).
    """

    state_total = np.asarray(state_total, dtype = complex)
    state_A = ptrace(state_total, indices[0])
    state_B = ptrace(state_total, indices[1])
    # ket_to_dm and von_neumann_entropy will validate states, no need for redundant check
    Na = nqubit(state_A)
    Nb = nqubit(state_B)
    Nab = nqubit(state_total)
    
    if Na + Nb != Nab:
        raise ValueError("The dimensions of the states do not match: "
                         "dim(A) + dim(B) must equal dim(total).")
    dimsA = state_A.ndim
    dimsB = state_B.ndim
    if dimsA != dimsB:
        raise ValueError("The dimensions of the states A and B must match.")
    state_A = ket_to_dm(state_A, batchmode = False)
    state_B = ket_to_dm(state_B, batchmode = False)
    state_total = ket_to_dm(state_total, batchmode = False)
    
    state_total = np.asarray(state_total, dtype = complex)
    state_A = np.asarray(state_A, dtype = complex)
    state_B = np.asarray(state_B, dtype = complex)
    
    return von_neumann_entropy(state_A) + von_neumann_entropy(state_B) - von_neumann_entropy(state_total)

def purity(state: MatrixLike, batchmode: bool):
    """
    Computes the purity of a quantum state.
    The purity is defined as the trace of the square of the density matrix.
    For a pure state, the purity is 1, and for a mixed state, it is less than 1 and decreases to a minimum of 1/2^Nqbits, where Nqbits is the number of qubits.
    Args:
        state (MatrixLike): The density matrix of the quantum state.
        batchmode (bool): If True, process a batch of states.
    Returns:
        float: The purity of the quantum state.
    """
    # ket_to_dm will validate the state, no need for redundant check
    state = ket_to_dm(state, batchmode = False)
    state = np.asarray(state, dtype = complex)
    return np.linalg.trace(state @ state)

def trace_distance(state1: MatrixLike, state2: MatrixLike | None = None):
    """
    Calculate the trace distance between two quantum states.
    The trace distance is defined as:
    T(ρ, σ) = 1/2 * ||ρ - σ||_1

    where ||.||_1 is the trace norm.
    Args:
        state1 (MatrixLike): First quantum state (density matrix) or difference between two quantum states.
        state2 (MatrixLike | None): Second quantum state (density matrix). If None, state1 should be already the difference between two quantum states.
    Returns:
        float: Trace distance between the two quantum states.
    """

    if not isinstance(state1, (list,np.ndarray)):
        raise TypeError("State 1 must be a list or numpy array")
    if not isinstance(state2, (np.ndarray, (list, type(None)))):
        raise TypeError("State 2 must be a list or numpy array or None")
    if state2 is not None:
        # ket_to_dm will validate the states, no need for redundant check
        if np.shape(state1) != np.shape(state2):
            raise ValueError("Both states must have the same dimensions")
        
        state1 = np.asarray(state1, dtype = complex)
        state2 = np.asarray(state2, dtype = complex)
        states = ket_to_dm([state1,state2], batchmode = True)
        dist = states[0] - states[1]
    else:
        state1 = np.asarray(state1, dtype = complex)
        dist = state1
        if len(np.shape(dist)) == 2:
            return 0.5 * np.trace(sqrtm(dist @ dag(dist))).real
        else:
            dists = np.array([sqrtm(dist[i] @ dag(dist[i])) for i in range(len(dist))])
            return np.real(0.5*(np.linalg.trace(dists)).real)

def von_neumann_entropy(state: MatrixLike):
    """
    Calculate the von Neumann entropy of a quantum state.
    The von Neumann entropy is defined as:
        S(ρ) = -Tr(ρ log2(ρ))

    where ρ is the density matrix of the quantum state.
    Args:
        state (MatrixLike): The quantum state, can be a density matrix or a pure state.
        ax (int): Axis along which to compute the entropy. Default is -1.
    Returns:
        np.ndarray: The von Neumann entropy of the state.
    """

    state = np.asarray(state, dtype = complex)
    # ket_to_dm will validate the state, no need for redundant check
    state = ket_to_dm(state, batchmode = False)
    eigs = np.linalg.eigvalsh(state)
    entropy = np.sum(-eigs * np.log(eigs), axis = -1)
    if np.shape(entropy) == ():
        entropy = np.array([entropy])
    entropy[np.where(np.isnan(entropy))] = 0
    return entropy