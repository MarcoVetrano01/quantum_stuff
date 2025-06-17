import numpy as np
from .utils import dag, is_state, is_herm, is_norm, tensor_product, ket_to_dm, ptrace
import warnings

warnings.filterwarnings(
    action='ignore',
    category=RuntimeWarning,
    message=r'.*invalid value encountered in log2.*'
)

def fidelity(state1: np.ndarray | list, state2: np.ndarray | list):
    """
    Calculate the fidelity between two quantum states.
    The fidelity is defined as:
    F(ρ, σ) = Tr(sqrt(sqrt(ρ) σ sqrt(ρ)))
    Args:
        state1 (np.ndarray | list): First quantum state or list of quantum states.
        state2 (np.ndarray | list): Second quantum state or list of quantum states.
    Returns:
        float: Fidelity between the two quantum states.
    """
    check1 = is_state(state1)
    check2 = is_state(state2)
    if not (check1[1] and check2[1]):
        raise TypeError("Both states must be quantum states.")
    if np.shape(state1) != np.shape(state2):
        raise ValueError("Both states must have the same shape.")
    
    state1 = np.asarray(state1, dtype = complex)
    state2 = np.asarray(state2, dtype = complex)

    if check1[0] == 1 or check1[0] == 2:
        state1 = ket_to_dm(state1)
        state2 = ket_to_dm(state2)

    eigs = np.linalg.eigvals(state1 @ state2)
    if len(state1.shape) != 2:
        return np.sum(np.sqrt(eigs), axis = 1)
    else:
        return(np.sum(np.sqrt(eigs)))

def Holevo_Info(states: np.ndarray | list, probabilities: np.ndarray | list):

    """
    Compute the Holevo information for an ensemble of quantum states.
    The Holevo information quantifies the amount of classical information that can be extracted from a quantum system.
    Given a set of quantum states and their corresponding probabilities, it measures the difference between the average entropy of the states and the entropy of the average state.
    η = ∑ p_i * ρ_i
    χ(η) = S(η) - ∑ p_i * S(ρ_i)
    where S(ρ) is the von Neumann entropy of the state ρ, p_i is the probability of state ρ_i, and η is the average state over the ensemble.
    Args:
        states (np.ndarray | list): A list or array of quantum states (density matrices).
        probabilities (np.ndarray | list): A list or array of probabilities corresponding to each state.
    Returns:
        float: The Holevo information of the ensemble.
    """
    check = is_state(states)
    if not check[1]:
        raise ValueError("states must be a list or array of quantum states.")
    states = ket_to_dm(states)
    if not isinstance(probabilities, (np.ndarray, list)):
        raise ValueError("Probabilities must be a list or array of probabilities.")
    if np.sum(probabilities) != 1:
        raise ValueError("Probabilities must sum to 1.")
    
    eta = np.average(states, weights= probabilities, axis = 0)
    return von_neumann_entropy(eta) - np.average(von_neumann_entropy(states), weights = probabilities, axis = 0)


def mutual_info(state_total: np.ndarray | list, state_A: np.ndarray | list, state_B: np.ndarray | list):
    """
    Calculate the mutual information between two subsystems A and B given a total state.
    The mutual information is defined as:
    I(A:B) = S(A) + S(B) - S(A:B)

    where S(X) is the von Neumann entropy of subsystem X.
    Args:
        state_total (np.ndarray | list): The total quantum state, can be a density matrix or a ket.
        state_A (np.ndarray | list): The quantum state of subsystem A.
        state_B (np.ndarray | list): The quantum state of subsystem B.
    Returns:
        float: The mutual information I(A:B).
    """
    check1 = is_state(state_A)
    check2 = is_state(state_B)
    check3 = is_state(state_total)
    if not (check1[1] and check2[1] and check3[1]):
        raise ValueError("All states must be valid quantum states.")
    Na = int(np.log2(state_A.shape[-1]))
    Nb = int(np.log2(state_B.shape[-1]))
    if Na + Nb != int(np.log2(state_total.shape[-1])):
        raise ValueError("The dimensions of the states do not match: "
                         "dim(A) + dim(B) must equal dim(total).")
    dimsA = state_A.ndim
    dimsB = state_B.ndim
    if dimsA != dimsB:
        raise ValueError("The dimensions of the states A and B must match.")
    state_A = ket_to_dm(state_A)
    state_B = ket_to_dm(state_B)
    state_total = ket_to_dm(state_total)
    
    state_total = np.asarray(state_total, dtype = complex)
    state_A = np.asarray(state_A, dtype = complex)
    state_B = np.asarray(state_B, dtype = complex)
    
    return von_neumann_entropy(state_A) + von_neumann_entropy(state_B) - von_neumann_entropy(state_total)

def purity(state: np.ndarray | list):
    """
    Computes the purity of a quantum state.
    The purity is defined as the trace of the square of the density matrix.
    For a pure state, the purity is 1, and for a mixed state, it is less than 1 and decreases to a minimum of 1/2^Nqbits, where Nqbits is the number of qubits.
    Args:
        state (np.ndarray | list): The density matrix of the quantum state.
    Returns:
        float: The purity of the quantum state.
    """
    check = is_state(state)
    if not check[1]:
        raise Exception("Input must be a quantum state or a list of quantum states.")
    if check[0] != 3:
        state = ket_to_dm(state)
    state = np.asarray(state, dtype = complex)
    return np.linalg.trace(state @ state)

def trace_distance(state1: np.ndarray | list, state2: np.ndarray | list | None = None):
    """
    Calculate the trace distance between two quantum states.
    The trace distance is defined as:
    T(ρ, σ) = 1/2 * ||ρ - σ||_1

    where ||.||_1 is the trace norm.
    Args:
        state1 (np.ndarray | list): First quantum state (density matrix) or difference between two quantum states.
        state2 (np.ndarray | list | None): Second quantum state (density matrix). If None, state1 should be already the difference between two quantum states.
    Returns:
        float: Trace distance between the two quantum states.
    """

    if not isinstance(state1, (list,np.ndarray)):
        raise TypeError("State 1 must be a list or numpy array")
    if not isinstance(state2, (np.ndarray, list, None)):
        raise TypeError("State 2 must be a list or numpy array or None")
    if state2 is not None:
        check1 = is_state(state1)
        check2 = is_state(state2)
        if not (check1[1] or check2[1]):
            raise TypeError("Both states must be valid quantum states")
        if np.shape(state1) != np.shape(state2):
            raise ValueError("Both states must have the same dimensions")
        
        else:
            state1 = np.asarray(state1, dtype = complex)
            state2 = np.asarray(state2, dtype = complex) 
            state1 = ket_to_dm(state1)
            state2 = ket_to_dm(state2)
            dist = state1 - state2
    else:
        state1 = np.asarray(state1, dtype = complex)
        dist = state1
    return 0.5*np.sqrt(np.linalg.trace(dist @ dag(dist)))

def von_neumann_entropy(state: np.ndarray):
    """
    Calculate the von Neumann entropy of a quantum state.
    The von Neumann entropy is defined as:
        S(ρ) = -Tr(ρ log2(ρ))

    where ρ is the density matrix of the quantum state.
    Args:
        state (np.ndarray): The quantum state, can be a density matrix or a pure state.
        ax (int): Axis along which to compute the entropy. Default is -1.
    Returns:
        np.ndarray: The von Neumann entropy of the state.
    """

    check = is_state(state)
    if not check[1]:
        raise Exception("Input is not a valid quantum state")
    if check[0] != 3:
        state = ket_to_dm(state)

    state = np.asarray(state, dtype = complex)

    eigs = np.linalg.eigvalsh(state)
    entropy = np.sum(-eigs * np.log2(eigs), axis = -1)
    if np.shape(entropy) == ():
        entropy = np.array([entropy])
    entropy[np.where(np.isnan(entropy))] = 0
    return entropy