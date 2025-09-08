import numpy as np
from .utils import is_state, tensor_product, ket_to_dm, ptrace, nqubit
from itertools import combinations
from scipy.sparse import csc_array, kron, csc_matrix

def anticommutator(A: np.ndarray | list, B: np.ndarray | list):
    """
    Anticommutator of two matrices A and B.
    Args:
        A (np.ndarray): First matrix.
        B (np.ndarray): Second matrix.
    Returns:
        np.ndarray: The anticommutator of A and B.
    """
    if not isinstance(A, (list, np.ndarray)):
        raise Exception("Both A and B must be numpy arrays or lists of numpy arrays.")
    if not isinstance(B, (list, np.ndarray)):
        raise Exception("Both A and B must be numpy arrays or lists of numpy arrays.")
    
    A = np.asarray(A, dtype = complex)
    B = np.asarray(B, dtype = complex)
    return A @ B + B @ A

def commutator(A: np.ndarray | list, B: np.ndarray | list):
    """
    Commutator of two matrices A and B.
    Args:
        A (np.ndarray): First matrix.
        B (np.ndarray): Second matrix.
    Returns:
        np.ndarray: The commutator of A and B.
    """
    if not isinstance(A, (list, np.ndarray)):
        raise Exception("Both A and B must be numpy arrays or lists of numpy arrays.")
    if not isinstance(B, (list, np.ndarray)):
        raise Exception("Both A and B must be numpy arrays or lists of numpy arrays.")
    if A.shape != B.shape:
        raise Exception("A and B must have the same shape.")
    
    A = np.asarray(A, dtype=complex)
    B = np.asarray(B, dtype=complex)
    
    return A @ B - B @ A

def haar_random_unitary(n_qubits):
    """
    Generate a Haar random unitary matrix for n_qubits.
    
    Parameters:
    - n_qubits (int): Number of qubits
    
    Returns:
    - numpy.ndarray: A Haar random unitary matrix
    """
    if (not isinstance(n_qubits, int) or n_qubits<1):
        raise Exception("n_qubits must be a positive integer")
    dim = 2**n_qubits
    
    # Generate Haar random unitary
    Z = (np.random.randn(dim, dim) + 1j*np.random.randn(dim, dim)) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    # Fix phases to get Haar distribution
    phases = np.diag([R[i,i]/abs(R[i,i]) for i in range(dim)])
    U = Q @ phases
    
    return U

def local_operators(operator: np.ndarray | csc_array | csc_matrix, N: int):
    """
    Creates a list of local operators for a given operator and number of qubits.
    Args:
        operator (np.ndarray): The operator to be replicated.
        N (int): The number of qubits.
    Returns:
        np.ndarray: A list of local operators for the given operator and number of qubits.
    """
    if not isinstance(operator, (np.ndarray, csc_array, csc_matrix)):
        raise Exception("Operator must be a numpy array.")
    if not isinstance(N, int) or N <= 0:
        raise Exception("N must be a positive integer.")
    if isinstance(operator, (np.ndarray)):
        op = [np.eye(2)]*N
        result = np.zeros((N, 2**N, 2**N), dtype = np.complex128)
        for i in range(N):
            op[i] = operator
            result[i] = tensor_product(op)
            op[i] = np.eye(2)
    else:
        op = [csc_array(np.eye(2))]*N
        result = []
        for i in range(N):
            op[i] = operator
            result.append(tensor_product(op))
            op[i] = csc_array(np.eye(2))
            result[i] = csc_array(result[i])
    
    return result

def measure(states: list | np.ndarray, operators: list, indices_list: list):
    """
    Measure a list of quantum states with a list of operators on multiple sets of indices.

    Args:
        states (list of np.ndarray): List of density matrices representing quantum states.
        operators (list of np.ndarray): List of operators to measure.
        indices_list (list of list of list of int): 
            For each operator, a list of index sets where the operator acts.
            Example: [ [[0], [1], [2]], [[0,1]], ... ]
        partial_trace_fn (function): Function that takes (states, indices_to_keep)
                                     and returns the reduced density matrices.

    Returns:
        np.array: For each operator, a list of lists of measurement results (one list per index set).
              Shape: [operator][index_set][state]
    """
    
    if not is_state(states)[1]:
        raise ValueError("Input must be a quantum state or a list of quantum states.")
    states = ket_to_dm(states)
    if not isinstance(operators, list) or not all(isinstance(op, np.ndarray) for op in operators):
        raise ValueError("Operators must be a list of numpy arrays.")
    if not isinstance(indices_list, list) or not all(isinstance(indices, list) for indices in indices_list):
        raise ValueError("Indices list must be a list of lists of indices.")
    if len(operators) != len(indices_list):
        raise ValueError("Operators and indices_list must have the same length.")
    if len(np.shape(states)) == 2:
        states = states[np.newaxis]
    for i in range(len(indices_list)):
        if len(indices_list[i]) == 0:
            nq = nqubit(operators[i])
            nrho = nqubit(states)
            indices_list[i] = combinations(range(nrho), nq)
    all_measurements = []

    for operator, list_of_index_groups in zip(operators, indices_list):
        for indices in list_of_index_groups:
            reduced_states = ptrace(states, indices)
            exp_vals = [np.linalg.trace(state @ operator) for state in reduced_states]
            all_measurements.append(exp_vals)

    return np.array(all_measurements).T


def sigmap():
    '''
    Creates the raising operator (sigma plus) for a qubit.
    Returns:
        np.ndarray: The raising operator (sigma plus) for a qubit.
    '''
    sp = np.array([[0,1],[0,0]], dtype = complex)
    return sp

def sigmam():
    '''
    Creates the lowering operator (sigma minus) for a qubit.
    Returns:
        np.ndarray: The lowering operator (sigma minus) for a qubit.
    '''

    sm = np.array([[0,0],[1,0]], dtype = complex)
    return sm

def sigmax():
    '''
    Creates the Pauli X operator (sigma x) for a qubit.
    Returns:
        np.ndarray: The Pauli X operator (sigma x) for a qubit.
    '''

    sx = np.array([[0,1], [1,0]], dtype = complex)
    return sx

def sigmay():
    '''
    Creates the Pauli Y operator (sigma y) for a qubit.
    Returns:
        np.ndarray: The Pauli Y operator (sigma y) for a qubit.
    '''

    sy = np.array([[0,-1j],[1j,0]], dtype = complex)
    return sy

def sigmaz():
    '''
    Creates the Pauli Z operator (sigma z) for a qubit.
    Returns:
        np.ndarray: The Pauli Z operator (sigma z) for a qubit.
    '''

    sz = np.array([[1,0],[0,-1]], dtype = complex)
    return sz