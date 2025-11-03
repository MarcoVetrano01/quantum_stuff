"""
Quantum Operators and Measurements Module

This module provides functions for quantum operators, measurements, and algebraic operations.
Functions are organized by functionality for better code navigation.
"""

import numpy as np
from .utils import is_state, tensor_product, ket_to_dm, ptrace, nqubit, dag, validate_matrix_types, MatrixLike, MatrixOrSparse, SparseLike
from itertools import combinations
from scipy.special import comb
from scipy.sparse import csc_array, kron, csc_matrix, dia_matrix
from typing import Union
import itertools

# =============================================================================
# PAULI OPERATORS
# =============================================================================

def sigmam() -> np.ndarray:
    """
    Create the Pauli raising operator σ₋ = (σₓ + iσᵧ)/2.
    
    Returns:
        np.ndarray: 2×2 raising operator matrix |0⟩⟨1|.
    """
    return np.array([[0, 1], [0, 0]], dtype=complex)

def sigmap() -> np.ndarray:
    """
    Create the Pauli lowering operator σ₊ = (σₓ - iσᵧ)/2.
    
    Returns:
        np.ndarray: 2×2 lowering operator matrix |1⟩⟨0|.
    """
    return np.array([[0, 0], [1, 0]], dtype=complex)

def sigmax() -> np.ndarray:
    """
    Create the Pauli X operator (bit-flip gate).
    
    Returns:
        np.ndarray: 2×2 Pauli X matrix [[0,1],[1,0]].
    """
    return np.array([[0, 1], [1, 0]], dtype=complex)

def sigmay() -> np.ndarray:
    """
    Create the Pauli Y operator (bit and phase flip gate).
    
    Returns:
        np.ndarray: 2×2 Pauli Y matrix [[0,-i],[i,0]].
    """
    return np.array([[0, -1j], [1j, 0]], dtype=complex)

def sigmaz() -> np.ndarray:
    """
    Create the Pauli Z operator (phase-flip gate).
    
    Returns:
        np.ndarray: 2×2 Pauli Z matrix [[1,0],[0,-1]].
    """
    return np.array([[1, 0], [0, -1]], dtype=complex)

def pauli_basis(N, normalized=True):
    """
    Build the full N-qubit Pauli basis.

    Args:
        N (int): Number of qubits
        normalized (bool): If True, returns orthonormal basis w.r.t. Hilbert–Schmidt inner product

    Returns:
        basis (list of np.ndarray): List of 2^N x 2^N matrices forming the basis
        labels (list of str): Corresponding Pauli string labels
    """
    # Single-qubit Paulis
    I = np.eye(2, dtype=complex)
    X = sigmax()
    Y = sigmay()
    Z = sigmaz()

    paulis = [I, X, Y, Z]
    labels_single = ["I", "X", "Y", "Z"]

    basis = []
    labels = []

    # Cartesian product of N choices from {I,X,Y,Z}
    for prod in itertools.product(range(4), repeat=N):
        mat = paulis[prod[0]]
        label = labels_single[prod[0]]
        for idx in prod[1:]:
            mat = np.kron(mat, paulis[idx])
            label += labels_single[idx]
        if normalized:
            mat = mat / np.sqrt(2**N)  # ensure orthonormality
        basis.append(mat)
        labels.append(label)

    return basis, labels

def hadamard() -> np.ndarray:
    """
    Create the Hadamard operator H.
    
    Returns:
        np.ndarray: 2×2 Hadamard matrix (1/sqrt(2))*[[1,1],[1,-1]].
    """
    return (1/np.sqrt(2)) * (sigmax() + sigmaz())

def proj1() -> np.ndarray:
    """
    Create the projection operator |1⟩⟨1|.
    
    Returns:
        np.ndarray: 2×2 projection matrix [[0,0],[0,1]].
    """
    return 0.5 * (np.eye(2) - sigmaz())

def proj0() -> np.ndarray:
    """
    Create the projection operator |0⟩⟨0|.
    
    Returns:
        np.ndarray: 2×2 projection matrix [[1,0],[0,0]].
    """
    return 0.5 * (np.eye(2) + sigmaz())

def CNOT() -> np.ndarray:
    """
    Create the CNOT (Controlled-NOT) gate for two qubits.
    
    Returns:
        np.ndarray: 4×4 CNOT gate matrix.
    """
    return tensor_product([proj0(), np.eye(2)]) + tensor_product([proj1(), sigmax()])

# =============================================================================
# ALGEBRAIC OPERATIONS
# =============================================================================

def anticommutator(A:MatrixOrSparse, B:MatrixOrSparse) -> np.ndarray:
    """
    Compute the anticommutator {A, B} = AB + BA of two matrices.
    
    Args:
        A (MatrixOrSparse): First matrix operator.
        B (MatrixOrSparse): Second matrix operator.
        
    Returns:
        np.ndarray: The anticommutator {A, B} = AB + BA.
        
    Raises:
        Exception: If inputs are not numpy arrays or lists.
    """
    from scipy.sparse import issparse
    if not isinstance(A, (list, np.ndarray)) and not issparse(A):
        raise Exception("Both A and B must be numpy arrays, lists, or sparse matrices.")
    if not isinstance(B, (list, np.ndarray)) and not issparse(B):
        raise Exception("Both A and B must be numpy arrays, lists, or sparse matrices.")
    if A.shape != B.shape:
        raise Exception("A and B must have the same shape.")
    if issparse(A) or issparse(B):
        A = csc_array(A, dtype=complex)
        B = csc_array(B, dtype=complex)
        return A @ B + B @ A
    A = np.asarray(A, dtype=complex)
    B = np.asarray(B, dtype=complex)
    return A @ B + B @ A
    return A @ B + B @ A

def commutator(A:MatrixOrSparse, B:MatrixOrSparse) -> np.ndarray:
    """
    Compute the commutator [A, B] = AB - BA of two matrices.
    
    Args:
        A (MatrixOrSparse): First matrix operator.
        B (MatrixOrSparse): Second matrix operator.
        
    Returns:
        np.ndarray: The commutator [A, B] = AB - BA.
        
    Raises:
        Exception: If inputs are not numpy arrays/lists or have different shapes.
    """
    from scipy.sparse import issparse
    if not isinstance(A, (list, np.ndarray)) and not issparse(A):
        raise Exception("Both A and B must be numpy arrays, lists, or sparse matrices.")
    if not isinstance(B, (list, np.ndarray)) and not issparse(B):
        raise Exception("Both A and B must be numpy arrays, lists, or sparse matrices.")
    if A.shape != B.shape:
        raise Exception("A and B must have the same shape.")
    if issparse(A) or issparse(B):
        A = csc_array(A, dtype=complex)
        B = csc_array(B, dtype=complex)
        return A @ B - B @ A  # Fixed: was + instead of -
    A = np.asarray(A, dtype=complex)
    B = np.asarray(B, dtype=complex)
    
    return A @ B - B @ A

# =============================================================================
# EXPECTATION VALUES AND MEASUREMENTS
# =============================================================================

def expect(state: MatrixLike, op: np.ndarray, batchmode: bool = True, tol: float = 1e-7) -> Union[float, np.ndarray]:
    """
    Calculate the expectation value ⟨ψ|O|ψ⟩ or Tr(ρO) of an operator.
    
    Args:
        state (MatrixLike): Quantum state as ket vector or density matrix.
        op (np.ndarray): Operator to measure.
        batchmode (bool): If True, allows batch processing of multiple states.
        
    Returns:
        Union[float, np.ndarray]: Expectation value(s) - scalar for single state,
                                 array for batch processing.
                                 
    Raises:
        ValueError: If input is not a valid quantum state.
    """
    state = np.array(state, dtype = complex)
    state = ket_to_dm(state, batchmode, tol = tol)
    l = np.shape(state) 

    if len(l) == 2:
        return np.trace(np.matmul(op, state))
    else:
        return np.einsum('ijk,kj->i', state, op)

def local_measurements(rho: np.ndarray, operators: list = [sigmax(), sigmay(), sigmaz()], batchmode: bool = True) -> np.ndarray:
    """
    Perform optimized local Pauli measurements on each qubit.
    
    Args:
        rho (np.ndarray): Density matrix of quantum state. Shape (d,d) or (n,d,d).
        
    Returns:
        np.ndarray: Measurement results array with shape (n_states, n_qubits*3).
                   For each qubit: [⟨σx⟩, ⟨σy⟩, ⟨σz⟩]
    """
    rho = np.array(rho, dtype = complex)
    
    Nq = int(np.log2(rho.shape[1]))
    shape = rho.shape
    if len(shape) > 2:
        dim = shape[0]
    else:
        rho = rho[np.newaxis]
        dim = 1
    out = np.zeros((dim, Nq, 3), dtype = complex)
    out = []
    for i in range(Nq):
        rho_red = ptrace(rho, [i])
        for k in range(len(operators)):        
            out.append(expect(rho_red, operators[k]))
    return np.array(out).T

def measure(states: MatrixLike, operators: list, indices_list: list, batchmode: bool = True) -> np.ndarray:
    """
    Measure quantum states with operators on specified qubit indices.

    Args:
        states (MatrixLike): List of density matrices representing quantum states.
        operators (list): List of operators to measure.
        indices_list (list): For each operator, list of index sets where operator acts.
                           Example: [[[0], [1], [2]], [[0,1]], ...]
        batchmode (bool): Whether to use batch processing mode.

    Returns:
        np.ndarray: Measurement results with shape [n_states, n_measurements].
                   Each column represents measurement results for one operator-index combination.
                   
    Raises:
        ValueError: If inputs are invalid quantum states or incompatible dimensions.
    """
    states = ket_to_dm(states, batchmode)
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

def two_qubits_measurements(ρ: np.ndarray, operators: list) -> np.ndarray:
    """
    Optimized two-qubit correlation measurements on all qubit pairs.
    
    Args:
        ρ (np.ndarray): Density matrix with shape (d,d) or (n,d,d) for n states.
        operators (list): List of two-qubit operators (4×4 matrices).
        
    Returns:
        np.ndarray: Measurement results with shape (n_states, n_pairs*n_operators).
                   Results for all C(n_qubits,2) pairs and all operators.
    """
    Nq = int(np.log2(ρ.shape[1]))
    shape = ρ.shape
    if len(shape) > 2:
        dim = shape[0]
    else:
        ρ = ρ[np.newaxis]
        dim = 1
    out = np.zeros((dim, int(len(operators)*comb(Nq, 2))), dtype = complex)
    for i, j in enumerate(combinations(range(Nq),2)):
        ρ_red = ptrace(ρ, list(j))
        for k in range(len(operators)):
            out[:, int(comb(Nq,2)) * k + i] = np.real(np.trace(operators[k]@ρ_red, axis1 = 1, axis2 = 2))
    return out

# =============================================================================
# OPERATOR CONSTRUCTION AND MANIPULATION  
# =============================================================================

def local_operators(operator: MatrixOrSparse, N: int) -> Union[np.ndarray, list]:
    """
    Create local operators acting on individual qubits in N-qubit system.
    
    Args:
        operator (MatrixOrSparse): Single-qubit operator to apply locally.
        N (int): Total number of qubits in the system.
        
    Returns:
        Union[np.ndarray, list]: Array of local operators, one for each qubit.
                                For sparse inputs, returns list of sparse matrices.
                                
    Raises:
        Exception: If operator type is invalid or N is not a positive integer.
    """
    if not isinstance(operator, (np.ndarray, csc_array, csc_matrix)):
        raise Exception("Operator must be a numpy array or sparse matrix.")
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

def haar_random_unitary(n_qubits: int) -> np.ndarray:
    """
    Generate a Haar random unitary matrix for quantum systems.
    
    Args:
        n_qubits (int): Number of qubits (must be positive integer).
        
    Returns:
        np.ndarray: Haar random unitary matrix of dimension 2^n_qubits.
        
    Raises:
        Exception: If n_qubits is not a positive integer.
    """
    if (not isinstance(n_qubits, int) or n_qubits<1):
        raise Exception("n_qubits must be a positive integer")
    dim = 2**n_qubits
    
    # Generate Haar random unitary using QR decomposition
    Z = (np.random.randn(dim, dim) + 1j*np.random.randn(dim, dim)) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    # Fix phases to get proper Haar distribution
    phases = np.diag([R[i,i]/abs(R[i,i]) for i in range(dim)])
    U = Q @ phases
    
    return U