"""
Quantum Computing Utility Functions

This module provides essential utility functions for quantum computing operations.
Functions are organized by functionality for better code navigation.
"""

import numpy as np
from functools import reduce
from itertools import combinations
import string
import random
from scipy.sparse import csc_array, csc_matrix, kron
import itertools
from scipy.sparse.linalg import norm
from typing import Union

# Type aliases for better code readability
MatrixLike = Union[np.ndarray, list]
SparseLike = Union[csc_array, csc_matrix]
MatrixOrSparse = Union[MatrixLike, SparseLike]


# =============================================================================
# MATRIX OPERATIONS
# =============================================================================

def dag(op: MatrixOrSparse):
    """
    Returns the adjoint (Hermitian conjugate) of a given operator.
    
    Supports dense matrices, sparse matrices, and batches of operators.
    
    Args:
        op (MatrixOrSparse): The operator to be adjointed.
            - Single matrix: 2D array or sparse matrix
            - Batch of matrices: list or 3D array with shape (M,N,N)
    Returns:
        MatrixOrSparse: The adjoint of the operator(s).
    """
    if not isinstance(op, (list, np.ndarray, csc_array, csc_matrix)):
        return Exception("op must be a numpy array or a list of numpy arrays.")
    if isinstance(op, (list, np.ndarray)) and len(op) == 0:
        return Exception("op cannot be empty.")
    
    # Handle sparse matrices directly
    if isinstance(op, (csc_array, csc_matrix)):
        return op.conj().T
    
    # Handle lists
    if isinstance(op, list):
        return [np.conj(a).T if isinstance(a, (np.ndarray, csc_array, csc_matrix)) else np.conj(np.asarray(a, dtype = complex)).T for a in op]
    
    # Handle numpy arrays
    if len(op.shape) == 1:
        op = ket_to_dm(op)
    if len(op.shape) == 2:
        return np.conj(op).T
    elif len(op.shape) == 3:
        # 3D array: batch of matrices
        if op.shape[1] != op.shape[2]:
            raise Exception("op must be a square matrix or a list of square matrices.")
        return np.conj(np.transpose(op, (0, 2, 1)))
    else:
        raise Exception("op must be a 2D or 3D array, or a list.")

def is_herm(A: MatrixOrSparse):
    """
    Check if a matrix (or batch of matrices) is Hermitian.
    
    A matrix is Hermitian if A = A† (equals its conjugate transpose).
    
    Args:
        A (MatrixOrSparse): The matrix/matrices to check.
    Returns:
        bool | list: True if Hermitian, False otherwise. For batches, returns list of bools.
    """
    if not isinstance(A, (np.ndarray, list, csc_array, csc_matrix)):
        raise TypeError("Input must be a numpy array, a list of arrays or a list of csc_matrices.")
    
    # Handle single sparse matrix
    if isinstance(A, (csc_array, csc_matrix)):
        return np.allclose(A.toarray(), A.conj().T.toarray())
    if isinstance(A, np.ndarray):
        return np.allclose(A, dag(A))
    # Handle list of matrices
    if isinstance(A, list):
        result = []
        for matrix in A:
            if isinstance(matrix, (csc_array, csc_matrix)):
                result.append(np.allclose(matrix.toarray(), matrix.conj().T.toarray()))
            else:
                matrix = np.asarray(matrix, dtype=complex)
                result.append(np.allclose(matrix, matrix.conj().T))
        return result
    
    # Handle numpy arrays
    A = np.asarray(A, dtype=complex)
    if len(A.shape) == 2:
        # Single matrix
        return np.allclose(A, A.conj().T)
    elif len(A.shape) == 3:
        # Batch of matrices
        return [np.allclose(A[i], A[i].conj().T) for i in range(A.shape[0])]
    else:
        raise ValueError("Input must be a 2D matrix or 3D array of matrices.")


# =============================================================================
# STATE VALIDATION
# =============================================================================

def is_norm(A: MatrixOrSparse):
    """
    Check if a vector or matrix (or batch) is normalized (has unit norm).
    
    Args:
        A (MatrixOrSparse): The vector(s)/matrix(es) to check.
    Returns:
        bool | np.ndarray: True if all normalized, otherwise array indicating which are not.
    """
    if not isinstance(A, (np.ndarray, list, csc_array, csc_matrix)):
        raise TypeError("Input must be a numpy array, a list of arrays, a csc_array or a list of csc_arrays.")
    check_type = isinstance(A, (list, np.ndarray))
    check_csr = isinstance(A, (csc_array, csc_matrix))
    if check_type and np.shape(A[0]) == ():
        check = np.isclose(np.linalg.norm(A), 1)
        return check
    if check_csr:
        check = np.isclose(norm(A), 1)
        return check
    if check_type and isinstance(A[0], (csc_array, csc_matrix)):
        check = np.array([np.isclose(norm(A[i]), 1) for i in range(len(A))])
    elif check_type and isinstance(A[0], (np.ndarray, list)):
        check = np.isclose(np.linalg.norm(A, axis=1), 1)
    if np.any(check == False):
        return check
    else:
        return True
    
    

def is_state(state: np.ndarray, batchmode: bool = True, tol: int = 1e-6):
    """
    Checks if a given state is a valid quantum state and categorizes it.
    
    Categories:
    - Category 1: Single ket (normalized vector)  
    - Category 2: Batch of kets (converted to density matrices)
    - Category 3: Density matrices (single or batch)
    
    Args:
        state (MatrixLike): The quantum state(s) to be checked.
        batchmode (bool): If True, treats 2D arrays as batches of kets. Defaults to True.
    Returns:
        tuple: (category, is_valid)
            - category (int): 1 for ket, 2 for batch of kets, 3 for density matrices
            - is_valid (bool | list): True if all valid, or validity indicators for invalid states
    """
    if not isinstance(state, (list, np.ndarray)):
        raise Exception("State must be a numpy array or a list of numpy arrays.")
    if len(state) == 0:
        raise Exception("State cannot be empty.")
    shape = len(np.shape(state))

    try:
        state = np.array(state, dtype=complex)
    except ValueError:
        raise Exception("States must all have the same dimension.")
    category = 0
    if shape > 3:
        raise Exception("State must be a vector, a square matrix or a list of square matrices.")
    if shape == 3 and state.shape[1] != state.shape[2]:
        raise Exception("State must be a square matrix or a list of square matrices.")

    if shape == 1:
        category = 1
        return category, (len(state.shape) == 1 and np.isclose(np.linalg.norm(state), 1))
    if shape == 2:
        if batchmode:
            state = state[:,np.newaxis] * state[:,:,np.newaxis].conj()
            category = 2
        else:
            category = 3
            state = state[np.newaxis,:,:]
            shape = 3
           
    
    
    check = np.isclose(np.linalg.trace(state), 1)
    check1 = np.all(check)
    if not check1:
        return check
    eigs = np.linalg.eigvalsh(state)
    eigs[np.abs(eigs) < tol] = 0
    check = np.all(eigs>= 0)
    if not check:
        return eigs > 0
    check2 = np.all(check)
    check = is_herm(state)
    if not check:
        return check
    check3 = np.all(check)
    if category == 0:
        category = 3
    return category, (check1 and check2 and check3)

# =============================================================================
# STATE CONVERSION
# =============================================================================
    
def ket_to_dm(state: MatrixLike, batchmode: bool, tol:float = 1e-7) -> np.ndarray:
    """
    Convert a ket to a density matrix.
    
    Args:
        state (MatrixLike): The quantum state in ket form.
        
    Returns:
        np.ndarray: The corresponding density matrix.
    """
    
    check = is_state(state, batchmode, tol)
    if check[0] == 3:
        return state
    if not isinstance(check[0], int):
        raise ValueError("Input contains invalid quantum states at indices: " + str(np.where(check == False)))
    if np.False_ in check:
        raise ValueError("Input must be a valid quantum state.")
    if check[0] == 3:
        return state
    state = np.asarray(state, dtype=complex)
    if state.ndim == 1:
        return np.outer(state, state.conj())
    elif state.ndim == 2 and state.shape[0] == 1:
        return np.outer(state[0], state[0].conj())
    else:
        return np.einsum('ni,nj->nij', state, state.conj())


def operator2vector(state: MatrixLike):
    """
    Vectorizes a quantum state (density matrix).
    Args:
        state (np.ndarray | list): The quantum state to be vectorized in density matrix form.
    Returns:
        np.ndarray: The vectorized form of the quantum state.
    """
    if not isinstance(state, (np.ndarray, list)):
        raise TypeError("Input must be a numpy array or a list of arrays.")
    is_list_of_state = len(np.shape(state)) == 3
    state = np.asarray(state, dtype=complex)
    shape = np.shape(state)[1]
    if is_list_of_state:
        state = np.array([state[i].ravel('F').reshape((shape**2, 1)) for i in range(len(state))], dtype = complex)
    else:
        state = state.ravel('F').reshape((shape**2, 1))
    return state

def vector2operator(state: MatrixLike):
    """
    Converts a vectorized quantum state back to its operator form.
    Args:
        state (MatrixLike): The vectorized quantum state.
    Returns:
        np.ndarray: The operator form of the quantum state.
    """
    if not isinstance(state, (np.ndarray, list)):
        raise TypeError("Input must be a numpy array or a list of arrays.")
    
    state = np.asarray(state, dtype=complex)
    is_list_of_state = len(np.shape(state)) == 3
    if is_list_of_state:
        N = int(0.5*np.log2(np.shape(state)[1]))
        state = np.array([state[i].reshape((2**N, 2**N), order = 'F') for i in range(len(state))], dtype = complex)
    else:
        N = int(0.5*np.log2(np.shape(state)[0]))
        state = state.reshape((2**N,2**N), order='F')
    return state


# =============================================================================
# QUANTUM OPERATIONS
# =============================================================================
    
def nqubit(op: MatrixLike) -> int:
    """
    Returns the number of qubits in a given quantum operator.
    
    Determines the number of qubits from the operator dimension: 2^N × 2^N.
    
    Args:
        op (MatrixLike): The quantum operator (density matrix or state).
    Returns:
        int: The number of qubits in the system.
    """
    if not isinstance(op, (np.ndarray, list)):
        raise TypeError("Input must be a numpy array or a list of arrays.")
    op = np.asarray(op, dtype=complex)
    if len(op.shape) == 1:
        op = ket_to_dm(op, batchmode = False)
    return int(np.log2(op.shape[1])) if isinstance(op, (np.ndarray, (list, np.array))) else 0

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
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

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


def tensor_product(operators: list):
    """
    Computes the tensor product of a list of operators.
    Args:
        operators (list): A list of operators (numpy arrays) to be tensor multiplied.
    Returns:
        np.ndarray: The resulting tensor product of the operators.
    """
    if isinstance(operators[0], (csc_array, csc_matrix)):
        return(reduce(kron, operators))
    else:
        return reduce(np.kron, operators)


def ptrace(rho: MatrixLike, index: list):
    """
    Partial trace of a density matrix rho. The specified indeces are left untraced.
    The remaining indices are traced out.
    Args:
        rho (MatrixLike): The density matrix to be traced.
        index (list): List of indices to keep untraced.
    Returns:
        np.ndarray: The resulting density matrix after partial trace.
    """
    if not isinstance(rho, (list, np.ndarray)):
        raise TypeError("Input must be a list or numpy array.")
    rho = np.asarray(rho, dtype = complex)
    shape = rho.shape

    if len(shape)>2:
        dim = shape[0]

    N = int(np.log2(shape[1]))
    ab = list(string.ascii_lowercase)
    stringa = list(''.join(random.sample(string.ascii_lowercase, N)))
    diff = list(set(ab)-set(stringa))
    stringa = stringa*2
    out = []

    for i in range(len(index)):
        stringa[index[i]] = diff[i]
        out.append(diff[i])

    for i in range(len(index)):
        out.append(stringa[index[i]+N])

    new_shape = [2]*2*N
    shape1 = [2**len(index)]*2

    if len(shape)>2:
        stringa.insert(0, diff[len(index)+1])
        out.insert(0, diff[len(index)+1])
        new_shape.insert(0,dim)
        shape1.insert(0, dim)
        
    out = ''.join(out)
    stringa = ''.join(stringa)
    rho = np.einsum(stringa+'->'+out, rho.reshape(new_shape)).reshape(shape1)
    return rho

# =============================================================================
# ARRAY VALIDATION AND CONVERSION UTILITIES
# =============================================================================

def ensure_complex_array(x: MatrixLike, name: str = "input") -> np.ndarray:
    """
    Convert input to complex numpy array with validation.
    
    Common utility to replace repeated pattern of:
    - Type checking with isinstance
    - Converting to complex numpy array with np.asarray(x, dtype=complex)
    
    Args:
        x (MatrixLike): Input to convert (numpy array or list).
        name (str): Name of the input for error messages.
        
    Returns:
        np.ndarray: Complex numpy array.
        
    Raises:
        TypeError: If input is not a valid array-like type.
    """
    if not isinstance(x, (np.ndarray, list)):
        raise TypeError(f"{name} must be a numpy array or list.")
    return np.asarray(x, dtype=complex)

def validate_matrix_types(*args, names: list = None) -> tuple:
    """
    Validate and convert multiple matrix inputs to complex arrays.
    
    Args:
        *args: Variable number of matrix inputs.
        names (list): Optional names for error messages.
        
    Returns:
        tuple: Converted complex numpy arrays.
        
    Raises:
        TypeError: If any input is not a valid matrix type.
    """
    if names is None:
        names = [f"argument_{i}" for i in range(len(args))]
    elif len(names) != len(args):
        raise ValueError("Number of names must match number of arguments.")
    
    return tuple(ensure_complex_array(arg, name) for arg, name in zip(args, names))