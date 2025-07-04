import numpy as np
from functools import reduce
from itertools import combinations
import string
import random
from scipy.sparse import csc_array, csc_matrix, kron

def dag(op: np.ndarray | list | csc_array | csc_matrix):
    """
    Returns the adjoint (or Hermitian conjugate) of a given operator.
    Args:
        op (np.ndarray): The operator to be adjointed. If op is a list of arrays it is must be in shape (M,N,N)
        the function will return the adjoint of each operator in the list.
    Returns:
        np.ndarray: The adjoint of the operator.
    """
    if not isinstance(op, (list, np.ndarray, csc_array, csc_matrix)):
        return Exception("op must be a numpy array or a list of numpy arrays.")
    if isinstance(op, (list)):
        op = np.asarray(op, dtype=complex)
    shape = len(np.shape(op)) > 2

    if shape:
        shape = np.shape(op)

        if shape[1] != shape[2]:
            raise Exception("op must be a square matrix or a list of square matrices.")
        
        return np.conj(np.transpose(op, (0,2,1)))
    
    else:
        return np.conj(op).T

def is_herm(A: np.ndarray | list | csc_array | csc_matrix):
    """
    Check if a matrix is Hermitian.
    Args:
        A (np.ndarray or list): The matrix to check.
    Returns:
        bool: True if the matrix is Hermitian, False otherwise.
    """

    if not isinstance(A, (np.ndarray,list, csc_array, csc_matrix)):
        raise TypeError("Input must be a numpy array, a list of arrays or a list of csc_matrices.")
    if isinstance(A, (csc_array, csc_matrix)):
        B = A.toarray()
    elif isinstance(A, list) and isinstance(A[0], (csc_array, csc_matrix)):
        B = [a.toarray() for a in A]
    else:
        B = A
    B = np.asarray(B, dtype=complex)
    return(np.allclose(B, dag(B)))

def is_norm(A: np.ndarray | list):
    """
    Check if a vector or matrix is normalized.
    Args:
        A (np.ndarray): The vector or matrix to check.
        ax (tuple): Axis along which to compute the norm.
    Returns:
        bool: True if the vector or matrix is normalized, False otherwise.
    """
    if not isinstance(A, (np.ndarray, list)):
        raise TypeError("Input must be a numpy array or a list of arrays.")
    
    category = is_state(A)[0]
    if category == 1:
        return np.isclose(np.linalg.norm(A), 1)
    elif category == 2:
        A = np.asarray(A, dtype=complex)
        return np.all(np.isclose(np.linalg.norm(A, axis = 1), 1))
    else:
        return np.all(np.isclose(np.linalg.norm(A, axis = (1,2)), 1))

def is_state(state: np.ndarray | list):
    """
    Checks if a given state is a valid quantum state.
    Args:
        state (np.ndarray): The quantum state to be checked.
    Returns:
        bool: True if the state is a valid quantum state, False otherwise.
    """
    if not isinstance(state, (list, np.ndarray)):
        raise Exception("state must be a numpy array or a list of numpy arrays.")
    shape = len(np.shape(state))
    state = np.array(state, dtype=complex)
    category = 0
    if shape > 3:
        raise Exception("State must be a vector, a square matrix or a list of square matrices.")
    
    if shape == 1:
        category = 1
        return category, (len(state.shape) == 1 and np.isclose(np.linalg.norm(state), 1))
    
    if shape > 1:

        if  shape == 2 and state.shape[0] != state.shape[1]:
            state = state[:,np.newaxis] * state[:,:,np.newaxis].conj()
            category = 2
        if shape == 3 and state.shape[1] != state.shape[2]:
            raise Exception("State must be a square matrix or a list of square matrices.")

        check1 = np.all(np.isclose(np.linalg.trace(state), 1))
        eigs = np.linalg.eigvalsh(state)
        tol = 1e-10
        eigs[np.abs(eigs) < tol] = 0
        check2 = np.all(eigs>= 0)
        check3 = is_herm(state)
        if category == 0:
            category = 3
        return category, (check1 and check2 and check3)
    
def ket_to_dm(state: np.ndarray | list) -> np.ndarray:
    """
    Convert a ket to a density matrix.
    
    Args:
        state (np.ndarray | list): The quantum state in ket form.
        
    Returns:
        np.ndarray: The corresponding density matrix.
    """
    
    check = is_state(state)
    if not check[1]:
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
    
def nqubit(op: np.ndarray | list) -> int:
    """
    Returns the number of qubits in a given density matrix.
    Args:
        op (np.ndarray | list): The density matrix to check.
    Returns:
        int: The number of qubits in the density matrix."""
    if not isinstance(op, (np.ndarray, list)):
        raise TypeError("Input must be a numpy array or a list of arrays.")
    op = np.asarray(op, dtype=complex)
    if is_state(op)[1]:
        op = ket_to_dm(op)
    
    return int(np.log2(op.shape[1])) if isinstance(op, (np.ndarray, (list, np.array))) else 0

def ptrace(rho: np.ndarray | list, index: list):
    """
    Partial trace of a density matrix rho. The specified indeces are left untraced.
    The remaining indices are traced out.
    Args:
        rho (np.ndarray | list): The density matrix to be traced.
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
