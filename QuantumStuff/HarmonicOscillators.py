import numpy as np
from .utils import dag
from strawberryfields.decompositions import williamson
from scipy.linalg import expm
# To be updated and tested in the future
hbar = 1
m = 1

def create(size: int):
    """
    Creates the ladder operator of a given size.
    Args:
        size (int): The size of the ladder operator.
    Returns:
        np.ndarray: The ladder operator of the given size.
    """
    if not isinstance(size, int) or size < 1:
        raise Exception("Size must be a positive integer.")
    
    a = np.diag(np.sqrt(np.arange(1, size)), k=1)
    a = np.asarray(a, dtype=complex)
    return a

def destroy(size: int):
    """
    Creates the annihilation operator of a given size.
    Args:
        size (int): The size of the annihilation operator.
    Returns:
        np.ndarray: The annihilation operator of the given size.
    """
    if not isinstance(size, int) or size < 1:
        raise Exception("Size must be a positive integer.")
    
    return dag(create(size))

def is_gaussian(cov: np.ndarray):
    th, s = williamson(cov)
    return np.allclose(s@th@s.T, cov)

def momentum(omega: np.ndarray, size: int):
    """
    Creates the momentum operator for a harmonic oscillator.
    Args:
        omega (np.ndarray): The angular frequency of the harmonic oscillator.
        size (int): The size of the momentum operator.
    Returns:
        np.ndarray: The momentum operator for a harmonic oscillator.
    """
    return np.imag(1)*np.sqrt(hbar*m*omega/2)*(create(size) - destroy(size))

def number_operator(size: int):
    """
    Creates the number operator of a given size.
    Args:
        size (int): The size of the number operator.
    Returns:
        np.ndarray: The number operator of the given size.
    """
    if not isinstance(size, int) or size <= 0:
        raise Exception("Size must be a positive integer.")
    
    return destroy(size) @ create(size)

def position(omega: np.ndarray, size: int):
    """
    Creates the position operator for a harmonic oscillator.
    Args:
        omega (np.ndarray): The angular frequency of the harmonic oscillator.
        size (int): The size of the position operator.
    Returns:
        np.ndarray: The position operator for a harmonic oscillator.
    """
    return np.sqrt(hbar/(2*m*omega))*(create(size) + destroy(size))

def ptr_cov(cov: np.ndarray, nmode: int):
    return np.array([cov[2*nmode,2*nmode:2*nmode+2], cov[2*nmode+1,2*nmode:2*nmode+2]])

def random_gaussian_state(size: int):
    mean_xp = np.random.randn(2 * size)
    M = np.random.randn(2 * size, 2 * size)
    cov_matrix = M @ M.T
    epsilon = 0.1
    cov_matrix += epsilon * np.eye(2 * size)
    initial_state = np.random.multivariate_normal(mean_xp, cov_matrix)
    return initial_state, cov_matrix

def quantum_network_hamiltonian(omega: np.ndarray, L: np.ndarray, n_modes: int, dt: float):

    J = np.block([
        [np.zeros((n_modes, n_modes)), np.eye(n_modes)],
        [np.eye(n_modes), np.zeros((n_modes,n_modes))]
    ])

    H = np.block([
        [omega**2 + L, np.zeros((n_modes,n_modes))],
        [np.zeros((n_modes,n_modes)), np.eye(n_modes)]
    ])

    H = J@H
    S = expm(H*dt)
    return H, S
def random_gaussian_state(size: int):
    mean_xp = np.random.randn(2 * size)
    M = np.random.randn(2 * size, 2 * size)
    cov_matrix = M @ M.T
    epsilon = 0.1
    cov_matrix += epsilon * np.eye(2 * size)
    initial_state = np.random.multivariate_normal(mean_xp, cov_matrix)
    return initial_state, cov_matrix