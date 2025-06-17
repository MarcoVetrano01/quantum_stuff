import numpy as np
from .Operators import haar_random_unitary, measure, sigmax, sigmay, sigmaz
from .utils import dag, is_state
from functools import reduce
import plotly.graph_objects as go
import string
import random

def BlochSpherePlot(states: np.ndarray | list):
    """
    Plot points on a Bloch sphere using Plotly.
    Args:
        states : np.ndarray or list
            A list or array of quantum states (density matrices or pure states).
    """
    if not isinstance(states, (np.ndarray, list)):
        raise TypeError("Input states must be a numpy array or a list.")
    
    # Sphere mesh (low resolution for better performance)
    vectors = bloch_vector(states).T
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    u, v = np.meshgrid(u, v)
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    # Sphere surface
    sphere = go.Surface(x=x, y=y, z=z, opacity=0.3, colorscale='Blues', showscale=False)

    # Extract vector components
    vx, vy, vz = zip(*vectors)

    # Points on the sphere
    points = go.Scatter3d(
        x=vx, y=vy, z=vz,
        mode='markers',
        marker=dict(size=2),
        name='Points'
    )

    fig = go.Figure(data=[sphere, points])
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[-1.2, 1.2]),
            yaxis=dict(nticks=4, range=[-1.2, 1.2]),
            zaxis=dict(nticks=4, range=[-1.2, 1.2]),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title='Bloch Sphere Plot',
    )
    fig.show()

def bloch_vector(rho: np.ndarray | list):
    """
    Calculate the Bloch vector for a given quantum state ρ.
    The state can be a pure state (ket) or a mixed state (density matrix).
    Args:
        ρ (np.ndarray): The quantum state, either a ket or a density matrix or a list of states.
    Returns:
        np.ndarray: The Bloch vector components (x, y, z).
    """

    return np.transpose(np.real(measure(rho, [sigmax(), sigmay(), sigmaz()])), (0,2,1))



def left(dm = False, N: int = 1):
    """
    Creates the left state for a qubit.
    Args:
        dm (bool): If True, returns the density matrix representation.
        N (int): Number of qubits.
    Returns:
        np.ndarray: The left state for a qubit, either as a vector or a density matrix.
    """

    l = (1/np.sqrt(2))*(zero()+1j*one())
    if N != 1:
        l = [l]*N
        l = reduce(np.kron, l)
    if dm == False:
        return l
    else:
        return np.outer(l, l.conj())
    
def minus(dm = False, N: int = 1):
    """
    Creates the minus state for a qubit.
    Args:
        dm (bool): If True, returns the density matrix representation.
        N (int): Number of qubits.
    Returns:
        np.ndarray: The minus state for a qubit, either as a vector or a density matrix.
    """

    meno = 1/np.sqrt(2)*(zero()-one())
    if N != 1:
        meno = [meno]*N
        meno = reduce(np.kron, meno)
    if dm == False:
        return meno
    else:
        return np.outer(meno, meno.conj())
    
def one(dm = False, N: int = 1):
    """
    Creates the one state for a qubit.
    Args:
        dm (bool): If True, returns the density matrix representation.
        N (int): Number of qubits.
    Returns:
        np.ndarray: The one state for a qubit, either as a vector or a density matrix.
    """

    one = np.array([0,1], dtype = complex)
    if N != 1:
        one = [one]*N
        one = reduce(np.kron, one)
    if dm == False:
        return one
    else:
        return np.outer(one, one.conj())



def plus(dm = False, N: int = 1):
    """
    Creates the plus state for a qubit.
    Args:
        dm (bool): If True, returns the density matrix representation.
        N (int): Number of qubits.
    Returns:
        np.ndarray: The plus state for a qubit, either as a vector or a density matrix.
    """

    p = 1/np.sqrt(2)*(zero()+one())
    if N != 1:
        p = [p]*N
        p = reduce(np.kron, p)
    if dm == False:
        return p
    else:
        return np.outer(p, p.conj())

def random_qubit(n_qubits, pure=False, dm=True):
    """
    Generate a random quantum state for n_qubits using Haar random unitaries.
    
    Parameters:
    - n_qubits (int): Number of qubits
    - pure (bool): If True, generate a pure state. If False, generate a mixed state.
    - dm (bool): If True, return density matrix representation. If False, return state vector (only for pure states).
    
    Returns:
    - numpy.ndarray: A random quantum state
    """
    dim = 2**n_qubits
    U = haar_random_unitary(n_qubits)
    
    if pure:
        # Apply unitary to |0⟩ state to get random pure state
        state_vector = U[:, 0]  # First column of unitary
        
        if dm:
            # Convert to density matrix
            density_matrix = np.outer(state_vector, np.conjugate(state_vector))
            return density_matrix
        else:
            return state_vector
    else:
        # For mixed states, generate random eigenvalues that sum to 1
        eigenvalues = np.random.dirichlet([1]*dim)
        
        # Create diagonal matrix of eigenvalues and rotate with Haar unitary
        D = np.diag(eigenvalues)
        density_matrix = U @ D @ U.conj().T
        
        return density_matrix

def right(dm = False, N: int = 1):
    """
    Creates the right state for a qubit.
    Args:
        dm (bool): If True, returns the density matrix representation.
        N (int): Number of qubits.
    Returns:
        np.ndarray: The right state for a qubit, either as a vector or a density matrix.
    """

    r = (1/np.sqrt(2))*(zero()-1j*one())
    if N != 1:
        r = [r]*N
        r = reduce(np.kron, r)
    if dm == False:
        return r
    else:
        return np.outer(r, r.conj())
    
def zero(dm=False, N=1):
    """
    Creates the zero state for a qubit.
    Args:
        dm (bool): If True, returns the density matrix representation.
        N (int): Number of qubits.
    Returns:
        np.ndarray: The zero state for a qubit, either as a vector or a density matrix.
    """

    zero = np.array([1, 0], dtype = complex)
    if N > 1:
        zero = reduce(np.kron, [zero] * N)
    if dm:
        return np.outer(zero, zero.conj())
    return zero
