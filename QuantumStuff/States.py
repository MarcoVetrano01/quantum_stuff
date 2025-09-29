"""
Quantum State Generation and Manipulation Functions

This module provides functions for creating and manipulating quantum states,
including computational basis states, superposition states, and random states.
Functions are organized by functionality for better code navigation.
"""

import numpy as np
from .Operators import haar_random_unitary, local_measurements
from functools import reduce
import plotly.graph_objects as go
import string
import random
from typing import Union

# Type aliases for better code readability
MatrixLike = Union[np.ndarray, list]
StateVector = np.ndarray
DensityMatrix = np.ndarray


# =============================================================================
# COMPUTATIONAL BASIS STATES
# =============================================================================

def zero(dm: bool = False, N: int = 1) -> Union[StateVector, DensityMatrix]:
    """
    Creates the zero state |0⟩ for one or multiple qubits.
    
    Args:
        dm (bool): If True, returns the density matrix representation.
        N (int): Number of qubits.
    Returns:
        Union[StateVector, DensityMatrix]: The zero state, either as a vector or density matrix.
    """
    zero_state = np.array([1, 0], dtype=complex)
    if N > 1:
        zero_state = reduce(np.kron, [zero_state] * N)
    if dm:
        return np.outer(zero_state, zero_state.conj())
    return zero_state


def one(dm: bool = False, N: int = 1) -> Union[StateVector, DensityMatrix]:
    """
    Creates the one state |1⟩ for one or multiple qubits.
    
    Args:
        dm (bool): If True, returns the density matrix representation.
        N (int): Number of qubits.
    Returns:
        Union[StateVector, DensityMatrix]: The one state, either as a vector or density matrix.
    """
    one_state = np.array([0, 1], dtype=complex)
    if N != 1:
        one_state = reduce(np.kron, [one_state] * N)
    if dm:
        return np.outer(one_state, one_state.conj())
    return one_state


# =============================================================================
# SUPERPOSITION STATES
# =============================================================================

def plus(dm: bool = False, N: int = 1) -> Union[StateVector, DensityMatrix]:
    """
    Creates the plus state |+⟩ = (|0⟩ + |1⟩)/√2 for one or multiple qubits.
    
    Args:
        dm (bool): If True, returns the density matrix representation.
        N (int): Number of qubits.
    Returns:
        Union[StateVector, DensityMatrix]: The plus state, either as a vector or density matrix.
    """
    plus_state = (1/np.sqrt(2)) * (zero() + one())
    if N != 1:
        plus_state = reduce(np.kron, [plus_state] * N)
    if dm:
        return np.outer(plus_state, plus_state.conj())
    return plus_state


def minus(dm: bool = False, N: int = 1) -> Union[StateVector, DensityMatrix]:
    """
    Creates the minus state |-⟩ = (|0⟩ - |1⟩)/√2 for one or multiple qubits.
    
    Args:
        dm (bool): If True, returns the density matrix representation.
        N (int): Number of qubits.
    Returns:
        Union[StateVector, DensityMatrix]: The minus state, either as a vector or density matrix.
    """
    minus_state = (1/np.sqrt(2)) * (zero() - one())
    if N != 1:
        minus_state = reduce(np.kron, [minus_state] * N)
    if dm:
        return np.outer(minus_state, minus_state.conj())
    return minus_state


def right(dm: bool = False, N: int = 1) -> Union[StateVector, DensityMatrix]:
    """
    Creates the right state |R⟩ = (|0⟩ + i|1⟩)/√2 for one or multiple qubits.
    
    Args:
        dm (bool): If True, returns the density matrix representation.
        N (int): Number of qubits.
    Returns:
        Union[StateVector, DensityMatrix]: The right state, either as a vector or density matrix.
    """
    right_state = (1/np.sqrt(2)) * (zero() + 1j * one())
    if N != 1:
        right_state = reduce(np.kron, [right_state] * N)
    if dm:
        return np.outer(right_state, right_state.conj())
    return right_state


def left(dm: bool = False, N: int = 1) -> Union[StateVector, DensityMatrix]:
    """
    Creates the left state |L⟩ = (|0⟩ - i|1⟩)/√2 for one or multiple qubits.
    
    Args:
        dm (bool): If True, returns the density matrix representation.
        N (int): Number of qubits.
    Returns:
        Union[StateVector, DensityMatrix]: The left state, either as a vector or density matrix.
    """
    left_state = (1/np.sqrt(2)) * (zero() - 1j * one())
    if N != 1:
        left_state = reduce(np.kron, [left_state] * N)
    if dm:
        return np.outer(left_state, left_state.conj())
    return left_state


# =============================================================================
# RANDOM STATE GENERATION
# =============================================================================

def random_qubit(n_qubits: int, pure: bool = False, dm: bool = True) -> Union[StateVector, DensityMatrix]:
    """
    Generate a random quantum state for n_qubits using Haar random unitaries.
    
    Args:
        n_qubits (int): Number of qubits
        pure (bool): If True, generate a pure state. If False, generate a mixed state.
        dm (bool): If True, return density matrix representation. If False, return state vector (only for pure states).
    
    Returns:
        Union[StateVector, DensityMatrix]: A random quantum state
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


# =============================================================================
# BLOCH SPHERE REPRESENTATION
# =============================================================================

def bloch_vector(rho: MatrixLike, batchmode: bool) -> np.ndarray:
    """
    Calculate the Bloch vector for a given quantum state ρ.
    
    The state can be a pure state (ket) or a mixed state (density matrix).
    For single-qubit states, returns the 3D Bloch vector (x, y, z).
    
    Args:
        rho (MatrixLike): The quantum state, either a ket or a density matrix or a list of states.
        batchmode (Bool): If True, process a batch of states.
    Returns:
        np.ndarray: The Bloch vector components (x, y, z).
    """
    rho_array = np.array(rho, dtype = complex)
    bloch_vec = np.real(local_measurements(rho_array, batchmode = batchmode))
    return bloch_vec


def BlochSpherePlot(states: MatrixLike):
    """
    Plot points on a Bloch sphere using Plotly.
    
    Args:
        states (MatrixLike): A list or array of quantum states (density matrices or pure states).
    """
    if not isinstance(states, (np.ndarray, list)):
        raise TypeError("Input states must be a numpy array or a list.")
    
    # Calculate Bloch vectors for all states
    vectors = bloch_vector(states).T
    
    # Create sphere mesh (low resolution for better performance)
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
