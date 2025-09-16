import numpy as np
import pytest
import sys
import os

# Add the parent directory to the path so we can import QuantumStuff
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from QuantumStuff.States import (
    zero, one, plus, minus, left, right, random_qubit, bloch_vector
)
from QuantumStuff.utils import is_state, is_norm
from QuantumStuff.Operators import sigmax, sigmay, sigmaz, expect

class TestBasicStates:
    def test_zero_state(self):
        """Test |0⟩ state"""
        psi = zero()
        expected = np.array([1, 0], dtype=complex)
        np.testing.assert_array_almost_equal(psi, expected)
        assert is_norm(psi)
    
    def test_one_state(self):
        """Test |1⟩ state"""
        psi = one()
        expected = np.array([0, 1], dtype=complex)
        np.testing.assert_array_almost_equal(psi, expected)
        assert is_norm(psi)
    
    def test_plus_state(self):
        """Test |+⟩ state"""
        psi = plus()
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        np.testing.assert_array_almost_equal(psi, expected)
        assert is_norm(psi)
    
    def test_minus_state(self):
        """Test |-⟩ state"""
        psi = minus()
        expected = np.array([1/np.sqrt(2), -1/np.sqrt(2)], dtype=complex)
        np.testing.assert_array_almost_equal(psi, expected)
        assert is_norm(psi)
    
    def test_right_state(self):
        """Test |R⟩ state (|+i⟩)"""
        psi = right()
        expected = np.array([1/np.sqrt(2), 1j/np.sqrt(2)], dtype=complex)
        np.testing.assert_array_almost_equal(psi, expected)
        assert is_norm(psi)
    
    def test_left_state(self):
        """Test |L⟩ state (|-i⟩)"""
        psi = left()
        expected = np.array([1/np.sqrt(2), -1j/np.sqrt(2)], dtype=complex)
        np.testing.assert_array_almost_equal(psi, expected)
        assert is_norm(psi)

class TestDensityMatrices:
    def test_zero_density_matrix(self):
        """Test |0⟩⟨0| density matrix"""
        rho = zero(dm=True)
        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        np.testing.assert_array_almost_equal(rho, expected)
        is_valid, is_dm = is_state(rho, batchmode=False)
        assert is_valid and is_dm
    
    def test_plus_density_matrix(self):
        """Test |+⟩⟨+| density matrix"""
        rho = plus(dm=True)
        expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        np.testing.assert_array_almost_equal(rho, expected)
        is_valid, is_dm = is_state(rho, batchmode=False)
        assert is_valid and is_dm

class TestMultiQubitStates:
    def test_two_qubit_zero(self):
        """Test two-qubit |00⟩ state"""
        psi = zero(N=2)
        expected = np.array([1, 0, 0, 0], dtype=complex)
        np.testing.assert_array_almost_equal(psi, expected)
        assert is_norm(psi)
    
    def test_three_qubit_plus(self):
        """Test three-qubit |+++⟩ state"""
        psi = plus(N=3)
        expected = np.ones(8, dtype=complex) / np.sqrt(8)
        np.testing.assert_array_almost_equal(psi, expected)
        assert is_norm(psi)
    
    def test_multi_qubit_dimensions(self):
        """Test correct dimensions for multi-qubit states"""
        for N in range(1, 5):
            psi = zero(N=N)
            assert len(psi) == 2**N
            
            rho = zero(dm=True, N=N)
            assert rho.shape == (2**N, 2**N)

class TestRandomStates:
    def test_random_pure_state(self):
        """Test random pure state generation"""
        psi = random_qubit(1, pure=True, dm=False)
        assert is_norm(psi)
        assert len(psi) == 2
    
    def test_random_density_matrix(self):
        """Test random density matrix generation"""
        rho = random_qubit(1, pure=False, dm=True)
        is_valid, is_dm = is_state(rho, batchmode=False)
        assert is_valid and is_dm
        assert rho.shape == (2, 2)
    
    def test_random_multi_qubit(self):
        """Test random multi-qubit states"""
        for n_qubits in range(1, 4):
            psi = random_qubit(n_qubits, pure=True, dm=False)
            assert is_norm(psi)
            assert len(psi) == 2**n_qubits
            
            rho = random_qubit(n_qubits, pure=False, dm=True)
            is_valid, is_dm = is_state(rho, False)
            assert is_valid and is_dm
            assert rho.shape == (2**n_qubits, 2**n_qubits)

class TestBlochVector:
    def test_bloch_vector_computational_basis(self):
        """Test Bloch vectors for computational basis states"""
        # |0⟩ state should give (0, 0, 1)
        rho0 = zero(dm=True)
        bloch0 = bloch_vector(rho0)
        expected0 = np.array([0, 0, 1])
        np.testing.assert_array_almost_equal(bloch0.flatten(), expected0, decimal=10)
        
        # |1⟩ state should give (0, 0, -1)
        rho1 = one(dm=True)
        bloch1 = bloch_vector(rho1)
        expected1 = np.array([0, 0, -1])
        np.testing.assert_array_almost_equal(bloch1.flatten(), expected1, decimal=10)
    
    def test_bloch_vector_superposition_states(self):
        """Test Bloch vectors for superposition states"""
        # |+⟩ state should give (1, 0, 0)
        rho_plus = plus(dm=True)
        bloch_plus = bloch_vector(rho_plus)
        expected_plus = np.array([1, 0, 0])
        np.testing.assert_array_almost_equal(bloch_plus.flatten(), expected_plus, decimal=10)
        
        # |-⟩ state should give (-1, 0, 0)
        rho_minus = minus(dm=True)
        bloch_minus = bloch_vector(rho_minus)
        expected_minus = np.array([-1, 0, 0])
        np.testing.assert_array_almost_equal(bloch_minus.flatten(), expected_minus, decimal=10)
    
    def test_bloch_vector_y_eigenstates(self):
        """Test Bloch vectors for Y-basis states"""
        # |R⟩ state should give (0, 1, 0)
        rho_right = right(dm=True)
        bloch_right = bloch_vector(rho_right)
        expected_right = np.array([0, 1, 0])
        np.testing.assert_array_almost_equal(bloch_right.flatten(), expected_right, decimal=10)
        
        # |L⟩ state should give (0, -1, 0)
        rho_left = left(dm=True)
        bloch_left = bloch_vector(rho_left)
        expected_left = np.array([0, -1, 0])
        np.testing.assert_array_almost_equal(bloch_left.flatten(), expected_left, decimal=10)
    
    def test_bloch_vector_batch(self):
        """Test Bloch vector calculation for batch of states"""
        states = np.array([zero(dm=True), one(dm=True), plus(dm=True)])
        bloch_vectors = bloch_vector(states)
        
        assert bloch_vectors.shape == (3, 3)
        
        # Check individual vectors
        np.testing.assert_array_almost_equal(bloch_vectors[0], [0, 0, 1], decimal=10)
        np.testing.assert_array_almost_equal(bloch_vectors[1], [0, 0, -1], decimal=10)
        np.testing.assert_array_almost_equal(bloch_vectors[2], [1, 0, 0], decimal=10)
    
    def test_bloch_vector_unit_length(self):
        """Test that Bloch vectors for pure states have unit length"""
        pure_states = [zero(dm=True), one(dm=True), plus(dm=True), minus(dm=True), 
                      right(dm=True), left(dm=True)]
        
        for rho in pure_states:
            bloch = bloch_vector(rho)
            length = np.linalg.norm(bloch)
            assert np.isclose(length, 1.0, atol=1e-10)
    
    def test_bloch_vector_single_mixed_state(self):
        """Test Bloch vector for a single mixed state"""
        # Maximally mixed state (identity/2) should give (0, 0, 0)
        rho_mixed = np.eye(2, dtype=complex) / 2
        bloch_mixed = bloch_vector(rho_mixed)
        expected_mixed = np.array([0, 0, 0])
        np.testing.assert_array_almost_equal(bloch_mixed.flatten(), expected_mixed, decimal=10)
        
        # Partially mixed state: 0.8|0⟩⟨0| + 0.2|1⟩⟨1|
        rho_partial = 0.8 * zero(dm=True) + 0.2 * one(dm=True)
        bloch_partial = bloch_vector(rho_partial)
        expected_partial = np.array([0, 0, 0.6])  # (0.8 - 0.2) = 0.6 along z-axis
        np.testing.assert_array_almost_equal(bloch_partial.flatten(), expected_partial, decimal=10)
        
        # Check that mixed state Bloch vectors have length ≤ 1
        length_mixed = np.linalg.norm(bloch_mixed)
        length_partial = np.linalg.norm(bloch_partial)
        assert length_mixed <= 1.0
        assert length_partial <= 1.0
        assert length_mixed < 1.0  # Strictly less than 1 for non-pure states
        assert length_partial < 1.0
    
    def test_bloch_vector_batch_mixed_states(self):
        """Test Bloch vector calculation for batch of mixed states"""
        # Create a batch of mixed states
        rho_maximally_mixed = np.eye(2, dtype=complex) / 2
        rho_partial_z = 0.7 * zero(dm=True) + 0.3 * one(dm=True)
        rho_partial_x = 0.6 * plus(dm=True) + 0.4 * minus(dm=True)
        
        mixed_states = np.array([rho_maximally_mixed, rho_partial_z, rho_partial_x])
        bloch_vectors = bloch_vector(mixed_states)
        
        assert bloch_vectors.shape == (3, 3)
        
        # Check individual vectors
        # Maximally mixed: (0, 0, 0)
        np.testing.assert_array_almost_equal(bloch_vectors[0], [0, 0, 0], decimal=10)
        
        # Partial z-mix: (0, 0, 0.4) since (0.7 - 0.3) = 0.4
        np.testing.assert_array_almost_equal(bloch_vectors[1], [0, 0, 0.4], decimal=10)
        
        # Partial x-mix: (0.2, 0, 0) since (0.6 - 0.4) = 0.2
        np.testing.assert_array_almost_equal(bloch_vectors[2], [0.2, 0, 0], decimal=10)
        
        # Check that all mixed state Bloch vectors have length ≤ 1
        for i in range(3):
            length = np.linalg.norm(bloch_vectors[i])
            assert length <= 1.0
            if i > 0:  # Non-maximally mixed states should have length < 1
                assert length < 1.0

class TestStateExpectations:
    def test_pauli_expectations(self):
        """Test Pauli operator expectations for known states"""
        # |0⟩ state
        psi0 = zero()
        assert np.isclose(expect(psi0, sigmaz()), 1.0)
        assert np.isclose(expect(psi0, sigmax()), 0.0)
        assert np.isclose(expect(psi0, sigmay()), 0.0)
        
        # |1⟩ state
        psi1 = one()
        assert np.isclose(expect(psi1, sigmaz()), -1.0)
        assert np.isclose(expect(psi1, sigmax()), 0.0)
        assert np.isclose(expect(psi1, sigmay()), 0.0)
        
        # |+⟩ state
        psi_plus = plus()
        assert np.isclose(expect(psi_plus, sigmax()), 1.0)
        assert np.isclose(expect(psi_plus, sigmay()), 0.0)
        assert np.isclose(expect(psi_plus, sigmaz()), 0.0)

if __name__ == "__main__":
    pytest.main([__file__])
