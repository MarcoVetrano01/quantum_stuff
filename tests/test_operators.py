import numpy as np
import pytest
import sys
import os
from scipy.sparse import csc_array

# Add the parent directory to the path so we can import QuantumStuff
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from QuantumStuff.Operators import (
    anticommutator, commutator, expect, haar_random_unitary,
    local_measurements, local_operators, sigmax, sigmay, sigmaz,
    sigmap, sigmam
)
from QuantumStuff.utils import dag, is_herm

class TestPauliOperators:
    def test_sigmax(self):
        """Test Pauli-X operator"""
        sx = sigmax()
        expected = np.array([[0, 1], [1, 0]], dtype=complex)
        np.testing.assert_array_almost_equal(sx, expected)
        assert is_herm(sx)
    
    def test_sigmay(self):
        """Test Pauli-Y operator"""
        sy = sigmay()
        expected = np.array([[0, -1j], [1j, 0]], dtype=complex)
        np.testing.assert_array_almost_equal(sy, expected)
        assert is_herm(sy)
    
    def test_sigmaz(self):
        """Test Pauli-Z operator"""
        sz = sigmaz()
        expected = np.array([[1, 0], [0, -1]], dtype=complex)
        np.testing.assert_array_almost_equal(sz, expected)
        assert is_herm(sz)
    
    def test_sigmap(self):
        """Test raising operator σ+"""
        sp = sigmap()
        expected = np.array([[0, 1], [0, 0]], dtype=complex)
        np.testing.assert_array_almost_equal(sp, expected)
    
    def test_sigmam(self):
        """Test lowering operator σ-"""
        sm = sigmam()
        expected = np.array([[0, 0], [1, 0]], dtype=complex)
        np.testing.assert_array_almost_equal(sm, expected)
    
    def test_pauli_relations(self):
        """Test Pauli operator relations"""
        sx, sy, sz = sigmax(), sigmay(), sigmaz()
        
        # Test anticommutation relations
        assert np.allclose(anticommutator(sx, sy), np.zeros((2, 2)))
        assert np.allclose(anticommutator(sx, sz), np.zeros((2, 2)))
        assert np.allclose(anticommutator(sy, sz), np.zeros((2, 2)))
        
        # Test commutation relations
        assert np.allclose(commutator(sx, sy), 2j * sz)
        assert np.allclose(commutator(sy, sz), 2j * sx)
        assert np.allclose(commutator(sz, sx), 2j * sy)

class TestCommutators:
    def test_commutator_basic(self):
        """Test basic commutator [A, B] = AB - BA"""
        A = np.array([[1, 2], [3, 4]], dtype=complex)
        B = np.array([[0, 1], [1, 0]], dtype=complex)
        result = commutator(A, B)
        expected = A @ B - B @ A
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_anticommutator_basic(self):
        """Test basic anticommutator {A, B} = AB + BA"""
        A = np.array([[1, 2], [3, 4]], dtype=complex)
        B = np.array([[0, 1], [1, 0]], dtype=complex)
        result = anticommutator(A, B)
        expected = A @ B + B @ A
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_commutator_zero(self):
        """Test commutator of commuting operators"""
        A = np.array([[1, 0], [0, 2]], dtype=complex)
        B = np.array([[3, 0], [0, 4]], dtype=complex)
        result = commutator(A, B)
        expected = np.zeros((2, 2), dtype=complex)
        np.testing.assert_array_almost_equal(result, expected)

class TestExpectation:
    def test_expect_pure_state(self):
        """Test expectation value with pure state"""
        psi = np.array([1, 0], dtype=complex)  # |0⟩ state
        sz = sigmaz()
        result = expect(psi, sz)
        expected = 1.0  # ⟨0|σz|0⟩ = 1
        assert np.isclose(result, expected)
    
    def test_expect_density_matrix(self):
        """Test expectation value with density matrix"""
        rho = np.array([[0.5, 0], [0, 0.5]], dtype=complex)  # Mixed state
        sz = sigmaz()
        result = expect(rho, sz)
        expected = 0.0  # Tr(ρσz) = 0 for this mixed state
        assert np.isclose(result, expected)
    
    def test_expect_batch(self):
        """Test expectation value with batch of states"""
        states = np.array([[[1], [0]], [[0], [1]]], dtype=complex)  # |0⟩ and |1⟩
        sz = sigmaz()
        results = expect(states, sz)
        expected = np.array([1.0, -1.0])
        np.testing.assert_array_almost_equal(results, expected)

class TestHaarRandomUnitary:
    def test_unitary_property(self):
        """Test that generated matrix is unitary"""
        U = haar_random_unitary(2)
        identity = U @ dag(U)
        expected = np.eye(4)
        np.testing.assert_array_almost_equal(identity, expected, decimal=10)
    
    def test_determinant_one(self):
        """Test that determinant is 1 (up to phase)"""
        U = haar_random_unitary(1)
        det = np.linalg.det(U)
        assert np.isclose(abs(det), 1.0)
    
    def test_dimension(self):
        """Test correct dimensions"""
        for n in range(1, 4):
            U = haar_random_unitary(n)
            expected_dim = 2**n
            assert U.shape == (expected_dim, expected_dim)
    
    def test_invalid_input(self):
        """Test error handling for invalid inputs"""
        with pytest.raises(Exception):
            haar_random_unitary(0)
        with pytest.raises(Exception):
            haar_random_unitary(-1)
        with pytest.raises(Exception):
            haar_random_unitary(1.5)

class TestLocalOperators:
    def test_single_qubit_operators(self):
        """Test generation of local single-qubit operators"""
        sx = sigmax()
        N = 3
        local_ops = local_operators(sx, N)
        
        # Should have N operators
        assert len(local_ops) == N
        
        # Each should be 2^N × 2^N
        for op in local_ops:
            assert op.shape == (2**N, 2**N)
    
    def test_sparse_operators(self):
        """Test with sparse matrices"""
        sx = csc_array(sigmax())
        N = 2
        local_ops = local_operators(sx, N)
        assert len(local_ops) == N

class TestLocalMeasurements:
    def test_single_qubit_measurements(self):
        """Test local measurements on single qubit"""
        # |+⟩ state
        psi = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        rho = np.outer(psi, np.conj(psi))
        
        results = local_measurements(rho[np.newaxis])
        
        # Should return measurements for x, y, z
        assert results.shape[1] == 3
        
        # For |+⟩ state: ⟨σx⟩ = 1, ⟨σy⟩ = 0, ⟨σz⟩ = 0
        np.testing.assert_almost_equal(results[0, 0], 1.0, decimal=5)  # σx
        np.testing.assert_almost_equal(results[0, 1], 0.0, decimal=5)  # σy
        np.testing.assert_almost_equal(results[0, 2], 0.0, decimal=5)  # σz
    
    def test_two_qubit_measurements(self):
        """Test local measurements on two-qubit system"""
        # |00⟩ state
        psi = np.array([1, 0, 0, 0], dtype=complex)
        rho = np.outer(psi, np.conj(psi))
        
        results = local_measurements(rho[np.newaxis])
        
        # Should have measurements for 2 qubits, 3 operators each
        assert results.shape == (1, 2, 3)
        
        # For |00⟩: both qubits should give ⟨σz⟩ = 1
        np.testing.assert_almost_equal(results[0, 0, 2], 1.0, decimal=5)  # First qubit σz
        np.testing.assert_almost_equal(results[0, 1, 2], 1.0, decimal=5)  # Second qubit σz

if __name__ == "__main__":
    pytest.main([__file__])
