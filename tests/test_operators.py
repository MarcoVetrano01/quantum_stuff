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
    sigmap, sigmam, measure, two_qubits_measurements
)
from QuantumStuff.utils import dag, is_herm
from QuantumStuff.States import zero, one, plus, minus

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

class TestSparseCommutators:
    """Test commutators and anticommutators with sparse matrices"""
    
    def test_commutator_sparse_dense(self):
        """Test commutator with sparse and dense matrices"""
        # Create sparse Pauli-X matrix
        A_sparse = csc_array([[0, 1], [1, 0]], dtype=complex)
        # Create dense Pauli-Z matrix
        B_dense = np.array([[1, 0], [0, -1]], dtype=complex)
        
        result = commutator(A_sparse, B_dense)
        # [σx, σz] = σxσz - σzσx
        expected = A_sparse @ B_dense - B_dense @ A_sparse
        np.testing.assert_array_almost_equal(result.toarray(), expected)
    
    def test_commutator_dense_sparse(self):
        """Test commutator with dense and sparse matrices"""
        # Create dense Pauli-Y matrix
        A_dense = np.array([[0, -1j], [1j, 0]], dtype=complex)
        # Create sparse Pauli-Z matrix
        B_sparse = csc_array([[1, 0], [0, -1]], dtype=complex)
        
        result = commutator(A_dense, B_sparse)
        # [σy, σz] = σyσz - σzσy
        expected = A_dense @ B_sparse - B_sparse @ A_dense
        np.testing.assert_array_almost_equal(result.toarray(), expected)
    
    def test_commutator_both_sparse(self):
        """Test commutator with both matrices sparse"""
        # Create sparse Pauli matrices
        A_sparse = csc_array([[0, 1], [1, 0]], dtype=complex)  # σx
        B_sparse = csc_array([[0, -1j], [1j, 0]], dtype=complex)  # σy
        
        result = commutator(A_sparse, B_sparse)
        # [σx, σy] = σxσy - σyσx
        expected = A_sparse @ B_sparse - B_sparse @ A_sparse
        np.testing.assert_array_almost_equal(result.toarray(), expected.toarray())
    
    def test_anticommutator_sparse_dense(self):
        """Test anticommutator with sparse and dense matrices"""
        # Create sparse Pauli-X matrix
        A_sparse = csc_array([[0, 1], [1, 0]], dtype=complex)
        # Create dense Pauli-Z matrix
        B_dense = np.array([[1, 0], [0, -1]], dtype=complex)
        
        result = anticommutator(A_sparse, B_dense)
        # {σx, σz} = 0 (Pauli matrices anticommute)
        expected = np.zeros((2, 2), dtype=complex)
        np.testing.assert_array_almost_equal(result.toarray(), expected)
    
    def test_anticommutator_dense_sparse(self):
        """Test anticommutator with dense and sparse matrices"""
        # Create dense Pauli-Y matrix
        A_dense = np.array([[0, -1j], [1j, 0]], dtype=complex)
        # Create sparse Pauli-X matrix
        B_sparse = csc_array([[0, 1], [1, 0]], dtype=complex)
        
        result = anticommutator(A_dense, B_sparse)
        # {σy, σx} = 0 (Pauli matrices anticommute)
        expected = np.zeros((2, 2), dtype=complex)
        np.testing.assert_array_almost_equal(result.toarray(), expected)
    
    def test_anticommutator_both_sparse(self):
        """Test anticommutator with both matrices sparse"""
        # Create sparse Pauli matrices
        A_sparse = csc_array([[0, 1], [1, 0]], dtype=complex)  # σx
        B_sparse = csc_array([[0, 1], [1, 0]], dtype=complex)  # σx
        
        result = anticommutator(A_sparse, B_sparse)
        # {σx, σx} = 2I
        expected = 2 * np.eye(2, dtype=complex)
        np.testing.assert_array_almost_equal(result.toarray(), expected)
    
    def test_commutator_sparse_larger_matrices(self):
        """Test commutator with larger sparse matrices"""
        # Create 4x4 sparse matrices (for 2-qubit system)
        from scipy.sparse import diags, csc_array
        
        # Create diagonal sparse matrix
        A_sparse = diags([1, -1, 1, -1], shape=(4, 4), dtype=complex, format='csc')
        # Create another diagonal sparse matrix (will commute)
        B_sparse = diags([2, 2, -2, -2], shape=(4, 4), dtype=complex, format='csc')
        
        result = commutator(A_sparse, B_sparse)
        # Diagonal matrices commute, so commutator should be zero
        expected = np.zeros((4, 4), dtype=complex)
        np.testing.assert_array_almost_equal(result.toarray(), expected)
    
    def test_anticommutator_sparse_larger_matrices(self):
        """Test anticommutator with larger sparse matrices"""
        # Create 4x4 sparse matrices
        from scipy.sparse import diags
        
        # Create diagonal sparse matrix
        A_sparse = diags([1, -1, 1, -1], shape=(4, 4), dtype=complex, format='csc')
        # Create another diagonal sparse matrix
        B_sparse = diags([1, 1, -1, -1], shape=(4, 4), dtype=complex, format='csc')
        
        result = anticommutator(A_sparse, B_sparse)
        expected = A_sparse @ B_sparse + B_sparse @ A_sparse
        np.testing.assert_array_almost_equal(result.toarray(), expected.toarray())
    
    def test_commutator_sparse_zero_result(self):
        """Test commutator of commuting sparse matrices (should be zero)"""
        # Create commuting sparse matrices (both diagonal)
        from scipy.sparse import diags
        
        A_sparse = diags([1, 2], shape=(2, 2), dtype=complex, format='csc')
        B_sparse = diags([3, 4], shape=(2, 2), dtype=complex, format='csc')
        
        result = commutator(A_sparse, B_sparse)
        expected = np.zeros((2, 2), dtype=complex)
        np.testing.assert_array_almost_equal(result.toarray(), expected)
    
    def test_sparse_matrix_type_preservation(self):
        """Test that operations with sparse matrices preserve sparsity when appropriate"""
        # Create sparse matrices
        A_sparse = csc_array([[1, 0], [0, 0]], dtype=complex)
        B_sparse = csc_array([[0, 0], [0, 1]], dtype=complex)
        
        # Commutator should be sparse
        result_comm = commutator(A_sparse, B_sparse)
        assert hasattr(result_comm, 'toarray'), "Result should be sparse"
        
        # Anticommutator should be sparse
        result_anticomm = anticommutator(A_sparse, B_sparse)
        assert hasattr(result_anticomm, 'toarray'), "Result should be sparse"

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
        result = expect(rho, sz, batchmode = False)
        expected = 0.0  # Tr(ρσz) = 0 for this mixed state
        assert np.isclose(result, expected)
    
    def test_expect_batch(self):
        """Test expectation value with batch of states"""
        states = np.array([[1, 0], [0, 1]], dtype=complex)  # |0⟩ and |1⟩
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
        assert results.shape == (1, 6)
        
        # For |00⟩: both qubits should give ⟨σz⟩ = 1
        np.testing.assert_almost_equal(results[0, 2], 1.0, decimal=5)  # First qubit σz
        np.testing.assert_almost_equal(results[0, 5], 1.0, decimal=5)  # Second qubit σz

class TestBatchDensityMatrices:
    """Test expectation values with batches of density matrices"""
    
    def test_batch_pure_states(self):
        """Test batch processing with pure state density matrices"""
        # Create batch of pure states: |0⟩, |1⟩, |+⟩
        psi0 = zero()
        psi1 = one()
        psi_plus = plus()
        
        # Convert to density matrices
        rho0 = np.outer(psi0, np.conj(psi0))
        rho1 = np.outer(psi1, np.conj(psi1))
        rho_plus = np.outer(psi_plus, np.conj(psi_plus))
        
        batch_states = np.array([rho0, rho1, rho_plus])
        
        # Test with σz
        sz = sigmaz()
        results = expect(batch_states, sz, batchmode = True)
        expected = np.array([1.0, -1.0, 0.0])  # ⟨0|σz|0⟩, ⟨1|σz|1⟩, ⟨+|σz|+⟩
        np.testing.assert_array_almost_equal(results, expected, decimal=10)
        
        # Test with σx
        sx = sigmax()
        results = expect(batch_states, sx, batchmode = True)
        expected = np.array([0.0, 0.0, 1.0])  # ⟨0|σx|0⟩, ⟨1|σx|1⟩, ⟨+|σx|+⟩
        np.testing.assert_array_almost_equal(results, expected, decimal=10)
    
    def test_batch_mixed_states(self):
        """Test batch processing with mixed state density matrices"""
        # Create various mixed states
        rho1 = 0.5 * np.array([[1, 0], [0, 1]], dtype=complex)  # Maximally mixed
        rho2 = 0.8 * np.array([[1, 0], [0, 0]], dtype=complex) + 0.2 * np.array([[0, 0], [0, 1]], dtype=complex)  # Mixed
        rho3 = np.array([[0.6, 0.2], [0.2, 0.4]], dtype=complex)  # General mixed state
        
        batch_states = np.array([rho1, rho2, rho3])
        
        # Test with σz
        sz = sigmaz()
        results = expect(batch_states, sz, batchmode = True)
        expected = np.array([0.0, 0.6, 0.2])  # Expected values for these mixed states
        np.testing.assert_array_almost_equal(results, expected, decimal=10)
    
    def test_batch_different_sizes(self):
        """Test batch processing with different batch sizes"""
        # Single state (test with batchmode=False)
        rho_single = np.array([[1, 0], [0, 0]], dtype=complex)
        result = expect(rho_single, sigmaz(), batchmode=False)
        assert np.isclose(result, 1.0)
        
        # Large batch
        batch_size = 10
        batch_states = np.array([rho_single for _ in range(batch_size)])
        results = expect(batch_states, sigmaz(), batchmode = True)
        expected = np.ones(batch_size)
        np.testing.assert_array_almost_equal(results, expected, decimal=10)
    
    def test_batch_multi_qubit(self):
        """Test batch processing with multi-qubit density matrices"""
        # Create two-qubit states |00⟩ and |11⟩
        psi00 = np.array([1, 0, 0, 0], dtype=complex)
        psi11 = np.array([0, 0, 0, 1], dtype=complex)
        
        rho00 = np.outer(psi00, np.conj(psi00))
        rho11 = np.outer(psi11, np.conj(psi11))
        
        batch_states = np.array([rho00, rho11])
        
        # Test with σz ⊗ I
        sz_I = np.kron(sigmaz(), np.eye(2))
        results = expect(batch_states, sz_I, batchmode = True)
        expected = np.array([1.0, -1.0])  # First qubit measurement
        np.testing.assert_array_almost_equal(results, expected, decimal=10)

class TestMeasureFunction:
    """Test the general measure function"""
    
    def test_measure_single_qubit_local(self):
        """Test measuring single qubit with local operators"""
        # Prepare states: |0⟩, |1⟩, |+⟩
        states = [
            np.outer(zero(), np.conj(zero())),
            np.outer(one(), np.conj(one())),
            np.outer(plus(), np.conj(plus()))
        ]
        
        # Measure σz on qubit 0
        operators = [sigmaz()]
        indices_list = [[[0]]]  # Single operator on single qubit
        
        results = measure(states, operators, indices_list, batchmode = True)
        
        # Shape should be [num_states, num_measurements]
        assert results.shape == (3, 1)
        
        # Expected values
        expected = np.array([[1.0], [-1.0], [0.0]])
        np.testing.assert_array_almost_equal(results, expected, decimal=10)
    
    def test_measure_two_qubit_system(self):
        """Test measuring two-qubit system"""
        # Prepare Bell states
        bell_00 = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)  # |Φ+⟩
        bell_01 = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)  # |Φ-⟩
        
        states = [
            np.outer(bell_00, np.conj(bell_00)),
            np.outer(bell_01, np.conj(bell_01))
        ]
        
        # Measure σz on each qubit separately
        operators = [sigmaz(), sigmaz()]
        indices_list = [[[0]], [[1]]]  # First σz on qubit 0, second σz on qubit 1
        
        results = measure(states, operators, indices_list, batchmode = True)
        
        # Shape should be [num_states, num_measurements]
        assert results.shape == (2, 2)
        
        # For Bell states, local measurements should give 0
        expected = np.array([[0.0, 0.0], [0.0, 0.0]])
        np.testing.assert_array_almost_equal(results, expected, decimal=10)
    
    def test_measure_multi_qubit_operators(self):
        """Test measuring with multi-qubit operators"""
        # Prepare three-qubit GHZ state: (|000⟩ + |111⟩)/√2
        ghz = np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho_ghz = np.outer(ghz, np.conj(ghz))
        
        states = [rho_ghz]
        
        # Measure σz ⊗ σz on qubits (0,1)
        sz_sz = np.kron(sigmaz(), sigmaz())
        operators = [sz_sz]
        indices_list = [[[0, 1]]]
        
        results = measure(states, operators, indices_list, batchmode = False)
        
        # For GHZ state, ⟨σz ⊗ σz⟩ = 1
        assert results.shape == (1, 1)
        np.testing.assert_almost_equal(results[0, 0], 1.0, decimal=10)
    
    def test_measure_multiple_operators(self):
        """Test measuring with multiple different operators"""
        # Single qubit state |+⟩
        psi_plus = plus()
        rho_plus = np.outer(psi_plus, np.conj(psi_plus))
        
        states = [rho_plus]
        
        # Measure all Pauli operators
        operators = [sigmax(), sigmay(), sigmaz()]
        indices_list = [[[0]], [[0]], [[0]]]
        
        results = measure(states, operators, indices_list, batchmode = True)
        
        # Expected: ⟨σx⟩ = 1, ⟨σy⟩ = 0, ⟨σz⟩ = 0 for |+⟩
        expected = np.array([[1.0, 0.0, 0.0]])
        np.testing.assert_array_almost_equal(results, expected, decimal=10)

class TestTwoQubitMeasurements:
    """Test the optimized two_qubits_measurements function"""
    
    def test_two_qubit_measurements_single_state(self):
        """Test two-qubit measurements on single state"""
        # Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        bell_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho_bell = np.outer(bell_plus, np.conj(bell_plus))
        
        # Test with Pauli operators
        operators = [
            np.kron(sigmaz(), sigmaz()),  # σz ⊗ σz
            np.kron(sigmax(), sigmax()),  # σx ⊗ σx
            np.kron(sigmay(), sigmay())   # σy ⊗ σy
        ]
        
        results = two_qubits_measurements(rho_bell, operators)
        
        # For Bell state: ⟨σz⊗σz⟩ = 1, ⟨σx⊗σx⟩ = 1, ⟨σy⊗σy⟩ = -1
        expected = np.array([[1.0, 1.0, -1.0]])
        np.testing.assert_array_almost_equal(results.real, expected, decimal=10)
    
    def test_two_qubit_measurements_batch(self):
        """Test two-qubit measurements on batch of states"""
        # Create different two-qubit states
        # |00⟩ and |11⟩ states
        state_00 = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
        state_11 = np.array([0, 0, 0, 1], dtype=complex)  # |11⟩
        
        rho1 = np.outer(state_00, np.conj(state_00))
        rho2 = np.outer(state_11, np.conj(state_11))
        
        batch_states = np.array([rho1, rho2])
        
        operators = [np.kron(sigmaz(), sigmaz())]
        
        results = two_qubits_measurements(batch_states, operators)
        
        # For |00⟩: ⟨σz⊗σz⟩ = 1, for |11⟩: ⟨σz⊗σz⟩ = 1
        expected = np.array([[1.0], [1.0]])
        np.testing.assert_array_almost_equal(results.real, expected, decimal=10)
    
    def test_three_qubit_system_pairs(self):
        """Test two-qubit measurements on three-qubit system (all pairs)"""
        # Create three-qubit GHZ state: (|000⟩ + |111⟩)/√2
        ghz = np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho_ghz = np.outer(ghz, np.conj(ghz))
        
        operators = [np.kron(sigmaz(), sigmaz())]
        
        results = two_qubits_measurements(rho_ghz, operators)
        
        # For GHZ state, all two-qubit correlations ⟨σz⊗σz⟩ should be 1
        # Three qubits give 3 choose 2 = 3 pairs: (0,1), (0,2), (1,2)
        expected = np.array([[1.0, 1.0, 1.0]])
        np.testing.assert_array_almost_equal(results.real, expected, decimal=10)
    
    def test_mixed_state_measurements(self):
        """Test two-qubit measurements on mixed states"""
        # Create a mixed state: 0.5 * |00⟩⟨00| + 0.5 * |11⟩⟨11|
        rho00 = np.outer([1, 0, 0, 0], [1, 0, 0, 0])
        rho11 = np.outer([0, 0, 0, 1], [0, 0, 0, 1])
        mixed_state = 0.5 * (rho00 + rho11)
        
        operators = [
            np.kron(sigmaz(), sigmaz()),
            np.kron(sigmax(), sigmax())
        ]
        
        results = two_qubits_measurements(mixed_state, operators)
        
        # For this mixed state: ⟨σz⊗σz⟩ = 1, ⟨σx⊗σx⟩ = 0
        expected = np.array([[1.0, 0.0]])
        np.testing.assert_array_almost_equal(results.real, expected, decimal=10)

if __name__ == "__main__":
    pytest.main([__file__])
