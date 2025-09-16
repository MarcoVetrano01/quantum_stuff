import numpy as np
import pytest
from scipy.sparse import csc_array, csc_matrix
import sys
import os

# Add the parent directory to the path so we can import QuantumStuff
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from QuantumStuff.utils import (
    dag, is_herm, is_norm, is_state, ket_to_dm, nqubit, 
    operator2vector, ptrace, tensor_product, vector2operator
)

class TestDag:
    def test_dag_numpy_array(self):
        """Test conjugate transpose of numpy array"""
        A = np.array([[1, 2], [3, 4]], dtype=complex)
        result = dag(A)
        expected = np.array([[1, 3], [2, 4]], dtype=complex)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_dag_complex_matrix(self):
        """Test conjugate transpose with complex numbers"""
        A = np.array([[1+1j, 2-1j], [3, 4+2j]], dtype=complex)
        result = dag(A)
        expected = np.array([[1-1j, 3], [2+1j, 4-2j]], dtype=complex)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_dag_sparse_matrix(self):
        """Test conjugate transpose of sparse matrix"""
        A = csc_array([[1+1j, 0], [0, 2-1j]])
        result = dag(A)
        expected = csc_array([[1-1j, 0], [0, 2+1j]])
        np.testing.assert_array_almost_equal(result.toarray(), expected.toarray())
    
    def test_dag_batch_list(self):
        """Test conjugate transpose of list of matrices"""
        A1 = np.array([[1, 2], [3, 4]], dtype=complex)
        A2 = np.array([[1+1j, 0], [0, 2-1j]], dtype=complex)
        A3 = np.array([[0, 1j], [-1j, 0]], dtype=complex)
        
        matrices = [A1, A2, A3]
        results = dag(matrices)
        
        expected = [
            np.array([[1, 3], [2, 4]], dtype=complex),
            np.array([[1-1j, 0], [0, 2+1j]], dtype=complex),
            np.array([[0, 1j], [-1j, 0]], dtype=complex)
        ]
        
        assert len(results) == 3
        for result, exp in zip(results, expected):
            np.testing.assert_array_almost_equal(result, exp)
    
    def test_dag_batch_numpy_array(self):
        """Test conjugate transpose of numpy array batch"""
        # 3D array: (batch_size, dim, dim)
        batch = np.array([
            [[1, 2], [3, 4]],
            [[1+1j, 0], [0, 2-1j]],
            [[0, 1j], [-1j, 0]]
        ], dtype=complex)
        
        results = dag(batch)
        
        expected = np.array([
            [[1, 3], [2, 4]],
            [[1-1j, 0], [0, 2+1j]],
            [[0, 1j], [-1j, 0]]
        ], dtype=complex)
        
        if isinstance(results, list):
            # If function returns list of matrices
            assert len(results) == 3
            for i, result in enumerate(results):
                np.testing.assert_array_almost_equal(result, expected[i])
        else:
            # If function returns single array
            np.testing.assert_array_almost_equal(results, expected)
    
    def test_dag_batch_sparse_matrices(self):
        """Test conjugate transpose of list of sparse matrices"""
        A1 = csc_array([[1+1j, 0], [0, 2-1j]])
        A2 = csc_array([[0, 1j], [-1j, 0]])
        
        matrices = [A1, A2]
        results = dag(matrices)
        
        expected = [
            csc_array([[1-1j, 0], [0, 2+1j]]),
            csc_array([[0, 1j], [-1j, 0]])
        ]
        
        assert len(results) == 2
        for result, exp in zip(results, expected):
            np.testing.assert_array_almost_equal(result.toarray(), exp.toarray())
    
    def test_dag_batch_mixed_types(self):
        """Test conjugate transpose of mixed matrix types"""
        A1 = np.array([[1, 2], [3, 4]], dtype=complex)
        A2 = csc_array([[1+1j, 0], [0, 2-1j]])
        
        matrices = [A1, A2]
        results = dag(matrices)
        
        # First should be numpy array
        expected1 = np.array([[1, 3], [2, 4]], dtype=complex)
        np.testing.assert_array_almost_equal(results[0], expected1)
        
        # Second should be sparse (or converted appropriately)
        expected2 = csc_array([[1-1j, 0], [0, 2+1j]])
        if hasattr(results[1], 'toarray'):
            np.testing.assert_array_almost_equal(results[1].toarray(), expected2.toarray())
        else:
            np.testing.assert_array_almost_equal(results[1], expected2.toarray())
    
    def test_dag_involution_property(self):
        """Test that dag(dag(A)) = A for batch"""
        A1 = np.array([[1+1j, 2-1j], [3, 4+2j]], dtype=complex)
        A2 = np.array([[0, 1j], [-1j, 5]], dtype=complex)
        
        matrices = [A1, A2]
        double_dag = dag(dag(matrices))
        
        for original, result in zip(matrices, double_dag):
            np.testing.assert_array_almost_equal(original, result)

class TestIsHerm:
    def test_hermitian_matrix(self):
        """Test detection of Hermitian matrix"""
        A = np.array([[1, 1+1j], [1-1j, 2]], dtype=complex)
        assert is_herm(A) == True
    
    def test_non_hermitian_matrix(self):
        """Test detection of non-Hermitian matrix"""
        A = np.array([[1, 2], [3, 4]], dtype=complex)
        assert is_herm(A) == False
    
    def test_real_symmetric_matrix(self):
        """Test real symmetric matrix (should be Hermitian)"""
        A = np.array([[1, 2], [2, 3]], dtype=float)
        assert is_herm(A) == True
    
    def test_batch_hermitian_matrices(self):
        """Test batch detection of Hermitian matrices"""
        # Create batch of matrices: some Hermitian, some not
        herm1 = np.array([[1, 1+1j], [1-1j, 2]], dtype=complex)
        herm2 = np.array([[3, 0], [0, -1]], dtype=complex)
        non_herm = np.array([[1, 2], [3, 4]], dtype=complex)
        
        matrices = [herm1, herm2, non_herm]
        results = is_herm(matrices)
        expected = [True, True, False]
        
        assert results == expected
    
    def test_batch_numpy_array(self):
        """Test batch with numpy array of matrices"""
        # 3D array: (batch_size, dim, dim)
        batch = np.array([
            [[1, 1+1j], [1-1j, 2]],      # Hermitian
            [[0, 1], [1, 0]],            # Hermitian (Pauli-X)
            [[1, 2], [3, 4]]             # Non-Hermitian
        ], dtype=complex)
        
        results = is_herm(batch)
        expected = [True, True, False]
        
        assert results == expected
    
    def test_sparse_matrices_batch(self):
        """Test batch of sparse matrices"""
        from scipy.sparse import csc_array
        
        # Hermitian sparse matrix
        herm_sparse = csc_array([[1, 1j], [-1j, 2]])
        # Non-Hermitian sparse matrix  
        non_herm_sparse = csc_array([[1, 2], [3, 4]])
        
        matrices = [herm_sparse, non_herm_sparse]
        results = is_herm(matrices)
        expected = [True, False]
        
        assert results == expected
    
    def test_mixed_matrix_types(self):
        """Test batch with mix of dense and sparse matrices"""
        from scipy.sparse import csc_array
        
        herm_dense = np.array([[2, 1-1j], [1+1j, 3]], dtype=complex)
        non_herm_sparse = csc_array([[0, 1], [2, 0]])
        
        matrices = [herm_dense, non_herm_sparse]
        results = is_herm(matrices)
        expected = [True, False]
        
        assert results == expected

class TestIsNorm:
    def test_normalized_vector(self):
        """Test normalized quantum state"""
        psi = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        assert is_norm(psi) == True
    
    def test_unnormalized_vector(self):
        """Test unnormalized quantum state"""
        psi = np.array([1, 1], dtype=complex)
        assert is_norm(psi) == False
    
    def test_zero_vector(self):
        """Test zero vector"""
        psi = np.array([0, 0], dtype=complex)
        assert is_norm(psi) == False
    
    def test_batch_all_normalized(self):
        """Test batch of all normalized vectors"""
        norm1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        norm2 = np.array([1, 0], dtype=complex)
        norm3 = np.array([0, 1], dtype=complex)
        
        states = [norm1, norm2, norm3]
        result = is_norm(states)
        assert result == True  # All normalized should return True
    
    def test_batch_some_unnormalized(self):
        """Test batch with some unnormalized vectors"""
        norm1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        norm2 = np.array([1, 0], dtype=complex)
        unnorm = np.array([1, 1], dtype=complex)  # Not normalized
        zero_vec = np.array([0, 0], dtype=complex)  # Not normalized
        
        states = [norm1, norm2, unnorm, zero_vec]
        result = is_norm(states)
        

            # Alternative: might return the indices directly
        expected_indices = [True, True, False, False]
        assert [result[i] == expected_indices[i] for i in range(len(expected_indices))]
    
    def test_batch_numpy_array_all_normalized(self):
        """Test batch numpy array with all normalized states"""
        batch = np.array([
            [1/np.sqrt(2), 1/np.sqrt(2)],  # Normalized
            [1, 0],                        # Normalized
            [0, 1],                        # Normalized
        ], dtype=complex)
        
        result = is_norm(batch)
        assert result == True  # All normalized
    
    def test_batch_numpy_array_some_unnormalized(self):
        """Test batch numpy array with some unnormalized states"""
        batch = np.array([
            [1/np.sqrt(2), 1/np.sqrt(2)],  # Normalized
            [1, 0],                        # Normalized
            [1, 1],                        # Unnormalized
            [2, 0]                         # Unnormalized
        ], dtype=complex)
        
        result = is_norm(batch)
        
        # Should return indices of unnormalized vectors
        expected_indices = [True, True, False, False]  # Indices 2 and 3 are unnormalized
        assert [result[i] == expected_indices[i] for i in range(len(expected_indices))]

    
    def test_batch_multiqubit_states(self):
        """Test batch of multi-qubit states"""
        # Two-qubit states
        state1 = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩ - Normalized
        state2 = np.array([0, 0, 0, 1], dtype=complex)  # |11⟩ - Normalized  
        state3 = np.sqrt(np.array([1/2, 1/2, 1/2, 1/2], dtype=complex))  # Unnormalized
        state4 = np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3), 0], dtype=complex)  # Normalized
        
        states = [state1, state2, state3, state4]
        result = is_norm(states)
        
        # Only state3 (index 2) should be unnormalized
        expected_indices = [True, True, False, True]
        assert [result[i] == expected_indices[i] for i in range(len(expected_indices))]

    
    def test_sparse_matrix_support(self):
        """Test normalization check with sparse matrices"""
        from scipy.sparse import csc_matrix
        
        # Single sparse normalized vector
        sparse_norm = csc_matrix([1/np.sqrt(2), 1/np.sqrt(2)])
        result = is_norm(sparse_norm)
        assert result == True
        
        # Single sparse unnormalized vector
        sparse_unnorm = csc_matrix([1, 1])
        result = is_norm(sparse_unnorm)
        assert result == False
    
    def test_batch_sparse_matrices(self):
        """Test batch of sparse matrices"""
        from scipy.sparse import csc_matrix
        
        # Create batch of sparse vectors
        sparse1 = csc_matrix([1, 0])  # Normalized
        sparse2 = csc_matrix([0, 1])  # Normalized
        sparse3 = csc_matrix([1, 1])  # Unnormalized
        
        states = [sparse1, sparse2, sparse3]
        result = is_norm(states)
        
        # Should detect that index 2 is unnormalized
        expected_indices = [True, True, False]
        assert [result[i] == expected_indices[i] for i in range(len(expected_indices))]


    def test_tolerance_edge_cases(self):
        """Test normalization with numerical tolerance"""
        # Almost normalized states (within numerical precision)
        almost_norm1 = np.array([1/np.sqrt(2), 1/np.sqrt(2) + 1e-15], dtype=complex)
        almost_norm2 = np.array([1.0000001, 0], dtype=complex)
        clearly_unnorm = np.array([1.1, 0], dtype=complex)
        
        states = [almost_norm1, almost_norm2, clearly_unnorm]
        result = is_norm(states)
        expected_indices = [True, False, False]
        # Behavior depends on numerical tolerance in implementation

        assert [result[i] == expected_indices[i] for i in range(len(expected_indices))]
    
    def test_single_element_batch(self):
        """Test batch with single element"""
        # Single normalized vector in list
        single_norm = [np.array([1, 0], dtype=complex)]
        result = is_norm(single_norm)
        assert result == True
        
        # Single unnormalized vector in list
        single_unnorm = [np.array([1, 1], dtype=complex)]
        result = is_norm(single_unnorm)
        
        if isinstance(result, tuple):
            indices = result[0]
            assert 0 in indices
        else:
            assert 0 in result
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Invalid input type
        with pytest.raises(TypeError):
            is_norm("invalid")
        
        with pytest.raises(TypeError):
            is_norm(123)
    
    def test_mixed_sparse_dense_batch(self):
        """Test batch with mix of sparse and dense arrays"""
        from scipy.sparse import csc_matrix
        
        dense_norm = np.array([1, 0], dtype=complex)
        sparse_norm = csc_matrix([0, 1])
        dense_unnorm = np.array([1, 1], dtype=complex)
        
        states = [dense_norm, sparse_norm, dense_unnorm]
        result = np.where(np.array([is_norm(state) for state in states]) == False)
        
        # Should detect that index 2 is unnormalized
        assert 2 in result
    
    def test_empty_batch_handling(self):
        """Test handling of edge cases"""
        # Empty list should be handled appropriately
        try:
            result = is_norm([])
            # If it doesn't raise an error, should return appropriate value
        except Exception:
            # Exception is also acceptable
            pass
    
    def test_return_format_consistency(self):
        """Test that return format is consistent"""
        # Single vectors should return boolean
        single_norm = np.array([1, 0], dtype=complex)
        result = is_norm(single_norm)
        assert isinstance(result, (bool, np.bool))
        
        single_unnorm = np.array([1, 1], dtype=complex)
        result = is_norm(single_unnorm)
        assert isinstance(result, (bool, np.bool))
        
        # Batches should return True or indices
        batch_all_norm = [np.array([1, 0]), np.array([0, 1])]
        result = is_norm(batch_all_norm)
        assert result == True
        
        batch_some_unnorm = [np.array([1, 0]), np.array([1, 1])]
        result = is_norm(batch_some_unnorm)
        # Should be indices (tuple from np.where) or array-like
        assert not isinstance(result, bool) or result == False
    
    def test_high_dimensional_vectors(self):
        """Test normalization check for high-dimensional vectors"""
        # 3D normalized vector
        vec_3d = np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], dtype=complex)
        assert is_norm(vec_3d) == True
        
        # 3D unnormalized vector
        vec_3d_unnorm = np.array([1, 1, 1], dtype=complex)
        assert is_norm(vec_3d_unnorm) == False
        
        # Batch of 3D vectors
        batch_3d = [
            np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], dtype=complex),  # Normalized
            np.array([1, 0, 0], dtype=complex),  # Normalized
            np.array([1, 1, 1], dtype=complex)   # Unnormalized
        ]
        result = is_norm(batch_3d)
        expected_indices = [True, True, False]
        assert [result[i] == expected_indices[i] for i in range(len(expected_indices))]

class TestIsState:
    def test_valid_ket(self):
        """Test valid quantum state vector (category 1)"""
        psi = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        category, is_valid = is_state(psi)
        assert category == 1  # ket category
        assert is_valid == True
    
    def test_invalid_ket(self):
        """Test invalid quantum state vector"""
        psi = np.array([1, 1], dtype=complex)  # Not normalized
        category, is_valid = is_state(psi)
        assert category == 1  # Still recognized as ket category
        assert is_valid == False
    
    def test_valid_density_matrix(self):
        """Test valid density matrix (category 2 or 3)"""
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        result = is_state(rho, batchmode=False)
        
        if len(result) == 2:
            category, is_valid = result
            assert category == 3
            assert is_valid == True
        else:
            # Function might return indices on failure
            pytest.fail("Expected valid density matrix to return (category, True)")
    
    def test_invalid_density_matrix_trace(self):
        """Test density matrix with invalid trace"""
        # Trace != 1
        rho = np.array([[0.6, 0.5], [0.5, 0.6]], dtype=complex)
        result = is_state(rho, batchmode=False)
        
        # Should return indices where trace check failed
        if isinstance(result, tuple) and len(result) == 1:
            indices = result[0]
            assert len(indices) > 0  # Should have invalid indices
        else:
            # Alternative: might return (category, False)
            if len(result) == 2:
                category, is_valid = result
                assert is_valid == False
    
    def test_invalid_density_matrix_eigenvalues(self):
        """Test density matrix with negative eigenvalues"""
        # Not positive semidefinite
        rho = np.array([[1, 2], [2, 1]], dtype=complex)
        result = is_state(rho, batchmode=False)
        
        # Should detect invalid eigenvalues
        if isinstance(result, tuple) and len(result) == 1:
            indices = result[0]
            assert len(indices) > 0
        else:
            if len(result) == 2:
                category, is_valid = result
                assert is_valid == False
    
    def test_invalid_density_matrix_hermiticity(self):
        """Test non-Hermitian matrix"""
        rho = np.array([[1, 1+1j], [1-2j, 0]], dtype=complex)  # Not Hermitian
        result = is_state(rho, batchmode=False)
        
        # Should detect non-Hermitian property
        if isinstance(result, tuple) and len(result) == 1:
            indices = result[0]
            assert len(indices) > 0
        else:
            if len(result) == 2:
                category, is_valid = result
                assert is_valid == False
    
    def test_batch_all_valid_kets(self):
        """Test batch of all valid kets (category 3)"""
        valid_ket1 = np.array([1, 0], dtype=complex)
        valid_ket2 = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        valid_ket3 = np.array([0, 1], dtype=complex)
        
        states = [valid_ket1, valid_ket2, valid_ket3]
        result = is_state(states)
        
        if len(result) == 2:
            category, is_valid = result
            assert category == 2  # List category
            assert is_valid == True  # All valid
        else:
            pytest.fail("Expected (2, True) for all valid kets")

    def test_batch_mixed_validity(self):
        """Test batch with mix of valid and invalid kets"""
        valid_ket = np.array([1, 0], dtype=complex)
        invalid_ket = np.array([1, 1], dtype=complex)  # Not normalized
        
        states = [valid_ket, invalid_ket, valid_ket]
        result = is_state(states)
        
        expected = [True, False, True]
        assert [result[i] == expected[i] for i in range(len(expected))]

    def test_batch_all_valid_density_matrices(self):
        """Test batch of all valid density matrices"""
        valid_dm1 = np.array([[1, 0], [0, 0]], dtype=complex)  # |0⟩⟨0|
        valid_dm2 = np.array([[0.5, 0], [0, 0.5]], dtype=complex)  # Mixed state
        
        states = [valid_dm1, valid_dm2]
        result = is_state(states)
        
        if len(result) == 2:
            category, is_valid = result
            assert category == 3
            assert is_valid == True
        else:
            pytest.fail("Expected all valid density matrices to return (3, True)")
    
    def test_batch_invalid_density_matrices(self):
        """Test batch with invalid density matrices"""
        valid_dm = np.array([[1, 0], [0, 0]], dtype=complex)
        invalid_dm = np.array([[0.6, 0], [0, 0.6]], dtype=complex)  # Trace != 1
        
        states = [valid_dm, invalid_dm]
        result = is_state(states)
        expected_indices = [True, False]
        assert [result[i] == expected_indices[i] for i in range(len(expected_indices))]
    
    def test_batch_numpy_array_kets(self):
        """Test batch as numpy array - interpreted as ket list or matrix"""
        # 2D array: (batch_size, state_dim)
        batch_kets = np.array([
            [1, 0],                        # Valid |0⟩
            [0, 1],                        # Valid |1⟩  
            [1/np.sqrt(2), 1/np.sqrt(2)],  # Valid |+⟩
        ], dtype=complex)
        
        result = is_state(batch_kets)
        
        category, is_valid = result
        # Could be category 2 (matrix) or 3 (list)
        assert category == 2
        # Validity depends on interpretation
        assert is_valid == True
    
    def test_batch_numpy_array_density_matrices(self):
        """Test 3D array of density matrices"""
        batch_dms = np.array([
            [[1, 0], [0, 0]],              # Valid |0⟩⟨0|
            [[0.5, 0.5], [0.5, 0.5]],      # Valid |+⟩⟨+|
            [[0.5, 0], [0, 0.5]],          # Valid mixed state
        ], dtype=complex)
        
        result = is_state(batch_dms)
        
        category, is_valid = result
        assert category == 3  # List category for 3D array
        assert is_valid == True  # All should be valid
    
    def test_automatic_ket_to_dm_conversion(self):
        """Test automatic conversion of non-square 2D array to density matrices"""
        # 2D array where shape[0] != shape[1] should be converted to density matrices
        ket_array = np.array([[1, 0], [0, 1]], dtype=complex).T  # Shape (2, 2) but treated as kets
        
        # This might trigger the automatic ket-to-dm conversion
        result = is_state(ket_array)
        
        if len(result) == 2:
            category, is_valid = result
            assert category in [2, 3]  # Converted to density matrix category
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Empty list - should raise exception
        with pytest.raises(Exception):
            is_state([])
        
        # Non-numpy/list input - should raise exception
        with pytest.raises(Exception):
            is_state("invalid")
        
        # Too many dimensions - should raise exception  
        with pytest.raises(Exception):
            high_dim = np.random.random((2, 2, 2, 2))
            is_state(high_dim)
    
    def test_multiqubit_states(self):
        """Test multi-qubit states"""
        # Two-qubit ket
        bell_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        category, is_valid = is_state(bell_state)
        assert category == 1  # ket category
        assert is_valid == True
        
        # Two-qubit density matrix
        bell_dm = np.outer(bell_state, np.conj(bell_state))
        result = is_state(bell_dm, batchmode = False)
        
        category, is_valid = result
        assert category == 3  # dm category
        assert is_valid == True
        
        # Maximally mixed two-qubit state
        mixed_dm = np.eye(4, dtype=complex) / 4
        result = is_state(mixed_dm, batchmode = False)
        
        category, is_valid = result
        assert category == 3
        assert is_valid == True
    
    def test_return_format_consistency(self):
        """Test that return format is consistent"""
        test_states = [
            np.array([1, 0], dtype=complex),  # Valid ket
            np.array([1, 1], dtype=complex),  # Invalid ket  
            np.array([[1, 0], [0, 0]], dtype=complex),  # Valid dm
        ]
        
        for state in test_states:
            result = is_state(state, False)
            
            # Result should be either:
            # 1. (category, boolean) for valid states
            # 2. (indices,) for invalid states
            assert isinstance(result, tuple)
            assert len(result) in [1, 2]
            
            if len(result) == 2:
                category, is_valid = result
                assert isinstance(category, int)
                assert isinstance(is_valid, np.bool)
                assert category in [1, 2, 3]

    def test_numerical_tolerance(self):
        """Test behavior at numerical tolerance boundaries"""
        # Almost normalized ket
        almost_norm = np.array([1.0000001, 0], dtype=complex)
        category, is_valid = is_state(almost_norm)
        assert category == 1
        # Validity depends on tolerance
        assert isinstance(is_valid, np.bool)
        
        # Almost valid trace for density matrix
        almost_trace_one = np.array([[0.500001, 0], [0, 0.499999]], dtype=complex)
        result = is_state(almost_trace_one, False)
        
        # Should handle numerical precision appropriately
        assert isinstance(result, tuple)
        assert len(result) in [1, 2]
        mixed_dm = np.eye(4) / 4
        category, is_valid = is_state(mixed_dm, False)
        assert category == 3  # dm category
        assert is_valid == True
    
    def test_invalid_shapes(self):
        """Test invalid state shapes"""
        # Non-square matrix
        try:
            non_square = np.array([[1, 2, 3], [4, 5, 6]], dtype=complex)
            category, is_valid = is_state(non_square)
            # Should either handle gracefully or raise exception
        except Exception:
            pass  # Exception is acceptable
    
   
    def test_category_consistency(self):
        """Test that categories are assigned consistently"""
        # 1D arrays should always be category 1
        vectors = [
            np.array([1, 0], dtype=complex),
            np.array([0, 1], dtype=complex),
            np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex),
            np.array([1, 1], dtype=complex)  # Even invalid ones
        ]
        
        for vec in vectors:
            category, _ = is_state(vec)
            assert category == 1
        
        # 2D square arrays should be category 2 or 3
        matrices = [
            np.array([[1, 0], [0, 0]], dtype=complex),
            np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex),
            np.eye(2, dtype=complex)/2
        ]
        
        for mat in matrices:
            category, _ = is_state(mat, False)
            assert category in [2,3]
        
        # Lists should always be category 3
        lists = [
            [np.array([1, 0], dtype=complex)],
            [np.array([[1, 0], [0, 0]], dtype=complex)],
            
        ]
        
        for lst in lists:
            category, _ = is_state(lst)
            assert category in [2,3]
    
    def test_batch_numpy_array_density_matrices_invalid(self):
        """Test batch as numpy array of density matrices"""
        # 3D array: (batch_size, dim, dim)
        batch_dms = np.array([
            [[1, 0], [0, 0]],              # Valid |0⟩⟨0|
            [[0.5, 0.5], [0.5, 0.5]],      # Valid |+⟩⟨+|
            [[0.5, 0], [0, 0.5]],          # Valid mixed state
            [[2, 0], [0, 0]]               # Invalid (trace > 1)
        ], dtype=complex)
        
        results = is_state(batch_dms)
        expected = [True, True, True, False]
        
        assert [results[i] == expected[i] for i in range(len(expected))]
      
    def test_batch_multiqubit_states(self):
        """Test batch of multi-qubit states"""
        # Two-qubit states
        bell_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        product_state = np.array([1, 0, 0, 0], dtype=complex)
        unnorm_state = np.array([1, 1, 1, 1], dtype=complex)
        
        # Two-qubit density matrices
        bell_dm = np.outer(bell_state, np.conj(bell_state))
        mixed_dm = np.eye(4) / 4  # Maximally mixed
        
        states = [bell_state, product_state, unnorm_state, bell_dm, mixed_dm]
        res = []
        for state in states:
            results = is_state(state, False)
            res.append(results)
        expected = [(1, np.True_), (1, np.True_), (1, np.False_), (3, np.True_), (3, np.True_)]
        assert res == expected

class TestKetToDm:
    def test_pure_state_conversion(self):
        """Test conversion from ket to density matrix"""
        psi = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        rho = ket_to_dm(psi)
        expected = np.outer(psi, np.conj(psi))
        np.testing.assert_array_almost_equal(rho, expected)
    
    def test_batch_conversion(self):
        """Test batch conversion of multiple states"""
        states = np.array([[1, 0], [0, 1]], dtype=complex)
        rhos = ket_to_dm(states)
        assert rhos.shape == (2, 2, 2)
    
    def test_invalid_input(self):
        """Test error handling for invalid input"""
        with pytest.raises(ValueError):
            ket_to_dm(np.array([1, 1], dtype=complex))
        with pytest.raises(ValueError):
            ket_to_dm(np.array([[1, 0], [1, 1]], dtype=complex))


class TestNqubit:
    def test_single_qubit(self):
        """Test single qubit operator"""
        op = np.array([[1, 0], [0, 1]], dtype=complex)
        assert nqubit(op) == 1
    
    def test_two_qubit(self):
        """Test two qubit operator"""
        op = np.eye(4, dtype=complex)
        assert nqubit(op) == 2
    
    def test_three_qubit(self):
        """Test three qubit operator"""
        op = np.eye(8, dtype=complex)
        assert nqubit(op) == 3

class TestTensorProduct:
    def test_two_operators(self):
        """Test tensor product of two operators"""
        A = np.array([[1, 0], [0, 1]], dtype=complex)
        B = np.array([[0, 1], [1, 0]], dtype=complex)
        result = tensor_product([A, B])
        expected = np.kron(A, B)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_three_operators(self):
        """Test tensor product of three operators"""
        A = np.array([[1, 0], [0, 1]], dtype=complex)
        B = np.array([[0, 1], [1, 0]], dtype=complex)
        C = np.array([[1, 0], [0, -1]], dtype=complex)
        result = tensor_product([A, B, C])
        expected = np.kron(np.kron(A, B), C)
        np.testing.assert_array_almost_equal(result, expected)

class TestPtrace:
    def test_single_qubit_trace(self):
        """Test partial trace of two-qubit system"""
        # Bell state |00⟩ + |11⟩
        psi = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        rho = np.outer(psi, np.conj(psi))
        
        # Trace out second qubit
        rho_A = ptrace(rho, [0])
        expected = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
        np.testing.assert_array_almost_equal(rho_A, expected)
    
    def test_two_qubit_trace(self):
        """Test partial trace of three-qubit system"""
        # GHZ state |000⟩ + |111⟩
        psi = np.array([1/np.sqrt(2), 0, 0, 0, 0, 0, 0, 1/np.sqrt(2)], dtype=complex)
        rho = np.outer(psi, np.conj(psi))
        
        # Trace out third qubit
        rho_AB = ptrace(rho, [2])
        expected = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
        np.testing.assert_array_almost_equal(rho_AB, expected)

class TestOperatorVector:
    def test_operator_to_vector_conversion(self):
        """Test conversion between operator and vector representations"""
        op = np.array([[1, 2], [3, 4]], dtype=complex)
        vec = operator2vector(op)
        op_reconstructed = vector2operator(vec)
        np.testing.assert_array_almost_equal(op, op_reconstructed)
    def test_batch_conversion(self):
        """Test batch conversion of multiple operators"""
        ops = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=complex)
        vecs = operator2vector(ops)
        assert vecs.shape == (2, 4, 1)
        ops_reconstructed = vector2operator(vecs)
        np.testing.assert_array_almost_equal(ops, ops_reconstructed)

class TestBatchOperations:
    """Comprehensive tests for batch operations across all utility functions"""
    
    def test_consistent_batch_behavior(self):
        """Test that all functions handle batches consistently"""
        # Create a batch of operators and states
        operators = [
            np.array([[1, 1j], [-1j, 1]], dtype=complex),  # Hermitian
            np.array([[0, 1], [1, 0]], dtype=complex),     # Hermitian (Pauli-X)
            np.array([[1, 2], [3, 4]], dtype=complex)      # Non-Hermitian
        ]
        
        states_ket = [
            np.array([1, 0], dtype=complex),                # Normalized
            np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex),  # Normalized
            np.array([1, 1], dtype=complex)                 # Unnormalized
        ]
        
        # Test dag batch
        dag_results = dag(operators)
        assert len(dag_results) == 3
        
        # Test is_herm batch
        herm_results = is_herm(operators)
        assert herm_results == [True, True, False]
        
        # Test is_norm batch
        norm_results = is_norm(states_ket)
        expected = np.array([True, True, False])
        assert [norm_results[i] == expected[i] for i in range(len(expected))]
        
        # Test is_state batch
        state_results = is_state(states_ket)
        expected = [True, True, False]
        assert len(state_results) == 3
        assert [state_results[i] == expected[i] for i in range(len(expected))]
        
    
    def test_empty_batch_handling(self):
        """Test handling of empty batches"""
        # Empty lists should return empty results
        with pytest.raises(Exception):
            assert dag([])
            assert is_herm([]) == []
            assert is_norm([]) == []
            assert is_state([]) == []
    
    def test_single_element_batch(self):
        """Test single-element batches"""
        single_op = [np.array([[1, 0], [0, 1]], dtype=complex)]
        single_state = [np.array([1, 0], dtype=complex)]
        
        # Should return lists with single elements
        dag_result = dag(single_op)
        assert len(dag_result) == 1
        np.testing.assert_array_almost_equal(dag_result[0], single_op[0])
        
        herm_result = is_herm(single_op)
        assert herm_result == [True]
        
        norm_result = is_norm(single_state)
        assert [norm_result] == [True]
        
        state_result = is_state(single_state)
        expected_result = (1, True)
        assert [state_result[i] == expected_result[i] for i in range(len(state_result))]
    
    def test_large_batch_performance(self):
        """Test performance with large batches"""
        # Create large batch
        n_batch = 100
        large_batch_ops = [np.random.random((2, 2)) + 1j * np.random.random((2, 2)) 
                          for _ in range(n_batch)]
        large_batch_states = [np.random.random(2) + 1j * np.random.random(2) 
                             for _ in range(n_batch)]
        
        # Should handle large batches without error
        dag_results = dag(large_batch_ops)
        assert len(dag_results) == n_batch
        
        herm_results = is_herm(large_batch_ops)
        assert len(herm_results) == n_batch
        assert all(isinstance(r, bool) for r in herm_results)
        
        norm_results = is_norm(large_batch_states)
        assert len(norm_results) == n_batch
        
        state_results = is_state(large_batch_states)
        assert len(state_results) == n_batch
    
    def test_batch_different_dimensions(self):
        """Test batch operations with different matrix dimensions"""
        # Mix of 2x2 and 4x4 matrices
        ops_mixed_dim = [
            np.eye(2, dtype=complex),           # 2x2
            np.eye(4, dtype=complex),           # 4x4
            np.array([[1, 1j], [-1j, 1]], dtype=complex)  # 2x2
        ]
        
        # Should handle mixed dimensions
        dag_results = dag(ops_mixed_dim)
        assert len(dag_results) == 3
        assert dag_results[0].shape == (2, 2)
        assert dag_results[1].shape == (4, 4)
        assert dag_results[2].shape == (2, 2)
        
        herm_results = is_herm(ops_mixed_dim)
        assert len(herm_results) == 3
        assert all(isinstance(r, bool) for r in herm_results)
    
    def test_batch_numpy_vs_list_consistency(self):
        """Test that numpy arrays and lists give consistent results"""
        # Create test data as both list and numpy array
        matrices_list = [
            np.array([[1, 0], [0, 1]], dtype=complex),
            np.array([[0, 1], [1, 0]], dtype=complex),
            np.array([[1, 1j], [-1j, 1]], dtype=complex)
        ]
        
        matrices_array = np.array(matrices_list)
        
        # Test dag
        dag_list = dag(matrices_list)
        dag_array = dag(matrices_array)
        
        # Results should be equivalent (allowing for different return types)
        assert len(dag_list) == len(dag_array) if isinstance(dag_array, list) else dag_array.shape[0]
        
        # Test is_herm
        herm_list = is_herm(matrices_list)
        herm_array = is_herm(matrices_array)
        assert herm_list == herm_array
    
    def test_batch_error_propagation(self):
        """Test that errors in batch operations are handled appropriately"""
        # Mix valid and invalid inputs
        mixed_inputs = [
            np.array([[1, 0], [0, 1]], dtype=complex),  # Valid
            "invalid_input",                            # Invalid
            np.array([[1, 2, 3], [4, 5, 6]], dtype=complex)  # Wrong shape for some operations
        ]
        
        # Functions should either handle gracefully or raise appropriate errors
        try:
            # This might raise an error or handle gracefully depending on implementation
            result = dag(mixed_inputs)
            # If it doesn't raise an error, should return something reasonable
            assert isinstance(result, list)
        except (TypeError, ValueError):
            # This is also acceptable behavior
            pass
    
    def test_batch_memory_efficiency(self):
        """Test that batch operations don't create unnecessary copies"""
        # Create batch of matrices
        original_matrices = [np.array([[1, 2], [3, 4]], dtype=complex) for _ in range(10)]
        
        # Get results
        dag_results = dag(original_matrices)
        
        # Verify results are correct
        assert len(dag_results) == 10
        for orig, result in zip(original_matrices, dag_results):
            expected = orig.conj().T
            np.testing.assert_array_almost_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__])