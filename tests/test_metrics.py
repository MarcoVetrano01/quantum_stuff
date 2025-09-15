import numpy as np
import pytest
import sys
import os

# Add the parent directory to the path so we can import QuantumStuff
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from QuantumStuff.Metrics import (
    fidelity, trace_distance, von_neumann_entropy, 
    mutual_info, purity
)
from QuantumStuff.States import zero, one, plus, minus
from QuantumStuff.utils import ket_to_dm, tensor_product

class TestFidelity:
    def test_identical_pure_states(self):
        """Test fidelity between identical pure states"""
        psi = zero()
        rho1 = ket_to_dm(psi)
        rho2 = ket_to_dm(psi)
        f = fidelity(rho1, rho2)
        assert np.isclose(f, 1.0)
    
    def test_orthogonal_pure_states(self):
        """Test fidelity between orthogonal pure states"""
        rho1 = ket_to_dm(zero())
        rho2 = ket_to_dm(one())
        f = fidelity(rho1, rho2)
        assert np.isclose(f, 0.0)
    
    def test_superposition_states(self):
        """Test fidelity between superposition states"""
        rho1 = ket_to_dm(plus())
        rho2 = ket_to_dm(minus())
        f = fidelity(rho1, rho2)
        assert np.isclose(f, 0.0)  # |+⟩ and |-⟩ are orthogonal
    
    def test_mixed_states(self):
        """Test fidelity between mixed states"""
        # Maximally mixed state
        rho1 = np.eye(2) / 2
        # Pure |0⟩ state
        rho2 = ket_to_dm(zero())
        f = fidelity(rho1, rho2)
        expected = 1/np.sqrt(2)  # Known result
        assert np.isclose(f, expected, atol=1e-10)
    
    def test_fidelity_symmetry(self):
        """Test that fidelity is symmetric"""
        rho1 = ket_to_dm(zero())
        rho2 = ket_to_dm(plus())
        f12 = fidelity(rho1, rho2)
        f21 = fidelity(rho2, rho1)
        assert np.isclose(f12, f21)
    
    def test_fidelity_bounds(self):
        """Test that fidelity is between 0 and 1"""
        states = [zero(), one(), plus(), minus()]
        for i, psi1 in enumerate(states):
            for j, psi2 in enumerate(states):
                rho1 = ket_to_dm(psi1)
                rho2 = ket_to_dm(psi2)
                f = fidelity(rho1, rho2)
                assert 0 <= f <= 1

class TestTraceDistance:
    def test_identical_states(self):
        """Test trace distance between identical states"""
        rho = ket_to_dm(zero())
        d = trace_distance(rho, rho)
        assert np.isclose(d, 0.0)
    
    def test_orthogonal_pure_states(self):
        """Test trace distance between orthogonal pure states"""
        rho1 = ket_to_dm(zero())
        rho2 = ket_to_dm(one())
        d = trace_distance(rho1, rho2)
        assert np.isclose(d, 1.0)
    
    def test_superposition_states(self):
        """Test trace distance between superposition states"""
        rho1 = ket_to_dm(plus())
        rho2 = ket_to_dm(minus())
        d = trace_distance(rho1, rho2)
        assert np.isclose(d, 1.0)  # Orthogonal states
    
    def test_mixed_states(self):
        """Test trace distance with mixed states"""
        # Maximally mixed state
        rho1 = np.eye(2) / 2
        # Pure |0⟩ state  
        rho2 = ket_to_dm(zero())
        d = trace_distance(rho1, rho2)
        expected = 0.5  # Known result
        assert np.isclose(d, expected, atol=1e-10)
    
    def test_trace_distance_symmetry(self):
        """Test that trace distance is symmetric"""
        rho1 = ket_to_dm(zero())
        rho2 = ket_to_dm(plus())
        d12 = trace_distance(rho1, rho2)
        d21 = trace_distance(rho2, rho1)
        assert np.isclose(d12, d21)
    
    def test_trace_distance_bounds(self):
        """Test that trace distance is between 0 and 1"""
        states = [zero(), one(), plus(), minus()]
        for i, psi1 in enumerate(states):
            for j, psi2 in enumerate(states):
                rho1 = ket_to_dm(psi1)
                rho2 = ket_to_dm(psi2)
                d = trace_distance(rho1, rho2)
                assert 0 <= d <= 1

class TestVonNeumannEntropy:
    def test_pure_state_entropy(self):
        """Test that pure states have zero entropy"""
        pure_states = [zero(), one(), plus(), minus()]
        for psi in pure_states:
            rho = ket_to_dm(psi)
            S = von_neumann_entropy(rho)
            assert np.isclose(S, 0.0, atol=1e-10)
    
    def test_maximally_mixed_entropy(self):
        """Test entropy of maximally mixed state"""
        rho = np.eye(2) / 2  # Maximally mixed single qubit
        S = von_neumann_entropy(rho)
        expected = np.log(2)  # log(d) for d-dimensional system
        assert np.isclose(S, expected, atol=1e-10)
    
    def test_two_qubit_maximally_mixed(self):
        """Test entropy of two-qubit maximally mixed state"""
        rho = np.eye(4) / 4
        S = von_neumann_entropy(rho)
        expected = np.log(4)  # log(4) = 2*log(2)
        assert np.isclose(S, expected, atol=1e-10)
    
    def test_entropy_non_negative(self):
        """Test that entropy is always non-negative"""
        # Test various mixed states
        for p in np.linspace(0, 1, 11):
            rho = p * ket_to_dm(zero()) + (1-p) * ket_to_dm(one())
            S = von_neumann_entropy(rho)
            assert S >= -1e-10  # Allow small numerical errors
    
    def test_bell_state_entropy(self):
        """Test entropy of Bell state (entangled pure state)"""
        # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        bell = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        rho = ket_to_dm(bell)
        S = von_neumann_entropy(rho)
        assert np.isclose(S, 0.0, atol=1e-10)  # Pure state

class TestPurity:
    def test_pure_state_purity(self):
        """Test that pure states have purity = 1"""
        pure_states = [zero(), one(), plus(), minus()]
        for psi in pure_states:
            rho = ket_to_dm(psi)
            P = purity(rho)
            assert np.isclose(P, 1.0, atol=1e-10)
    
    def test_maximally_mixed_purity(self):
        """Test purity of maximally mixed state"""
        rho = np.eye(2) / 2
        P = purity(rho)
        expected = 0.5  # 1/d for d-dimensional maximally mixed state
        assert np.isclose(P, expected, atol=1e-10)
    
    def test_two_qubit_maximally_mixed_purity(self):
        """Test purity of two-qubit maximally mixed state"""
        rho = np.eye(4) / 4
        P = purity(rho)
        expected = 0.25  # 1/4
        assert np.isclose(P, expected, atol=1e-10)
    
    def test_purity_bounds(self):
        """Test that purity is between 1/d and 1"""
        # Single qubit mixed states
        for p in np.linspace(0, 1, 11):
            rho = p * ket_to_dm(zero()) + (1-p) * ket_to_dm(one())
            P = purity(rho)
            assert 0.5 <= P <= 1.0  # 1/2 ≤ P ≤ 1 for qubit

class TestMutualInformation:
    def test_product_state_mutual_info(self):
        """Test mutual information for product states (should be zero)"""
        # |0⟩⊗|0⟩ state
        psi1 = zero()
        psi2 = zero()
        psi_product = tensor_product([psi1, psi2])
        rho = ket_to_dm(psi_product)
        
        MI = mutual_info(rho, [0], [1])
        assert np.isclose(MI, 0.0, atol=1e-10)
    
    def test_bell_state_mutual_info(self):
        """Test mutual information for Bell state (maximally entangled)"""
        # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        bell = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        rho = ket_to_dm(bell)
        
        MI = mutual_info(rho, [0], [1])
        expected = 2 * np.log(2)  # Maximum mutual information for qubits
        assert np.isclose(MI, expected, atol=1e-10)
    
    def test_mutual_info_symmetry(self):
        """Test that mutual information is symmetric in subsystems"""
        # Create some entangled state
        bell = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        rho = ket_to_dm(bell)
        
        MI_AB = mutual_info(rho, [0], [1])
        MI_BA = mutual_info(rho, [1], [0])
        assert np.isclose(MI_AB, MI_BA, atol=1e-10)
    
    def test_mutual_info_non_negative(self):
        """Test that mutual information is non-negative"""
        # Test various two-qubit states
        states = [
            tensor_product([zero(), zero()]),
            tensor_product([zero(), one()]),
            tensor_product([plus(), plus()]),
            np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)  # Bell
        ]
        
        for psi in states:
            rho = ket_to_dm(psi)
            MI = mutual_info(rho, [0], [1])
            assert MI >= -1e-10  # Allow small numerical errors

class TestMetricRelations:
    def test_fidelity_trace_distance_relation(self):
        """Test relation between fidelity and trace distance"""
        # For pure states: D(ρ,σ) = √(1 - F²(ρ,σ))
        rho1 = ket_to_dm(zero())
        rho2 = ket_to_dm(plus())
        
        f = fidelity(rho1, rho2)
        d = trace_distance(rho1, rho2)
        
        # For pure states: D = √(2(1-F))
        expected_d = np.sqrt(2 * (1 - f))
        assert np.isclose(d, expected_d, atol=1e-10)
    
    def test_entropy_purity_relation(self):
        """Test relation between entropy and purity"""
        # For qubit: if purity P, then S ≤ -P*log(P) - (1-P)*log(1-P) (for appropriate mixed state)
        for p in np.linspace(0.1, 0.9, 9):
            rho = p * ket_to_dm(zero()) + (1-p) * ket_to_dm(one())
            S = von_neumann_entropy(rho)
            P = purity(rho)
            
            # For this specific form of mixed state
            expected_S = -p * np.log(p) - (1-p) * np.log(1-p)
            assert np.isclose(S, expected_S, atol=1e-10)
            
            # And purity should be p² + (1-p)²
            expected_P = p**2 + (1-p)**2
            assert np.isclose(P, expected_P, atol=1e-10)

if __name__ == "__main__":
    pytest.main([__file__])
