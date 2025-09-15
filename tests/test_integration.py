"""
Integration tests for QuantumStuff library.
These tests verify that different modules work correctly together.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import QuantumStuff
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from QuantumStuff.States import zero, one, plus, random_qubit
from QuantumStuff.Operators import sigmax, sigmay, sigmaz, expect, commutator
from QuantumStuff.Evolution import evolve_unitary, evolve_lindblad
from QuantumStuff.Metrics import fidelity, von_neumann_entropy, purity
from QuantumStuff.utils import ket_to_dm, tensor_product, ptrace

class TestQuantumCircuitSimulation:
    """Test complete quantum circuit simulation workflows"""
    
    def test_single_qubit_circuit(self):
        """Test a simple single-qubit circuit: H-Z-H"""
        # Start with |0⟩
        psi = zero()
        
        # Apply Hadamard: |0⟩ → |+⟩
        H = (sigmax() + sigmaz()) / np.sqrt(2)
        psi = evolve_unitary(H, psi)
        
        # Check we have |+⟩ state
        expected_plus = plus()
        assert np.isclose(abs(np.vdot(psi, expected_plus)), 1.0, atol=1e-10)
        
        # Apply Z gate: |+⟩ → |-⟩
        psi = evolve_unitary(sigmaz(), psi)
        
        # Apply Hadamard again: |-⟩ → |1⟩
        psi = evolve_unitary(H, psi)
        
        # Check we have |1⟩ state
        expected_one = one()
        assert np.isclose(abs(np.vdot(psi, expected_one)), 1.0, atol=1e-10)
    
    def test_bell_state_preparation(self):
        """Test Bell state preparation: H⊗I followed by CNOT"""
        # Start with |00⟩
        psi00 = tensor_product([zero(), zero()])
        
        # Apply H⊗I: |00⟩ → (|00⟩ + |10⟩)/√2
        H = (sigmax() + sigmaz()) / np.sqrt(2)
        I = np.eye(2)
        H_I = tensor_product([H, I])
        psi = evolve_unitary(H_I, psi00)
        
        # Apply CNOT gate
        CNOT = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0], 
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]], dtype=complex)
        psi_bell = evolve_unitary(CNOT, psi)
        
        # Check we have Bell state (|00⟩ + |11⟩)/√2
        expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        assert np.allclose(np.abs(psi_bell), np.abs(expected), atol=1e-10)
        
        # Verify entanglement by checking reduced density matrices
        rho_bell = ket_to_dm(psi_bell)
        rho_A = ptrace(rho_bell, [0])  # First qubit
        
        # Reduced state should be maximally mixed
        expected_mixed = np.eye(2) / 2
        assert np.allclose(rho_A, expected_mixed, atol=1e-10)
        
        # Entropy should be log(2)
        S = von_neumann_entropy(rho_A)
        assert np.isclose(S, np.log(2), atol=1e-10)

class TestDecoherenceSimulation:
    """Test simulation of decoherence processes"""
    
    def test_t1_relaxation(self):
        """Test T1 (amplitude damping) process"""
        # Start in excited state |1⟩
        psi0 = one()
        H = np.zeros((2, 2), dtype=complex)  # No free evolution
        
        # Simulate amplitude damping
        from QuantumStuff.Operators import sigmam
        gamma = 1.0  # Decay rate
        c_ops = [np.sqrt(gamma) * sigmam()]
        
        t = np.array([0, 1, 2, 5])
        rho_t = evolve_lindblad(psi0, H, t, c_ops=c_ops)
        
        # Check that excited state population decays exponentially
        pop_excited = [np.real(rho[1, 1]) for rho in rho_t]
        
        # At t=0, should be in |1⟩
        assert np.isclose(pop_excited[0], 1.0, atol=1e-10)
        
        # Population should decrease monotonically
        for i in range(len(pop_excited) - 1):
            assert pop_excited[i] >= pop_excited[i+1]
        
        # At long times, should approach |0⟩
        assert pop_excited[-1] < 0.1
    
    def test_t2_dephasing(self):
        """Test T2 (pure dephasing) process"""
        # Start in superposition |+⟩
        psi0 = plus()
        H = np.zeros((2, 2), dtype=complex)
        
        # Simulate pure dephasing
        gamma_phi = 1.0
        c_ops = [np.sqrt(gamma_phi) * sigmaz()]
        
        t = np.array([0, 0.5, 1.0, 2.0])
        rho_t = evolve_lindblad(psi0, H, t, c_ops=c_ops)
        
        # Check that coherences decay while populations remain constant
        coherence = [np.abs(rho[0, 1]) for rho in rho_t]
        pop_0 = [np.real(rho[0, 0]) for rho in rho_t]
        pop_1 = [np.real(rho[1, 1]) for rho in rho_t]
        
        # Initial coherence should be 0.5
        assert np.isclose(coherence[0], 0.5, atol=1e-10)
        
        # Coherence should decay
        for i in range(len(coherence) - 1):
            assert coherence[i] >= coherence[i+1]
        
        # Populations should remain roughly constant (0.5 each for |+⟩)
        for pop in pop_0:
            assert np.isclose(pop, 0.5, atol=1e-2)
        for pop in pop_1:
            assert np.isclose(pop, 0.5, atol=1e-2)

class TestQuantumMetricsWorkflow:
    """Test workflows involving quantum metrics and information theory"""
    
    def test_state_discrimination(self):
        """Test quantum state discrimination using fidelity"""
        # Prepare a set of quantum states
        states = [zero(), one(), plus()]
        target_state = plus()
        
        # Calculate fidelities
        fidelities = []
        for state in states:
            rho1 = ket_to_dm(target_state)
            rho2 = ket_to_dm(state)
            f = fidelity(rho1, rho2)
            fidelities.append(f)
        
        # Should have maximum fidelity with itself
        max_fidelity_idx = np.argmax(fidelities)
        assert max_fidelity_idx == 2  # plus() is at index 2
        assert np.isclose(fidelities[max_fidelity_idx], 1.0)
    
    def test_entanglement_measures(self):
        """Test entanglement quantification"""
        # Compare different two-qubit states
        
        # Product state |00⟩
        psi_product = tensor_product([zero(), zero()])
        rho_product = ket_to_dm(psi_product)
        
        # Bell state (|00⟩ + |11⟩)/√2
        psi_bell = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        rho_bell = ket_to_dm(psi_bell)
        
        # Partially entangled state
        psi_partial = np.array([np.sqrt(0.8), 0, 0, np.sqrt(0.2)], dtype=complex)
        rho_partial = ket_to_dm(psi_partial)
        
        # Calculate entanglement entropy (entropy of reduced state)
        entropies = []
        for rho in [rho_product, rho_partial, rho_bell]:
            rho_A = ptrace(rho, [0])
            S = von_neumann_entropy(rho_A)
            entropies.append(S)
        
        # Product state should have zero entanglement
        assert np.isclose(entropies[0], 0.0, atol=1e-10)
        
        # Bell state should have maximum entanglement
        assert np.isclose(entropies[2], np.log(2), atol=1e-10)
        
        # Partial entanglement should be between 0 and log(2)
        assert 0 < entropies[1] < np.log(2)
        
        # Order should be: product < partial < Bell
        assert entropies[0] < entropies[1] < entropies[2]

class TestRandomStateProperties:
    """Test properties of randomly generated quantum states"""
    
    def test_random_state_validity(self):
        """Test that random states satisfy quantum state properties"""
        for n_qubits in range(1, 4):
            for pure in [True, False]:
                # Generate random state
                if pure:
                    psi = random_qubit(n_qubits, pure=True, dm=False)
                    rho = ket_to_dm(psi)
                else:
                    rho = random_qubit(n_qubits, pure=False, dm=True)
                
                # Check normalization
                assert np.isclose(np.trace(rho), 1.0, atol=1e-10)
                
                # Check positive semidefiniteness
                eigenvals = np.linalg.eigvals(rho)
                assert np.all(eigenvals >= -1e-10)
                
                # Check Hermiticity
                assert np.allclose(rho, rho.conj().T, atol=1e-10)
                
                # Check purity
                P = purity(rho)
                if pure:
                    assert np.isclose(P, 1.0, atol=1e-10)
                else:
                    assert P <= 1.0
    
    def test_measurement_statistics(self):
        """Test measurement statistics on random states"""
        # Generate many random single-qubit states
        n_states = 50
        expectation_values = {'x': [], 'y': [], 'z': []}
        
        for _ in range(n_states):
            psi = random_qubit(1, pure=True, dm=False)
            
            # Measure Pauli expectations
            exp_x = expect(psi, sigmax())
            exp_y = expect(psi, sigmay()) 
            exp_z = expect(psi, sigmaz())
            
            expectation_values['x'].append(np.real(exp_x))
            expectation_values['y'].append(np.real(exp_y))
            expectation_values['z'].append(np.real(exp_z))
        
        # Check that expectation values are within bounds [-1, 1]
        for pauli in ['x', 'y', 'z']:
            values = expectation_values[pauli]
            assert all(-1 <= v <= 1 for v in values)
        
        # Check that |⟨σ⟩|² = ⟨σx⟩² + ⟨σy⟩² + ⟨σz⟩² ≤ 1 for pure states
        for i in range(n_states):
            norm_squared = (expectation_values['x'][i]**2 + 
                          expectation_values['y'][i]**2 + 
                          expectation_values['z'][i]**2)
            assert norm_squared <= 1.0 + 1e-10

def run_integration_tests():
    """Run all integration tests"""
    import unittest
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestQuantumCircuitSimulation,
        TestDecoherenceSimulation, 
        TestQuantumMetricsWorkflow,
        TestRandomStateProperties
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
