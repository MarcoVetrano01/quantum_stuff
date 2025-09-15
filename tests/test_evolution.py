import numpy as np
import pytest
import sys
import os

# Add the parent directory to the path so we can import QuantumStuff
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from QuantumStuff.Evolution import (
    evolve_unitary, dissipator, evolve_lindblad
)
from QuantumStuff.States import zero, one, plus
from QuantumStuff.Operators import sigmax, sigmay, sigmaz, sigmap, sigmam
from QuantumStuff.utils import is_state, dag

class TestUnitaryEvolution:
    def test_identity_evolution(self):
        """Test evolution with identity operator"""
        psi0 = zero()
        U = np.eye(2, dtype=complex)
        psi_final = evolve_unitary(U, psi0)
        np.testing.assert_array_almost_equal(psi_final, psi0)
    
    def test_pauli_x_evolution(self):
        """Test evolution with Pauli-X (bit flip)"""
        psi0 = zero()
        U = sigmax()
        psi_final = evolve_unitary(U, psi0)
        expected = one()
        np.testing.assert_array_almost_equal(psi_final, expected)
    
    def test_hadamard_evolution(self):
        """Test evolution with Hadamard gate"""
        psi0 = zero()
        H = (sigmax() + sigmaz()) / np.sqrt(2)
        psi_final = evolve_unitary(H, psi0)
        expected = plus()
        np.testing.assert_array_almost_equal(psi_final, expected)
    
    def test_unitary_preservation(self):
        """Test that unitary evolution preserves normalization"""
        psi0 = np.array([0.6, 0.8], dtype=complex)
        theta = np.pi/4
        U = np.cos(theta/2) * np.eye(2) - 1j * np.sin(theta/2) * sigmax()
        psi_final = evolve_unitary(U, psi0)
        
        # Check normalization is preserved
        assert np.isclose(np.linalg.norm(psi_final), 1.0)
    
    def test_batch_evolution(self):
        """Test batch evolution of multiple states"""
        states = np.array([zero(), one()], dtype=complex)
        U = sigmax()
        evolved_states = evolve_unitary(U, states)
        
        # |0⟩ should become |1⟩ and vice versa
        np.testing.assert_array_almost_equal(evolved_states[0], one())
        np.testing.assert_array_almost_equal(evolved_states[1], zero())

class TestDissipator:
    def test_amplitude_damping(self):
        """Test amplitude damping dissipator"""
        rho = np.array([[0, 0], [0, 1]], dtype=complex)  # |1⟩⟨1|
        L = sigmam()  # Lowering operator
        drho_dt = dissipator(rho, L)
        
        # Should have negative contribution to |1⟩⟨1| population
        assert drho_dt[1, 1] < 0
        # Should have positive contribution to |0⟩⟨0| population
        assert drho_dt[0, 0] > 0
    
    def test_dephasing(self):
        """Test pure dephasing"""
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)  # |+⟩⟨+|
        L = sigmaz()
        drho_dt = dissipator(rho, L)
        
        # Diagonal elements should be preserved
        assert np.isclose(drho_dt[0, 0], 0.0, atol=1e-10)
        assert np.isclose(drho_dt[1, 1], 0.0, atol=1e-10)
        
        # Off-diagonal elements should decay
        assert drho_dt[0, 1] < 0
        assert drho_dt[1, 0] < 0
    
    def test_multiple_lindblad_operators(self):
        """Test dissipator with multiple Lindblad operators"""
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        L_list = [sigmam(), sigmaz()]
        drho_dt = dissipator(rho, L_list)
        
        # Should be sum of individual dissipators
        drho_dt_individual = sum(dissipator(rho, L) for L in L_list)
        np.testing.assert_array_almost_equal(drho_dt, drho_dt_individual)

class TestLindbladEvolution:
    def test_free_evolution(self):
        """Test Lindblad evolution with only Hamiltonian (no dissipation)"""
        psi0 = zero()
        H = sigmaz()  # Energy splitting
        t = np.linspace(0, np.pi, 10)
        
        # No collapse operators
        rho_t = evolve_lindblad(psi0, H, t, c_ops=[])
        
        # Should evolve unitarily
        for i, rho in enumerate(rho_t):
            # Trace should be preserved
            assert np.isclose(np.trace(rho), 1.0)
            # Should remain pure (Tr(ρ²) = 1)
            assert np.isclose(np.trace(rho @ rho), 1.0, atol=1e-6)
    
    def test_amplitude_damping_evolution(self):
        """Test evolution with amplitude damping"""
        psi0 = one()  # Start in excited state
        H = np.zeros((2, 2), dtype=complex)  # No Hamiltonian
        t = np.linspace(0, 5, 50)
        gamma = 1.0
        c_ops = [np.sqrt(gamma) * sigmam()]
        
        rho_t = evolve_lindblad(psi0, H, t, c_ops=c_ops)
        
        # Population should decay from |1⟩ to |0⟩
        pop_excited = [np.real(rho[1, 1]) for rho in rho_t]
        pop_ground = [np.real(rho[0, 0]) for rho in rho_t]
        
        # Excited state population should decrease
        assert pop_excited[0] > pop_excited[-1]
        # Ground state population should increase
        assert pop_ground[0] < pop_ground[-1]
        # Total population should be conserved
        for rho in rho_t:
            assert np.isclose(np.trace(rho), 1.0, atol=1e-6)
    
    def test_dephasing_evolution(self):
        """Test evolution with pure dephasing"""
        psi0 = plus()  # Start in superposition
        H = np.zeros((2, 2), dtype=complex)
        t = np.linspace(0, 2, 20)
        gamma = 1.0
        c_ops = [np.sqrt(gamma) * sigmaz()]
        
        rho_t = evolve_lindblad(psi0, H, t, c_ops=c_ops)
        
        # Coherences should decay while populations remain constant
        coherence = [np.abs(rho[0, 1]) for rho in rho_t]
        pop_0 = [np.real(rho[0, 0]) for rho in rho_t]
        pop_1 = [np.real(rho[1, 1]) for rho in rho_t]
        
        # Coherence should decay
        assert coherence[0] > coherence[-1]
        # Populations should remain roughly constant
        assert np.isclose(pop_0[0], pop_0[-1], atol=0.1)
        assert np.isclose(pop_1[0], pop_1[-1], atol=0.1)

class TestQuantumStates:
    def test_state_preservation_properties(self):
        """Test that evolution preserves quantum state properties"""
        # Start with various initial states
        initial_states = [zero(), one(), plus(), 
                         np.array([0.6, 0.8], dtype=complex)]
        
        H = 0.5 * sigmaz()
        t = np.array([0, 1, 2])
        
        for psi0 in initial_states:
            rho_t = evolve_lindblad(psi0, H, t, c_ops=[])
            
            for rho in rho_t:
                # Trace should be 1
                assert np.isclose(np.trace(rho), 1.0, atol=1e-10)
                # Should be positive semidefinite
                eigenvals = np.linalg.eigvals(rho)
                assert np.all(eigenvals >= -1e-10)  # Allow small numerical errors
                # Hermiticity
                assert np.allclose(rho, dag(rho), atol=1e-10)

if __name__ == "__main__":
    pytest.main([__file__])
