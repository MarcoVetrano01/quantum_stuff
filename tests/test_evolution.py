import numpy as np
import pytest
import sys
import os
from scipy.sparse import csc_array, csc_matrix
from scipy.linalg import expm

# Add the parent directory to the path so we can import QuantumStuff
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from QuantumStuff.Evolution import (
    evolve_unitary, dissipator, evolve_lindblad, Super_D
)
from QuantumStuff.States import zero, one, plus
from QuantumStuff.Operators import sigmax, sigmay, sigmaz, sigmap, sigmam, hadamard
from QuantumStuff.utils import is_state, dag, ket_to_dm

# Try to import QuTiP for comparison tests
try:
    import qutip as qt
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False
    qt = None

class TestUnitaryEvolution:
    def test_identity_evolution(self):
        """Test evolution with identity operator"""
        psi0 = zero()
        U = np.eye(2, dtype=complex)
        psi_final = evolve_unitary(U, psi0, batchmode=False)
        np.testing.assert_array_almost_equal(psi_final, psi0)
    
    def test_pauli_x_evolution(self):
        """Test evolution with Pauli-X (bit flip)"""
        psi0 = zero()
        U = sigmax()
        psi_final = evolve_unitary(U, psi0, batchmode=False)
        expected = one()
        np.testing.assert_array_almost_equal(psi_final, expected)
    
    def test_hadamard_evolution(self):
        """Test evolution with Hadamard gate"""
        psi0 = zero()
        H = hadamard()
        psi_final = evolve_unitary(H, psi0, batchmode=False)
        expected = plus()
        np.testing.assert_array_almost_equal(psi_final, expected)
    
    def test_unitary_preservation(self):
        """Test that unitary evolution preserves normalization"""
        psi0 = np.array([0.6, 0.8], dtype=complex)
        theta = np.pi/4
        U = np.cos(theta/2) * np.eye(2) - 1j * np.sin(theta/2) * sigmax()
        psi_final = evolve_unitary(U, psi0, batchmode=False)
        
        # Check normalization is preserved
        assert np.isclose(np.linalg.norm(psi_final), 1.0)
    
    def test_batch_evolution(self):
        """Test batch evolution of multiple states"""
        states = np.array([zero(), one()], dtype=complex)
        U = sigmax()
        evolved_states = evolve_unitary(U, states, batchmode=True)
        
        # |0⟩ should become |1⟩ and vice versa
        np.testing.assert_array_almost_equal(evolved_states[0], one())
        np.testing.assert_array_almost_equal(evolved_states[1], zero())

    def test_csc_array_conversion_evolution(self):
        """Test evolution using matrix converted from sparse csc_array"""
        psi0 = zero()
        # Create sparse Pauli-X matrix and convert to dense
        U_sparse = csc_array([[0, 1], [1, 0]], dtype=complex)
        U_dense = U_sparse.toarray()
        psi_final = evolve_unitary(U_dense, psi0, batchmode=False)
        expected = one()
        np.testing.assert_array_almost_equal(psi_final, expected)
    
    def test_csc_matrix_conversion_evolution(self):
        """Test evolution using matrix converted from sparse csc_matrix"""
        psi0 = zero()
        # Create sparse Pauli-X matrix and convert to dense
        U_sparse = csc_matrix([[0, 1], [1, 0]], dtype=complex)
        U_dense = U_sparse.toarray()
        psi_final = evolve_unitary(U_dense, psi0, batchmode=False)
        expected = one()
        np.testing.assert_array_almost_equal(psi_final, expected)
    
    def test_sparse_hadamard_conversion_evolution(self):
        """Test evolution using Hadamard gate converted from sparse"""
        psi0 = zero()
        # Create sparse Hadamard matrix and convert to dense
        H_data = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        H_sparse = csc_array(H_data)
        H_dense = H_sparse.toarray()
        psi_final = evolve_unitary(H_dense, psi0, batchmode=False)
        expected = plus()
        np.testing.assert_array_almost_equal(psi_final, expected)
    
    def test_sparse_unitary_conversion_preservation(self):
        """Test that unitary evolution preserves normalization (sparse converted)"""
        psi0 = np.array([0.6, 0.8], dtype=complex)
        # Create sparse rotation matrix and convert to dense
        theta = np.pi/4
        cos_half = np.cos(theta/2)
        sin_half = np.sin(theta/2)
        U_data = np.array([[cos_half, -1j*sin_half], [-1j*sin_half, cos_half]], dtype=complex)
        U_sparse = csc_array(U_data)
        U_dense = U_sparse.toarray()
        psi_final = evolve_unitary(U_dense, psi0, batchmode=False)
        
        # Check normalization is preserved
        assert np.isclose(np.linalg.norm(psi_final), 1.0)
    
    def test_batch_evolution_sparse_conversion(self):
        """Test batch evolution using matrix converted from sparse"""
        states = np.array([zero(), one()], dtype=complex)
        # Create sparse Pauli-X matrix and convert to dense
        U_sparse = csc_array([[0, 1], [1, 0]], dtype=complex)
        U_dense = U_sparse.toarray()
        evolved_states = evolve_unitary(U_dense, states, batchmode=True)
        
        # |0⟩ should become |1⟩ and vice versa
        np.testing.assert_array_almost_equal(evolved_states[0], one())
        np.testing.assert_array_almost_equal(evolved_states[1], zero())
    
    def test_batch_evolution_mixed_states(self):
        """Test batch evolution with mixed quantum states"""
        # Create superposition states
        psi1 = (zero() + one()) / np.sqrt(2)  # |+⟩
        psi2 = (zero() - one()) / np.sqrt(2)  # |-⟩
        states = np.array([psi1, psi2], dtype=complex)
        
        # Apply Pauli-Z (phase flip)
        U = sigmaz()
        evolved_states = evolve_unitary(U, states, batchmode=True)
        
        # |+⟩ should become |-⟩ and |-⟩ should become |+⟩
        expected1 = (zero() - one()) / np.sqrt(2)  # |-⟩
        expected2 = (zero() + one()) / np.sqrt(2)  # |+⟩
        np.testing.assert_array_almost_equal(evolved_states[0], expected1)
        np.testing.assert_array_almost_equal(evolved_states[1], expected2)
    
    def test_batch_evolution_large_sparse_conversion(self):
        """Test batch evolution with larger matrix converted from sparse (2-qubit system)"""
        # Create 2-qubit states |00⟩ and |11⟩
        psi_00 = np.array([1, 0, 0, 0], dtype=complex)
        psi_11 = np.array([0, 0, 0, 1], dtype=complex)
        states = np.array([psi_00, psi_11], dtype=complex)
        
        # Create sparse CNOT gate and convert to dense
        cnot_data = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0], 
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        U_cnot_sparse = csc_array(cnot_data)
        U_cnot_dense = U_cnot_sparse.toarray()
        
        evolved_states = evolve_unitary(U_cnot_dense, states, batchmode=True)
        
        # CNOT: |00⟩ → |00⟩, |11⟩ → |10⟩
        expected_00 = np.array([1, 0, 0, 0], dtype=complex)
        expected_10 = np.array([0, 0, 1, 0], dtype=complex)
        np.testing.assert_array_almost_equal(evolved_states[0], expected_00)
        np.testing.assert_array_almost_equal(evolved_states[1], expected_10)
    
    def test_sparse_vs_dense_consistency(self):
        """Test that sparse and dense evolution give the same results"""
        psi0 = (zero() + 1j * one()) / np.sqrt(2)
        
        # Dense Pauli-Y
        U_dense = sigmay()
        # Sparse Pauli-Y converted to dense
        U_sparse = csc_array(U_dense)
        U_sparse_converted = U_sparse.toarray()
        
        psi_dense = evolve_unitary(U_dense, psi0, batchmode=False)
        psi_sparse_converted = evolve_unitary(U_sparse_converted, psi0, batchmode=False)
        
        np.testing.assert_array_almost_equal(psi_dense, psi_sparse_converted)
    
    def test_batch_sparse_vs_dense_consistency(self):
        """Test batch evolution consistency between sparse and dense"""
        # Create multiple test states
        states = np.array([
            zero(),
            one(), 
            (zero() + one()) / np.sqrt(2),
            (zero() + 1j * one()) / np.sqrt(2)
        ], dtype=complex)
        
        # Test with rotation gate
        theta = np.pi/3
        U_dense = np.cos(theta/2) * np.eye(2) - 1j * np.sin(theta/2) * sigmaz()
        U_sparse = csc_array(U_dense)
        U_sparse_converted = U_sparse.toarray()
        
        evolved_dense = evolve_unitary(U_dense, states, batchmode=True)
        evolved_sparse_converted = evolve_unitary(U_sparse_converted, states, batchmode=True)
        
        np.testing.assert_array_almost_equal(evolved_dense, evolved_sparse_converted)
    
    def test_sparse_matrix_properties(self):
        """Test that sparse matrices maintain their mathematical properties"""
        # Create sparse Pauli matrices
        sx_sparse = csc_array(sigmax())
        sy_sparse = csc_array(sigmay()) 
        sz_sparse = csc_array(sigmaz())
        
        # Test unitarity: U @ U.H = I
        sx_dense = sx_sparse.toarray()
        assert np.allclose(sx_dense @ sx_dense.conj().T, np.eye(2))
        
        # Test Hermiticity of Pauli matrices
        assert np.allclose(sx_dense, sx_dense.conj().T)
        
        # Test that sparse conversion preserves properties
        assert np.allclose(sx_sparse.toarray(), sigmax())
        assert np.allclose(sy_sparse.toarray(), sigmay())
        assert np.allclose(sz_sparse.toarray(), sigmaz())
    
    def test_batch_evolution_different_sparse_types(self):
        """Test batch evolution with different sparse matrix types"""
        states = np.array([zero(), one()], dtype=complex)
        
        # Test with csc_array
        U_csc_array = csc_array([[0, 1], [1, 0]], dtype=complex)
        evolved_csc_array = evolve_unitary(U_csc_array.toarray(), states, batchmode=True)
        
        # Test with csc_matrix
        U_csc_matrix = csc_matrix([[0, 1], [1, 0]], dtype=complex)
        evolved_csc_matrix = evolve_unitary(U_csc_matrix.toarray(), states, batchmode=True)
        
        # Both should give same results
        np.testing.assert_array_almost_equal(evolved_csc_array, evolved_csc_matrix)
        
        # Both should flip the states
        np.testing.assert_array_almost_equal(evolved_csc_array[0], one())
        np.testing.assert_array_almost_equal(evolved_csc_array[1], zero())
    
    def test_large_sparse_unitary_efficiency(self):
        """Test that large sparse unitaries can be handled efficiently"""
        from scipy.sparse import diags
        
        # Create a large sparse diagonal unitary (phase gates)
        n = 8  # 3-qubit system
        phases = np.exp(1j * np.linspace(0, 2*np.pi, n))
        U_sparse_large = diags(phases, shape=(n, n), dtype=complex, format='csc')
        U_dense_large = U_sparse_large.toarray()
        
        # Create a simple 3-qubit state |000⟩
        psi0_large = np.zeros(n, dtype=complex)
        psi0_large[0] = 1.0
        
        # Test evolution
        psi_evolved = evolve_unitary(U_dense_large, psi0_large, batchmode=False)
        
        # Should just apply phase to |000⟩
        expected = phases[0] * psi0_large
        np.testing.assert_array_almost_equal(psi_evolved, expected)
        
        # Check unitarity is preserved
        assert np.isclose(np.linalg.norm(psi_evolved), 1.0)

class TestDissipator:
    def test_amplitude_damping(self):
        """Test amplitude damping dissipator"""
        rho = np.array([[1, 0], [0, 0]], dtype=complex)  # |0⟩⟨0|
        L = sigmam()  # Lowering operator
        drho_dt = dissipator(rho, L)
        
        # Should have negative contribution to |0⟩⟨0| population
        assert drho_dt[0, 0] < 0
        # Should have positive contribution to |1⟩⟨1| population
        assert drho_dt[1, 1] > 0
    
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
        psi0 = zero()  # Start in excited state
        H = np.zeros((2, 2), dtype=complex)  # No Hamiltonian
        t = np.linspace(0, 5, 50)
        gamma = 1.0
        c_ops = [np.sqrt(gamma) * sigmam()]
        
        rho_t = evolve_lindblad(psi0, H, t, c_ops=c_ops)
        
        # Population should decay from |1⟩ to |0⟩
        pop_excited = [np.real(rho[0, 0]) for rho in rho_t]
        pop_ground = [np.real(rho[1, 1]) for rho in rho_t]
        
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
        t = np.linspace(0,1, 20)
        
        for psi0 in initial_states:
            rho_t = evolve_lindblad(psi0, H, t, c_ops=[])
            
            assert np.False_ not in is_state(rho_t) 

@pytest.mark.skipif(not QUTIP_AVAILABLE, reason="QuTiP not available")
class TestQuTiPComparison:
    """Compare QuantumStuff evolution with QuTiP results"""
    
    def test_unitary_evolution_comparison(self):
        """Test unitary evolution against QuTiP"""
        # Initial state |0⟩
        psi0_qs = zero()
        psi0_qt = qt.basis(2, 0)
        
        # Convert to density matrices for comparison
        rho0_qs = ket_to_dm(psi0_qs, batchmode=False)
        rho0_qt = psi0_qt * psi0_qt.dag()
        
        # Pauli-X evolution
        U_qs = sigmax()
        U_qt = qt.sigmax()
        
        # Evolve with QuantumStuff
        rho_final_qs = evolve_unitary(U_qs, rho0_qs, batchmode = False)
        
        # Evolve with QuTiP
        rho_final_qt = U_qt * rho0_qt * U_qt.dag()
        
        # Compare results
        np.testing.assert_array_almost_equal(
            rho_final_qs, rho_final_qt.full(), decimal=10
        )
    
    def test_hadamard_evolution_comparison(self):
        """Test Hadamard gate evolution comparison"""
        # Initial state |0⟩
        psi0_qs = zero(dm = True)
        psi0_qt = qt.basis(2, 0)
        
        # Convert to density matrix for QuTiP
        rho0_qt = psi0_qt * psi0_qt.dag()
        
        # Hadamard gate
        H_qs = hadamard()
        H_qt = (qt.sigmax() + qt.sigmaz()) / np.sqrt(2)
        
        # Evolve
        rho_final_qs = evolve_unitary(H_qs, psi0_qs, batchmode = False)
        rho_final_qt = H_qt * rho0_qt * H_qt.dag()
        
        np.testing.assert_array_almost_equal(
            rho_final_qs, rho_final_qt.full(), decimal=10
        )
    
    def test_rotation_evolution_comparison(self):
        """Test arbitrary rotation evolution"""
        # Rotation around Z-axis
        theta = np.pi / 3
        
        # QuantumStuff
        U_qs = np.cos(theta/2) * np.eye(2, dtype=complex) - 1j * np.sin(theta/2) * sigmaz()
        psi0_qs = plus(dm = True)
        rho_final_qs = evolve_unitary(U_qs, psi0_qs, batchmode = False)
        
        # QuTiP
        U_qt = (-1j * theta/2 * qt.sigmaz()).expm()
        psi0_qt = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        rho0_qt = psi0_qt * psi0_qt.dag()
        rho_final_qt = U_qt * rho0_qt * U_qt.dag()
        
        np.testing.assert_array_almost_equal(
            rho_final_qs, rho_final_qt.full(), decimal=8
        )
    
    def test_free_evolution_comparison(self):
        """Test free evolution under Hamiltonian"""
        # Simple Hamiltonian: H = ωσz
        omega = 1.0
        H_qs = omega * sigmaz()
        H_qt = omega * qt.sigmaz()
        
        # Time evolution
        t = 0.5
        
        # Initial state
        psi0_qs = plus(dm = True)
        psi0_qt = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        rho0_qt = psi0_qt * psi0_qt.dag()
        
        # QuantumStuff evolution (manual time evolution)
        U_qs = expm(-1j * H_qs * t)
        rho_final_qs = evolve_unitary(U_qs, psi0_qs, batchmode=False)
        
        # QuTiP evolution
        U_qt = (-1j * H_qt * t).expm()
        rho_final_qt = U_qt * rho0_qt * U_qt.dag()
        
        np.testing.assert_array_almost_equal(
            rho_final_qs, rho_final_qt.full(), decimal=8
        )
    
    def test_amplitude_damping_comparison(self):
        """Test amplitude damping evolution comparison"""
        # Parameters
        gamma = 0.1
        
        # Initial state |1⟩
        psi0_qs = one()
        psi0_qt = qt.basis(2, 1)
        rho0_qs = ket_to_dm(psi0_qs, batchmode=False)
        rho0_qt = psi0_qt * psi0_qt.dag()
        
        # Jump operators
        L_qs = np.sqrt(gamma) * sigmam()
        L_qt = np.sqrt(gamma) * qt.sigmam()
        
        # QuantumStuff dissipator
        drho_dt_qs = Super_D([L_qs])
        
        # QuTiP dissipator  
        drho_dt_qt = qt.lindblad_dissipator(L_qt, L_qt)
        
        np.testing.assert_array_almost_equal(
            drho_dt_qs, drho_dt_qt.full(), decimal=8
        )
    
    def test_dephasing_comparison(self):
        """Test dephasing evolution comparison"""
        # Parameters
        gamma = 0.05
        
        # Initial state |+⟩
        psi0_qs = plus()
        psi0_qt = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        rho0_qs = ket_to_dm(psi0_qs, batchmode=False)
        rho0_qt = psi0_qt * psi0_qt.dag()
        
        # Jump operators for dephasing
        L_qs = np.sqrt(gamma) * sigmaz()
        L_qt = np.sqrt(gamma) * qt.sigmaz()
        
        # Compare dissipators
        drho_dt_qs = Super_D([L_qs])
        drho_dt_qt = qt.lindblad_dissipator(L_qt, L_qt)
        
        np.testing.assert_array_almost_equal(
            drho_dt_qs, drho_dt_qt.full(), decimal=8
        )
    
    def test_two_level_system_rabi_oscillations(self):
        """Test Rabi oscillations comparison"""
        # Two-level system with driving
        omega_0 = 1.0  # Qubit frequency
        omega_R = 0.5  # Rabi frequency
        
        # Hamiltonians
        H_qs = 0.5 * omega_0 * sigmaz() + 0.5 * omega_R * sigmax()
        H_qt = 0.5 * omega_0 * qt.sigmaz() + 0.5 * omega_R * qt.sigmax()
        
        # Initial state |0⟩
        psi0_qs = zero()
        psi0_qt = qt.basis(2, 0)
        rho0_qs = ket_to_dm(psi0_qs, batchmode=False)
        rho0_qt = psi0_qt * psi0_qt.dag()
        
        # Short time evolution (manual unitary)
        t = 0.1
        U_qs = expm(-1j * H_qs * t)
        U_qt = (-1j * H_qt * t).expm()
        
        # Evolve
        rho_final_qs = evolve_unitary(U_qs, rho0_qs, batchmode = False)
        rho_final_qt = U_qt * rho0_qt * U_qt.dag()
        
        # Should be approximately equal for small t
        np.testing.assert_array_almost_equal(
            rho_final_qs, rho_final_qt.full(), decimal=3
        )
    
    def test_bloch_vector_evolution_comparison(self):
        """Test Bloch vector evolution comparison"""
        # Initial state with Bloch vector [1, 0, 0] (|+⟩ state)
        psi0_qs = plus()
        psi0_qt = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        rho0_qs = ket_to_dm(psi0_qs, batchmode=False)
        rho0_qt = psi0_qt * psi0_qt.dag()
        
        # Rotation around Y-axis
        theta = np.pi/4
        U_qs = np.cos(theta/2) * np.eye(2, dtype=complex) - 1j * np.sin(theta/2) * sigmay()
        U_qt = (-1j * theta/2 * qt.sigmay()).expm()
        
        # Evolve
        rho_final_qs = evolve_unitary(U_qs, rho0_qs, batchmode = False)
        rho_final_qt = U_qt * rho0_qt * U_qt.dag()
        
        np.testing.assert_array_almost_equal(
            rho_final_qs, rho_final_qt.full(), decimal=10
        )
    
    def test_multi_jump_operator_comparison(self):
        """Test multiple jump operators comparison"""
        # Multiple decay channels
        gamma1, gamma2 = 0.1, 0.05
        
        # Initial state
        psi0_qs = one()
        psi0_qt = qt.basis(2, 1)
        rho0_qs = ket_to_dm(psi0_qs, batchmode=False)
        rho0_qt = psi0_qt * psi0_qt.dag()
        
        # Multiple jump operators
        L1_qs = np.sqrt(gamma1) * sigmam()
        L2_qs = np.sqrt(gamma2) * sigmaz()
        L1_qt = np.sqrt(gamma1) * qt.sigmam()
        L2_qt = np.sqrt(gamma2) * qt.sigmaz()
        
        # Compare individual dissipators
        drho_total_qs = Super_D([L1_qs, L2_qs])
     
        drho1_qt = qt.lindblad_dissipator(L1_qt, L1_qt)
        drho2_qt = qt.lindblad_dissipator(L2_qt, L2_qt)
        drho_total_qt = drho1_qt + drho2_qt
        
        np.testing.assert_array_almost_equal(
            drho_total_qs, drho_total_qt.full(), decimal=8
        )

if __name__ == "__main__":
    pytest.main([__file__])
