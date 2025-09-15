# QuantumStuff Test Suite

This directory contains comprehensive tests for the QuantumStuff quantum computing library.

## Test Structure

- **`test_utils.py`**: Tests for utility functions (dag, is_herm, tensor_product, etc.)
- **`test_operators.py`**: Tests for quantum operators and measurements
- **`test_states.py`**: Tests for quantum state generation and manipulation  
- **`test_evolution.py`**: Tests for quantum evolution (unitary and Lindblad)
- **`test_metrics.py`**: Tests for quantum information metrics (fidelity, entropy, etc.)
- **`test_integration.py`**: Integration tests that test modules working together

## Running Tests

### Option 1: Using the test runner (recommended)
```bash
python run_tests.py
```

### Option 2: Using pytest (if installed)
```bash
# Install test dependencies
pip install -r test_requirements.txt

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=QuantumStuff

# Run specific test file
pytest tests/test_utils.py

# Run tests with verbose output
pytest tests/ -v
```

### Option 3: Using unittest
```bash
python -m unittest discover tests/
```

## Test Categories

### Unit Tests
Each module has comprehensive unit tests covering:
- Basic functionality
- Edge cases and error handling
- Mathematical properties and relationships
- Input validation

### Integration Tests
The integration test suite (`test_integration.py`) includes:
- **Quantum Circuit Simulation**: Complete workflows like Bell state preparation
- **Decoherence Simulation**: T1 and T2 processes
- **Quantum Metrics Workflows**: State discrimination and entanglement measures
- **Random State Properties**: Validation of randomly generated states

## Test Coverage

The tests aim to cover:
- ✅ All public functions and methods
- ✅ Mathematical correctness (quantum mechanical properties)
- ✅ Error handling and edge cases
- ✅ Integration between modules
- ✅ Physical constraints (normalization, unitarity, etc.)

## Key Test Examples

### Quantum State Properties
```python
# Test that all quantum states are properly normalized
assert is_norm(zero())
assert is_norm(plus())

# Test quantum state validity
is_valid, is_dm = is_state(rho)
assert is_valid
```

### Operator Mathematics
```python
# Test Pauli algebra: [σx, σy] = 2iσz
assert np.allclose(commutator(sigmax(), sigmay()), 2j * sigmaz())

# Test unitarity of random unitary matrices
U = haar_random_unitary(2)
assert np.allclose(U @ dag(U), np.eye(4))
```

### Evolution Correctness
```python
# Test that unitary evolution preserves normalization
psi_final = evolve_unitary(U, psi_initial)
assert is_norm(psi_final)

# Test that Lindblad evolution preserves trace
rho_t = evolve_lindblad(psi0, H, t, c_ops)
for rho in rho_t:
    assert np.isclose(np.trace(rho), 1.0)
```

### Information Theory
```python
# Test entropy bounds
S = von_neumann_entropy(rho)
assert 0 <= S <= np.log(dim)

# Test fidelity properties
f = fidelity(rho1, rho2)
assert 0 <= f <= 1
```

## Error Handling Tests

The test suite includes comprehensive error handling tests:
- Invalid input types and shapes
- Non-physical quantum states
- Dimension mismatches
- Numerical edge cases

## Performance Considerations

Some tests involve:
- Large matrix operations (controlled with smaller test cases)
- Random state generation (seeded for reproducibility when needed)
- Numerical integration (limited time ranges for speed)

## Dependencies

Core dependencies:
- `numpy`: Numerical computations
- `scipy`: Scientific computing (sparse matrices, integration)

Test dependencies:
- `pytest`: Test framework (optional, falls back to unittest)
- `pytest-cov`: Coverage reporting (optional)

## Contributing

When adding new functions to QuantumStuff:

1. Add corresponding tests in the appropriate test file
2. Include both positive and negative test cases  
3. Test mathematical properties specific to quantum mechanics
4. Add integration tests if the function interacts with other modules
5. Update this README if adding new test categories

## Debugging Failed Tests

If tests fail:

1. Check the specific error message and assertion
2. Verify that your QuantumStuff installation is up to date
3. Check for circular import issues (common cause of import errors)
4. Ensure all dependencies are installed with correct versions
5. Run individual test files to isolate issues

Example debugging:
```bash
# Run just one test file with verbose output
pytest tests/test_utils.py -v

# Run just one test class
pytest tests/test_utils.py::TestDag -v

# Run just one test method
pytest tests/test_utils.py::TestDag::test_dag_complex_matrix -v
```
