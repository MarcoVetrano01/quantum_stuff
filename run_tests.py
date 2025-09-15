"""
Test runner for QuantumStuff library.
Run this file to execute all tests.
"""

import sys
import os
import subprocess

def run_tests():
    """Run all tests in the test directory"""
    # Add the parent directory to the path
    sys.path.insert(0, os.path.dirname(__file__))
    
    try:
        # Try to run with pytest if available
        result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v'], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to unittest if pytest is not available
        print("pytest not found, using unittest...")
        
        # Import and run tests manually
        import unittest
        
        # Discover and run tests
        loader = unittest.TestLoader()
        start_dir = os.path.join(os.path.dirname(__file__), 'tests')
        suite = loader.discover(start_dir, pattern='test_*.py')
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
