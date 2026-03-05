#!/usr/bin/env python
"""
Test suite for GroverSearchEngine.
"""

import numpy as np
import pytest
from grover_search import GroverSearchEngine


class TestBasicFunctionality:
    """Test basic functionality of the Grover search engine."""

    def test_initialization(self):
        """Test that the class initializes correctly."""
        grover = GroverSearchEngine(n_qubits=3, marked_states=[2, 6])

        assert grover.n_qubits == 3
        assert len(grover.marked_states) == 2
        assert grover.optimal_iterations == 1

    def test_oracle_creation(self):
        """Test oracle operator creation."""
        grover = GroverSearchEngine(n_qubits=3, marked_states=[5])
        oracle = grover._create_oracle()

        assert oracle is not None
        assert oracle.num_qubits == 3

    def test_diffusion_creation(self):
        """Test diffusion operator creation."""
        grover = GroverSearchEngine(n_qubit=3, marked_states=5)
        diffusion = grover._create_diffusion_operator()

        assert diffusion is not None
        assert diffusion.num_qubits == 3

    def test_quantum_advantage(self):
        """Test that quantum search outperforms classical."""
        grover = GroverSearchEngine(n_qubits=4, marked_states=10)

        quantum = grover.run_simulation(shots=2048)
        classical = grover.classical_search_comparison(shots=2048)

        # Quantum should significantly outperform classical expected
        assert quantum['success_probability'] > classical['expected_probability'] * 5


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_qubit_system(self):
        """Test with just 1 qubit (2 states)."""
        grover = GroverSearchEngine(n_qubits=1, marked_states=1)
        result = grover.run_simulation(shots=1024)

        # Should work, though not very useful
        assert result['success_probability'] >= 0

    def test_all_states_marked(self):
        """Test when all states are marked."""
        grover = GroverSearchEngine(n_qubits=2, marked_states=[0, 1, 2, 3])
        result = grover.run_simulation(shots=1024)

        # Should find a marked state with certainty
        assert result['success_probability'] > 0.95

    def test_half_states_marked(self):
        """Test when exactly half of states are marked."""
        grover = GroverSearchEngine(n_qubits=3, marked_states=[0, 1, 2, 3])
        result = grover.run_simulation(shots=1024)

        # M/N = 0.5, should still work well
        assert result['success_probability'] > 0.8

    def test_zero_iterations(self):
        """Test running with zero Grover iterations."""
        grover = GroverSearchEngine(n_qubits=3, marked_states=5)
        result = grover.run_simulation(iterations=0, shots=1024)

        # With no iterations, should get uniform distribution ~1/8
        assert 0.05 < result['success_probability'] < 0.20


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self):
        """Test a complete analysis workflow."""
        grover = GroverSearchEngine(n_qubits=3, marked_states=[2, 6])

        # Run quantum
        quantum = grover.run_simulation(shots=2048)

        # Run classical
        classical = grover.classical_search_comparison(shots=2048)

        # Analyze iterations
        analysis = grover.analyze_iterations(max_iterations=5, shots=512)

        # Verify all components work together
        assert quantum['success_probability'] > classical['expected_probability']
        assert len(analysis['probabilities']) == 6  # 0 to 5
        assert analysis['optimal_iteration'] == grover.optimal_iterations

    def test_reproducibility(self):
        """Test that results are consistent across runs."""
        grover = GroverSearchEngine(n_qubits=4, marked_states=7)

        # Run multiple times with same parameters
        results = [grover.run_simulation(shots=2048) for _ in range(3)]

        # Probabilities should be similar (within sampling error)
        probs = [r['success_probability'] for r in results]
        mean_prob = np.mean(probs)

        for prob in probs:
            assert abs(prob - mean_prob) < 0.1  # Within 10% variation


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
