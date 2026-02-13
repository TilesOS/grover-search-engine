"""
Tests for Grover Search Engine
Run with: pytest test_grover_search.py -v
"""

import pytest
import numpy as np
from grover_search import GroverSearchEngine


class TestGroverSearchEngine:
    """Test suite for GroverSearchEngine class."""
    
    def test_initialization_single_state(self):
        """Test initialization with a single marked state."""
        grover = GroverSearchEngine(n_qubits=3, marked_states=5)
        
        assert grover.n_qubits == 3
        assert grover.N == 8
        assert grover.marked_states == [5]
        assert grover.M == 1
        assert grover.optimal_iterations > 0
    
    def test_initialization_multiple_states(self):
        """Test initialization with multiple marked states."""
        grover = GroverSearchEngine(n_qubits=4, marked_states=[1, 3, 5, 7])
        
        assert grover.n_qubits == 4
        assert grover.N == 16
        assert grover.marked_states == [1, 3, 5, 7]
        assert grover.M == 4
    
    def test_initialization_list_input(self):
        """Test that single int converts to list."""
        grover = GroverSearchEngine(n_qubits=2, marked_states=3)
        assert isinstance(grover.marked_states, list)
        assert grover.marked_states == [3]
    
    def test_invalid_marked_state_too_large(self):
        """Test that marked state out of range raises error."""
        with pytest.raises(ValueError, match="out of range"):
            GroverSearchEngine(n_qubits=3, marked_states=10)  # Max is 7
    
    def test_invalid_marked_state_negative(self):
        """Test that negative marked state raises error."""
        with pytest.raises(ValueError, match="out of range"):
            GroverSearchEngine(n_qubits=3, marked_states=-1)
    
    def test_optimal_iterations_calculation(self):
        """Test optimal iterations formula."""
        # For N=8, M=1: optimal = (π/4) * sqrt(8) ≈ 2.22 → 2
        grover = GroverSearchEngine(n_qubits=3, marked_states=0)
        assert grover.optimal_iterations == 2
        
        # For N=16, M=1: optimal = (π/4) * sqrt(16) ≈ 3.14 → 3
        grover = GroverSearchEngine(n_qubits=4, marked_states=0)
        assert grover.optimal_iterations == 3
    
    def test_optimal_iterations_multiple_marked(self):
        """Test optimal iterations with multiple marked states."""
        # For N=16, M=4: optimal = (π/4) * sqrt(16/4) = π/4 * 2 ≈ 1.57 → 2
        grover = GroverSearchEngine(n_qubits=4, marked_states=[0, 1, 2, 3])
        assert grover.optimal_iterations == 2
    
    def test_build_circuit(self):
        """Test circuit building."""
        grover = GroverSearchEngine(n_qubits=3, marked_states=5)
        circuit = grover.build_circuit()
        
        assert circuit.num_qubits == 3
        assert circuit.num_clbits == 3
        assert circuit.depth() > 0
    
    def test_build_circuit_custom_iterations(self):
        """Test circuit building with custom iterations."""
        grover = GroverSearchEngine(n_qubits=3, marked_states=5)
        circuit = grover.build_circuit(iterations=5)
        
        # Circuit should exist and have gates
        assert circuit.count_ops() is not None
    
    def test_run_simulation_basic(self):
        """Test basic simulation run."""
        grover = GroverSearchEngine(n_qubits=3, marked_states=5)
        result = grover.run_simulation(shots=1024)
        
        assert 'counts' in result
        assert 'success_probability' in result
        assert 'quantum_time' in result
        assert 'shots' in result
        assert 'iterations' in result
        
        assert result['shots'] == 1024
        assert 0 <= result['success_probability'] <= 1
    
    def test_run_simulation_high_success(self):
        """Test that optimal iterations give high success probability."""
        grover = GroverSearchEngine(n_qubits=3, marked_states=5)
        result = grover.run_simulation(shots=4096)
        
        # With optimal iterations, should get >80% success
        assert result['success_probability'] > 0.8
    
    def test_classical_search_comparison(self):
        """Test classical search simulation."""
        grover = GroverSearchEngine(n_qubits=3, marked_states=5)
        result = grover.classical_search_comparison(shots=1024)
        
        assert 'success_probability' in result
        assert 'expected_probability' in result
        assert 'classical_time' in result
        assert 'shots' in result
        
        # Expected probability should be M/N = 1/8 = 0.125
        assert abs(result['expected_probability'] - 0.125) < 0.001
    
    def test_analyze_iterations(self):
        """Test iteration analysis."""
        grover = GroverSearchEngine(n_qubits=3, marked_states=5)
        analysis = grover.analyze_iterations(max_iterations=4, shots=512)
        
        assert 'iterations' in analysis
        assert 'probabilities' in analysis
        assert 'optimal_iteration' in analysis
        assert 'max_probability' in analysis
        
        assert len(analysis['iterations']) == len(analysis['probabilities'])
        assert len(analysis['iterations']) == 5  # 0 to 4 inclusive
    
    def test_probability_increases_then_decreases(self):
        """Test that probability follows expected pattern."""
        grover = GroverSearchEngine(n_qubits=3, marked_states=5)
        analysis = grover.analyze_iterations(max_iterations=6, shots=1024)
        
        probs = analysis['probabilities']
        optimal = analysis['optimal_iteration']
        
        # Probability should generally increase up to optimal
        assert probs[optimal] > probs[0]
        
        # Probability should decrease after over-rotation
        # (may not be strictly true due to sampling, but should hold on average)
        if optimal + 2 < len(probs):
            # Check that we're past the peak
            assert max(probs) > probs[-1]
    
    def test_multiple_marked_states_simulation(self):
        """Test simulation with multiple marked states."""
        grover = GroverSearchEngine(n_qubits=4, marked_states=[2, 5, 8, 11])
        result = grover.run_simulation(shots=2048)
        
        # With 4 marked states out of 16, should have good success rate
        assert result['success_probability'] > 0.5
    
    def test_small_search_space(self):
        """Test with minimal search space (2 qubits)."""
        grover = GroverSearchEngine(n_qubits=2, marked_states=3)
        result = grover.run_simulation(shots=1024)
        
        assert result['success_probability'] > 0.7
    
    def test_oracle_creation(self):
        """Test oracle circuit creation."""
        grover = GroverSearchEngine(n_qubits=3, marked_states=5)
        oracle = grover._create_oracle()
        
        assert oracle is not None
        assert oracle.num_qubits == 3
    
    def test_diffusion_creation(self):
        """Test diffusion operator creation."""
        grover = GroverSearchEngine(n_qubits=3, marked_states=5)
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

