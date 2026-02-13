"""
Generalized Grover Search Engine
An implementation of Grover's quantum search algorithm that automatically
constructs oracle, diffusion operator, and determines optimal iteration counts.

Author: Tyler McClure
Date: February 2026
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union
import time


class GroverSearchEngine:
    """
    A generalized and scalable Grover search engine which searches for any marked state(s)
    in a quantum search space.
    
    Attributes:
        n_qubits (int): Number of qubits in the search space
        marked_states (List[int]): List of marked states to search for
        N (int): Total size of search space (2^n_qubits)
        M (int): Number of marked states
        optimal_iterations (int): Calculated optimal number of Grover iterations
    """
    
    def __init__(self, n_qubits: int, marked_states: Union[int, List[int]]):
        """
        Initialize the Grover Search Engine.
        
        Args:
            n_qubits: Number of qubits (search space size = 2^n_qubits)
            marked_states: Single marked state (int) or list of marked states
        
        Raises:
            ValueError: If marked states are out of range or invalid
        """
        self.n_qubits = n_qubits
        self.N = 2 ** n_qubits
        
        # Handle both single state and list of states
        if isinstance(marked_states, int):
            self.marked_states = [marked_states]
        else:
            self.marked_states = list(marked_states)
        
        self.M = len(self.marked_states)
        
        # Validate marked states
        for state in self.marked_states:
            if state < 0 or state >= self.N:
                raise ValueError(f"Marked state {state} out of range [0, {self.N-1}]")
        
        # Calculate optimal number of iterations
        self.optimal_iterations = self._calculate_optimal_iterations()
        
        print(f"Grover Search Engine Initialized:")
        print(f"  - Qubits: {self.n_qubits}")
        print(f"  - Search space size: {self.N}")
        print(f"  - Marked states: {self.marked_states}")
        print(f"  - Number of solutions: {self.M}")
        print(f"  - Optimal iterations: {self.optimal_iterations}")
    
    def _calculate_optimal_iterations(self) -> int:
        """
        Calculate the optimal number of Grover iterations.
        
        For M marked items out of N total items:
        k ≈ (π/4) * sqrt(N/M)
        
        Returns:
            Optimal number of iterations (rounded to nearest integer)
        """
        if self.M == 0:
            raise ValueError("Must have at least one marked state")
        
        optimal = (np.pi / 4) * np.sqrt(self.N / self.M)
        return int(np.round(optimal))
    
    def _create_oracle(self) -> QuantumCircuit:
        """
        Create the oracle that marks the target states.
        
        The oracle flips the phase of the marked states:
        |x⟩ → -|x⟩ if x is marked
        |x⟩ → |x⟩ otherwise
        
        Returns:
            QuantumCircuit implementing the oracle
        """
        qr = QuantumRegister(self.n_qubits, 'q')
        oracle = QuantumCircuit(qr, name='Oracle')
        
        for marked_state in self.marked_states:
            # Convert marked state to binary string
            binary_str = format(marked_state, f'0{self.n_qubits}b')
            
            # Apply X gates to flip qubits that should be 0
            for i, bit in enumerate(binary_str):
                if bit == '0':
                    oracle.x(i)
            
            # Multi-controlled Z gate (marks the state)
            if self.n_qubits == 1:
                oracle.z(0)
            elif self.n_qubits == 2:
                oracle.cz(0, 1)
            else:
                # For n > 2, use multi-controlled Z
                oracle.h(self.n_qubits - 1)
                oracle.mcx(list(range(self.n_qubits - 1)), self.n_qubits - 1)
                oracle.h(self.n_qubits - 1)
            
            # Undo the X gates
            for i, bit in enumerate(binary_str):
                if bit == '0':
                    oracle.x(i)
        
        return oracle
    
    def _create_diffusion_operator(self) -> QuantumCircuit:
        """
        Create the Grover diffusion operator (inversion about average).
        
        The diffusion operator reflects about the average amplitude:
        D = 2|s⟩⟨s| - I
        where |s⟩ is the equal superposition state
        
        Returns:
            QuantumCircuit implementing the diffusion operator
        """
        qr = QuantumRegister(self.n_qubits, 'q')
        diffusion = QuantumCircuit(qr, name='Diffusion')
        
        # Apply Hadamard to all qubits
        diffusion.h(range(self.n_qubits))
        
        # Apply X to all qubits
        diffusion.x(range(self.n_qubits))
        
        # Multi-controlled Z gate
        if self.n_qubits == 1:
            diffusion.z(0)
        elif self.n_qubits == 2:
            diffusion.cz(0, 1)
        else:
            diffusion.h(self.n_qubits - 1)
            diffusion.mcx(list(range(self.n_qubits - 1)), self.n_qubits - 1)
            diffusion.h(self.n_qubits - 1)
        
        # Apply X to all qubits
        diffusion.x(range(self.n_qubits))
        
        # Apply Hadamard to all qubits
        diffusion.h(range(self.n_qubits))
        
        return diffusion
    
    def build_circuit(self, iterations: int = None) -> QuantumCircuit:
        """
        Build the Grover search circuit.
        
        Args:
            iterations: Number of Grover iterations (uses optimal if None)
        
        Returns:
            Complete quantum circuit ready for execution
        """
        if iterations is None:
            iterations = self.optimal_iterations
        
        qr = QuantumRegister(self.n_qubits, 'q')
        cr = ClassicalRegister(self.n_qubits, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        # Initialize to uniform superposition
        circuit.h(range(self.n_qubits))
        circuit.barrier()
        
        # Create oracle and diffusion operator
        oracle = self._create_oracle()
        diffusion = self._create_diffusion_operator()
        
        # Apply Grover iterations
        for i in range(iterations):
            circuit.compose(oracle, inplace=True)
            circuit.barrier()
            circuit.compose(diffusion, inplace=True)
            circuit.barrier()
        
        # Measure all qubits
        circuit.measure(range(self.n_qubits), range(self.n_qubits))
        
        return circuit
    
    def run_simulation(self, iterations: int = None, shots: int = 1024) -> Dict:
        """
        Run the Grover search simulation.
        
        Args:
            iterations: Number of Grover iterations (uses optimal if None)
            shots: Number of measurement shots
        
        Returns:
            Dictionary containing simulation results
        """
        circuit = self.build_circuit(iterations)
        
        # Run simulation
        simulator = AerSimulator()
        start_time = time.time()
        job = simulator.run(circuit, shots=shots)
        result = job.result()
        quantum_time = time.time() - start_time
        
        counts = result.get_counts()
        
        # Calculate success probability
        success_count = sum(counts.get(format(state, f'0{self.n_qubits}b'), 0) 
                           for state in self.marked_states)
        success_probability = success_count / shots
        
        return {
            'counts': counts,
            'success_probability': success_probability,
            'quantum_time': quantum_time,
            'shots': shots,
            'iterations': iterations if iterations else self.optimal_iterations
        }
    
    def classical_search_comparison(self, shots: int = 1024) -> Dict:
        """
        Simulate classical random search for comparison.
        
        Args:
            shots: Number of random samples
        
        Returns:
            Dictionary with classical search results
        """
        start_time = time.time()
        
        # Simulate random sampling
        samples = np.random.randint(0, self.N, shots)
        success_count = sum(1 for s in samples if s in self.marked_states)
        success_probability = success_count / shots
        
        classical_time = time.time() - start_time
        
        # Expected probability for classical random search
        expected_probability = self.M / self.N
        
        return {
            'success_probability': success_probability,
            'expected_probability': expected_probability,
            'classical_time': classical_time,
            'shots': shots
        }
    
    def analyze_iterations(self, max_iterations: int = None, shots: int = 1024) -> Dict:
        """
        Analyze probability amplification across different iteration counts.
        
        Args:
            max_iterations: Maximum iterations to test (defaults to 2x optimal)
            shots: Number of shots per iteration count
        
        Returns:
            Dictionary with iteration analysis data
        """
        if max_iterations is None:
            max_iterations = min(2 * self.optimal_iterations + 1, int(np.sqrt(self.N)) + 5)
        
        iteration_range = range(0, max_iterations + 1)
        probabilities = []
        
        print(f"\nAnalyzing probability amplification (0 to {max_iterations} iterations)...")
        
        for i in iteration_range:
            result = self.run_simulation(iterations=i, shots=shots)
            probabilities.append(result['success_probability'])
            print(f"  Iteration {i}: Success probability = {result['success_probability']:.4f}")
        
        return {
            'iterations': list(iteration_range),
            'probabilities': probabilities,
            'optimal_iteration': self.optimal_iterations,
            'max_probability': max(probabilities)
        }
    
    def plot_results(self, quantum_result: Dict, classical_result: Dict, 
                     save_path: str = None):
        """
        Plot comparison between quantum and classical search results.
        
        Args:
            quantum_result: Results from run_simulation()
            classical_result: Results from classical_search_comparison()
            save_path: Path to save the figure (displays if None)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Measurement counts histogram
        counts = quantum_result['counts']
        states = sorted(counts.keys())
        values = [counts[s] for s in states]
        
        # Color marked states differently
        colors = ['red' if int(s, 2) in self.marked_states else 'blue' for s in states]
        
        ax1.bar(range(len(states)), values, color=colors, alpha=0.7)
        ax1.set_xlabel('Measurement Outcome', fontsize=12)
        ax1.set_ylabel('Counts', fontsize=12)
        ax1.set_title(f'Grover Search Results ({quantum_result["iterations"]} iterations)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(states)))
        ax1.set_xticklabels([f'{int(s, 2)}' for s in states], rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Marked States'),
                          Patch(facecolor='blue', alpha=0.7, label='Other States')]
        ax1.legend(handles=legend_elements)
        
        # Plot 2: Quantum vs Classical comparison
        methods = ['Quantum\nSearch', 'Classical\nSearch', 'Classical\nExpected']
        probs = [quantum_result['success_probability'], 
                classical_result['success_probability'],
                classical_result['expected_probability']]
        colors_comp = ['green', 'orange', 'red']
        
        bars = ax2.bar(methods, probs, color=colors_comp, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Success Probability', fontsize=12)
        ax2.set_title('Quantum vs Classical Search', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1.1])
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
    
    def plot_probability_amplification(self, analysis_data: Dict, save_path: str = None):
        """
        Plot probability amplification across iterations.
        
        Args:
            analysis_data: Results from analyze_iterations()
            save_path: Path to save the figure (displays if None)
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        iterations = analysis_data['iterations']
        probabilities = analysis_data['probabilities']
        optimal = analysis_data['optimal_iteration']
        
        # Plot probability curve
        ax.plot(iterations, probabilities, 'b-o', linewidth=2, markersize=6, 
               label='Success Probability')
        
        # Mark optimal iteration
        optimal_prob = probabilities[optimal]
        ax.axvline(x=optimal, color='red', linestyle='--', linewidth=2, 
                  label=f'Optimal Iteration ({optimal})')
        ax.plot(optimal, optimal_prob, 'r*', markersize=20, 
               label=f'Max Probability ({optimal_prob:.3f})')
        
        # Classical probability line
        classical_prob = self.M / self.N
        ax.axhline(y=classical_prob, color='orange', linestyle=':', linewidth=2,
                  label=f'Classical Expected ({classical_prob:.3f})')
        
        ax.set_xlabel('Number of Grover Iterations', fontsize=12)
        ax.set_ylabel('Success Probability', fontsize=12)
        ax.set_title('Probability Amplification in Grover\'s Algorithm', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1.1])
        
        # Add theoretical curve
        iter_theory = np.linspace(0, max(iterations), 100)
        theta = np.arcsin(np.sqrt(self.M / self.N))
        prob_theory = np.sin((2 * iter_theory + 1) * theta) ** 2
        ax.plot(iter_theory, prob_theory, 'g--', alpha=0.5, linewidth=1.5,
               label='Theoretical Curve')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
    
    def print_summary(self, quantum_result: Dict, classical_result: Dict):
        """
        Print a comprehensive summary of the search results.
        
        Args:
            quantum_result: Results from run_simulation()
            classical_result: Results from classical_search_comparison()
        """
        print("\n" + "="*70)
        print("GROVER SEARCH ENGINE - RESULTS SUMMARY")
        print("="*70)
        
        print(f"\nSearch Space Configuration:")
        print(f"  • Number of qubits: {self.n_qubits}")
        print(f"  • Search space size: {self.N}")
        print(f"  • Marked states: {self.marked_states}")
        print(f"  • Number of solutions: {self.M}")
        
        print(f"\nQuantum Search Results:")
        print(f"  • Iterations used: {quantum_result['iterations']}")
        print(f"  • Success probability: {quantum_result['success_probability']:.4f}")
        print(f"  • Execution time: {quantum_result['quantum_time']:.6f} seconds")
        print(f"  • Measurement shots: {quantum_result['shots']}")
        
        print(f"\nClassical Random Search:")
        print(f"  • Success probability: {classical_result['success_probability']:.4f}")
        print(f"  • Expected probability: {classical_result['expected_probability']:.4f}")
        print(f"  • Execution time: {classical_result['classical_time']:.6f} seconds")
        
        speedup = quantum_result['success_probability'] / classical_result['expected_probability']
        print(f"\nQuantum Advantage:")
        print(f"  • Probability speedup: {speedup:.2f}x")
        print(f"  • Theoretical speedup: ~{np.sqrt(self.N / self.M):.2f}x")
        
        print("\n" + "="*70)


def main():
    """
    Example usage of this Grover Search Engine.
    """
    print("Generalized Grover Search Engine - Demo\n")
    
    # Example 1: Search for a single state
    print("\n--- Example 1: Single marked state ---")
    grover1 = GroverSearchEngine(n_qubits=3, marked_states=5)
    
    # Run quantum search
    quantum_result1 = grover1.run_simulation(shots=2048)
    
    # Run classical search
    classical_result1 = grover1.classical_search_comparison(shots=2048)
    
    # Print summary
    grover1.print_summary(quantum_result1, classical_result1)
    
    # Plot results
    grover1.plot_results(quantum_result1, classical_result1, 
                        save_path='/home/claude/grover_single_state.png')
    
    # Example 2: Multiple marked states
    print("\n\n--- Example 2: Multiple marked states ---")
    grover2 = GroverSearchEngine(n_qubits=4, marked_states=[3, 7, 11])
    
    quantum_result2 = grover2.run_simulation(shots=2048)
    classical_result2 = grover2.classical_search_comparison(shots=2048)
    
    grover2.print_summary(quantum_result2, classical_result2)
    grover2.plot_results(quantum_result2, classical_result2,
                        save_path='/home/claude/grover_multiple_states.png')
    
    # Example 3: Probability amplification analysis
    print("\n\n--- Example 3: Probability amplification ---")
    analysis_data = grover2.analyze_iterations(max_iterations=12, shots=1024)
    grover2.plot_probability_amplification(analysis_data,
                                          save_path='/home/claude/probability_amplification.png')


if __name__ == "__main__":
    main()

