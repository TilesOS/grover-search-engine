"""
Grover Search Engine - Detailed examples/tutorials
Shows the various use cases and advanced features of the
Generalized Grover Search Engine.
"""

from grover_search import GroverSearchEngine
import numpy as np
import matplotlib.pyplot as plt


def tutorial_1_basic_search():
    """
    Tutorial 1: Basic single-state search
    Shows the fundamental usage of Grover's algorithm.
    """
    print("\n" + "="*80)
    print("TUTORIAL 1: Basic Single-State Search")
    print("="*80)
    print("\nScenario: Find the number 7 in a database of 16 numbers (0-15)")
    print("This requires 4 qubits since 2^4 = 16\n")
    
    # Initialize search engine
    grover = GroverSearchEngine(n_qubits=4, marked_states=7)
    
    # Run quantum search
    quantum_result = grover.run_simulation(shots=4096)
    
    # Classical comparison
    classical_result = grover.classical_search_comparison(shots=4096)
    
    # Display results
    grover.print_summary(quantum_result, classical_result)
    
    print("\n💡 Key Takeaway:")
    print("   With just 3 Grover iterations, we achieve ~98% success probability")
    print("   Classical random search would only give us 1/16 = 6.25% per try")
    
    return grover, quantum_result, classical_result


def tutorial_2_multiple_targets():
    """
    Tutorial 2: Searching for multiple marked states
    Demonstrates Grover's algorithm with multiple solutions.
    """
    print("\n" + "="*80)
    print("TUTORIAL 2: Multiple Target States")
    print("="*80)
    print("\nScenario: Find prime numbers less than 16: [2, 3, 5, 7, 11, 13]")
    print("This is 6 marked states out of 16 total states\n")
    
    # Prime numbers less than 16
    primes = [2, 3, 5, 7, 11, 13]
    grover = GroverSearchEngine(n_qubits=4, marked_states=primes)
    
    # Run simulation
    quantum_result = grover.run_simulation(shots=4096)
    classical_result = grover.classical_search_comparison(shots=4096)
    
    grover.print_summary(quantum_result, classical_result)
    
    print("\n💡 Key Takeaway:")
    print("   More marked states = fewer iterations needed")
    print("   Optimal iterations scales as √(N/M) where M = number of marked states")
    
    return grover, quantum_result, classical_result


def tutorial_3_iteration_analysis():
    """
    Tutorial 3: Iteration count importance
    Shows what happens with too few or too many iterations.
    """
    print("\n" + "="*80)
    print("TUTORIAL 3: Iteration Count Analysis")
    print("="*80)
    print("\nScenario: Analyze how success probability changes with iterations")
    print("Search for state |10⟩ in 4-qubit system\n")
    
    grover = GroverSearchEngine(n_qubits=4, marked_states=10)
    
    # Test different iteration counts
    iteration_counts = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    results = []
    
    print("Testing different iteration counts:")
    print(f"{'Iterations':>12} | {'Success Prob':>14} | {'Comment':>30}")
    print("-" * 60)
    
    for iterations in iteration_counts:
        result = grover.run_simulation(iterations=iterations, shots=2048)
        prob = result['success_probability']
        results.append(prob)
        
        # Determine comment
        if iterations == 0:
            comment = "No amplification yet"
        elif iterations == grover.optimal_iterations:
            comment = "✓ OPTIMAL"
        elif iterations < grover.optimal_iterations:
            comment = "Under-rotated"
        else:
            comment = "Over-rotated"
        
        print(f"{iterations:>12} | {prob:>14.4f} | {comment:>30}")
    
    print(f"\nOptimal iterations: {grover.optimal_iterations}")
    print(f"Maximum probability achieved: {max(results):.4f}")
    
    print("\n💡 Key Takeaway:")
    print("   Too few iterations: probability not fully amplified")
    print("   Optimal iterations: maximum probability")
    print("   Too many iterations: probability decreases (over-rotation)")
    
    return grover, iteration_counts, results


def tutorial_4_scaling_analysis():
    """
    Tutorial 4: Scaling behavior
    Demonstrates how the algorithm scales with problem size.
    """
    print("\n" + "="*80)
    print("TUTORIAL 4: Scaling Analysis")
    print("="*80)
    print("\nScenario: Compare algorithm performance for different search space sizes\n")
    
    qubit_range = [2, 3, 4, 5, 6]
    scaling_data = []
    
    print(f"{'Qubits':>8} | {'Space Size':>12} | {'Optimal Iter':>14} | {'Success Prob':>15}")
    print("-" * 65)
    
    for n_qubits in qubit_range:
        # Always search for state 0 for consistency
        grover = GroverSearchEngine(n_qubits=n_qubits, marked_states=0)
        result = grover.run_simulation(shots=2048)
        
        scaling_data.append({
            'qubits': n_qubits,
            'space_size': grover.N,
            'optimal_iter': grover.optimal_iterations,
            'success_prob': result['success_probability']
        })
        
        print(f"{n_qubits:>8} | {grover.N:>12} | {grover.optimal_iterations:>14} | "
              f"{result['success_probability']:>15.4f}")
    
    print("\n💡 Key Takeaway:")
    print("   Iterations grow as √N (square root of search space)")
    print("   Classical search would need O(N) operations")
    print("   This is the quadratic speedup of Grover's algorithm!")
    
    return scaling_data


def tutorial_5_visualization_deep_dive():
    """
    Tutorial 5: Comprehensive visualization
    Creates all available visualizations and explains them.
    """
    print("\n" + "="*80)
    print("TUTORIAL 5: Visualization Deep Dive")
    print("="*80)
    print("\nScenario: Generate all visualization types for analysis\n")
    
    # Create search engine
    grover = GroverSearchEngine(n_qubits=3, marked_states=[3, 5])
    
    # Run simulations
    quantum_result = grover.run_simulation(shots=4096)
    classical_result = grover.classical_search_comparison(shots=4096)
    
    print("Generating visualizations...")
    
    # Visualization 1: Results comparison
    print("\n1. Creating quantum vs classical comparison plot...")
    grover.plot_results(quantum_result, classical_result, 
                       save_path='/home/claude/tutorial_results_comparison.png')
    
    # Visualization 2: Probability amplification
    print("2. Analyzing probability amplification...")
    analysis = grover.analyze_iterations(max_iterations=8, shots=1024)
    grover.plot_probability_amplification(analysis,
                                         save_path='/home/claude/tutorial_amplification.png')
    
    print("\n✓ All visualizations saved!")
    print("\nVisualization Guide:")
    print("  📊 Results comparison: Shows measurement distribution and quantum advantage")
    print("  📈 Amplification curve: Shows how probability increases with iterations")
    
    return grover, quantum_result, classical_result, analysis


def tutorial_6_edge_cases():
    """
    Tutorial 6: Edge cases and special scenarios
    Explores boundary conditions and interesting cases.
    """
    print("\n" + "="*80)
    print("TUTORIAL 6: Edge Cases and Special Scenarios")
    print("="*80)
    
    # Case 1: Very small search space (2 qubits)
    print("\n--- Case 1: Minimal search space (2 qubits, 4 states) ---")
    grover1 = GroverSearchEngine(n_qubits=2, marked_states=2)
    result1 = grover1.run_simulation(shots=2048)
    print(f"Success probability: {result1['success_probability']:.4f}")
    print(f"Optimal iterations: {grover1.optimal_iterations}")
    
    # Case 2: Half the states are marked
    print("\n--- Case 2: Half of search space is marked ---")
    grover2 = GroverSearchEngine(n_qubits=3, marked_states=[0, 1, 2, 3])
    result2 = grover2.run_simulation(shots=2048)
    print(f"Success probability: {result2['success_probability']:.4f}")
    print(f"Optimal iterations: {grover2.optimal_iterations}")
    print(f"Note: When M=N/2, only 1 iteration is needed!")
    
    # Case 3: Single marked state in larger space
    print("\n--- Case 3: Needle in haystack (1 in 64 states) ---")
    grover3 = GroverSearchEngine(n_qubits=6, marked_states=42)
    result3 = grover3.run_simulation(shots=2048)
    classical3 = grover3.classical_search_comparison(shots=2048)
    print(f"Quantum success: {result3['success_probability']:.4f}")
    print(f"Classical expected: {classical3['expected_probability']:.4f}")
    print(f"Speedup: {result3['success_probability']/classical3['expected_probability']:.1f}x")
    
    print("\n💡 Key Takeaway:")
    print("   Grover's algorithm works for any search space size and marked state count")
    print("   The speedup is most dramatic for finding rare items (small M/N ratio)")


def advanced_custom_circuit_inspection():
    """
    Tutorial 7: Inspect the quantum circuit structure
    Circuit-level details.
    """
    print("\n" + "="*80)
    print("ADVANCED: Circuit Structure Inspection")
    print("="*80)
    
    grover = GroverSearchEngine(n_qubits=3, marked_states=5)
    
    # Build circuit
    circuit = grover.build_circuit()
    
    print("\nCircuit Statistics:")
    print(f"  • Qubits: {circuit.num_qubits}")
    print(f"  • Classical bits: {circuit.num_clbits}")
    print(f"  • Circuit depth: {circuit.depth()}")
    print(f"  • Gate counts: {circuit.count_ops()}")
    print(f"  • Number of Grover iterations: {grover.optimal_iterations}")
    
    # Get individual components
    oracle = grover._create_oracle()
    diffusion = grover._create_diffusion_operator()
    
    print("\nOracle Circuit:")
    print(f"  • Depth: {oracle.depth()}")
    print(f"  • Gates: {oracle.count_ops()}")
    
    print("\nDiffusion Operator Circuit:")
    print(f"  • Depth: {diffusion.depth()}")
    print(f"  • Gates: {diffusion.count_ops()}")
    
    print("\nFull Circuit Diagram:")
    print(circuit.draw(output='text', fold=-1))
    
    return circuit, oracle, diffusion


def run_all_tutorials():
    """
    Run all tutorials in sequence.
    """
    print("\n" + "="*80)
    print("GROVER SEARCH ENGINE - COMPREHENSIVE TUTORIAL SUITE")
    print("="*80)
    print("\nThis tutorial will guide you through all features of the Grover Search Engine.")
    print("Each tutorial demonstrates different aspects and use cases.\n")
    
    input("Press Enter to start Tutorial 1...")
    tutorial_1_basic_search()
    
    input("\nPress Enter to continue to Tutorial 2...")
    tutorial_2_multiple_targets()
    
    input("\nPress Enter to continue to Tutorial 3...")
    tutorial_3_iteration_analysis()
    
    input("\nPress Enter to continue to Tutorial 4...")
    tutorial_4_scaling_analysis()
    
    input("\nPress Enter to continue to Tutorial 5...")
    tutorial_5_visualization_deep_dive()
    
    input("\nPress Enter to continue to Tutorial 6...")
    tutorial_6_edge_cases()
    
    input("\nPress Enter to see Advanced Circuit Inspection...")
    advanced_custom_circuit_inspection()
    
    print("\n" + "="*80)
    print("TUTORIAL COMPLETE!")
    print("="*80)
    print("\nYou've now seen all the major features of the Grover Search Engine.")
    print("Feel free to experiment with your own parameters and scenarios!")
    print("\nFor more information, check the README.md file.")


if __name__ == "__main__":
    # You can run all tutorials or individual ones
    import sys
    
    if len(sys.argv) > 1:
        tutorial_num = sys.argv[1]
        
        tutorials = {
            '1': tutorial_1_basic_search,
            '2': tutorial_2_multiple_targets,
            '3': tutorial_3_iteration_analysis,
            '4': tutorial_4_scaling_analysis,
            '5': tutorial_5_visualization_deep_dive,
            '6': tutorial_6_edge_cases,
            'advanced': advanced_custom_circuit_inspection,
        }
        
        if tutorial_num in tutorials:
            tutorials[tutorial_num]()
        else:
            print(f"Unknown tutorial: {tutorial_num}")
            print(f"Available: {', '.join(tutorials.keys())}")
    else:
        # Run all tutorials
        run_all_tutorials()

