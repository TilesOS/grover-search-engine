"""
Test script to verify Grover's algorithm implementation
before running on real IBM Quantum hardware.
"""

from qiskit import QuantumCircuit, execute
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def grovers_algorithm(search_space):
    """
    Implements Grover's algorithm to find a marked item in a search space.

    Args:
        search_space: List of items to search through

    Returns:
        Dictionary with measurement counts
    """
    n = len(search_space)
    qubits_needed = 0
    while (1 << qubits_needed) < n:
        qubits_needed += 1

    # Create quantum circuit
    qc = QuantumCircuit(qubits_needed, qubits_needed)

    # Apply Hadamard gates to all qubits for superposition
    qc.h(range(qubits_needed))

    # Oracle: Mark the target item (in this case, we'll mark "grover")
    # For simplicity, let's assume we're searching for "grover"
    target_index = 0  # Index of "grover" in search space

    # Apply X gate to target qubits to mark the state
    qc.x(target_index)
    qc.h(qubits_needed - 1)  # Apply Hadamard to last qubit for phase flip
    qc.ccx(0, 1, qubits_needed - 1) if qubits_needed > 2 else None  # Controlled-NOT
    qc.h(qubits_needed - 1)
    qc.x(target_index)

    # Diffusion operator (inversion about mean)
    qc.h(range(qubits_needed))
    qc.x(range(qubits_needed))

    # Controlled-Z gate (multi-controlled Z gate)
    if qubits_needed == 2:
        qc.cz(0, 1)
    elif qubits_needed == 3:
        qc.ccz(0, 1, 2)

    qc.x(range(qubits_needed))
    qc.h(range(qubits_needed))

    # Measure all qubits
    qc.measure(range(qubits_needed), range(qubits_needed))

    return qc

def test_grover_algorithm():
    """Test Grover's algorithm with a small search space."""
    # Simple search space
    search_space = ["grover", "search", "engine"]

    print(f"Testing Grover's algorithm on search space: {search_space}")

    # Create and run quantum circuit
    qc = grovers_algorithm(search_space)

    # Simulate the circuit
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=1024).result()
    counts = result.get_counts(qc)

    print("\nMeasurement results:")
    for state, count in sorted(counts.items()):
        prob = count / 1024 * 100
        print(f"State {state}: {count} times ({prob:.1f}%)")

    # Plot the results
    plot_histogram(counts, title="Grover's Algorithm Results")
    plt.savefig("grover_test_results.png", dpi=300, bbox_inches='tight')
    print("\nResults saved to grover_test_results.png")

    return counts

if __name__ == "__main__":
    test_grover_algorithm()
