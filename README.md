# Generalized Grover Search Engine

A comprehensive implementation of Grover's quantum search algorithm using Qiskit that automatically constructs oracles, diffusion operators, and determines optimal iteration counts.

![Grover's Algorithm](https://img.shields.io/badge/Quantum-Grover's_Algorithm-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Qiskit](https://img.shields.io/badge/Qiskit-Latest-purple)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Theory](#theory)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Results & Visualization](#results--visualization)
- [Performance Analysis](#performance-analysis)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a **generalized Grover search engine** that can search for any number of marked states in a quantum search space of arbitrary size. The implementation automatically:

- Constructs the oracle for any marked state(s)
- Builds the Grover diffusion operator
- Calculates the optimal number of iterations
- Runs quantum simulations
- Compares against classical random search
- Visualizes probability amplification

## Features

### Core Functionality
- **Automatic Oracle Construction**: Given marked states, automatically builds the phase-flipping oracle
- **Diffusion Operator**: Implements the Grover diffusion operator (inversion about average)
- **Optimal Iterations**: Calculates and uses the theoretically optimal number of Grover iterations
- **Multiple Marked States**: Supports searching for single or multiple target states simultaneously

### Analysis & Visualization
- **Quantum vs Classical Comparison**: Direct comparison with random classical search
- **Probability Amplification**: Tracks success probability across different iteration counts
- **Comprehensive Plotting**: Publication-quality visualizations of results
- **Performance Metrics**: Detailed timing and probability statistics

## Theory

### Grover's Algorithm

Grover's algorithm provides a **quadratic speedup** for unstructured search problems:
- **Classical search**: O(N) queries needed
- **Quantum search**: O(√N) queries needed

### Algorithm Steps

1. **Initialization**: Create uniform superposition of all states
   ```
   |ψ⟩ = H⊗n|0⟩ = (1/√N) ∑|x⟩
   ```

2. **Oracle**: Flip phase of marked states
   ```
   O|x⟩ = (-1)^f(x)|x⟩ where f(x) = 1 if x is marked, 0 otherwise
   ```

3. **Diffusion Operator**: Reflect about the average
   ```
   D = 2|ψ⟩⟨ψ| - I
   ```

4. **Iteration**: Repeat oracle + diffusion ~π/4 √(N/M) times

5. **Measurement**: Measure to obtain marked state with high probability

### Optimal Iterations

For M marked items out of N total items, the optimal number of iterations is:

```
k_optimal ≈ (π/4) √(N/M)
```

The success probability after k iterations is:

```
P(success) = sin²((2k+1)θ)
where θ = arcsin(√(M/N))
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/TilesOS/grover-search-engine.git
cd grover-search-engine
```

2. **Create virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from grover_search import GroverSearchEngine

# Create a search engine for 3 qubits (8 states) with marked state 5
grover = GroverSearchEngine(n_qubits=3, marked_states=5)

# Run quantum search
quantum_result = grover.run_simulation(shots=2048)

# Compare with classical search
classical_result = grover.classical_search_comparison(shots=2048)

# Print summary
grover.print_summary(quantum_result, classical_result)

# Plot results
grover.plot_results(quantum_result, classical_result)
```

### Run Demo

```bash
python grover_search.py
```

This will run three examples:
1. Single marked state search
2. Multiple marked states search
3. Probability amplification analysis

## Usage Examples

### Example 1: Search for Single State

```python
# Search for state |5⟩ in a 3-qubit system (8 total states)
grover = GroverSearchEngine(n_qubits=3, marked_states=5)

# Run with optimal iterations
result = grover.run_simulation(shots=2048)
print(f"Success probability: {result['success_probability']:.4f}")
```

**Expected Output:**
```
Grover Search Engine Initialized:
  - Qubits: 3
  - Search space size: 8
  - Marked states: [5]
  - Number of solutions: 1
  - Optimal iterations: 2

Success probability: 0.9453
```

### Example 2: Multiple Marked States

```python
# Search for states |3⟩, |7⟩, and |11⟩ in a 4-qubit system
grover = GroverSearchEngine(n_qubits=4, marked_states=[3, 7, 11])

result = grover.run_simulation(shots=2048)
classical = grover.classical_search_comparison(shots=2048)

print(f"Quantum success: {result['success_probability']:.4f}")
print(f"Classical expected: {classical['expected_probability']:.4f}")
print(f"Speedup: {result['success_probability']/classical['expected_probability']:.2f}x")
```

### Example 3: Analyze Probability Amplification

```python
# Analyze how probability changes with iteration count
grover = GroverSearchEngine(n_qubits=3, marked_states=5)

analysis = grover.analyze_iterations(max_iterations=6, shots=1024)

# Plot the amplification curve
grover.plot_probability_amplification(analysis, 
                                     save_path='amplification.png')
```

### Example 4: Custom Iterations

```python
# Use custom number of iterations instead of optimal
grover = GroverSearchEngine(n_qubits=4, marked_states=10)

# Try different iteration counts
for iterations in [1, 2, 3, 4, 5]:
    result = grover.run_simulation(iterations=iterations, shots=1024)
    print(f"Iterations: {iterations}, Probability: {result['success_probability']:.4f}")
```

### Example 5: Build and Examine Circuit

```python
# Build the circuit without running
grover = GroverSearchEngine(n_qubits=3, marked_states=7)
circuit = grover.build_circuit()

# Print circuit information
print(f"Circuit depth: {circuit.depth()}")
print(f"Gate count: {circuit.count_ops()}")

# Draw the circuit (requires matplotlib)
print(circuit.draw(output='text'))
```

## API Reference

### `GroverSearchEngine` Class

#### Constructor

```python
GroverSearchEngine(n_qubits: int, marked_states: Union[int, List[int]])
```

**Parameters:**
- `n_qubits`: Number of qubits (search space size = 2^n_qubits)
- `marked_states`: Single marked state (int) or list of marked states

**Attributes:**
- `n_qubits`: Number of qubits
- `N`: Total search space size (2^n_qubits)
- `marked_states`: List of marked states
- `M`: Number of marked states
- `optimal_iterations`: Calculated optimal iterations

#### Methods

##### `build_circuit(iterations: int = None) -> QuantumCircuit`
Build the complete Grover search circuit.

**Parameters:**
- `iterations`: Number of Grover iterations (uses optimal if None)

**Returns:** Qiskit QuantumCircuit object

---

##### `run_simulation(iterations: int = None, shots: int = 1024) -> Dict`
Run the quantum simulation.

**Parameters:**
- `iterations`: Number of iterations (default: optimal)
- `shots`: Number of measurement shots

**Returns:** Dictionary with keys:
- `counts`: Measurement outcome counts
- `success_probability`: Probability of measuring marked state
- `quantum_time`: Execution time
- `shots`: Number of shots used
- `iterations`: Iterations used

---

##### `classical_search_comparison(shots: int = 1024) -> Dict`
Simulate classical random search.

**Parameters:**
- `shots`: Number of random samples

**Returns:** Dictionary with keys:
- `success_probability`: Observed success probability
- `expected_probability`: Theoretical probability (M/N)
- `classical_time`: Execution time
- `shots`: Number of shots

---

##### `analyze_iterations(max_iterations: int = None, shots: int = 1024) -> Dict`
Analyze probability across different iteration counts.

**Parameters:**
- `max_iterations`: Maximum iterations to test (default: 2x optimal)
- `shots`: Shots per iteration count

**Returns:** Dictionary with keys:
- `iterations`: List of iteration counts tested
- `probabilities`: Success probability for each count
- `optimal_iteration`: Calculated optimal iteration
- `max_probability`: Maximum observed probability

---

##### `plot_results(quantum_result: Dict, classical_result: Dict, save_path: str = None)`
Plot quantum vs classical comparison.

**Parameters:**
- `quantum_result`: Output from `run_simulation()`
- `classical_result`: Output from `classical_search_comparison()`
- `save_path`: Path to save figure (displays if None)

---

##### `plot_probability_amplification(analysis_data: Dict, save_path: str = None)`
Plot probability amplification curve.

**Parameters:**
- `analysis_data`: Output from `analyze_iterations()`
- `save_path`: Path to save figure (displays if None)

---

##### `print_summary(quantum_result: Dict, classical_result: Dict)`
Print comprehensive results summary.

**Parameters:**
- `quantum_result`: Output from `run_simulation()`
- `classical_result`: Output from `classical_search_comparison()`

## Results & Visualization

The engine produces three types of visualizations:

### 1. Measurement Results
Shows the distribution of measurement outcomes, with marked states highlighted in red.

### 2. Quantum vs Classical Comparison
Bar chart comparing:
- Quantum search success probability
- Classical random search (observed)
- Classical expected probability (M/N)

### 3. Probability Amplification Curve
Line plot showing:
- Success probability vs iteration count
- Optimal iteration point
- Theoretical curve
- Classical baseline

## Performance Analysis

### Complexity Comparison

| Method | Time Complexity | Success Probability |
|--------|----------------|---------------------|
| Classical Random | O(N) | M/N |
| Classical Exhaustive | O(N) | 1 (guaranteed) |
| Grover Quantum | O(√N) | ~1 (high prob.) |

### Example Benchmarks (3 qubits, 1 marked state)

```
Search Space: 8 states
Marked States: [5]
Classical Expected: 0.125 (12.5%)

Grover Results:
- Optimal Iterations: 2
- Success Probability: 0.945 (94.5%)
- Speedup: ~7.6x probability improvement
```

## Advanced Usage

### Accessing Internal Components

```python
grover = GroverSearchEngine(n_qubits=3, marked_states=5)

# Get the oracle circuit
oracle = grover._create_oracle()
print(oracle.draw())

# Get the diffusion operator
diffusion = grover._create_diffusion_operator()
print(diffusion.draw())
```

### Custom Analysis Pipeline

```python
import numpy as np

grover = GroverSearchEngine(n_qubits=4, marked_states=[5, 10, 15])

# Test custom iteration range
results = []
for i in range(0, 8):
    res = grover.run_simulation(iterations=i, shots=2048)
    results.append({
        'iterations': i,
        'probability': res['success_probability'],
        'time': res['quantum_time']
    })

# Analyze results
best = max(results, key=lambda x: x['probability'])
print(f"Best iteration count: {best['iterations']}")
print(f"Best probability: {best['probability']:.4f}")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
git clone https://github.com/TilesOS/grover-search-engine.git
cd grover-search-engine
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Additional dev dependencies
```

### Running Tests

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

1. Grover, L.K. (1996). "A fast quantum mechanical algorithm for database search". Proceedings of the 28th Annual ACM Symposium on Theory of Computing.

2. Nielsen, M.A. & Chuang, I.L. (2010). "Quantum Computation and Quantum Information". Cambridge University Press.

3. Qiskit Documentation: https://qiskit.org/documentation/

## Author

Tyler McClure - [@TilesOS](https://github.com/TilesOS)

---

**Note**: This is a simulation using Qiskit Aer. For actual quantum hardware execution, modify the backend configuration to use IBM Quantum systems via `IBMProvider`.
