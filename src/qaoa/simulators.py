"""
QAOA circuit construction and simulation for cortical depth experiments.
Uses Qiskit's statevector simulator for ideal performance testing.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
try:
    # if you’ve got the unified primitives API in Terra ≥0.45
    from qiskit.primitives import Sampler
except ImportError:
    # fall back to Aer’s own primitive (class name is “Sampler,” not “AerSampler”)
    from qiskit_aer.primitives import Sampler

from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from typing import List, Tuple, Dict, Callable, Optional
import time


class QAOACircuit:
    """Encapsulates QAOA circuit construction and parameter binding."""
    
    def __init__(self, cost_hamiltonian, p: int = 1):
        """
        Initialize QAOA circuit builder.
        
        Args:
            cost_hamiltonian: CorticalDepthHamiltonian object
            p: Number of QAOA layers (ansatz depth)
        """
        self.cost_ham = cost_hamiltonian
        self.n_qubits = cost_hamiltonian.n_qubits
        self.p = p
        
        # Create parameters
        self.gamma_params = [Parameter(f'γ_{i}') for i in range(p)]
        self.beta_params = [Parameter(f'β_{i}') for i in range(p)]
        
        # Build the circuit template
        self.circuit = self._build_circuit()
    
    def _build_circuit(self) -> QuantumCircuit:
        """Build the parameterized QAOA circuit."""
        qc = QuantumCircuit(self.n_qubits)
        
        # Initial state: |+>^n
        for i in range(self.n_qubits):
            qc.h(i)
        
        # Apply p QAOA layers
        for layer in range(self.p):
            # Cost layer: exp(-i γ H_C)
            self._add_cost_layer(qc, self.gamma_params[layer])
            
            # Mixer layer: exp(-i β H_B)
            self._add_mixer_layer(qc, self.beta_params[layer])
        
        # Add measurements
        qc.measure_all()
        
        return qc
    
    def _add_cost_layer(self, qc: QuantumCircuit, gamma: Parameter):
        """Add the cost Hamiltonian evolution layer."""
        # For each ZZ term in the Hamiltonian
        for term, coeff in zip(self.cost_ham.pauli_terms, self.cost_ham.coefficients):
            # Find which qubits have Z (not I)
            z_qubits = [i for i, pauli in enumerate(term) if pauli == 'Z']
            
            if len(z_qubits) == 2:
                # Apply RZZ gate: exp(-i γ J_ij Z_i Z_j)
                i, j = z_qubits
                qc.rzz(2 * gamma * coeff, i, j)
    
    def _add_mixer_layer(self, qc: QuantumCircuit, beta: Parameter):
        """Add the mixer Hamiltonian evolution layer."""
        # Standard X-mixer: apply RX to each qubit
        for i in range(self.n_qubits):
            qc.rx(2 * beta, i)
    
    def bind_parameters(self, params: np.ndarray) -> QuantumCircuit:
        """
        Bind parameter values to the circuit.
        
        Args:
            params: Array of shape (2*p,) with [γ_0, ..., γ_{p-1}, β_0, ..., β_{p-1}]
        
        Returns:
            QuantumCircuit with bound parameters
        """
        if len(params) != 2 * self.p:
            raise ValueError(f"Expected {2*self.p} parameters, got {len(params)}")
        
        # Split into gamma and beta values
        gamma_vals = params[:self.p]
        beta_vals = params[self.p:]
        
        # Create parameter dictionary
        param_dict = {}
        for i in range(self.p):
            param_dict[self.gamma_params[i]] = gamma_vals[i]
            param_dict[self.beta_params[i]] = beta_vals[i]
        
        return self.circuit.bind_parameters(param_dict)


def build_qaoa_circuit(hamiltonian, p: int = 1) -> Callable:
    """
    Build a QAOA circuit function.
    
    Args:
        hamiltonian: CorticalDepthHamiltonian object
        p: Number of QAOA layers
    
    Returns:
        Function that takes (gamma, beta) and returns a parameterized circuit
    """
    qaoa = QAOACircuit(hamiltonian, p)
    
    def circuit_func(params):
        """Create circuit with given parameters."""
        return qaoa.bind_parameters(params)
    
    # Attach useful attributes
    circuit_func.n_qubits = qaoa.n_qubits
    circuit_func.p = p
    circuit_func.qaoa_obj = qaoa
    
    return circuit_func


def expectation_value(circuit_func: Callable, params: np.ndarray, 
                     backend: Optional[AerSimulator] = None,
                     shots: int = 8192) -> float:
    """
    Calculate expectation value <H_C> for given parameters.
    
    Args:
        circuit_func: Function that creates parameterized circuit
        params: Parameter values [gamma, beta]
        backend: Qiskit backend (if None, uses statevector)
        shots: Number of shots for sampling
    
    Returns:
        Expectation value of the cost Hamiltonian
    """
    # Get the circuit with bound parameters
    circuit = circuit_func(params)
    
    # Remove measurements for statevector calculation
    circuit_no_meas = circuit.remove_final_measurements(inplace=False)
    
    # Get the cost Hamiltonian
    hamiltonian = circuit_func.qaoa_obj.cost_ham
    
    if backend is None:
        # Use exact statevector calculation
        sv = Statevector(circuit_no_meas)
        expectation = sv.expectation_value(hamiltonian.operator).real
    else:
        # Use sampling-based estimation
        # This would require more complex implementation with Estimator primitive
        # For now, we'll use statevector even with backend specified
        sv = Statevector(circuit_no_meas)
        expectation = sv.expectation_value(hamiltonian.operator).real
    
    return expectation


def optimize_qaoa(circuit_func: Callable, 
                 init_guess: Optional[np.ndarray] = None,
                 backend: Optional[AerSimulator] = None,
                 method: str = 'COBYLA',
                 maxiter: int = 100,
                 verbose: bool = True) -> Dict:
    """
    Optimize QAOA parameters to minimize <H_C>.
    
    Args:
        circuit_func: QAOA circuit function
        init_guess: Initial parameter guess (if None, uses grid search)
        backend: Qiskit backend
        method: Optimization method
        maxiter: Maximum iterations
        verbose: Print optimization progress
    
    Returns:
        Dict with optimal parameters, final cost, and optimization history
    """
    p = circuit_func.p
    
    # Initialize parameters if not provided
    if init_guess is None:
        if p == 1:
            # Simple grid search for p=1
            init_guess = grid_search_init(circuit_func, backend)
        else:
            # Random initialization for p>1
            init_guess = np.random.uniform(0, 2*np.pi, size=2*p)
    
    # Track optimization history
    history = {'costs': [], 'params': [], 'times': []}
    start_time = time.time()
    
    def cost_function(params):
        """Cost function for optimizer."""
        cost = expectation_value(circuit_func, params, backend)
        
        # Record history
        current_time = time.time() - start_time
        history['costs'].append(cost)
        history['params'].append(params.copy())
        history['times'].append(current_time)
        
        if verbose and len(history['costs']) % 10 == 0:
            print(f"Iteration {len(history['costs'])}: Cost = {cost:.6f}")
        
        return cost
    
    # Run optimization
    if verbose:
        print(f"Starting {method} optimization with p={p}")
        print(f"Initial guess: {init_guess}")
    
    result = minimize(
        cost_function,
        init_guess,
        method=method,
        options={'maxiter': maxiter}
    )
    
    if verbose:
        print(f"\nOptimization completed in {time.time() - start_time:.2f} seconds")
        print(f"Final cost: {result.fun:.6f}")
        print(f"Optimal parameters: {result.x}")
    
    return {
        'optimal_params': result.x,
        'optimal_cost': result.fun,
        'optimization_result': result,
        'history': history,
        'p': p
    }


def grid_search_init(circuit_func: Callable, backend: Optional[AerSimulator] = None,
                    n_points: int = 20) -> np.ndarray:
    """
    Perform grid search to find good initial parameters for p=1.
    
    Args:
        circuit_func: QAOA circuit function
        backend: Qiskit backend
        n_points: Number of grid points per dimension
    
    Returns:
        Best initial parameters found
    """
    if circuit_func.p != 1:
        raise ValueError("Grid search only implemented for p=1")
    
    # Define grid
    gamma_range = np.linspace(0, 2*np.pi, n_points)
    beta_range = np.linspace(0, np.pi, n_points)
    
    best_cost = np.inf
    best_params = None
    
    # Search grid
    for gamma in gamma_range:
        for beta in beta_range:
            params = np.array([gamma, beta])
            cost = expectation_value(circuit_func, params, backend)
            
            if cost < best_cost:
                best_cost = cost
                best_params = params
    
    return best_params


def get_partition_from_counts(counts: Dict[str, int]) -> np.ndarray:
    """
    Extract partition labels from measurement counts.
    
    Args:
        counts: Dictionary of bitstring counts
    
    Returns:
        Binary label array based on most frequent bitstring
    """
    # Find most frequent bitstring
    best_bitstring = max(counts, key=counts.get)
    
    # Convert to label array (reverse order due to Qiskit convention)
    labels = np.array([int(bit) for bit in best_bitstring[::-1]])
    
    return labels


def run_qaoa_sampling(circuit_func: Callable, optimal_params: np.ndarray,
                     shots: int = 8192, backend: Optional[AerSimulator] = None) -> Dict:
    """
    Run QAOA circuit with optimal parameters and collect samples.
    
    Args:
        circuit_func: QAOA circuit function
        optimal_params: Optimized parameters
        shots: Number of measurement shots
        backend: Qiskit backend (if None, creates AerSimulator)
    
    Returns:
        Dict with counts, most frequent bitstring, and partition labels
    """
    if backend is None:
        backend = AerSimulator()
    
    # Get circuit with optimal parameters
    circuit = circuit_func(optimal_params)
    
    # Run circuit
    job = backend.run(circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    # Get partition from most frequent bitstring
    labels = get_partition_from_counts(counts)
    
    # Calculate cut value
    hamiltonian = circuit_func.qaoa_obj.cost_ham
    best_bitstring = max(counts, key=counts.get)
    cut_value = -hamiltonian.get_classical_energy(best_bitstring[::-1])
    
    return {
        'counts': counts,
        'labels': labels,
        'best_bitstring': best_bitstring,
        'cut_value': cut_value,
        'shots': shots
    }


def analyze_qaoa_results(results: Dict, hamiltonian) -> Dict:
    """
    Analyze QAOA results including cut quality and statistics.
    
    Args:
        results: Output from run_qaoa_sampling
        hamiltonian: CorticalDepthHamiltonian object
    
    Returns:
        Dict with analysis metrics
    """
    counts = results['counts']
    total_shots = results['shots']
    
    # Analyze energy distribution
    energies = []
    for bitstring, count in counts.items():
        energy = hamiltonian.get_classical_energy(bitstring[::-1])
        energies.extend([energy] * count)
    
    energies = np.array(energies)
    
    # Calculate cut values (negative of energies for max-cut)
    cut_values = -energies
    
    analysis = {
        'mean_energy': np.mean(energies),
        'std_energy': np.std(energies),
        'min_energy': np.min(energies),
        'max_cut_value': np.max(cut_values),
        'mean_cut_value': np.mean(cut_values),
        'best_solution_probability': counts[results['best_bitstring']] / total_shots,
        'n_unique_solutions': len(counts)
    }
    
    return analysis