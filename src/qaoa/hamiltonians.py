"""
QAOA Hamiltonian construction for cortical depth partitioning.
Converts correlation matrices into Ising/QUBO formulations.
"""


__all__ = [
    "CorticalDepthHamiltonian",
    "from_corr_matrix",
    "save_hamiltonian",
    "load_hamiltonian",
    "build_cost_hamiltonian",
    "build_penalty_hamiltonian",
    "build_x_mixer",
    "build_xy_mixer",
]

import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp, Pauli
from typing import List, Tuple, Dict
from itertools import combinations
from typing import Mapping


class CorticalDepthHamiltonian:
    """
    Encapsulates the cost Hamiltonian for cortical depth partitioning.
    
    The Hamiltonian is: H_C = -Σ_{i<j} J_ij * Z_i * Z_j
    where J_ij is the correlation between neurons i and j.
    """
    
    def __init__(self, corr_matrix: np.ndarray, threshold: float = 0.0):
        """
        Initialize the Hamiltonian from a correlation matrix.
        
        Args:
            corr_matrix: N x N correlation matrix
            threshold: Minimum absolute correlation to include (for sparsity)
        """
        self.n_qubits = corr_matrix.shape[0]
        self.corr_matrix = corr_matrix.copy()
        self.threshold = threshold
        
        # Build the Hamiltonian terms
        self.pauli_terms = []
        self.coefficients = []
        self._build_hamiltonian()
        
        # Create the SparsePauliOp
        self.operator = SparsePauliOp.from_list(
            [(term, coeff) for term, coeff in zip(self.pauli_terms, self.coefficients)]
        )
    
    def _build_hamiltonian(self):
        """Build the Pauli terms and coefficients from correlation matrix."""
        # Extract upper triangle (excluding diagonal)
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                J_ij = self.corr_matrix[i, j]
                
                # Skip weak correlations for efficiency
                if abs(J_ij) < self.threshold:
                    continue
                
                # Create Pauli string: Z_i * Z_j
                pauli_str = ['I'] * self.n_qubits
                pauli_str[i] = 'Z'
                pauli_str[j] = 'Z'
                
                # Note the negative sign: we want to minimize -J_ij * Z_i * Z_j
                self.pauli_terms.append(''.join(pauli_str))
                self.coefficients.append(-J_ij)
    
    def get_operator(self) -> SparsePauliOp:
        """Return the quantum operator representation."""
        return self.operator
    
    def get_classical_energy(self, bitstring: str) -> float:
        """
        Calculate the energy of a classical bitstring configuration.
        
        Args:
            bitstring: Binary string of length n_qubits (e.g., '0110')
        
        Returns:
            float: The energy of this configuration
        """
        # Convert bitstring to spin values (+1 for '0', -1 for '1')
        spins = np.array([1 if bit == '0' else -1 for bit in bitstring])
        
        # Calculate energy: H = -Σ_{i<j} J_ij * s_i * s_j
        energy = 0.0
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                energy -= self.corr_matrix[i, j] * spins[i] * spins[j]
        
        return energy
    
    def get_edge_list(self) -> List[Tuple[int, int, float]]:
        """
        Get list of edges with weights for visualization.
        
        Returns:
            List of (node_i, node_j, weight) tuples
        """
        edges = []
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                if abs(self.corr_matrix[i, j]) >= self.threshold:
                    edges.append((i, j, self.corr_matrix[i, j]))
        return edges
    
    def summary(self) -> Dict:
        """Return summary statistics of the Hamiltonian."""
        n_terms = len(self.pauli_terms)
        max_coeff = max(abs(c) for c in self.coefficients) if self.coefficients else 0
        
        return {
            'n_qubits': self.n_qubits,
            'n_terms': n_terms,
            'sparsity': 1 - (2 * n_terms) / (self.n_qubits * (self.n_qubits - 1)),
            'max_coefficient': max_coeff,
            'threshold': self.threshold
        }


def from_corr_matrix(corr_matrix: np.ndarray, threshold: float = 0.0) -> CorticalDepthHamiltonian:
    """
    Convenience function to create a Hamiltonian from a correlation matrix.
    
    Args:
        corr_matrix: N x N correlation matrix
        threshold: Minimum absolute correlation to include
    
    Returns:
        CorticalDepthHamiltonian object
    """
    return CorticalDepthHamiltonian(corr_matrix, threshold)


def build_cost_hamiltonian(
    triplet_weights: Mapping[Tuple[int, int, int], float]
) -> SparsePauliOp:
    """
    Builds the cost Hamiltonian H_C = -Σ w_abc Z_a Z_b Z_c from triplet weights.

    This is a pure function with no side-effects. The negative sign ensures
    that minimizing the energy corresponds to maximizing the score from
    activating highly-overlapping assembly triplets.

    Args:
        triplet_weights: A mapping from (a, b, c) tuples to their scaled weight.

    Returns:
        A SparsePauliOp representing the cost Hamiltonian.
    """
    if not triplet_weights:
        return SparsePauliOp.from_list([])

    n_qubits = max(max(t) for t in triplet_weights.keys()) + 1
    pauli_list = []
    coeffs = []
    
    # Iterate and build lists simultaneously to ensure order is preserved
    for (a, b, c), weight in triplet_weights.items():
        label = ['I'] * n_qubits
        label[a], label[b], label[c] = 'Z', 'Z', 'Z'
        pauli_list.append("".join(label))
        # Note the negative sign for the cost function
        coeffs.append(-weight)
        
    return SparsePauliOp.from_list(list(zip(pauli_list, coeffs)))


def build_penalty_hamiltonian(n_qubits: int, k: int, lam: float) -> SparsePauliOp:
    """
    Builds the penalty Hamiltonian for the k-hot constraint.

    This is a pure function with no side-effects. The Hamiltonian is
    H_pen = λ * (Σ_i (1 - Z_i)/2 - k)², which penalizes states that do not
    have exactly `k` qubits in the |1> state. It is constructed using
    Qiskit's SparsePauliOp algebra for clarity and correctness.

    Args:
        n_qubits: The total number of qubits (assemblies).
        k: The target cardinality (number of active assemblies).
        lam: The positive penalty strength (lambda).

    Returns:
        A SparsePauliOp representing the penalty Hamiltonian.
    """
    identity = SparsePauliOp.from_list([("I" * n_qubits, 1.0)])
    
    # Build the sum of Z_i terms correctly
    z_sum_list = [('I' * i + 'Z' + 'I' * (n_qubits - 1 - i), 1.0) for i in range(n_qubits)]
    z_sum = SparsePauliOp.from_list(z_sum_list)

    # Number operator: N_op = 0.5 * (n_qubits * I - Σ_i Z_i)
    num_op = 0.5 * (n_qubits * identity - z_sum)

    # The term inside the square: (N_op - k*I)
    penalty_term = num_op - (k * identity)
    
    # Square the term and multiply by lambda. The @ operator handles this.
    H_penalty = lam * (penalty_term @ penalty_term)
    
    # Simplify to combine terms, making it more efficient for the solver.
    return H_penalty.simplify()


def build_x_mixer(n_qubits: int) -> SparsePauliOp:
    """
    Creates the standard transverse-field (X) mixer Hamiltonian: H_M = Σ_i X_i.
    
    This is a pure function with no side-effects.

    Args:
        n_qubits: Number of qubits.

    Returns:
        SparsePauliOp representing the X mixer.
    """
    pauli_list = [('I' * i + 'X' + 'I' * (n_qubits - 1 - i)) for i in range(n_qubits)]
    return SparsePauliOp(pauli_list)


def build_xy_mixer(n_qubits: int) -> SparsePauliOp:
    """
    Creates the constraint-preserving XY-mixer Hamiltonian.

    This is a pure function with no side-effects. The mixer is
    H_M = Σ_{i<j} (X_i X_j + Y_i Y_j). It commutes with Σ_i Z_i, thus
    preserving the k-hot subspace. The sum is over a fully-connected graph.

    Args:
        n_qubits: Number of qubits.

    Returns:
        SparsePauliOp representing the XY mixer.
    """
    pauli_list = []
    for i, j in combinations(range(n_qubits), 2):
        # Create X_i X_j term
        x_label = ['I'] * n_qubits
        x_label[i], x_label[j] = 'X', 'X'
        pauli_list.append("".join(x_label))
        
        # Create Y_i Y_j term
        y_label = ['I'] * n_qubits
        y_label[i], y_label[j] = 'Y', 'Y'
        pauli_list.append("".join(y_label))

    # The simplify() method will correctly sum terms with the same label
    return SparsePauliOp(pauli_list).simplify()


def save_hamiltonian(hamiltonian: CorticalDepthHamiltonian, filepath: str):
    """
    Save Hamiltonian data for later use.
    
    Args:
        hamiltonian: The Hamiltonian object
        filepath: Path to save the data
    """
    import pickle
    
    data = {
        'corr_matrix': hamiltonian.corr_matrix,
        'threshold': hamiltonian.threshold,
        'pauli_terms': hamiltonian.pauli_terms,
        'coefficients': hamiltonian.coefficients,
        'n_qubits': hamiltonian.n_qubits
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_hamiltonian(filepath: str) -> CorticalDepthHamiltonian:

    """
    Load a saved Hamiltonian.
    
    Args:
        filepath: Path to the saved data
    
    Returns:
        Reconstructed CorticalDepthHamiltonian
    """
    import pickle
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Reconstruct the Hamiltonian
    hamiltonian = CorticalDepthHamiltonian.__new__(CorticalDepthHamiltonian)
    hamiltonian.corr_matrix = data['corr_matrix']
    hamiltonian.threshold = data['threshold']
    hamiltonian.pauli_terms = data['pauli_terms']
    hamiltonian.coefficients = data['coefficients']
    hamiltonian.n_qubits = data['n_qubits']
    
    # Recreate the operator
    hamiltonian.operator = SparsePauliOp.from_list(
        [(term, coeff) for term, coeff in zip(hamiltonian.pauli_terms, hamiltonian.coefficients)]
    )
    
    return hamiltonian