"""
Tests for QAOA simulators module.
"""

import pytest
import numpy as np
from src.qaoa.hamiltonians import from_corr_matrix, CorticalDepthHamiltonian
from src.qaoa.simulators import (
    QAOACircuit,
    build_qaoa_circuit,
    expectation_value,
    optimize_qaoa,
    get_partition_from_counts,
    grid_search_init
)


class TestQAOACircuitConstruction:
    """Test QAOA circuit building."""
    
    def test_build_qaoa_circuit_structure(self):
        """Test that QAOA circuit has correct structure."""
        # Simple 2-qubit correlation matrix
        corr_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        hamiltonian = from_corr_matrix(corr_matrix)
        
        circuit_func = build_qaoa_circuit(hamiltonian, p=1)
        
        # Check attributes
        assert circuit_func.n_qubits == 2
        assert circuit_func.p == 1
        
        # Create circuit with specific parameters
        params = np.array([0.5, 0.3])  # gamma, beta
        circuit = circuit_func(params)
        
        # Should have measurements
        assert circuit.num_qubits == 2
        assert circuit.num_clbits == 2
    
    def test_qaoa_circuit_gates(self):
        """Test that circuit contains expected gates."""
        # 3-qubit system
        corr_matrix = np.array([
            [1.0, 0.5, 0.0],
            [0.5, 1.0, -0.3],
            [0.0, -0.3, 1.0]
        ])
        hamiltonian = from_corr_matrix(corr_matrix)
        
        qaoa_circuit = QAOACircuit(hamiltonian, p=1)
        circuit = qaoa_circuit.bind_parameters(np.array([0.5, 0.3]))
        
        # Remove measurements for analysis
        circuit_no_meas = circuit.remove_final_measurements(inplace=False)
        
        # Count gate types
        gate_counts = circuit_no_meas.count_ops()
        
        # Should have Hadamards
        assert gate_counts.get('h', 0) == 3
        
        # Should have RZZ gates for correlations
        assert gate_counts.get('rzz', 0) == 2  # Two non-zero correlations
        
        # Should have RX gates for mixer
        assert gate_counts.get('rx', 0) == 3


class TestExpectationValue:
    """Test expectation value calculation."""
    
    def test_expectation_value_at_zero(self):
        """Test expectation value at gamma=0, beta=0."""
        # At these parameters, circuit is just Hadamards
        corr_matrix = np.array([
            [1.0, 0.8],
            [0.8, 1.0]
        ])
        hamiltonian = from_corr_matrix(corr_matrix)
        circuit_func = build_qaoa_circuit(hamiltonian, p=1)
        
        # At gamma=0, beta=0, we're in equal superposition
        # Expectation value should be 0 for any ZZ term
        exp_val = expectation_value(circuit_func, np.array([0.0, 0.0]))
        
        assert abs(exp_val) < 1e-10
    
    def test_expectation_value_max_cut(self):
        """Test expectation value for known max-cut configuration."""
        # Simple 2-node graph with negative correlation
        # Max cut is when nodes are in different partitions
        corr_matrix = np.array([
            [1.0, -1.0],
            [-1.0, 1.0]
        ])
        hamiltonian = from_corr_matrix(corr_matrix)
        circuit_func = build_qaoa_circuit(hamiltonian, p=1)
        
        # The minimum energy should be -(-1) = 1 for opposite spins
        # Test with parameters that should give good cut
        exp_val = expectation_value(circuit_func, np.array([np.pi/2, np.pi/4]))
        
        # Should be negative (we minimize -J_ij * Z_i * Z_j)
        assert exp_val < 0


class TestOptimization:
    """Test QAOA optimization."""
    
    def test_optimize_qaoa_simple(self):
        """Test optimization on simple 2-qubit system."""
        corr_matrix = np.array([
            [1.0, -0.8],
            [-0.8, 1.0]
        ])
        hamiltonian = from_corr_matrix(corr_matrix)
        circuit_func = build_qaoa_circuit(hamiltonian, p=1)
        
        # Run optimization
        result = optimize_qaoa(circuit_func, maxiter=50, verbose=False)
        
        # Check output structure
        assert 'optimal_params' in result
        assert 'optimal_cost' in result
        assert 'history' in result
        
        # Optimal cost should be negative (good cut)
        assert result['optimal_cost'] < 0
        
        # History should show improvement
        history_costs = result['history']['costs']
        assert len(history_costs) > 0
        assert history_costs[-1] <= history_costs[0]  # Should improve or stay same
    
    def test_grid_search_init(self):
        """Test grid search initialization."""
        corr_matrix = np.eye(3)
        corr_matrix[0, 1] = corr_matrix[1, 0] = 0.5
        
        hamiltonian = from_corr_matrix(corr_matrix)
        circuit_func = build_qaoa_circuit(hamiltonian, p=1)
        
        # Run grid search
        init_params = grid_search_init(circuit_func, n_points=10)
        
        assert len(init_params) == 2
        assert 0 <= init_params[0] <= 2*np.pi  # gamma range
        assert 0 <= init_params[1] <= np.pi     # beta range


class TestPartitionExtraction:
    """Test extracting partition from measurement results."""
    
    def test_get_partition_from_counts(self):
        """Test partition extraction from counts."""
        # Mock counts with clear winner
        counts = {
            '00110': 500,
            '11001': 200,
            '10101': 100,
            '00111': 50
        }
        
        labels = get_partition_from_counts(counts)
        
        # Should pick most frequent: '00110'
        # Note: Qiskit reverses bit order
        expected = np.array([0, 1, 1, 0, 0])
        np.testing.assert_array_equal(labels, expected)
    
    def test_get_partition_single_bitstring(self):
        """Test with single bitstring."""
        counts = {'10101': 1000}
        
        labels = get_partition_from_counts(counts)
        expected = np.array([1, 0, 1, 0, 1])
        np.testing.assert_array_equal(labels, expected)


class TestEndToEnd:
    """Test complete QAOA workflow."""
    
    def test_small_cortical_depth_problem(self):
        """Test on small cortical depth-like problem."""
        # Create correlation matrix with depth structure
        # Neurons 0-2: shallow (correlated)
        # Neurons 3-5: deep (correlated)
        # Cross-depth: anti-correlated
        n = 6
        corr_matrix = np.eye(n)
        
        # Within-layer positive correlation
        for i in range(3):
            for j in range(3):
                if i != j:
                    corr_matrix[i, j] = 0.7
                    corr_matrix[i+3, j+3] = 0.7
        
        # Between-layer negative correlation
        for i in range(3):
            for j in range(3, 6):
                corr_matrix[i, j] = corr_matrix[j, i] = -0.5
        
        # Create Hamiltonian
        hamiltonian = from_corr_matrix(corr_matrix, threshold=0.1)
        
        # Build circuit
        circuit_func = build_qaoa_circuit(hamiltonian, p=1)
        
        # Optimize
        result = optimize_qaoa(circuit_func, maxiter=30, verbose=False)
        
        # Check that we found a good solution
        assert result['optimal_cost'] < -1.0  # Should find good cut
        