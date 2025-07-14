"""
Tests for analysis metrics module.
"""

import pytest
import numpy as np
from src.analysis.metrics import (
    calc_corr_matrix,
    classical_partition,
    compare_partitions,
    depth_stats,
    plot_depth_scatter,
    plot_correlation_matrix
)


class TestCorrelationMatrix:
    """Test correlation matrix calculation."""
    
    def test_calc_corr_matrix_identity(self):
        """Test that identical signals have correlation 1."""
        # Create identical signals
        n_neurons, n_timepoints = 5, 100
        activity = np.random.randn(n_neurons, n_timepoints)
        activity[1, :] = activity[0, :]  # Make neuron 1 identical to neuron 0
        
        corr_matrix = calc_corr_matrix(activity)
        
        # Check diagonal is 1
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), 1.0)
        
        # Check identical neurons have correlation 1
        assert abs(corr_matrix[0, 1] - 1.0) < 1e-10
    
    def test_calc_corr_matrix_anticorrelated(self):
        """Test that opposite signals have correlation -1."""
        n_neurons, n_timepoints = 3, 100
        activity = np.random.randn(n_neurons, n_timepoints)
        activity[1, :] = -activity[0, :]  # Make neuron 1 opposite to neuron 0
        
        corr_matrix = calc_corr_matrix(activity)
        
        # Check anti-correlation
        assert abs(corr_matrix[0, 1] - (-1.0)) < 1e-10
    
    def test_calc_corr_matrix_with_nans(self):
        """Test handling of NaN values."""
        activity = np.random.randn(10, 50)
        activity[0, 10:20] = np.nan
        
        corr_matrix = calc_corr_matrix(activity)
        
        # Should not have NaNs in output
        assert not np.any(np.isnan(corr_matrix))


class TestClassicalPartition:
    """Test classical clustering baseline."""
    
    def test_classical_partition_two_blocks(self):
        """Test partitioning of clear block structure."""
        # Create block correlation matrix
        n = 20
        corr_matrix = np.eye(n)
        
        # Block 1: neurons 0-9
        for i in range(10):
            for j in range(10):
                corr_matrix[i, j] = 0.8 if i != j else 1.0
        
        # Block 2: neurons 10-19
        for i in range(10, 20):
            for j in range(10, 20):
                corr_matrix[i, j] = 0.8 if i != j else 1.0
        
        # Weak inter-block correlation
        for i in range(10):
            for j in range(10, 20):
                corr_matrix[i, j] = corr_matrix[j, i] = 0.1
        
        labels = classical_partition(corr_matrix)
        
        # Check that we get two distinct groups
        assert len(np.unique(labels)) == 2
        
        # Check that neurons in same block get same label
        assert len(np.unique(labels[:10])) == 1
        assert len(np.unique(labels[10:20])) == 1
        
        # Check that blocks get different labels
        assert labels[0] != labels[10]


class TestComparePartitions:
    """Test partition comparison metrics."""
    
    def test_compare_partitions_identical(self):
        """Test that identical partitions have ARI=1."""
        labels1 = np.array([0, 0, 1, 1, 0, 1])
        labels2 = labels1.copy()
        
        ari = compare_partitions(labels1, labels2, metric='ari')
        assert ari == 1.0
        
        nmi = compare_partitions(labels1, labels2, metric='nmi')
        assert nmi == 1.0
    
    def test_compare_partitions_opposite(self):
        """Test that opposite partitions still have high similarity."""
        labels1 = np.array([0, 0, 1, 1, 0, 1])
        labels2 = 1 - labels1  # Flip labels
        
        ari = compare_partitions(labels1, labels2, metric='ari')
        assert ari == 1.0  # ARI is invariant to label permutation
    
    def test_compare_partitions_random(self):
        """Test that random partitions have low similarity."""
        np.random.seed(42)
        labels1 = np.random.randint(0, 2, size=100)
        labels2 = np.random.randint(0, 2, size=100)
        
        ari = compare_partitions(labels1, labels2, metric='ari')
        assert -0.1 < ari < 0.1  # Should be close to 0 for random


class TestDepthStats:
    """Test depth statistics calculation."""
    
    def test_depth_stats_separation(self):
        """Test depth statistics with clear separation."""
        # Create two groups with different depths
        labels = np.array([0, 0, 0, 1, 1, 1])
        coords_z = np.array([100, 120, 110, 300, 320, 310])  # Clear separation
        
        stats = depth_stats(labels, coords_z)
        
        # Check group statistics
        assert abs(stats['group_0']['mean_depth'] - 110) < 1e-10
        assert abs(stats['group_1']['mean_depth'] - 310) < 1e-10
        
        # Check separation metrics
        assert stats['separation']['mean_difference'] == 200
        assert stats['separation']['effect_size'] > 5  # Large effect size
    
    def test_depth_stats_single_group(self):
        """Test with only one group."""
        labels = np.zeros(5)  # All in group 0
        coords_z = np.array([100, 110, 120, 130, 140])
        
        stats = depth_stats(labels, coords_z)
        
        assert 'group_0' in stats
        assert 'group_1' not in stats
        assert 'separation' not in stats


class TestVisualization:
    """Test visualization functions (smoke tests)."""
    
    def test_plot_depth_scatter_smoke(self):
        """Test that depth scatter plot runs without error."""
        coords_3d = np.random.randn(50, 3) * 100
        labels = np.random.randint(0, 2, size=50)
        
        # Should run without error
        fig = plot_depth_scatter(coords_3d, labels)
        assert fig is not None
    
    def test_plot_correlation_matrix_smoke(self):
        """Test that correlation matrix plot runs without error."""
        corr_matrix = np.random.randn(20, 20)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(corr_matrix, 1.0)
        
        labels = np.random.randint(0, 2, size=20)
        
        # Should run without error
        fig = plot_correlation_matrix(corr_matrix, labels)
        assert fig is not None