"""
Analysis metrics 
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Any

def calc_corr_matrix(activity_matrix):
    """
    Calculate Pearson correlation matrix from neural activity data.
    
    Args:
        activity_matrix (np.ndarray): Shape (N_neurons, T_timepoints)
    
    Returns:
        np.ndarray: Correlation matrix of shape (N_neurons, N_neurons)
    """
    # Handle any NaN values
    activity_clean = np.nan_to_num(activity_matrix, nan=0.0)
    
    # Standardize each neuron's activity
    activity_centered = activity_clean - np.mean(activity_clean, axis=1, keepdims=True)
    activity_std = np.std(activity_centered, axis=1, keepdims=True)
    
    # Avoid division by zero
    activity_std[activity_std == 0] = 1.0
    activity_normalized = activity_centered / activity_std
    
    # Compute correlation matrix
    corr_matrix = np.dot(activity_normalized, activity_normalized.T) / activity_normalized.shape[1]
    
    # Ensure diagonal is exactly 1
    np.fill_diagonal(corr_matrix, 1.0)
    
    return corr_matrix


def classical_partition(corr_matrix, method='average'):
    """
    Perform hierarchical clustering to partition neurons into 2 groups.
    
    Args:
        corr_matrix (np.ndarray): Correlation matrix (N x N)
        method (str): Linkage method for hierarchical clustering
    
    Returns:
        np.ndarray: Binary labels (0 or 1) for each neuron
    """
    # Convert correlation to distance
    distance_matrix = 1 - np.abs(corr_matrix)
    
    # Ensure distance matrix is valid
    np.fill_diagonal(distance_matrix, 0)
    
    # Convert to condensed form for scipy
    distance_condensed = squareform(distance_matrix, checks=False)
    
    # Perform hierarchical clustering
    linkage_matrix = hierarchy.linkage(distance_condensed, method=method)
    
    # Get cluster labels for 2 clusters
    labels = hierarchy.fcluster(linkage_matrix, t=2, criterion='maxclust') - 1
    
    return labels.astype(int)


def compare_partitions(labels_qaoa, labels_classical, metric='ari'):
    """
    Compare two partitions using Adjusted Rand Index or Normalized Mutual Information.
    
    Args:
        labels_qaoa (np.ndarray): QAOA partition labels
        labels_classical (np.ndarray): Classical partition labels
        metric (str): 'ari' for Adjusted Rand Index or 'nmi' for Normalized Mutual Information
    
    Returns:
        float: Similarity score between partitions (higher is better)
    """
    if metric == 'ari':
        return adjusted_rand_score(labels_qaoa, labels_classical)
    elif metric == 'nmi':
        return normalized_mutual_info_score(labels_qaoa, labels_classical)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'ari' or 'nmi'.")

def analyze_session_quality(session) -> Dict[str, Any]:
    """
    Quick, composite ‘quality’ score for a 2‑photon session.

    Parameters
    ----------
    session : Any
        Object returned by ``load_session_complete()`` with at least
        the following attributes:
            • activity_matrix : (neurons × time) ndarray
            • neuron_coords   : (neurons × 3) ndarray  (x, y, z)
    
    Returns
    -------
    dict
        {
            "overall_score"  : float,
            "mean_abs_corr"  : float,
            "depth_stats"    : dict,
        }
    """
    # ------------------------------------------------------------------ 1. correlation
    # Uses low‑level helper already defined in this file
    corr = calc_corr_matrix(session.activity_matrix)
    # Only upper‑triangle, exclude diagonal
    iu = np.triu_indices_from(corr, k=1)
    mean_abs_corr = float(np.abs(corr[iu]).mean())

    # ------------------------------------------------------------------ 2. depth separation
    # Partition neurons into two communities
    labels = classical_partition(corr)
    # Z‑coordinate (depth) is column 2
    try:                        
        z = session.neuron_coords[:, 2]
    except AttributeError:      
        z = np.array([n.z for n in session.neurons], dtype=float)

    if (z <= 0).all():
        z = -z
    z = z - z.min()
    
    depth = depth_stats(labels, z)

    # ------------------------------------------------------------------ 3. overall score
    # Lower mean_abs_corr is better; higher separation effect‑size is better
    effect = depth.get("separation", {}).get("effect_size", 0.0)
    overall = (1.0 - mean_abs_corr) * (effect if np.isfinite(effect) else 0.0)

    return {
        "overall_score": overall,
        "mean_abs_corr": mean_abs_corr,
        "depth_stats": depth,
    }

# Add the helper to the public interface for convenient imports
try:
    __all__.append("analyze_session_quality")
except NameError:
    __all__ = ["analyze_session_quality"]

def depth_stats(labels, coords_z):
    """
    Calculate depth statistics for each partition group.
    
    Args:
        labels (np.ndarray): Partition labels (0 or 1)
        coords_z (np.ndarray): Z-coordinates (depth) for each neuron
    
    Returns:
        dict: Statistics for each group including mean, std, count
    """
    stats = {}
    
    for label in np.unique(labels):
        mask = labels == label
        z_values = coords_z[mask]
        
        stats[f'group_{label}'] = {
            'mean_depth': np.mean(z_values),
            'std_depth': np.std(z_values),
            'min_depth': np.min(z_values),
            'max_depth': np.max(z_values),
            'count': np.sum(mask),
            'fraction': np.mean(mask)
        }
    
    # Add separation metric
    if len(np.unique(labels)) == 2:
        mean_diff = abs(stats['group_0']['mean_depth'] - stats['group_1']['mean_depth'])
        pooled_std = np.sqrt(
            (stats['group_0']['std_depth']**2 + stats['group_1']['std_depth']**2) / 2
        )
        stats['separation'] = {
            'mean_difference': mean_diff,
            'effect_size': mean_diff / pooled_std if pooled_std > 0 else np.inf
        }
    
    return stats


def plot_depth_scatter(coords_3d, labels, title="Neuron Partitioning by Depth", 
                      save_path=None, figsize=(10, 8)):
    """
    Create 3D scatter plot of neurons colored by partition labels.
    
    Args:
        coords_3d (np.ndarray): Shape (N_neurons, 3) with x, y, z coordinates
        labels (np.ndarray): Partition labels for each neuron
        title (str): Plot title
        save_path (str): Optional path to save figure
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create colormap
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each group
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(coords_3d[mask, 0], 
                  coords_3d[mask, 1], 
                  coords_3d[mask, 2],
                  c=[colors[i]], 
                  label=f'Group {label}',
                  s=50, 
                  alpha=0.6,
                  edgecolors='k',
                  linewidth=0.5)
    
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_zlabel('Z depth (μm)')
    ax.set_title(title)
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust viewing angle for better depth visualization
    ax.view_init(elev=20, azim=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_correlation_matrix(corr_matrix, labels=None, title="Neural Correlation Matrix",
                          save_path=None, figsize=(10, 8)):
    """
    Plot correlation matrix with optional cluster ordering.
    
    Args:
        corr_matrix (np.ndarray): Correlation matrix
        labels (np.ndarray): Optional cluster labels for reordering
        title (str): Plot title
        save_path (str): Optional path to save figure
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Reorder matrix by clusters if labels provided
    if labels is not None:
        order = np.argsort(labels)
        corr_ordered = corr_matrix[order][:, order]
        
        # Add cluster boundaries
        unique_labels = np.unique(labels)
        boundaries = []
        for label in unique_labels[:-1]:
            boundary = np.sum(labels[order] <= label)
            boundaries.append(boundary)
    else:
        corr_ordered = corr_matrix
        boundaries = []
    
    # Plot correlation matrix
    im = ax.imshow(corr_ordered, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Add cluster boundaries
    for boundary in boundaries:
        ax.axhline(y=boundary-0.5, color='k', linewidth=2)
        ax.axvline(x=boundary-0.5, color='k', linewidth=2)
    
    ax.set_title(title)
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Neuron Index')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig