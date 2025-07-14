# src/assembly/clustering.py

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from typing import List

from features import pearson_corr_matrix

if TYPE_CHECKING:
    from dataio.data_structures import SessionData

__all__ = ["soft_cluster_neurons", "find_optimal_k"]

# Configure a basic logger
logger = logging.getLogger(__name__)


def find_optimal_k(
    corr: np.ndarray,
    k_min: int = 20,
    k_max: int = 80,
    step: int = 2,
    sample_size: int | None = 5000,
) -> int:
    """
    Finds the optimal number of clusters (k) using silhouette analysis.

    This function sweeps a range of k values, performs hierarchical clustering
    for each, and returns the k that yields the highest mean silhouette score
    on the correlation-distance matrix (1 – |r|).

    For performance, the silhouette score is calculated on a random subsample
    of the data points if the dataset is large.

    Parameters
    ----------
    corr : np.ndarray
        The neuron-neuron correlation matrix.
    k_min : int, optional
        The minimum number of clusters to test, by default 20.
    k_max : int, optional
        The maximum number of clusters to test, by default 80.
    step : int, optional
        The step size for sweeping k values, by default 2.
    sample_size : int | None, optional
        Number of data points to use for silhouette score calculation.
        If None, all points are used. By default 5000.

    Returns
    -------
    int
        The optimal number of clusters (k) found.
    """
    logger.info(f"Searching for optimal k in range [{k_min}, {k_max}]...")
    dist_matrix = 1 - np.abs(corr)
    np.fill_diagonal(dist_matrix, 0)

    k_range = range(k_min, k_max + 1, step)
    scores = []
    best_score = -1
    best_k = k_min

    for k in tqdm(k_range, desc="Finding optimal k"):
        model = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
        labels = model.fit_predict(dist_matrix)

        score = silhouette_score(
            dist_matrix, labels, metric="precomputed", sample_size=sample_size
        )
        scores.append(score)

        if score > best_score:
            best_score = score
            best_k = k

    logger.info(f"Optimal k found: {best_k} (Silhouette Score: {best_score:.4f})")
    return best_k


def soft_cluster_neurons(
    session: SessionData, k_override: int | None = None
) -> np.ndarray:
    """
    Performs soft clustering of neurons into overlapping assemblies using NMF.

    The method applies Non-negative Matrix Factorization (NMF) to the absolute
    correlation matrix. This yields a "loadings" matrix where each row
    represents an assembly and each column a neuron. To create binary,
    overlapping memberships, each assembly's loading vector is thresholded at
    1 standard deviation above its mean.

    Parameters
    ----------
    session : SessionData
        The session data object, used to compute the correlation matrix.
    k_override : int | None, optional
        If provided, this number of assemblies is used, overriding the
        automatic k selection, by default None.

    Returns
    -------
    np.ndarray
        A binary membership matrix `M` of shape (k, n_neurons), where `M[i, j] = 1`
        indicates that neuron `j` is a member of assembly `i`.
    """
    logger.info("Starting soft clustering of neurons into assemblies.")
    corr_matrix = pearson_corr_matrix(session.normalized_traces)
    n_neurons = corr_matrix.shape[0]

    if k_override:
        k = k_override
        logger.info(f"Using provided k_override: {k} assemblies.")
    else:
        k = find_optimal_k(corr_matrix, k_max=min(80, n_neurons - 1))

    # NMF works on non-negative matrices
    abs_corr = np.abs(corr_matrix)

    logger.info(f"Performing NMF with {k} components...")
    model = NMF(
        n_components=k, init="nndsvda", solver="cd", max_iter=1000, random_state=42
    )
    # W: (n_neurons, k), H: (k, n_neurons)
    # H represents the component loadings for each neuron
    H = model.fit_transform(abs_corr).T  # Transpose to get (k, n_neurons)

    # Binarize the membership matrix based on the thresholding rule
    logger.info("Binarizing membership matrix (μ + 1σ threshold)...")
    memberships = np.zeros_like(H, dtype=np.int8)
    for i in range(k):
        component_loadings = H[i, :]
        mean_loading = np.mean(component_loadings)
        std_loading = np.std(component_loadings)
        threshold = mean_loading + std_loading
        memberships[i, component_loadings >= threshold] = 1

    n_members = np.sum(memberships)
    logger.info(f"Clustering complete. Found {n_members} total assembly memberships.")
    return memberships

def _member_indices(memberships: np.ndarray) -> List[np.ndarray]:
    """Helper to get lists of neuron indices for each assembly."""
    return [np.where(memberships[i, :])[0] for i in range(memberships.shape[0])]

def rank_and_select_assemblies(
    memberships: np.ndarray,
    corr_matrix: np.ndarray,
    mode: str = "size",
    n_select: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Ranks assemblies and selects a subset.

    This function can rank assemblies by their size (number of member neurons)
    or their internal cohesion (average absolute correlation between members).
    It returns the subset of the membership matrix and the original indices
    of the selected assemblies.

    Args:
        memberships: The full (K_full, N_neurons) binary membership matrix.
        corr_matrix: The full (N_neurons, N_neurons) correlation matrix.
        mode: The ranking strategy. "size" or "cohesion".
        n_select: The number of top assemblies to select. If None, returns all.

    Returns:
        A tuple containing:
        - sub_memberships: The (n_select, N_neurons) membership matrix for the top assemblies.
        - sub_corr_matrix: The (n_select, n_select) correlation matrix between selected assemblies.
        - selected_indices: The original indices (from 0 to K_full-1) of the selected assemblies.
    """
    if n_select is None or n_select >= memberships.shape[0]:
        # Trivial case: return everything, no selection needed
        assembly_corr = memberships @ corr_matrix @ memberships.T
        return memberships, assembly_corr, list(range(memberships.shape[0]))

    logger.info(f"Ranking {memberships.shape[0]} assemblies by '{mode}' to select top {n_select}.")
    
    if mode == "size":
        scores = memberships.sum(axis=1)
    elif mode == "cohesion":
        neuron_indices_per_assembly = _member_indices(memberships)
        scores = np.array([
            np.mean(np.abs(corr_matrix[np.ix_(idxs, idxs)])) if len(idxs) > 1 else 0
            for idxs in neuron_indices_per_assembly
        ])
    else:
        raise ValueError("Invalid mode. Choose 'size' or 'cohesion'.")

    # Get indices of the top N assemblies in descending order of score
    selected_indices = scores.argsort()[-n_select:][::-1]
    selected_indices.sort() # Keep original relative order

    logger.info(f"Selected assembly indices: {selected_indices}")

    sub_memberships = memberships[selected_indices, :]
    
    assembly_corr = memberships @ corr_matrix @ memberships.T
    sub_assembly_corr = assembly_corr[np.ix_(selected_indices, selected_indices)]

    return sub_memberships, sub_assembly_corr, list(selected_indices)


__all__.append("rank_and_select_assemblies")