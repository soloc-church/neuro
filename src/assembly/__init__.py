# src/assembly/__init__.py

"""
High-level helper facade for assembly graph construction.

This package provides tools for Phase 1 (classical preprocessing) and
Phase 2 (pruning) of the meta-assembly QAOA workflow. The primary entry
point is the `build_assembly_graph` function, which orchestrates the
entire process from raw session data to a pruned, scaled set of triplet
weights ready for Hamiltonian construction.

Example
-------
>>> from dataio.loaders import load_session
>>> from assembly import build_assembly_graph
>>> session = load_session("some_session_id")
>>> scaled_triplets = build_assembly_graph(session, sigma_thresh=1.5, max_terms=100)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
#import logging

from .clustering import find_optimal_k, soft_cluster_neurons, rank_and_select_assemblies
from .overlap import compute_triplet_weights
from .pruning import prune_and_rescale
from features import pearson_corr_matrix

if TYPE_CHECKING:
    from dataio.data_structures import SessionData

# Define the public API of the 'assembly' package
__all__ = [
    "soft_cluster_neurons",
    "find_optimal_k",
    "compute_triplet_weights",
    "prune_and_rescale",
    "build_assembly_graph",
]


def build_assembly_graph(
    session_data: SessionData,
    *,
    k_assemblies: int | None = 50,
    select_mode: str | None = None,
    n_select: int | None = None,
    select_indices: List[int] | None = None,
    normalization_strategy: str | None = None,
    sigma_thresh: float = 1.0,
    max_terms: int | None = 100,
) -> dict[tuple[int, int, int], float]:
    """
    End-to-end utility for Phases 1 & 2, now with selection and normalization.

    This function performs the full classical preprocessing pipeline:
    1.  Clusters neurons into `k_assemblies` soft, overlapping assemblies.
    2.  Optionally selects a subset of `n_select` assemblies based on a
        `select_mode` or pre-defined `select_indices`.
    3.  Computes the raw triplet-overlap weights (w_abc) for the selected subset.
    4.  Optionally applies a `normalization_strategy` (e.g., "rich_club").
    5.  Prunes, caps, and rescales the final weights.

    Args:
        session_data: The SessionData object.
        k_assemblies: The initial number of assemblies to create.
        select_mode: Method to select top assemblies ('size', 'cohesion').
        n_select: Number of assemblies to down-select to.
        select_indices: A pre-defined list of assembly indices to use.
        normalization_strategy: Weight normalization method ('rich_club').
        sigma_thresh: The `μ + sigma_thresh·σ` pruning rule threshold.
        max_terms: A hard cap on the number of final triplet terms.

    Returns:
        A dictionary mapping the pruned triplet indices to their scaled weights.
    """
    # Phase 1a: Full Clustering
    full_memberships = soft_cluster_neurons(session_data, k_override=k_assemblies)
    
    # Phase 1b: Assembly Selection
    if select_indices is not None:
        # Use pre-selected indices (e.g., from random sampling)
        logger.info(f"Using pre-defined subset of {len(select_indices)} assemblies.")
        sub_memberships = full_memberships[select_indices, :]
    elif select_mode and n_select:
        # Rank and select top N assemblies
        corr_matrix = pearson_corr_matrix(session_data.normalized_traces)
        sub_memberships, _, selected_indices = rank_and_select_assemblies(
            full_memberships, corr_matrix, mode=select_mode, n_select=n_select
        )
        # Remap indices in weights dict later if needed, but overlap computation is local
    else:
        # No selection, use all assemblies
        sub_memberships = full_memberships

    # Phase 1c: Compute Triplet Weights on the (potentially smaller) subset
    # The indices (a,b,c) will be relative to the sub-matrix (0 to n_select-1)
    weights_raw = compute_triplet_weights(sub_memberships)

    # Phase 2: Prune, Normalize, and Rescale
    scaled_triplets = prune_and_rescale(
        weights_raw,
        session_id=session_data.session_id,
        sigma_thresh=sigma_thresh,
        max_terms=max_terms,
        normalization_strategy=normalization_strategy,
        full_memberships=full_memberships,  # Pass full matrix for normalization metrics
    )

    return scaled_triplets