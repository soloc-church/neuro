# src/assembly/overlap.py

from __future__ import annotations
import itertools
import logging

import numpy as np
from tqdm import tqdm

__all__ = ["compute_triplet_weights"]

logger = logging.getLogger(__name__)


def compute_triplet_weights(
    memberships: np.ndarray,
) -> dict[tuple[int, int, int], int]:
    """
    Computes the triplet-overlap weights (w_abc) from a membership matrix.

    w_abc is the number of individual neurons that are members of all three
    assemblies (a, b, and c) simultaneously. This implementation uses a
    vectorized `numpy.einsum` operation for high performance.

    Parameters
    ----------
    memberships : np.ndarray
        A binary membership matrix of shape (k, n_neurons), where k is the
        number of assemblies and n is the number of neurons.

    Returns
    -------
    dict[tuple[int, int, int], int]
        A dictionary mapping ordered triplets (a, b, c) with a < b < c to
        their integer overlap weight. Triplets with zero overlap are excluded.
    """
    n_assemblies, n_neurons = memberships.shape
    logger.info(
        f"Computing triplet weights for {n_assemblies} assemblies and {n_neurons} neurons."
    )

    if n_assemblies < 3:
        logger.warning("Fewer than 3 assemblies, no triplets can be formed.")
        return {}

    if memberships.dtype != np.int8:
        M = memberships.astype(np.int8)
    else:
        M = memberships

    # Vectorized computation of the entire (k, k, k) overlap tensor.
    # 'ik,jk,lk->ijl' sums the product over the neuron dimension 'k'.
    # This is equivalent to (M[i] * M[j] * M[l]).sum() for all i,j,l.
    logger.info("Building overlap tensor with `einsum`...")
    overlap_tensor = np.einsum('ik,jk,lk->ijl', M, M, M, optimize='optimal')
    logger.info("Overlap tensor built.")

    weights: dict[tuple[int, int, int], int] = {}
    triplet_indices = list(itertools.combinations(range(n_assemblies), 3))
    
    # Extract weights for the upper triangle (i < j < l)
    for a, b, c in tqdm(triplet_indices, desc="Extracting w_abc"):
        overlap = overlap_tensor[a, b, c]
        if overlap > 0:
            weights[(a, b, c)] = int(overlap)

    logger.info(
        f"Found {len(weights)} non-zero triplet overlaps out of "
        f"{len(triplet_indices)} possible combinations."
    )
    return weights