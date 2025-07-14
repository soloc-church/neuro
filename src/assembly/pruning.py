# src/assembly/pruning.py

from __future__ import annotations
import logging
from pathlib import Path
from typing import Mapping

import numpy as np

__all__ = ["prune_and_rescale"]

logger = logging.getLogger(__name__)

# Define the directory for derived artifacts, ensuring it exists
DERIVED_DIR = Path(__file__).resolve().parents[2] / "data" / "derived" / "qaoa_meta"
DERIVED_DIR.mkdir(parents=True, exist_ok=True)


def prune_and_rescale(
    weights: Mapping[tuple[int, int, int], int],
    session_id: str,
    *,
    sigma_thresh: float = 1.0,
    max_terms: int | None = 100,
    normalization_strategy: str | None = None,
    full_memberships: np.ndarray | None = None, # Needed for rich-club
) -> dict[tuple[int, int, int], float]:
    """
    Prunes, caps, and rescales triplet weights.

    The function first filters triplets, keeping only those with a weight
    greater than or equal to (mean + sigma_thresh * std_dev). It then caps
    the total number of terms to `max_terms` by taking the strongest ones.
    Finally, it rescales the pruned weights to the [0, 1] range.

    The resulting pruned and scaled weights are saved as a compressed NumPy
    archive (`.npz`) for efficient downstream reuse.

    Parameters
    ----------
    weights : Mapping[tuple[int, int, int], int]
        The raw integer overlap weights for all triplets.
    session_id : str
        The identifier for the session, used for artifact naming.
    sigma_thresh : float, optional
        The number of standard deviations above the mean to use as a
        pruning threshold, by default 1.0.
    max_terms : int | None, optional
        A hard cap on the number of triplets to keep after pruning. If None,
        no cap is applied, by default 100.

    Returns
    -------
    dict[tuple[int, int, int], float]
        A dictionary of the final pruned and rescaled triplet weights.
    """
    if not weights:
        logger.warning("Input weights dictionary is empty. Returning empty dict.")
        return {}
    
    normalized_weights = dict(weights) # Start with a copy
    if normalization_strategy == "rich_club":
        if full_memberships is None:
            raise ValueError("`full_memberships` matrix is required for rich-club normalization.")
        
        logger.info("Applying 'rich-club' normalization to weights.")
        assembly_sizes = full_memberships.sum(axis=1)
        n_total_neurons = full_memberships.shape[1]
        
        temp_weights = {}
        for (a, b, c), w_raw in weights.items():
            # Expected overlap under random model: |A|*|B|*|C| / N^2
            expected_overlap = (
                assembly_sizes[a] * assembly_sizes[b] * assembly_sizes[c]
            ) / (n_total_neurons ** 2)
            
            # Clamp denominator to avoid division by zero or huge weights
            clamped_expected = max(expected_overlap, 1.0)
            
            temp_weights[(a, b, c)] = w_raw / clamped_expected
        normalized_weights = temp_weights

    weight_values = np.array(list(normalized_weights.values()), dtype=np.float64)
    if len(weight_values) == 0:
        return {}
        
    mean_w = np.mean(weight_values)
    std_w = np.std(weight_values)
    threshold = mean_w + sigma_thresh * std_w

    logger.info(f"Pruning weights. μ={mean_w:.2f}, σ={std_w:.2f}, threshold (μ + {sigma_thresh}σ)={threshold:.2f}")

    # 1. Prune by statistical significance
    pruned_weights = {
        k: v for k, v in weights.items() if v >= threshold
    }
    logger.info(f"Kept {len(pruned_weights)} terms after σ-pruning.")

    # 2. Cap the number of terms
    if max_terms is not None and len(pruned_weights) > max_terms:
        sorted_items = sorted(pruned_weights.items(), key=lambda item: item[1], reverse=True)
        pruned_weights = dict(sorted_items[:max_terms])
        logger.info(f"Capped terms to {len(pruned_weights)} (max_terms={max_terms}).")

    if not pruned_weights:
        logger.warning("No weights remained after pruning. Returning empty dict.")
        return {}

    # 3. Rescale to [0, 1]
    final_values = np.array(list(pruned_weights.values()))
    w_max = np.max(final_values)
    
    if w_max == 0:
        return {k: 0.0 for k in pruned_weights}
        
    scaled_weights = {k: v / w_max for k, v in pruned_weights.items()}

    # --- Save artifact in .npz format ---
    artifact_path = DERIVED_DIR / f"pruned_triplets_{session_id}.npz"
    
    # Prepare data for npz storage
    final_keys_array = np.array(list(scaled_weights.keys()), dtype=int)
    final_values_array = np.array(list(scaled_weights.values()), dtype=float)
    
    metadata = {
        "session_id": session_id,
        "sigma_thresh": sigma_thresh,
        "max_terms": max_terms,
        "raw_stats": {"mean": mean_w, "std": std_w, "count": len(weights)},
        "pruning_threshold": threshold,
        "final_term_count": len(scaled_weights),
        "max_scaled_weight": 1.0,
    }

    np.savez_compressed(
        artifact_path,
        triplets=final_keys_array,
        weights=final_values_array,
        metadata=np.array(metadata) # Store dict as 0-dim object array
    )
    logger.info(f"Saved pruned and scaled weights to {artifact_path}")

    return scaled_weights