# src/qaoa/pipelines/meta_assembly.py

"""
Top-level driver for the meta-assembly QAOA workflow.

This module provides the main pipeline function, `run_meta_qaoa`, which is
intended to be called from a notebook like `notebooks/03_meta-assembly_qaoa.ipynb`.
It orchestrates the entire process from classical pre-processing to the final
fine-tuned QAOA result.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
# Use the backend-agnostic primitive from Qiskit for better compatibility
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import SPSA 

# --- Placeholder Imports ---
# These functions will be implemented in the hamiltonians and runners modules next.
# We import them here to define the pipeline's structure correctly.
from qaoa.hamiltonians import (
    build_cost_hamiltonian,
    build_penalty_hamiltonian,
    build_x_mixer,
    build_xy_mixer,
)
from qaoa.runners import execute_qaoa, get_ionq_aria_backend
# --- End Placeholder Imports ---

from assembly import build_assembly_graph

if TYPE_CHECKING:
    from qiskit_ibm_runtime import QiskitRuntimeService
    from dataio.data_structures import SessionData

logger = logging.getLogger(__name__)

# Ensure the directory for derived artifacts exists before any writes.
DERIVED_DIR = Path(__file__).resolve().parents[3] / "data" / "derived" / "qaoa_meta"
DERIVED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True, frozen=True)
class MetaAssemblyConfig:
    """Configuration for the end-to-end Meta-Assembly QAOA pipeline."""
    # --- Classical Preprocessing Configuration ---
    # The number of assemblies to cluster neurons into (becomes n_qubits).
    # If None, an optimal k is found automatically.
    k_assemblies: int | None = 50
    sigma_thresh: float = 1.0
    max_terms: int | None = 100

    # --- QAOA Problem & Solver Configuration ---
    # The target size of the meta-assembly (cardinality k for k-hot constraint).
    k_target: int = 3
    p_layers: int = 2
    num_warm_starts: int = 5  # Number of candidates to take from warm-up run

    # Hamiltonian penalty strength. Auto-calculates if None.
    lam_penalty: float | None = None

    # --- Warm-start Run Parameters (X-Mixer) ---
    warm_iters: int = 20
    shots_warm: int = 1024

    # --- Fine-tuning Run Parameters (XY-Mixer) ---
    fine_iters: int = 25
    shots_fine: int = 4096


def run_meta_qaoa(
    session_data: SessionData,
    *,
    config: MetaAssemblyConfig = MetaAssemblyConfig(),
    service: QiskitRuntimeService | None = None,
) -> dict:
    """
    Executes the full Phase 1-3 meta-assembly QAOA pipeline.

    This function performs:
    1.  Classical pre-processing to get pruned, scaled triplet weights.
    2.  A "warm-start" QAOA run with an X-mixer and penalty term to find
        promising candidate solutions in the k-hot subspace.
    3.  A "fine-tuning" QAOA run for each candidate using a constraint-
        preserving XY-mixer to find the true minimum.

    Returns
    -------
    dict
        A dictionary containing the results, including:
        - "best_bitstring": The optimal k-hot bitstring found.
        - "best_energy": The corresponding energy (from H_C).
        - "active_assemblies": A list of integer indices of the winning assemblies.
        - "warm_starts": A list of the top candidate bitstrings from the warm-up.
        - "config": The configuration used for the run.
    """
    logger.info(f"Starting meta-assembly QAOA pipeline with config: {config}")

    # === Phase 1 & 2: Classical Preprocessing ===
    # Note: `k_assemblies` is passed for clustering, while `k_target` is used later
    # for the QAOA penalty. This correctly separates the two concepts.
    scaled_triplets = build_assembly_graph(
        session_data,
        k_assemblies=config.k_assemblies,
        sigma_thresh=config.sigma_thresh,
        max_terms=config.max_terms,
    )

    if not scaled_triplets:
        logger.error("No significant triplets found after pruning. Aborting pipeline.")
        return {"error": "No triplets found", "config": asdict(config)}
    
    # Determine number of qubits from the largest assembly index found.
    n_qubits = max(max(t) for t in scaled_triplets.keys()) + 1
    logger.info(f"Built problem graph with {len(scaled_triplets)} triplets on {n_qubits} qubits.")

    # === Hamiltonian Construction ===
    H_cost = build_cost_hamiltonian(scaled_triplets)
    
    # Set penalty strength. Per the spec, it should outweigh the max reward.
    # Since weights are scaled to [0, 1], max reward is 1.0. A value of 1.5 is a sensible default.
    lam = config.lam_penalty if config.lam_penalty is not None else 1.5
    H_penalty = build_penalty_hamiltonian(n_qubits, config.k_target, lam)
    
    H_problem_warm = H_cost + H_penalty

    # === Phase 3a: Warm-Start Run with X-Mixer + Penalty ===
    logger.info("--- Starting Phase 3a: Warm-Start Run ---")
    # The `get_ionq_aria_backend` helper should handle service/provider logic.
    backend = get_ionq_aria_backend(service=service)
    # Use keyword `backend=` for forward-compatibility with Qiskit.
    estimator = Estimator(backend=backend)
    
    mixer_x = build_x_mixer(n_qubits)
    optimizer_warm = SPSA(maxiter=config.warm_iters)

    # The `execute_qaoa` function is expected to return a result object
    # that contains the measurement counts.
    warm_start_results = execute_qaoa(
        problem_hamiltonian=H_problem_warm,
        mixer_hamiltonian=mixer_x,
        p=config.p_layers,
        optimizer=optimizer_warm,
        estimator=estimator,
        shots=config.shots_warm,
    )

    # Safely extract counts, guarding against empty or malformed results.
    counts = warm_start_results.get("counts", {})
    k_hot_counts = {bs: count for bs, count in counts.items() if bs.count('1') == config.k_target}
    
    if not k_hot_counts:
        logger.error("Warm-start run did not produce any valid k-hot states. Aborting.")
        return {"error": "No valid k-hot states from warm-start", "warm_start_counts": counts, "config": asdict(config)}

    sorted_candidates = sorted(k_hot_counts.items(), key=lambda item: item[1], reverse=True)
    warm_starts = [bs for bs, _ in sorted_candidates[: config.num_warm_starts]]
    logger.info(f"Identified {len(warm_starts)} warm-start candidates: {warm_starts}")

    # === Phase 3b: Fine-Tuning Run with XY-Mixer ===
    logger.info("--- Starting Phase 3b: Fine-Tuning Run ---")
    best_overall_energy = float("inf")
    best_overall_bitstring = ""

    mixer_xy = build_xy_mixer(n_qubits)
    optimizer_fine = SPSA(maxiter=config.fine_iters)

    for i, start_bs in enumerate(warm_starts):
        logger.info(f"Fine-tuning run {i+1}/{len(warm_starts)} from initial state: |{start_bs}>")
        
        # This run is expected to return a result object with the final energy and bitstring.
        fine_tune_results = execute_qaoa(
            problem_hamiltonian=H_cost,  # Use original cost Hamiltonian
            mixer_hamiltonian=mixer_xy,
            p=config.p_layers,
            optimizer=optimizer_fine,
            estimator=estimator,
            shots=config.shots_fine,
            initial_state=start_bs, # Critical: start from a valid candidate
        )
        
        energy = fine_tune_results.get("energy")
        final_bs = fine_tune_results.get("bitstring")
        logger.info(f"Result for |{start_bs}>: Energy = {energy:.5f}")

        if energy is not None and energy < best_overall_energy:
            best_overall_energy = energy
            best_overall_bitstring = final_bs

    if not best_overall_bitstring:
         logger.error("Fine-tuning failed to produce a valid result.")
         return {"error": "Fine-tuning failed", "warm_starts": warm_starts, "config": asdict(config)}

    logger.info(f"Pipeline complete. Best bitstring: |{best_overall_bitstring}> with energy {best_overall_energy:.5f}")

    # === Final Result Aggregation ===
    active_assemblies = [i for i, bit in enumerate(best_overall_bitstring) if bit == "1"]
    
    result_dict = {
        "best_bitstring": best_overall_bitstring,
        "best_energy": best_overall_energy,
        "active_assemblies": active_assemblies,
        "warm_starts": warm_starts,
        "config": asdict(config),
    }

    return result_dict