"""
Phase 4 utilities – functional plausibility tests on held-out data.
Public API mirrors the notebook steps so researchers can script.
"""
import numpy as np
from sklearn.metrics import mutual_info_score
from typing import List, Dict, Optional, Tuple
from scipy import stats
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import from parent package
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from dataio.data_structures import SessionData

logger = logging.getLogger(__name__)

def hamming_cluster(bitstrings: List[str], radius: int = 2) -> List[List[str]]:
    """Group near-degenerate solutions by Hamming distance."""
    if not bitstrings:
        return []
    
    # Calculate pairwise Hamming distances
    n = len(bitstrings)
    clusters = []
    assigned = [False] * n
    
    for i in range(n):
        if assigned[i]:
            continue
        
        # Start a new cluster
        cluster = [bitstrings[i]]
        assigned[i] = True
        
        # Find all bitstrings within radius
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            
            # Calculate Hamming distance
            dist = sum(c1 != c2 for c1, c2 in zip(bitstrings[i], bitstrings[j]))
            
            if dist <= radius:
                cluster.append(bitstrings[j])
                assigned[j] = True
        
        clusters.append(cluster)
    
    logger.info(f"Found {len(clusters)} degenerate solution families from {len(bitstrings)} bitstrings")
    
    return clusters


def correlate_with_events(activity_matrix: np.ndarray,
                          event_times: np.ndarray, *,
                          fs: float, window: float = 0.5) -> float:
    """
    Returns peak correlation between meta-assembly mean activity and event raster.
    
    Parameters
    ----------
    activity_matrix : ndarray
        (n_neurons, n_timepoints) activity matrix
    event_times : ndarray
        Times of events in seconds
    fs : float
        Sampling frequency in Hz
    window : float
        Window size in seconds for correlation
        
    Returns
    -------
    float
        Peak correlation value
    """
    # Convert event times to frames
    event_frames = (event_times * fs).astype(int)
    
    # Create event raster
    n_frames = activity_matrix.shape[1]
    event_raster = np.zeros(n_frames)
    
    window_frames = int(window * fs)
    
    for event_frame in event_frames:
        if 0 <= event_frame < n_frames:
            # Add a window around each event
            start = max(0, event_frame - window_frames // 2)
            end = min(n_frames, event_frame + window_frames // 2)
            event_raster[start:end] = 1
    
    # Calculate mean activity
    mean_activity = np.mean(activity_matrix, axis=0)
    
    # Calculate correlation
    if np.std(event_raster) > 0 and np.std(mean_activity) > 0:
        correlation = np.corrcoef(mean_activity, event_raster)[0, 1]
    else:
        correlation = 0.0
    
    logger.info(f"Peak correlation with events: {correlation:.3f}")
    
    return correlation


def compare_to_random(meta_score: float,
                      random_scores: List[float]) -> float:
    """
    One-sided p-value.
    
    Parameters
    ----------
    meta_score : float
        Score (e.g., correlation) for the QAOA-found meta-assembly
    random_scores : list
        Scores from random control assemblies
        
    Returns
    -------
    float
        One-sided p-value
    """
    if not random_scores:
        logger.warning("No random scores provided for comparison")
        return 1.0
    
    # One-sided test: how many random scores are >= meta_score
    n_greater = sum(score >= meta_score for score in random_scores)
    p_value = (n_greater + 1) / (len(random_scores) + 1)  # +1 for continuity correction
    
    logger.info(f"Meta-assembly score: {meta_score:.3f}")
    logger.info(f"Random scores: mean={np.mean(random_scores):.3f}, "
                f"std={np.std(random_scores):.3f}")
    logger.info(f"One-sided p-value: {p_value:.4f}")
    
    return p_value


def validate_meta_assembly(session_data: SessionData,
                           winning_assemblies: List[int],
                           trial_condition: Optional[str] = None,
                           n_random_controls: int = 100) -> Dict:
    """
    Comprehensive validation of QAOA-found meta-assembly.
    
    Parameters
    ----------
    session_data : SessionData
        Complete session data
    winning_assemblies : list
        Indices of assemblies in the winning meta-assembly
    trial_condition : str, optional
        Specific trial condition to analyze
    n_random_controls : int
        Number of random control sets to generate
        
    Returns
    -------
    dict
        Validation results including p-values and visualizations
    """
    logger.info(f"Validating meta-assembly with assemblies: {winning_assemblies}")
    
    # Load membership matrix
    derived_dir = Path(__file__).resolve().parents[2] / "data" / "derived" / "qaoa_meta"
    membership_file = derived_dir / f"assemblies_{session_data.session_id}.npz"
    
    if not membership_file.exists():
        raise FileNotFoundError(f"Membership matrix not found at {membership_file}")
    
    data = np.load(membership_file)
    membership_matrix = data['membership_matrix']
    
    # Get neurons in winning meta-assembly
    meta_neurons = []
    for assembly_idx in winning_assemblies:
        neurons_in_assembly = np.where(membership_matrix[assembly_idx, :])[0]
        meta_neurons.extend(neurons_in_assembly)
    
    meta_neurons = list(set(meta_neurons))  # Remove duplicates
    logger.info(f"Meta-assembly contains {len(meta_neurons)} unique neurons")
    
    # Extract activity for meta-assembly
    meta_activity = session_data.activity_matrix[meta_neurons, :]
    
    # Analyze trial responses
    results = {
        'winning_assemblies': winning_assemblies,
        'n_neurons': len(meta_neurons),
        'neuron_indices': meta_neurons
    }
    
    # Get trials for specific condition if specified
    if trial_condition:
        trials = [t for t in session_data.trials if t.condition == trial_condition]
    else:
        trials = session_data.trials
    
    if not trials:
        logger.warning("No trials found for validation")
        return results
    
    # Calculate mean response during trials
    trial_responses = []
    for trial in trials:
        if trial.end_frame <= meta_activity.shape[1]:
            trial_activity = meta_activity[:, trial.start_frame:trial.end_frame]
            mean_response = np.mean(trial_activity)
            trial_responses.append(mean_response)
    
    results['mean_trial_response'] = np.mean(trial_responses)
    results['std_trial_response'] = np.std(trial_responses)
    
    # Compare to random controls
    random_scores = []
    n_assemblies = membership_matrix.shape[0]
    k = len(winning_assemblies)
    
    for _ in range(n_random_controls):
        # Random k assemblies
        random_assemblies = np.random.choice(n_assemblies, k, replace=False)
        
        # Get neurons in random meta-assembly
        random_neurons = []
        for assembly_idx in random_assemblies:
            neurons_in_assembly = np.where(membership_matrix[assembly_idx, :])[0]
            random_neurons.extend(neurons_in_assembly)
        
        random_neurons = list(set(random_neurons))
        
        # Calculate score for random set
        if random_neurons:
            random_activity = session_data.activity_matrix[random_neurons, :]
            random_trial_responses = []
            
            for trial in trials:
                if trial.end_frame <= random_activity.shape[1]:
                    trial_activity = random_activity[:, trial.start_frame:trial.end_frame]
                    mean_response = np.mean(trial_activity)
                    random_trial_responses.append(mean_response)
            
            if random_trial_responses:
                random_scores.append(np.mean(random_trial_responses))
    
    # Statistical comparison
    if random_scores:
        p_value = compare_to_random(results['mean_trial_response'], random_scores)
        results['p_value'] = p_value
        results['random_mean'] = np.mean(random_scores)
        results['random_std'] = np.std(random_scores)
        results['z_score'] = (results['mean_trial_response'] - results['random_mean']) / (results['random_std'] + 1e-10)
    
    # Create validation plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Meta-assembly activity heatmap
    ax = axes[0, 0]
    if meta_activity.shape[0] > 0:
        im = ax.imshow(meta_activity, aspect='auto', cmap='viridis')
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Neurons')
        ax.set_title('Meta-Assembly Activity')
        plt.colorbar(im, ax=ax)
    
    # Plot 2: Trial-averaged response
    ax = axes[0, 1]
    if trial_responses:
        ax.hist(trial_responses, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(np.mean(trial_responses), color='red', linestyle='--', 
                   label=f'Mean = {np.mean(trial_responses):.3f}')
        ax.set_xlabel('Mean Trial Response')
        ax.set_ylabel('Count')
        ax.set_title('Trial Response Distribution')
        ax.legend()
    
    # Plot 3: Comparison to random
    ax = axes[1, 0]
    if random_scores:
        ax.hist(random_scores, bins=30, alpha=0.7, color='gray', 
                label='Random', density=True)
        ax.axvline(results['mean_trial_response'], color='red', linestyle='--', 
                   linewidth=2, label='Meta-Assembly')
        ax.set_xlabel('Mean Response')
        ax.set_ylabel('Density')
        ax.set_title(f'Statistical Validation (p = {p_value:.4f})')
        ax.legend()
    
    # Plot 4: Assembly overlap visualization
    ax = axes[1, 1]
    overlap_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                neurons_i = set(np.where(membership_matrix[winning_assemblies[i], :])[0])
                neurons_j = set(np.where(membership_matrix[winning_assemblies[j], :])[0])
                overlap = len(neurons_i & neurons_j) / len(neurons_i | neurons_j)
                overlap_matrix[i, j] = overlap
            else:
                overlap_matrix[i, j] = 1.0
    
    im = ax.imshow(overlap_matrix, cmap='Blues', vmin=0, vmax=1)
    ax.set_xlabel('Assembly')
    ax.set_ylabel('Assembly')
    ax.set_title('Assembly Overlap (Jaccard)')
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(winning_assemblies)
    ax.set_yticklabels(winning_assemblies)
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = derived_dir / f"validation_{session_data.session_id}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Validation plot saved to {plot_file}")
    results['validation_plot'] = str(plot_file)
    
    return results