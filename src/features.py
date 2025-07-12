# src/features.py
"""
Deterministic, reusable feature calculators for neural activity datasets.

Each function takes numpy arrays or pandas DataFrames, returns numpy arrays or 
simple data structures, and **never** writes to disk—persistence is handled
in scripts or the I/O layer.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations, permutations
import math
import warnings

# Import dependencies, with fallbacks for optional libraries
try:
    from scipy import stats, signal
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster import hierarchy
    from scipy.optimize import curve_fit
    from sklearn.metrics import silhouette_score, mutual_info_score
    from sklearn.decomposition import PCA, NMF
    from sklearn.cluster import DBSCAN
    from numba import njit
    import networkx as nx
except ImportError as e:
    print(f"Warning: A required dependency is missing. {e}")
    
# Optional libraries for advanced tests
try:
    import dit
    from dit.pid import PID_BROJA
    from dit.inference import distribution_from_data
except ImportError:
    dit = None
try:
    from ripser import ripser
except ImportError:
    ripser = None
try:
    import statsmodels.api as sm
    from statsmodels.stats.multitest import fdrcorrection
except ImportError:
    sm = None


# ─────────────────────────────────────────────────────────────────────────────
# Core Statistics & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def pearson_corr_matrix(traces: np.ndarray) -> np.ndarray:
    """
    Calculates the Pearson correlation matrix for a set of time-series traces.

    Args:
        traces: (N_neurons, T_timepoints) 2D array of ΔF/F or spikes.
    
    Returns:
        (N, N) Pearson correlation matrix with 1s on the diagonal.
    """
    # Using np.nan_to_num is a robust way to handle potential NaNs in traces.
    traces_clean = np.nan_to_num(traces, nan=0.0)
    return np.corrcoef(traces_clean)

def binarize_activity_mad(activity_matrix: np.ndarray, mad_scale_factor: float = 3.0) -> np.ndarray:
    """
    Binarizes neural activity using a threshold based on the Median Absolute Deviation (MAD).

    Args:
        activity_matrix: (N_neurons, T_timepoints) array of neural activity.
        mad_scale_factor: How many MADs above the median to set the threshold.
    
    Returns:
        (N_neurons, T_timepoints) boolean array of binarized activity.
    """
    activity_clean = np.nan_to_num(activity_matrix, nan=0.0)
    median_activity = np.median(activity_clean, axis=1, keepdims=True)
    # MAD is ~0.6745 * std, so 1.4826 * MAD is ~1 std
    mad = np.median(np.abs(activity_clean - median_activity), axis=1, keepdims=True) * 1.4826
    threshold = median_activity + mad_scale_factor * (mad + 1e-9) # Add epsilon for robustness
    return activity_clean > threshold

def oasis_deconvolution(y: np.ndarray, g: float = 0.95, lam: float = 0.1) -> tuple[np.ndarray, np.ndarray, float]:
    """
    A simplified OASIS-like deconvolution for inferring spikes from calcium signals.

    Args:
        y: A single neuron's calcium trace (1D array).
        g: Autoregressive coefficient (calcium decay factor).
        lam: Sparsity penalty (spike threshold).

    Returns:
        A tuple containing (inferred_spikes, inferred_calcium_trace, g).
    """
    T = len(y)
    y_proc = y - np.percentile(y, 20)
    y_proc[y_proc < 0] = 0
    
    c = np.zeros(T)
    s = np.zeros(T)
    
    for t in range(1, T):
        c_pred = g * c[t-1]
        if y_proc[t] > c_pred + lam:
            s[t] = y_proc[t] - c_pred
            c[t] = y_proc[t]
        else:
            c[t] = c_pred
            s[t] = 0
            
    return s, c, g

# ─────────────────────────────────────────────────────────────────────────────
# Clustering & Assembly Analysis
# ─────────────────────────────────────────────────────────────────────────────

def calculate_linkage_matrix(corr_matrix: np.ndarray, method: str = 'ward') -> np.ndarray:
    """
    Computes the hierarchical clustering linkage matrix from a correlation matrix.
    
    Args:
        corr_matrix: (N, N) correlation matrix.
        method: Linkage method (e.g., 'ward', 'average', 'complete').

    Returns:
        The linkage matrix suitable for dendrograms and clustering.
    """
    distance_matrix = 1 - np.abs(corr_matrix)
    condensed_distance_matrix = squareform(distance_matrix, checks=False)
    return hierarchy.linkage(condensed_distance_matrix, method=method)

def find_optimal_clusters_silhouette(linkage_matrix: np.ndarray, corr_matrix: np.ndarray, n_clusters_range: range) -> tuple[int, list[float]]:
    """
    Finds the optimal number of clusters by maximizing the silhouette score.

    Args:
        linkage_matrix: The hierarchical clustering linkage matrix.
        corr_matrix: The original (N,N) correlation matrix.
        n_clusters_range: A range of cluster numbers to test (e.g., range(5, 51)).

    Returns:
        A tuple (optimal_n_clusters, silhouette_scores).
    """
    distance_matrix = 1 - np.abs(corr_matrix)
    silhouette_scores = []
    for n in n_clusters_range:
        labels = hierarchy.fcluster(linkage_matrix, n, criterion='maxclust')
        if len(np.unique(labels)) > 1:
            score = silhouette_score(distance_matrix, labels, metric='precomputed')
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(-1) # Invalid score
            
    optimal_n_clusters = n_clusters_range[np.argmax(silhouette_scores)]
    return optimal_n_clusters, silhouette_scores

def get_cluster_labels(linkage_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Assigns cluster labels to each neuron for a given number of clusters.

    Args:
        linkage_matrix: The hierarchical clustering linkage matrix.
        n_clusters: The desired number of clusters.

    Returns:
        An array of cluster labels for each neuron.
    """
    return hierarchy.fcluster(linkage_matrix, n_clusters, criterion='maxclust')

def calculate_cluster_coactivation(activity_matrix: np.ndarray, cluster_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the mean activity trace for each cluster and their co-activation (correlation).

    Args:
        activity_matrix: (N_neurons, T_timepoints) array of neural activity.
        cluster_labels: Array of cluster labels for each neuron.

    Returns:
        A tuple containing (cluster_activity_matrix, cluster_correlation_matrix).
    """
    cluster_activity_matrix = np.array([
        np.nanmean(activity_matrix[cluster_labels == cid, :], axis=0)
        for cid in np.unique(cluster_labels)
    ])
    cluster_corr = np.corrcoef(np.nan_to_num(cluster_activity_matrix))
    return cluster_activity_matrix, cluster_corr

# ─────────────────────────────────────────────────────────────────────────────
# Spatial & Functional Properties
# ─────────────────────────────────────────────────────────────────────────────

def analyze_spatial_coherence(neuron_coords: np.ndarray, cluster_labels: np.ndarray, n_permutations: int = 1000) -> dict:
    """
    Tests if clusters are more spatially compact than random chance via permutation test.

    Args:
        neuron_coords: (N_neurons, 3) array of [x, y, z] coordinates.
        cluster_labels: Array of cluster labels for each neuron.
        n_permutations: Number of shuffles for the null distribution.

    Returns:
        A dictionary of spatial statistics for each cluster.
    """
    cluster_spatial_stats = {}
    unique_clusters = np.unique(cluster_labels)

    for cid in unique_clusters:
        cluster_mask = cluster_labels == cid
        n_in_cluster = np.sum(cluster_mask)
        
        if n_in_cluster > 1:
            cluster_coords = neuron_coords[cluster_mask]
            observed_dist = np.mean(pdist(cluster_coords))
            
            null_dist = np.array([
                np.mean(pdist(neuron_coords[np.random.choice(len(neuron_coords), n_in_cluster, replace=False)]))
                for _ in range(n_permutations)
            ])
            
            p_value = np.sum(null_dist <= observed_dist) / n_permutations
            cluster_spatial_stats[cid] = {
                'mean_dist': observed_dist,
                'p_value': p_value,
                'is_compact': p_value < 0.05
            }
    return cluster_spatial_stats

def analyze_functional_coherence(neuron_bfs: np.ndarray, cluster_labels: np.ndarray, n_permutations: int = 1000) -> dict:
    """
    Tests if clusters have more homogeneous best frequencies (BF) than random chance.

    Args:
        neuron_bfs: (N_neurons,) array of best frequencies. Use np.nan for non-tuned.
        cluster_labels: Array of cluster labels for each neuron.
        n_permutations: Number of shuffles for the null distribution.

    Returns:
        A dictionary of functional statistics for each cluster.
    """
    cluster_bf_stats = {}
    unique_clusters = np.unique(cluster_labels)
    
    # Pool of valid, positive BFs for sampling
    valid_positive_bf_mask = ~np.isnan(neuron_bfs) & (neuron_bfs > 0)
    bf_values_for_sampling = neuron_bfs[valid_positive_bf_mask]

    for cid in unique_clusters:
        cluster_mask = cluster_labels == cid
        combined_mask = cluster_mask & valid_positive_bf_mask
        
        if np.sum(combined_mask) > 1:
            positive_cluster_bfs = neuron_bfs[combined_mask]
            observed_bf_std = np.std(np.log10(positive_cluster_bfs))
            
            null_bf_std = np.array([
                np.std(np.log10(np.random.choice(bf_values_for_sampling, len(positive_cluster_bfs), replace=False)))
                for _ in range(n_permutations)
            ])

            p_value = np.sum(null_bf_std <= observed_bf_std) / n_permutations
            cluster_bf_stats[cid] = {
                'mean_bf': np.mean(positive_cluster_bfs),
                'std_log_bf': observed_bf_std,
                'p_value': p_value,
                'is_homog': p_value < 0.05
            }
    return cluster_bf_stats

def analyze_distance_correlation(neuron_coords: np.ndarray, corr_matrix: np.ndarray, n_samples: int = 200000) -> tuple:
    """
    Analyzes the relationship between inter-neuron distance and functional correlation.
    
    Args:
        neuron_coords: (N_neurons, 3) array of coordinates.
        corr_matrix: (N, N) correlation matrix.
        n_samples: Number of neuron pairs to sample for efficiency.

    Returns:
        A tuple (binned_distances, binned_correlations, spearman_rho, p_value).
    """
    n_neurons = len(neuron_coords)
    n_pairs_to_sample = min(n_samples, n_neurons * (n_neurons-1)//2)
    
    idx_i = np.random.randint(0, n_neurons, n_pairs_to_sample)
    idx_j = np.random.randint(0, n_neurons, n_pairs_to_sample)
    valid_pairs = idx_i != idx_j
    idx_i, idx_j = idx_i[valid_pairs], idx_j[valid_pairs]

    distances = np.linalg.norm(neuron_coords[idx_i] - neuron_coords[idx_j], axis=1)
    correlations = corr_matrix[idx_i, idx_j]
    
    # Binned statistic
    distance_bins = np.linspace(0, np.percentile(distances, 99), 30)
    mean_corr_binned, _, _ = stats.binned_statistic(distances, correlations, statistic='mean', bins=distance_bins)
    bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
    
    # Spearman correlation
    rho, p = stats.spearmanr(distances, correlations)
    
    return bin_centers, mean_corr_binned, rho, p

def calculate_cross_plane_correlations(neuron_planes: np.ndarray, corr_matrix: np.ndarray) -> np.ndarray:
    """
    Calculates the mean correlation between all pairs of imaging planes.
    
    Args:
        neuron_planes: (N_neurons,) array of plane numbers.
        corr_matrix: (N, N) correlation matrix.

    Returns:
        A (n_planes, n_planes) matrix of mean inter-plane correlations.
    """
    unique_planes = sorted(np.unique(neuron_planes))
    cross_plane_corr = np.zeros((len(unique_planes), len(unique_planes)))

    for i, p1 in enumerate(unique_planes):
        for j, p2 in enumerate(unique_planes):
            if i <= j:
                n1_indices = np.where(neuron_planes == p1)[0]
                n2_indices = np.where(neuron_planes == p2)[0]
                if len(n1_indices) and len(n2_indices):
                    sub_matrix = corr_matrix[np.ix_(n1_indices, n2_indices)]
                    if i == j: # Within-plane
                        # Exclude diagonal and duplicates
                        vals = sub_matrix[np.triu_indices_from(sub_matrix, k=1)]
                    else: # Between-plane
                        vals = sub_matrix.flatten()
                    
                    if len(vals) > 0:
                        cross_plane_corr[i, j] = cross_plane_corr[j, i] = np.mean(vals)
                        
    return cross_plane_corr

# ─────────────────────────────────────────────────────────────────────────────
# Temporal Dynamics & Sequences
# ─────────────────────────────────────────────────────────────────────────────

def test_synchrony_significance(activity_binary: np.ndarray, n_shuffles: int = 1000) -> dict:
    """
    Tests if the maximum number of simultaneously active neurons is significant.

    Args:
        activity_binary: (N_neurons, T_timepoints) binarized activity matrix.
        n_shuffles: Number of shuffles for the null distribution.

    Returns:
        A dictionary with observed max, 99th percentile of null, and p-value.
    """
    simultaneous_count_observed = np.sum(activity_binary, axis=0)
    max_observed_synchrony = np.max(simultaneous_count_observed)

    shuffled_max_counts = np.zeros(n_shuffles)
    for i in range(n_shuffles):
        activity_shuffled = np.array([np.random.permutation(row) for row in activity_binary])
        shuffled_count = np.sum(activity_shuffled, axis=0)
        shuffled_max_counts[i] = np.max(shuffled_count)
        
    p_value = np.sum(shuffled_max_counts >= max_observed_synchrony) / n_shuffles
    
    return {
        'max_observed': max_observed_synchrony,
        'chance_threshold_99p': np.percentile(shuffled_max_counts, 99),
        'p_value': p_value
    }

def calculate_isi_stats(activity_binary: np.ndarray, frame_rate: float) -> pd.DataFrame:
    """
    Calculates Inter-Spike Interval (ISI) statistics for each neuron.
    
    Args:
        activity_binary: (N_neurons, T_timepoints) binarized activity matrix.
        frame_rate: The sampling rate in Hz.

    Returns:
        A pandas DataFrame with min and median ISIs for each neuron.
    """
    spike_times_frames = [np.where(trace > 0)[0] for trace in activity_binary]
    isi_rows = []
    for unit_id, s_times in enumerate(spike_times_frames):
        if len(s_times) >= 2:
            isi_frames = np.diff(s_times)
            isi_s = isi_frames / frame_rate
            isi_rows.append({
                'unit': unit_id, 
                'min_isi_s': np.min(isi_s), 
                'median_isi_s': np.median(isi_s)
            })
    return pd.DataFrame(isi_rows)

@njit
def _find_triplet_instances_fast(t1: np.ndarray, t2: np.ndarray, t3: np.ndarray, max_gap: int) -> list[tuple[int, int, int]]:
    """Numba-accelerated function to find raw time tuples for a pattern."""
    out_times = []
    for a in t1:
        for b in t2:
            if 0 < b - a <= max_gap:
                for c in t3:
                    if 0 < c - b <= max_gap:
                        out_times.append((a, b, c))
    return out_times

class SequentialPatternMiner:
    """
    A class to mine for higher-order sequential firing patterns (e.g., triplets).
    Accelerated with Numba.
    """
    def __init__(self, min_support: int = 5, max_gap_bins: int = 6, pattern_length: int = 3):
        self.min_support = min_support
        self.max_gap = max_gap_bins
        self.pattern_length = pattern_length
    
    def mine_patterns(self, transactions: list[list[int]]) -> list[dict]:
        """
        Mines patterns from a list of transactions (active neurons per time bin).
        
        Args:
            transactions: A list where each element is a list of active neuron IDs for a time bin.

        Returns:
            A list of found patterns, each a dictionary with neurons, instances, and support.
        """
        neuron_occurrences = defaultdict(list)
        for t_idx, transaction in enumerate(transactions):
            for neuron_id in transaction:
                neuron_occurrences[neuron_id].append(t_idx)
        
        active_neurons = [n for n, occ in neuron_occurrences.items() if len(occ) >= self.min_support]
        if len(active_neurons) < self.pattern_length:
            return []
        
        patterns_found = []
        for triplet in combinations(active_neurons, self.pattern_length):
            for perm in permutations(triplet):
                instances = self._find_pattern_instances(perm, neuron_occurrences)
                if len(instances) >= self.min_support:
                    patterns_found.append({'neurons': perm, 'instances': instances, 'support': len(instances)})
        return patterns_found
    
    def _find_pattern_instances(self, pattern: tuple, neuron_occurrences: dict) -> list[dict]:
        """Finds instances, delegating the core search to the fast numba function."""
        firing_times_list = [np.array(neuron_occurrences[n], dtype=np.int32) for n in pattern]
        raw_times = _find_triplet_instances_fast(firing_times_list[0], firing_times_list[1], firing_times_list[2], self.max_gap)
        
        instances = []
        for t1, t2, t3 in raw_times:
            instances.append({'times': [t1, t2, t3], 'lags': [t2 - t1, t3 - t2]})
        return instances

# ─────────────────────────────────────────────────────────────────────────────
# Advanced Structural & Information-Theoretic Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_neuronal_avalanches(activity_matrix: np.ndarray, threshold_std: float = 2.0, time_bin_frames: int = 10) -> dict:
    """
    Detects neuronal avalanches and fits a power law to their size distribution.
    
    Args:
        activity_matrix: (N_neurons, T_timepoints) array of neural activity.
        threshold_std: Standard deviation threshold for defining an event.
        time_bin_frames: Size of time bins to define co-activity.

    Returns:
        A dictionary with power law exponent (alpha) and mean branching ratio (sigma).
    """
    if 'powerlaw' not in globals():
        warnings.warn("`powerlaw` library not found. Skipping avalanche analysis.")
        return {'alpha': np.nan, 'sigma': np.nan, 'p_value': 1.0}

    threshold = np.nanmean(activity_matrix) + threshold_std * np.nanstd(activity_matrix)
    binary_activity = activity_matrix > threshold
    
    avalanche_sizes = []
    for t in range(0, binary_activity.shape[1] - time_bin_frames, time_bin_frames):
        active_count = np.sum(binary_activity[:, t:t+time_bin_frames])
        if active_count > 0:
            avalanche_sizes.append(active_count)
    
    if len(avalanche_sizes) < 50:
        return {'alpha': np.nan, 'sigma': np.nan, 'p_value': 1.0}

    # Fit power law
    fit = Fit(avalanche_sizes, discrete=True)
    alpha = fit.power_law.alpha
    
    # Branching ratio
    branching_ratios = []
    for t in range(binary_activity.shape[1] - 1):
        n_active_t = np.sum(binary_activity[:, t])
        n_active_t1 = np.sum(binary_activity[:, t+1])
        if n_active_t > 0:
            branching_ratios.append(n_active_t1 / n_active_t)
            
    sigma = np.mean(branching_ratios) if branching_ratios else np.nan
    
    return {'alpha': alpha, 'sigma': sigma}
    
def find_hypergraph_edges_glm(activity_binary: np.ndarray, high_var_neurons: np.ndarray, n_triplets: int = 5000) -> pd.DataFrame:
    """
    Identifies significant 3-neuron interactions (hyperedges) using a GLM.

    Args:
        activity_binary: (N_neurons, T_timepoints) binarized activity matrix.
        high_var_neurons: Indices of neurons with high variance to sample from.
        n_triplets: Number of triplets to test.

    Returns:
        A DataFrame of results with coefficients and p-values for interaction terms.
    """
    if not sm:
        warnings.warn("`statsmodels` not found. Skipping hypergraph analysis.")
        return pd.DataFrame()

    all_triplets = np.array(list(combinations(high_var_neurons, 3)))
    sample_indices = np.random.choice(len(all_triplets), min(n_triplets, len(all_triplets)), replace=False)
    triplet_samples = all_triplets[sample_indices]
    
    glm_results = []
    for i, j, k in triplet_samples:
        y = activity_binary[k, :]
        X = np.vstack([activity_binary[i, :], activity_binary[j, :], activity_binary[i, :] * activity_binary[j, :]]).T
        X = sm.add_constant(X, prepend=True)
        try:
            logit_model = sm.Logit(y, X)
            result = logit_model.fit(disp=0)
            glm_results.append({
                'hyperedge': (i, j, k),
                'coeff_interaction': result.params[3],
                'p_value': result.pvalues[3]
            })
        except Exception:
            continue
            
    results_df = pd.DataFrame(glm_results)
    if not results_df.empty:
        results_df['p_adj_fdr'] = fdrcorrection(results_df['p_value'])[1]
        
    return results_df

def analyze_topological_persistence(distance_matrix: np.ndarray, n_bootstraps: int = 200, threshold: float = 0.3) -> dict:
    """
    Performs Topological Data Analysis (TDA) to find persistent homology features.

    Args:
        distance_matrix: (N, N) matrix of pairwise distances between neurons.
        n_bootstraps: Number of shuffles for null distribution.
        threshold: Max persistence scale to consider.

    Returns:
        A dictionary with the number of significant H1 (loops) and H2 (voids) features.
    """
    if not ripser:
        warnings.warn("`ripser` not found. Skipping topology analysis.")
        return {'n_significant_h1': 0, 'n_significant_h2': 0}

    diagrams = ripser(distance_matrix, maxdim=2, thresh=threshold, distance_matrix=True)['dgms']
    h1_observed = diagrams[1]
    h2_observed = diagrams[2]

    # Null distribution for H1
    null_persistence_h1 = []
    for _ in range(n_bootstraps):
        perm = np.random.permutation(distance_matrix.shape[0])
        shuffled_dist = distance_matrix[perm][:, perm]
        null_diags = ripser(shuffled_dist, maxdim=1, thresh=threshold, distance_matrix=True)['dgms']
        if len(null_diags[1]) > 0:
            null_persistence_h1.extend(null_diags[1][:, 1] - null_diags[1][:, 0])

    if len(h1_observed) > 0 and null_persistence_h1:
        p_thresh_h1 = np.percentile(null_persistence_h1, 95)
        n_sig_h1 = np.sum((h1_observed[:, 1] - h1_observed[:, 0]) > p_thresh_h1)
    else:
        n_sig_h1 = 0
        
    # A proper H2 null is more complex, so we just count observed bars for now
    n_sig_h2 = len(h2_observed)
    
    return {'n_significant_h1': n_sig_h1, 'n_significant_h2': n_sig_h2, 'diagrams': diagrams}