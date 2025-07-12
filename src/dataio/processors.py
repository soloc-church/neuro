"""
Advanced feature extraction and processing module for calcium imaging data.

This module provides sophisticated analysis tools for extracting meaningful
features from neural activity data, including response characterization,
network analysis, and temporal dynamics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import warnings
from scipy import signal, stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx
from pathlib import Path
import logging

from .data_structures import SessionData, NeuronMetadata, TrialInfo
from .config          import EXPERIMENT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class ResponseCharacteristics:
    """Container for neural response characteristics."""
    neuron_id: str
    baseline_activity: float
    peak_response: float
    response_latency: float
    response_duration: float
    response_magnitude: float
    signal_to_noise: float
    response_reliability: float
    preferred_frequency: Optional[float] = None
    frequency_selectivity: Optional[float] = None
    temporal_pattern: Optional[str] = None


@dataclass
class NetworkMetrics:
    """Container for network analysis metrics."""
    clustering_coefficient: float
    path_length: float
    small_worldness: float
    modularity: float
    degree_distribution: Dict[str, float]
    centrality_measures: Dict[str, Dict[str, float]]
    community_structure: Dict[str, int]


@dataclass
class TemporalDynamics:
    """Container for temporal dynamics analysis."""
    autocorrelation: np.ndarray
    cross_correlations: Dict[str, np.ndarray]
    spectral_power: Dict[str, float]
    coherence: Dict[str, float]
    phase_relationships: Dict[str, float]


class ResponseProcessor:
    """Processes and characterizes neural responses."""
    
    def __init__(self, frame_rate: float = 30.0, baseline_window: Tuple[int, int] = (0, 30)):
        self.frame_rate = frame_rate
        self.baseline_window = baseline_window
    
    def analyze_trial_responses(self, session: SessionData, 
                              frequency_filter: Optional[float] = None) -> Dict[str, ResponseCharacteristics]:
        """Analyze neural responses across trials."""
        if session.activity_matrix.size == 0 or not session.trials:
            raise ValueError("Session must have activity data and trials")
        
        # Filter trials by frequency if specified
        trials_to_analyze = session.trials
        if frequency_filter is not None:
            trials_to_analyze = [t for t in session.trials if abs(t.frequency - frequency_filter) < 50]
        
        if not trials_to_analyze:
            raise ValueError(f"No trials found for frequency {frequency_filter}")
        
        response_chars = {}
        
        for neuron in session.neurons:
            if not neuron.is_valid:
                continue
            
            try:
                neuron_activity = session.get_activity_for_neuron(neuron.global_id)
                if neuron_activity is None:
                    continue
                
                characteristics = self._analyze_neuron_responses(
                    neuron, neuron_activity, trials_to_analyze
                )
                response_chars[neuron.global_id] = characteristics
                
            except Exception as e:
                logger.warning(f"Failed to analyze responses for neuron {neuron.global_id}: {str(e)}")
                continue
        
        return response_chars
    
    def _analyze_neuron_responses(self, neuron: NeuronMetadata, 
                                activity: np.ndarray, 
                                trials: List[TrialInfo]) -> ResponseCharacteristics:
        """Analyze responses for a single neuron with robust NaN handling."""
        trial_responses = []
        baselines = []
        peaks = []
        latencies = []
        durations = []
        
        # Track problematic trials
        skipped_trials = 0
        short_trials = 0
        nan_baselines = 0
        
        for trial in trials:
            try:
                # Extract trial activity
                trial_activity = activity[trial.start_frame:trial.end_frame]
                trial_duration = len(trial_activity)
                
                # Skip trials that are too short for baseline calculation
                min_required_frames = self.baseline_window[1]  # Default 30
                if trial_duration < min_required_frames:
                    short_trials += 1
                    logger.debug(f"Trial {trial.trial_idx} too short ({trial_duration} < {min_required_frames} frames)")
                    continue
                
                # Calculate baseline with NaN handling
                baseline_start = self.baseline_window[0]
                baseline_end = min(self.baseline_window[1], trial_duration)
                baseline_data = trial_activity[baseline_start:baseline_end]
                
                # Check if baseline has any valid (non-NaN) data
                if np.all(np.isnan(baseline_data)):
                    nan_baselines += 1
                    logger.debug(f"Trial {trial.trial_idx} has all-NaN baseline")
                    continue
                
                baseline = np.nanmean(baseline_data)
                
                # If baseline is still NaN after nanmean, skip this trial
                if np.isnan(baseline):
                    nan_baselines += 1
                    continue
                    
                baselines.append(baseline)
                
                # Find peak response (only in post-baseline period)
                response_start = self.baseline_window[1]
                response_window = trial_activity[response_start:]
                
                if len(response_window) > 0 and not np.all(np.isnan(response_window)):
                    # Use nanargmax to handle NaN values
                    valid_mask = ~np.isnan(response_window)
                    if np.any(valid_mask):
                        # Get peak among valid values
                        valid_values = response_window[valid_mask]
                        peak_idx_in_valid = np.argmax(valid_values)
                        # Map back to original indices
                        valid_indices = np.where(valid_mask)[0]
                        peak_idx = valid_indices[peak_idx_in_valid]
                        peak_value = response_window[peak_idx]
                        peaks.append(peak_value)
                        
                        # Calculate latency (frames to peak from stimulus onset)
                        latency = (peak_idx + response_start) / self.frame_rate  # Convert to seconds
                        latencies.append(latency)
                        
                        # Calculate response duration (above threshold)
                        # Use robust std calculation that ignores NaN
                        baseline_std = np.nanstd(baseline_data)
                        if baseline_std > 0:  # Avoid zero std
                            threshold = baseline + 2 * baseline_std
                        else:
                            # If baseline has no variance, use a fixed threshold
                            threshold = baseline + 0.1  # 10% above baseline
                        
                        above_threshold = response_window > threshold
                        if np.any(above_threshold):
                            duration = np.sum(above_threshold) / self.frame_rate
                            durations.append(duration)
                        else:
                            durations.append(0.0)
                    else:
                        # No valid data in response window
                        peaks.append(baseline)
                        latencies.append(0.0)
                        durations.append(0.0)
                else:
                    peaks.append(baseline)
                    latencies.append(0.0)
                    durations.append(0.0)
                
                # Store the cleaned trial response (with NaNs intact for correlation)
                trial_responses.append(trial_activity)
                
            except Exception as e:
                logger.warning(f"Failed to analyze trial {trial.trial_idx}: {str(e)}")
                skipped_trials += 1
                continue
        
        # Log statistics if many trials were problematic
        total_trials = len(trials)
        if short_trials > 0 or nan_baselines > 0 or skipped_trials > 0:
            logger.info(f"Neuron {neuron.global_id} trial analysis: "
                    f"{len(trial_responses)}/{total_trials} valid, "
                    f"{short_trials} too short, {nan_baselines} NaN baselines, "
                    f"{skipped_trials} errors")
        
        if not trial_responses:
            raise ValueError(f"No valid trial responses found for neuron {neuron.global_id}")
        
        # Calculate summary statistics with NaN handling
        baseline_activity = np.nanmean(baselines) if baselines else 0.0
        peak_response = np.nanmean(peaks) if peaks else 0.0
        response_latency = np.nanmean(latencies) if latencies else 0.0
        response_duration = np.nanmean(durations) if durations else 0.0
        response_magnitude = peak_response - baseline_activity
        
        # Calculate signal-to-noise ratio with NaN handling
        trial_means = []
        trial_vars = []
        for resp in trial_responses:
            if not np.all(np.isnan(resp)):
                trial_means.append(np.nanmean(resp))
                trial_vars.append(np.nanvar(resp))
        
        if trial_means and trial_vars:
            signal_power = np.nanvar(trial_means)  # Variance across trial means
            noise_power = np.nanmean(trial_vars)   # Average within-trial variance
            snr = signal_power / noise_power if noise_power > 0 else 0.0
        else:
            snr = 0.0
        
        # Calculate response reliability (correlation across trials) with robust handling
        if len(trial_responses) > 1:
            # Find minimum length and align trials
            min_length = min(len(resp) for resp in trial_responses)
            aligned_responses = np.array([resp[:min_length] for resp in trial_responses])
            
            # Remove trials that are all NaN
            valid_trial_mask = ~np.all(np.isnan(aligned_responses), axis=1)
            valid_responses = aligned_responses[valid_trial_mask]
            
            if len(valid_responses) > 1:
                # Calculate correlation matrix with NaN handling
                # Use custom correlation that handles NaN properly
                n_valid_trials = len(valid_responses)
                corr_matrix = np.zeros((n_valid_trials, n_valid_trials))
                
                for i in range(n_valid_trials):
                    for j in range(i+1, n_valid_trials):
                        # Find timepoints where both trials have valid data
                        valid_mask = ~(np.isnan(valid_responses[i]) | np.isnan(valid_responses[j]))
                        
                        if np.sum(valid_mask) > 10:  # Need at least 10 valid points
                            corr = np.corrcoef(valid_responses[i][valid_mask], 
                                            valid_responses[j][valid_mask])[0, 1]
                            if not np.isnan(corr):
                                corr_matrix[i, j] = corr
                                corr_matrix[j, i] = corr
                
                # Set diagonal to 1
                np.fill_diagonal(corr_matrix, 1.0)
                
                # Calculate reliability as average off-diagonal correlation
                mask = ~np.eye(n_valid_trials, dtype=bool)
                off_diagonal_corrs = corr_matrix[mask]
                # Only use non-zero correlations (zero means we couldn't calculate it)
                valid_corrs = off_diagonal_corrs[off_diagonal_corrs != 0]
                
                if len(valid_corrs) > 0:
                    reliability = np.mean(valid_corrs)
                else:
                    reliability = 0.0
            else:
                reliability = 0.0
        else:
            reliability = 0.0
        
        return ResponseCharacteristics(
            neuron_id=neuron.global_id,
            baseline_activity=baseline_activity,
            peak_response=peak_response,
            response_latency=response_latency,
            response_duration=response_duration,
            response_magnitude=response_magnitude,
            signal_to_noise=snr,
            response_reliability=reliability,
            preferred_frequency=neuron.best_frequency
        )
    
    def compute_frequency_tuning(self, session: SessionData) -> Dict[str, Dict[str, float]]:
        """Compute frequency tuning curves for all neurons."""
        tuning_curves = {}
        
        # Get unique frequencies
        frequencies = sorted(set(trial.frequency for trial in session.trials))
        
        for neuron in session.neurons:
            if not neuron.is_valid:
                continue
            
            neuron_activity = session.get_activity_for_neuron(neuron.global_id)
            if neuron_activity is None:
                continue
            
            tuning_curve = {}
            
            for freq in frequencies:
                freq_trials = [t for t in session.trials if abs(t.frequency - freq) < 1]
                if not freq_trials:
                    continue
                
                responses = []
                for trial in freq_trials:
                    trial_activity = neuron_activity[trial.start_frame:trial.end_frame]
                    # Use response window (skip baseline)
                    response_window = trial_activity[self.baseline_window[1]:]
                    if len(response_window) > 0:
                        responses.append(np.nanmean(response_window))
                
                if responses:
                    tuning_curve[freq] = np.nanmean(responses)
            
            if tuning_curve:
                tuning_curves[neuron.global_id] = tuning_curve
        
        return tuning_curves


class NetworkAnalyzer:
    """Analyzes functional network properties."""
    
    def __init__(self, correlation_threshold: float = 0.3):
        self.correlation_threshold = correlation_threshold
    
    def build_functional_network(self, session: SessionData, 
                                use_signal_correlations: bool = True) -> nx.Graph:
        """Build functional network from correlation data."""
        if use_signal_correlations and session.signal_correlations is not None:
            corr_matrix = session.signal_correlations
        elif session.noise_correlations is not None:
            corr_matrix = session.noise_correlations
        else:
            # Compute correlations from activity matrix
            corr_matrix = np.corrcoef(session.activity_matrix)
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes (neurons)
        for i, neuron in enumerate(session.neurons):
            G.add_node(neuron.global_id, **neuron.to_dict())
        
        # Add edges based on correlation threshold
        n_neurons = len(session.neurons)
        for i in range(n_neurons):
            for j in range(i + 1, n_neurons):
                if abs(corr_matrix[i, j]) >= self.correlation_threshold:
                    G.add_edge(
                        session.neurons[i].global_id,
                        session.neurons[j].global_id,
                        weight=abs(corr_matrix[i, j]),
                        correlation=corr_matrix[i, j]
                    )
        
        return G
    
    def analyze_network_properties(self, network: nx.Graph) -> NetworkMetrics:
        """Analyze network properties and compute metrics."""
        if len(network.nodes()) == 0:
            raise ValueError("Network has no nodes")
        
        # Basic network metrics
        clustering_coeff = nx.average_clustering(network)
        
        # Path length (only for connected components)
        if nx.is_connected(network):
            avg_path_length = nx.average_shortest_path_length(network)
        else:
            # Compute for largest connected component
            largest_cc = max(nx.connected_components(network), key=len)
            subgraph = network.subgraph(largest_cc)
            avg_path_length = nx.average_shortest_path_length(subgraph)
        
        # Small-worldness (Watts-Strogatz)
        # Generate random network with same degree sequence
        degree_sequence = [d for n, d in network.degree()]
        try:
            random_network = nx.configuration_model(degree_sequence)
            random_network = nx.Graph(random_network)  # Remove multi-edges
            random_clustering = nx.average_clustering(random_network)
            random_path_length = nx.average_shortest_path_length(random_network)
            
            small_worldness = (clustering_coeff / random_clustering) / (avg_path_length / random_path_length)
        except:
            small_worldness = np.nan
        
        # Modularity
        try:
            communities = nx.algorithms.community.greedy_modularity_communities(network)
            community_dict = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_dict[node] = i
            modularity = nx.algorithms.community.modularity(network, communities)
        except:
            community_dict = {}
            modularity = np.nan
        
        # Degree distribution
        degrees = [d for n, d in network.degree()]
        degree_dist = {
            'mean': np.mean(degrees),
            'std': np.std(degrees),
            'max': max(degrees) if degrees else 0,
            'min': min(degrees) if degrees else 0
        }
        
        # Centrality measures
        centrality_measures = {}
        try:
            centrality_measures['betweenness'] = nx.betweenness_centrality(network)
            centrality_measures['closeness'] = nx.closeness_centrality(network)
            centrality_measures['eigenvector'] = nx.eigenvector_centrality(network, max_iter=1000)
        except:
            logger.warning("Failed to compute some centrality measures")
        
        return NetworkMetrics(
            clustering_coefficient=clustering_coeff,
            path_length=avg_path_length,
            small_worldness=small_worldness,
            modularity=modularity,
            degree_distribution=degree_dist,
            centrality_measures=centrality_measures,
            community_structure=community_dict
        )
    
    def identify_network_hubs(self, network: nx.Graph, method: str = 'degree') -> List[str]:
        """Identify network hubs using various centrality measures."""
        if method == 'degree':
            centrality = dict(network.degree())
        elif method == 'betweenness':
            centrality = nx.betweenness_centrality(network)
        elif method == 'eigenvector':
            centrality = nx.eigenvector_centrality(network, max_iter=1000)
        else:
            raise ValueError(f"Unknown centrality method: {method}")
        
        # Identify top 10% as hubs
        threshold = np.percentile(list(centrality.values()), 90)
        hubs = [node for node, cent in centrality.items() if cent >= threshold]
        
        return hubs


class TemporalAnalyzer:
    """Analyzes temporal dynamics of neural activity."""
    
    def __init__(self, frame_rate: float = 30.0):
        self.frame_rate = frame_rate
    
    def analyze_temporal_dynamics(self, session: SessionData, 
                                max_lag: int = 100) -> Dict[str, TemporalDynamics]:
        """Analyze temporal dynamics for all neurons."""
        dynamics = {}
        
        for neuron in session.neurons:
            if not neuron.is_valid:
                continue
            
            neuron_activity = session.get_activity_for_neuron(neuron.global_id)
            if neuron_activity is None:
                continue
            
            try:
                # Autocorrelation
                autocorr = self._compute_autocorrelation(neuron_activity, max_lag)
                
                # Cross-correlations with other neurons
                cross_corrs = {}
                for other_neuron in session.neurons[:10]:  # Limit to first 10 for efficiency
                    if other_neuron.global_id != neuron.global_id and other_neuron.is_valid:
                        other_activity = session.get_activity_for_neuron(other_neuron.global_id)
                        if other_activity is not None:
                            cross_corr = self._compute_cross_correlation(
                                neuron_activity, other_activity, max_lag
                            )
                            cross_corrs[other_neuron.global_id] = cross_corr
                
                # Spectral analysis
                spectral_power = self._compute_spectral_power(neuron_activity)
                
                dynamics[neuron.global_id] = TemporalDynamics(
                    autocorrelation=autocorr,
                    cross_correlations=cross_corrs,
                    spectral_power=spectral_power,
                    coherence={},  # Placeholder
                    phase_relationships={}  # Placeholder
                )
                
            except Exception as e:
                logger.warning(f"Failed temporal analysis for neuron {neuron.global_id}: {str(e)}")
                continue
        
        return dynamics
    
    def _compute_autocorrelation(self, signal_data: np.ndarray, max_lag: int) -> np.ndarray:
        """Compute autocorrelation function."""
        # Remove NaNs
        valid_data = signal_data[~np.isnan(signal_data)]
        if len(valid_data) == 0:
            return np.zeros(2 * max_lag + 1)
        
        # Normalize
        normalized = (valid_data - np.mean(valid_data)) / np.std(valid_data)
        
        # Compute autocorrelation
        autocorr = np.correlate(normalized, normalized, mode='full')
        
        # Extract around zero lag
        center = len(autocorr) // 2
        start = max(0, center - max_lag)
        end = min(len(autocorr), center + max_lag + 1)
        
        return autocorr[start:end] / autocorr[center]  # Normalize by zero-lag
    
    def _compute_cross_correlation(self, signal1: np.ndarray, signal2: np.ndarray, 
                                 max_lag: int) -> np.ndarray:
        """Compute cross-correlation between two signals."""
        # Remove NaNs and align
        valid_mask = ~(np.isnan(signal1) | np.isnan(signal2))
        if np.sum(valid_mask) == 0:
            return np.zeros(2 * max_lag + 1)
        
        s1 = signal1[valid_mask]
        s2 = signal2[valid_mask]
        
        # Normalize
        s1_norm = (s1 - np.mean(s1)) / np.std(s1)
        s2_norm = (s2 - np.mean(s2)) / np.std(s2)
        
        # Compute cross-correlation
        cross_corr = np.correlate(s1_norm, s2_norm, mode='full')
        
        # Extract around zero lag
        center = len(cross_corr) // 2
        start = max(0, center - max_lag)
        end = min(len(cross_corr), center + max_lag + 1)
        
        return cross_corr[start:end] / len(s1_norm)
    
    def _compute_spectral_power(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Compute spectral power in different frequency bands."""
        # Remove NaNs
        valid_data = signal_data[~np.isnan(signal_data)]
        if len(valid_data) < 4: # Not enough data for Welch's method
            return {}

        # Compute power spectral density
        freqs, psd = signal.welch(valid_data, fs=self.frame_rate, nperseg=min(256, len(valid_data)))

        # Define frequency bands (for calcium imaging)
        bands = {
            'low': (0.01, 0.1),      # Very low frequency
            'medium': (0.1, 0.5),    # Medium frequency
            'high': (0.5, 2.0),      # High frequency (limited by sampling)
        }

        spectral_power = {}
        for band_name, (low_freq, high_freq) in bands.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(band_mask):
                # Use np.trapezoid instead of the deprecated np.trapz
                spectral_power[band_name] = np.trapezoid(psd[band_mask], freqs[band_mask])
            else:
                spectral_power[band_name] = 0.0

        return spectral_power


class DimensionalityReducer:
    """Performs dimensionality reduction and clustering analysis."""

    def __init__(self, method: str = 'pca'):
        self.method = method
        self.reducer = None
        self.scaler = StandardScaler()

    def fit_transform(self, session: SessionData, n_components: int = 10) -> np.ndarray:
        """Fit dimensionality reduction and transform data."""
        if session.activity_matrix.size == 0:
            raise ValueError("Session must have activity data")

        # Robust orientation and subsetting
        data = session.activity_matrix.copy()
        n_neurons = len(session.neurons)

        # Case A: rows may correspond to neurons (and might need subsetting)
        if data.shape[0] >= n_neurons and data.shape[1] != n_neurons:
            data = data[:n_neurons, :]  # Drop extra rows if any

        # Case B: columns may correspond to neurons (and might need subsetting)
        elif data.shape[1] >= n_neurons and data.shape[0] != n_neurons:
            data = data.T[:n_neurons, :]  # Transpose, then drop extras

        # Case C: A perfect row-wise match
        elif data.shape[0] == n_neurons:
            pass  # Data is already correctly oriented

        # If neither axis can represent the neuron count, bail out
        else:
            raise ValueError(
                f"Activity matrix shape {data.shape} is incompatible with "
                f"{n_neurons} neurons."
            )

        # Handle NaNs by replacing with row means (mean activity for each neuron)
        if np.isnan(data).any():
            row_means = np.nanmean(data, axis=1, keepdims=True)
            # Fill NaNs with 0 if an entire row is NaN
            row_means = np.nan_to_num(row_means)
            nan_mask = np.isnan(data)
            data[nan_mask] = np.broadcast_to(row_means, data.shape)[nan_mask]

        # Scale data (features are timepoints, so scale across them)
        data_scaled = self.scaler.fit_transform(data)

        # Apply dimensionality reduction
        if self.method == 'pca':
            self.reducer = PCA(n_components=n_components)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        transformed_data = self.reducer.fit_transform(data_scaled)

        return transformed_data

    def get_explained_variance(self) -> np.ndarray:
        """Get explained variance ratios for PCA."""
        if self.reducer is None or not hasattr(self.reducer, 'explained_variance_ratio_'):
            raise ValueError("Must fit reducer first")

        return self.reducer.explained_variance_ratio_

    def get_components(self) -> np.ndarray:
        """Get principal components for PCA."""
        if self.reducer is None or not hasattr(self.reducer, 'components_'):
            raise ValueError("Must fit reducer first")

        return self.reducer.components_

    def cluster_neurons(self, transformed_data: np.ndarray,
                       n_clusters: int = 5) -> Dict[str, int]:
        """Cluster neurons in reduced dimensional space."""
        # Note: n_clusters should be less than n_samples (n_neurons)
        if n_clusters > transformed_data.shape[0]:
            n_clusters = transformed_data.shape[0]
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(transformed_data)

        return {f"cluster_{i}": count for i, count in enumerate(np.bincount(cluster_labels))}


# Convenience functions
def analyze_responses(session: SessionData, **kwargs) -> Dict[str, ResponseCharacteristics]:
    """Convenience function for response analysis."""
    processor = ResponseProcessor(**kwargs)
    return processor.analyze_trial_responses(session)


def build_network(session: SessionData, **kwargs) -> Tuple[nx.Graph, NetworkMetrics]:
    """Convenience function for network analysis."""
    analyzer = NetworkAnalyzer(**kwargs)
    network = analyzer.build_functional_network(session)
    metrics = analyzer.analyze_network_properties(network)
    return network, metrics


def analyze_temporal_dynamics(session: SessionData, **kwargs) -> Dict[str, TemporalDynamics]:
    """Convenience function for temporal analysis."""
    analyzer = TemporalAnalyzer(**kwargs)
    return analyzer.analyze_temporal_dynamics(session)


def reduce_dimensions(session: SessionData, **kwargs) -> Tuple[np.ndarray, DimensionalityReducer]:
    """Convenience function for dimensionality reduction."""
    reducer = DimensionalityReducer(**kwargs)
    transformed_data = reducer.fit_transform(session)
    return transformed_data, reducer