from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import warnings
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Custom exception for data validation errors. Inherits from ValueError for broader compatibility."""
    pass


class DataValidator:
    """Utility class for validating data structures."""
    
    @staticmethod
    def validate_coordinates(x: float, y: float, z: float, 
                           min_val: float = -5000.0, max_val: float = 5000.0) -> None:
        """Validate 3D coordinates are within reasonable bounds."""
        coords = {'x': x, 'y': y, 'z': z}
        for name, val in coords.items():
            if not (min_val <= val <= max_val):
                raise ValidationError(f"Coordinate {name}={val} outside valid range [{min_val}, {max_val}]")
    
    @staticmethod
    def validate_quality_score(score: Optional[float]) -> None:
        """Validate quality score is within [0, 1] range."""
        if score is not None and not (0.0 <= score <= 1.0):
            raise ValidationError(f"Quality score {score} must be between 0 and 1")
    
    @staticmethod
    def validate_frequency(freq: float, allow_zero: bool = False) -> None:
        """Validate frequency is positive (or allow zero for silence trials)."""
        if allow_zero and freq == 0.0:
            return  # Allow zero for silence trials
        if freq <= 0:
            raise ValidationError(f"Frequency {freq} must be strictly positive")


@dataclass
class NeuronMetadata:
    """Rich metadata for each neuron with validation."""
    global_id: str
    session: str
    plane: int
    local_idx: int
    x: float
    y: float
    z: float
    best_frequency: Optional[float] = None
    quality_score: Optional[float] = None
    roi_size: Optional[float] = None
    snr: Optional[float] = None
    is_valid: bool = True
    
    def __post_init__(self):
        """Validate neuron metadata after initialization."""
        try:
            DataValidator.validate_coordinates(self.x, self.y, self.z)
            DataValidator.validate_quality_score(self.quality_score)
            if self.best_frequency is not None:
                DataValidator.validate_frequency(self.best_frequency, allow_zero=True)
            if not (1 <= self.plane <= 7):
                raise ValidationError(f"Plane {self.plane} must be between 1 and 7")
            if self.local_idx < 0:
                raise ValidationError(f"Local index {self.local_idx} must be non-negative")
        except ValidationError as e:
            # Re-raise with more context
            raise ValidationError(f"Invalid NeuronMetadata for {self.global_id}: {e}")

    @property
    def coords_3d(self) -> np.ndarray:
        """Get 3D coordinates as numpy array."""
        return np.array([self.x, self.y, self.z])
    
    @property
    def has_best_frequency(self) -> bool:
        """Check if neuron has a valid best frequency."""
        return self.best_frequency is not None and self.best_frequency >= 0
    
    @property
    def has_quality_metrics(self) -> bool:
        """Check if neuron has quality metrics."""
        return self.quality_score is not None or self.snr is not None
    
    def distance_to(self, other: 'NeuronMetadata') -> float:
        """Calculate Euclidean distance to another neuron."""
        return np.linalg.norm(self.coords_3d - other.coords_3d)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'global_id': self.global_id,
            'session': self.session,
            'plane': self.plane,
            'local_idx': self.local_idx,
            'x': self.x, 'y': self.y, 'z': self.z,
            'best_frequency': self.best_frequency,
            'quality_score': self.quality_score,
            'roi_size': self.roi_size,
            'snr': self.snr,
            'is_valid': self.is_valid
        }

@dataclass
class TrialInfo:
    """Structured trial information with validation."""
    trial_idx: int
    frequency: float
    level: float
    start_frame: int
    end_frame: int
    condition: str

    def __post_init__(self):
        """Validate trial information after initialization."""
        try:
            if self.trial_idx < 0:
                raise ValidationError(f"Trial index {self.trial_idx} must be non-negative")
            
            # Allow frequency of 0 for silence trials but validate positive otherwise
            DataValidator.validate_frequency(self.frequency, allow_zero=True)

            if self.start_frame < 0:
                raise ValidationError(f"Start frame {self.start_frame} must be non-negative")
            
            # End frame must be strictly after start frame
            if self.end_frame <= self.start_frame:
                raise ValidationError(f"End frame {self.end_frame} must be > start frame {self.start_frame}")
        except ValidationError as e:
            # Re-raise with more context
            raise ValidationError(f"Invalid TrialInfo for trial_idx={self.trial_idx}: {e}")

    @property
    def duration_frames(self) -> int:
        """Get trial duration in frames."""
        return self.end_frame - self.start_frame

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'trial_idx': self.trial_idx,
            'frequency': self.frequency,
            'level': self.level,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'condition': self.condition,
            'duration_frames': self.duration_frames
        }

@dataclass
class SessionData:
    """Complete session data with all metadata and enhanced functionality."""

    # Core identifiers
    session_id: str
    
    # Core data components
    neurons: List[NeuronMetadata]
    trials: List[TrialInfo]
    activity_matrix: np.ndarray # Shape: (n_neurons, n_timepoints)
    
    # Experimental parameters
    frame_rate: float
    experiment_vars: Dict[str, Any] = field(default_factory=dict)
    
    # Optional components
    spike_matrix: Optional[np.ndarray] = None
    signal_correlations: Optional[np.ndarray] = None
    noise_correlations: Optional[np.ndarray] = None
    correlation_metadata: Dict[str, Any] = field(default_factory=dict)
    preprocessing_params: Dict[str, Any] = field(default_factory=dict)

    # Internal cache
    _neuron_id_map: Dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        """Validate session data and build internal caches."""
        logger.debug(f"Initializing SessionData for {self.session_id}")
        # Build a map for fast neuron lookups
        self._neuron_id_map = {neuron.global_id: i for i, neuron in enumerate(self.neurons)}
        
        # Validate that the map is unique
        if len(self._neuron_id_map) != len(self.neurons):
            warnings.warn("Duplicate neuron IDs detected during SessionData creation. This should have been caught earlier.")

        # Validate basic structure
        if not self.neurons:
            warnings.warn(f"Session '{self.session_id}' has no neurons.")
        
        if not self.trials:
            warnings.warn(f"Session '{self.session_id}' has no trials.")
        
        # Validate activity matrix dimensions
        if self.activity_matrix.size > 0:
            expected_neurons = len(self.neurons)
            actual_neurons, actual_timepoints = self.activity_matrix.shape
            
            if expected_neurons != actual_neurons:
                # This is a critical error that makes the data unusable
                raise ValidationError(
                    f"Activity matrix neuron count mismatch for session '{self.session_id}': "
                    f"Matrix has {actual_neurons} neurons, but metadata has {expected_neurons}."
                )
            logger.debug(f"Activity matrix loaded with shape: ({actual_neurons}, {actual_timepoints})")
        else:
            if self.neurons: # If there are neurons, the matrix should not be empty
                warnings.warn(f"Session '{self.session_id}' has {len(self.neurons)} neurons but an empty activity matrix.")

    def get_activity_for_neuron(self, neuron_id: str) -> Optional[np.ndarray]:
        """Get activity time series for a specific neuron using a fast lookup."""
        neuron_idx = self._neuron_id_map.get(neuron_id)
        if neuron_idx is not None and self.activity_matrix.size > 0:
            return self.activity_matrix[neuron_idx, :]
        logger.warning(f"Could not find activity for neuron_id '{neuron_id}'")
        return None

    def get_trial_responses(self, trial_indices: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """Extract neural responses for specific trials."""
        if trial_indices is None:
            # Use all trial indices if none are provided
            trial_indices = [t.trial_idx for t in self.trials]
        
        responses = {}
        for trial_idx in trial_indices:
            if trial_idx >= len(self.trials) or trial_idx < 0:
                logger.warning(f"Requested trial_idx {trial_idx} is out of bounds. Skipping.")
                continue
                
            trial = self.trials[trial_idx]
            # Ensure trial frames are within the bounds of the activity matrix
            if self.activity_matrix.shape[1] >= trial.end_frame:
                # Slicing is exclusive of the end point, which is correct.
                trial_activity = self.activity_matrix[:, trial.start_frame:trial.end_frame]
                responses[f"trial_{trial_idx}"] = trial_activity
            else:
                 logger.warning(f"Trial {trial.trial_idx} (frames {trial.start_frame}-{trial.end_frame}) "
                                f"extends beyond activity matrix timepoints ({self.activity_matrix.shape[1]}). Skipping.")
        
        return responses

    def get_neuron_by_id(self, neuron_id: str) -> Optional[NeuronMetadata]:
        """Get neuron metadata by ID using the fast lookup map."""
        neuron_idx = self._neuron_id_map.get(neuron_id)
        if neuron_idx is not None:
            return self.neurons[neuron_idx]
        return None
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Export session data to a tidy pandas DataFrame.
        Each row represents one neuron's activity at one time point in one trial.
        """
        if not self.trials or self.activity_matrix.size == 0:
            warnings.warn("Cannot create DataFrame without trials and activity data.")
            return pd.DataFrame()

        records = []
        for trial in self.trials:
            trial_activity = self.get_trial_responses([trial.trial_idx]).get(f"trial_{trial.trial_idx}")
            if trial_activity is None:
                continue

            for neuron_idx, neuron in enumerate(self.neurons):
                activity_trace = trial_activity[neuron_idx, :]
                for frame_in_trial, activity in enumerate(activity_trace):
                    records.append({
                        "session_id": self.session_id,
                        "neuron_id": neuron.global_id,
                        "trial_idx": trial.trial_idx,
                        "condition": trial.condition,
                        "frequency": trial.frequency,
                        "level": trial.level,
                        "frame_global": trial.start_frame + frame_in_trial,
                        "frame_in_trial": frame_in_trial,
                        "activity": activity,
                        "neuron_plane": neuron.plane,
                        "neuron_x": neuron.x,
                        "neuron_y": neuron.y,
                        "neuron_z": neuron.z,
                    })
        return pd.DataFrame(records)

    def compute_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics for the session."""
        stats = {
            'session_id': self.session_id,
            'n_neurons': len(self.neurons),
            'n_trials': len(self.trials),
            'n_planes': len(set(n.plane for n in self.neurons)),
            'frame_rate': self.frame_rate,
            'duration_frames': self.activity_matrix.shape[1] if self.activity_matrix.size > 0 else 0,
        }
        
        # Quality metrics
        stats['n_valid_neurons'] = sum(1 for n in self.neurons if n.is_valid)
        
        # Best frequency coverage
        bf_neurons = [n for n in self.neurons if n.has_best_frequency]
        if bf_neurons:
            frequencies = [n.best_frequency for n in bf_neurons if n.best_frequency is not None]
            stats['bf_coverage'] = {
                'n_neurons_with_bf': len(bf_neurons),
                'min_frequency': min(frequencies) if frequencies else None,
                'max_frequency': max(frequencies) if frequencies else None,
                'n_unique_frequencies': len(set(frequencies))
            }
        
        # Trial statistics
        if self.trials:
            trial_durations = [t.duration_frames for t in self.trials]
            stats['trial_stats'] = {
                'mean_duration_frames': np.mean(trial_durations) if trial_durations else None,
                'std_duration_frames': np.std(trial_durations) if trial_durations else None,
                'cv': np.std(trial_durations) / np.mean(trial_durations) if trial_durations and np.mean(trial_durations) > 0 else 0,
                'unique_durations': list(np.unique(trial_durations)),
                'n_conditions': len(set(t.condition for t in self.trials))
            }
        
        return stats
    
    def save_summary(self, filepath: Union[str, Path]) -> None:
        """Save session summary to a human-readable JSON file."""
        import json
        
        stats = self.compute_summary_stats()
        filepath = Path(filepath)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        
        # Recursively apply the conversion
        def recursive_convert(data):
            if isinstance(data, dict):
                return {k: recursive_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_numpy_types(data)
        
        stats_converted = recursive_convert(stats)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(stats_converted, f, indent=4)
        logger.info(f"Session summary saved to {filepath}")