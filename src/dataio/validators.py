"""
Data validation and quality assurance module for calcium imaging data.

This module provides comprehensive validation tools for ensuring data integrity
and consistency across the pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings
import logging
from pathlib import Path

from .data_structures import SessionData, NeuronMetadata, TrialInfo
from .config          import VALIDATION_CONFIG, EXPERIMENT_CONFIG, DATA_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def add_warning(self, message: str) -> None:
        """Add a warning message and log it."""
        self.warnings.append(message)
        logger.warning(f"VALIDATION WARNING: {message}")
    
    def add_error(self, message: str) -> None:
        """Add an error message, mark as invalid, and log it."""
        self.errors.append(message)
        self.is_valid = False
        logger.error(f"VALIDATION ERROR: {message}")
    
    def add_metric(self, name: str, value: Any) -> None:
        """Add a validation metric."""
        self.metrics[name] = value


class DataIntegrityValidator:
    """Validates data integrity and consistency."""
    
    def __init__(self, config: Optional[object] = None):
        self.config = config or VALIDATION_CONFIG
    
    def validate_session(self, session: SessionData) -> ValidationResult:
        """Comprehensive validation of session data."""
        logger.info(f"--- Starting Data Integrity Validation for Session: {session.session_id} ---")
        result = ValidationResult(
            is_valid=True,
            warnings=[],
            errors=[],
            metrics={}
        )
        
        # Basic structure validation
        self._validate_basic_structure(session, result)
        
        # Neuron validation
        self._validate_neurons(session, result)
        
        # Trial validation
        self._validate_trials(session, result)
        
        # Activity matrix validation
        self._validate_activity_matrix(session, result)
        
        # Cross-validation between components
        self._validate_cross_consistency(session, result)
        
        logger.info(f"--- Validation Complete for {session.session_id}: "
                    f"Valid={result.is_valid}, Errors={len(result.errors)}, Warnings={len(result.warnings)} ---")
        return result
    
    def _validate_basic_structure(self, session: SessionData, result: ValidationResult) -> None:
        """Validate basic session structure."""
        logger.debug("Validating basic structure...")
        # Check session ID
        if not session.session_id or not isinstance(session.session_id, str):
            result.add_error("Invalid or missing session ID")
        
        # Check frame rate
        if session.frame_rate <= 0:
            result.add_error(f"Invalid frame rate: {session.frame_rate}")
        elif abs(session.frame_rate - EXPERIMENT_CONFIG.volume_scan_rate_hz) > self.config.frame_rate_tolerance:
            result.add_warning(f"Frame rate {session.frame_rate} differs significantly from expected "
                             f"{EXPERIMENT_CONFIG.volume_scan_rate_hz}")
        
        result.add_metric("frame_rate", session.frame_rate)
        result.add_metric("session_id", session.session_id)
    
    def _validate_neurons(self, session: SessionData, result: ValidationResult) -> None:
        """Validate neuron metadata."""
        logger.debug("Validating neuron metadata...")
        n_neurons = len(session.neurons)
        
        # Check neuron count
        if n_neurons == 0:
            result.add_error("No neurons found in session")
            return
        elif n_neurons < self.config.min_neurons_per_session:
            result.add_warning(f"Low neuron count: {n_neurons} (threshold: < {self.config.min_neurons_per_session})")
        elif n_neurons > self.config.max_neurons_per_session:
            result.add_warning(f"High neuron count: {n_neurons} (threshold: > {self.config.max_neurons_per_session})")
        
        # Validate individual neurons
        invalid_neurons = 0
        duplicate_ids = set()
        seen_ids = set()
        coordinate_issues = 0
        bf_neurons = 0
        quality_neurons = 0
        
        for neuron in session.neurons:
            # Check for duplicate IDs
            if neuron.global_id in seen_ids:
                duplicate_ids.add(neuron.global_id)
            seen_ids.add(neuron.global_id)
            
            # Validate coordinates
            coords = [neuron.x, neuron.y, neuron.z]
            if any(not (self.config.min_coordinate_value <= coord <= self.config.max_coordinate_value) 
                   for coord in coords):
                coordinate_issues += 1
            
            # Check if neuron is marked as invalid
            if not neuron.is_valid:
                invalid_neurons += 1
            
            # Count neurons with best frequency
            if neuron.has_best_frequency:
                bf_neurons += 1
            
            # Count quality neurons
            if neuron.has_quality_metrics:
                quality_neurons += 1
        
        # Report issues
        if duplicate_ids:
            result.add_error(f"Duplicate neuron IDs found: {list(duplicate_ids)[:5]}")
        
        if coordinate_issues > 0:
            result.add_warning(f"{coordinate_issues} neurons have coordinates outside valid range")
        
        if invalid_neurons > 0:
            result.add_warning(f"{invalid_neurons} neurons marked as invalid during loading")
        
        # Add metrics
        result.add_metric("n_neurons", n_neurons)
        result.add_metric("n_invalid_neurons", invalid_neurons)
        result.add_metric("n_bf_neurons", bf_neurons)
        result.add_metric("n_quality_neurons", quality_neurons)
        result.add_metric("bf_coverage", bf_neurons / n_neurons if n_neurons > 0 else 0)
    
    def _validate_trials(self, session: SessionData, result: ValidationResult) -> None:
        """Validate trial information including duration checks."""
        logger.debug("Validating trial information...")
        n_trials = len(session.trials)
        
        # Check trial count
        if n_trials == 0:
            result.add_warning("No trials found in session")
            return
        elif n_trials not in EXPERIMENT_CONFIG.expected_trial_counts:
            result.add_warning(f"Unexpected trial count: {n_trials}. Expected one of {EXPERIMENT_CONFIG.expected_trial_counts}")
        
        # Validate individual trials
        invalid_trials = 0
        overlapping_trials = 0
        duration_issues = 0
        short_trials = 0  # NEW: Track trials too short for baseline analysis
        
        # Get baseline window size from ResponseProcessor default
        min_baseline_frames = 30  # This should match ResponseProcessor.baseline_window[1]
        
        # Sort trials by start frame to check for overlaps correctly
        sorted_trials = sorted(session.trials, key=lambda t: t.start_frame)

        for i, trial in enumerate(sorted_trials):
            # Check frame range validity
            if trial.start_frame < 0 or trial.end_frame <= trial.start_frame:
                invalid_trials += 1
            
            # Check for overlapping trials with the *next* trial in the sorted list
            if i < len(sorted_trials) - 1:
                next_trial = sorted_trials[i+1]
                if trial.end_frame > next_trial.start_frame:
                    overlapping_trials += 1
                    # Log first occurrence for better debugging
                    if overlapping_trials == 1:
                        result.add_warning(f"Overlap detected: Trial ending at {trial.end_frame} overlaps with next trial starting at {next_trial.start_frame}")

            # Check trial duration
            duration = trial.duration_frames
            if duration <= 0:
                duration_issues += 1
            
            # NEW: Check if trial is too short for baseline analysis
            if 0 < duration < min_baseline_frames:
                short_trials += 1
                if short_trials == 1:  # Log details for first occurrence
                    result.add_warning(f"Trial {trial.trial_idx} has only {duration} frames, "
                                    f"less than minimum {min_baseline_frames} needed for baseline analysis")
        
        # Report issues
        if invalid_trials > 0:
            result.add_error(f"{invalid_trials} trials have invalid frame ranges (e.g., end <= start)")
        
        if overlapping_trials > 0:
            result.add_warning(f"{overlapping_trials} pairs of trials have overlapping frame ranges")
        
        if duration_issues > 0:
            result.add_error(f"{duration_issues} trials have non-positive durations")
        
        # NEW: Report short trials
        if short_trials > 0:
            result.add_warning(f"{short_trials} trials are shorter than {min_baseline_frames} frames "
                            f"and may cause issues in response analysis")
        
        # Add metrics
        result.add_metric("n_trials", n_trials)
        result.add_metric("n_invalid_trials", invalid_trials)
        result.add_metric("n_short_trials", short_trials)  # NEW metric
        
        if n_trials > 0:
            durations = [t.duration_frames for t in session.trials]
            result.add_metric("mean_trial_duration_frames", np.mean(durations))
            result.add_metric("std_trial_duration_frames", np.std(durations))
            result.add_metric("min_trial_duration_frames", np.min(durations))  # NEW metric
            result.add_metric("max_trial_duration_frames", np.max(durations))  # NEW metric
            result.add_metric("unique_trial_durations", list(np.unique(durations)))

    def _validate_activity_matrix(self, session: SessionData, result: ValidationResult) -> None:
        """Validate activity matrix with enhanced NaN and data quality checks."""
        logger.debug("Validating activity matrix...")
        if not hasattr(session, 'activity_matrix') or session.activity_matrix.size == 0:
            result.add_error("Activity matrix is missing or empty")
            return
        
        n_neurons_activity, n_timepoints = session.activity_matrix.shape
        n_neurons_meta = len(session.neurons)
        
        # Check dimension consistency
        if n_neurons_activity != n_neurons_meta:
            result.add_error(f"Shape mismatch: Activity matrix has {n_neurons_activity} rows (neurons), but metadata has {n_neurons_meta} neurons.")
        
        # Check for data quality issues
        nan_count = np.isnan(session.activity_matrix).sum()
        inf_count = np.isinf(session.activity_matrix).sum()
        
        # NEW: Check for extreme values that might cause issues
        finite_mask = np.isfinite(session.activity_matrix)
        if np.any(finite_mask):
            finite_values = session.activity_matrix[finite_mask]
            extreme_negative_count = np.sum(finite_values < -1000)
            extreme_positive_count = np.sum(finite_values > 1000)
        else:
            extreme_negative_count = 0
            extreme_positive_count = 0
        
        # Check for all-NaN rows (neurons with no data)
        all_nan_neurons = np.sum(np.all(np.isnan(session.activity_matrix), axis=1))
        
        # NEW: Check for high-NaN neurons (>50% NaN)
        nan_per_neuron = np.isnan(session.activity_matrix).sum(axis=1)
        high_nan_neurons = np.sum(nan_per_neuron > 0.5 * n_timepoints)

        # Check for zero variance neurons (ignoring NaNs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            zero_variance_neurons = np.sum(np.nanvar(session.activity_matrix, axis=1) == 0)

        # NEW: Check for all-NaN frames (timepoints with no data)
        all_nan_frames = np.sum(np.all(np.isnan(session.activity_matrix), axis=0))
        
        # Report issues
        if nan_count > 0:
            nan_fraction = nan_count / session.activity_matrix.size
            result.add_warning(f"Activity matrix contains {nan_count} NaN values ({nan_fraction:.2%}).")
            
            # NEW: More severe warning if NaN fraction is very high
            if nan_fraction > 0.1:
                result.add_warning(f"HIGH NaN fraction ({nan_fraction:.2%}) may cause analysis issues")
        
        if inf_count > 0:
            result.add_error(f"Activity matrix contains {inf_count} infinite values, which will break most analyses.")
        
        # NEW: Warn about extreme values
        if extreme_negative_count > 0:
            result.add_warning(f"{extreme_negative_count} values < -1000 detected (possible unhandled sentinel values)")
        
        if extreme_positive_count > 0:
            result.add_warning(f"{extreme_positive_count} values > 1000 detected (possible artifacts)")
        
        if all_nan_neurons > 0:
            result.add_warning(f"{all_nan_neurons} neurons have no activity data (all NaNs).")
        
        # NEW: Warn about high-NaN neurons
        if high_nan_neurons > all_nan_neurons:
            result.add_warning(f"{high_nan_neurons - all_nan_neurons} neurons have >50% NaN values")

        if zero_variance_neurons > all_nan_neurons:
            result.add_warning(f"{zero_variance_neurons - all_nan_neurons} neurons have constant (zero-variance) activity.")
        
        # NEW: Warn about all-NaN frames
        if all_nan_frames > 0:
            result.add_warning(f"{all_nan_frames} timepoints have no data across all neurons")
        
        # Add metrics
        result.add_metric("activity_matrix_shape", session.activity_matrix.shape)
        result.add_metric("n_nan_values", int(nan_count))
        result.add_metric("n_inf_values", int(inf_count))
        result.add_metric("n_extreme_negative", int(extreme_negative_count))  # NEW
        result.add_metric("n_extreme_positive", int(extreme_positive_count))  # NEW
        result.add_metric("n_all_nan_neurons", int(all_nan_neurons))
        result.add_metric("n_high_nan_neurons", int(high_nan_neurons))  # NEW
        result.add_metric("n_zero_variance_neurons", int(zero_variance_neurons))
        result.add_metric("n_all_nan_frames", int(all_nan_frames))  # NEW
        
        # Calculate data range only on finite values
        if np.any(finite_mask):
            result.add_metric("activity_data_range", {
                "min": float(np.min(finite_values)),
                "max": float(np.max(finite_values)),
                "mean": float(np.mean(finite_values)),
                "std": float(np.std(finite_values))
            })
        else:
            result.add_metric("activity_data_range", {
                "min": "NaN",
                "max": "NaN", 
                "mean": "NaN",
                "std": "NaN"
            })
    
    def _validate_cross_consistency(self, session: SessionData, result: ValidationResult) -> None:
        """Validate consistency between different components."""
        logger.debug("Validating cross-consistency...")
        # Check that all trial frames are within activity matrix bounds
        if hasattr(session, 'activity_matrix') and session.activity_matrix.size > 0 and session.trials:
            max_frame = session.activity_matrix.shape[1]
            invalid_frame_trials = []
            
            for trial in session.trials:
                # The end frame is exclusive in Python slicing, so it can be equal to the length
                if trial.end_frame > max_frame:
                    invalid_frame_trials.append(trial.trial_idx)
            
            if invalid_frame_trials:
                result.add_error(f"Trials extend beyond activity matrix timepoints ({max_frame}). "
                               f"Offending trial indices: {invalid_frame_trials[:5]}")
        
        # Check correlation matrix dimensions if present
        n_neurons = len(session.neurons)
        if session.signal_correlations is not None:
            actual_shape = session.signal_correlations.shape
            if actual_shape != (n_neurons, n_neurons):
                result.add_warning(f"Signal correlation matrix shape {actual_shape} doesn't match neuron count ({n_neurons})")
        
        if session.noise_correlations is not None:
            actual_shape = session.noise_correlations.shape
            if actual_shape != (n_neurons, n_neurons):
                result.add_warning(f"Noise correlation matrix shape {actual_shape} doesn't match neuron count ({n_neurons})")


class QualityAnalyzer:
    """Analyzes data quality and provides quality metrics."""
    
    def analyze_session_quality(self, session: SessionData) -> Dict[str, Any]:
        """Comprehensive quality analysis of session data."""
        quality_report = {
            "overall_score": 0.0,
            "neuron_quality": self._analyze_neuron_quality(session),
            "trial_quality": self._analyze_trial_quality(session),
            "activity_quality": self._analyze_activity_quality(session),
            "temporal_quality": self._analyze_temporal_quality(session),
            "recommendations": []
        }
        
        # Calculate overall quality score
        quality_report["overall_score"] = self._calculate_overall_score(quality_report)
        
        # Generate recommendations
        quality_report["recommendations"] = self._generate_recommendations(quality_report)
        
        return quality_report
    
    def _analyze_neuron_quality(self, session: SessionData) -> Dict[str, Any]:
        """Analyze neuron-level quality metrics."""
        if not session.neurons:
            return {"error": "No neurons to analyze"}
        
        n_neurons = len(session.neurons)
        quality_scores = [n.quality_score for n in session.neurons if n.quality_score is not None]
        snr_values = [n.snr for n in session.neurons if n.snr is not None]
        bf_coverage = sum(1 for n in session.neurons if n.has_best_frequency) / n_neurons
        
        # Spatial distribution analysis
        coords = np.array([[n.x, n.y, n.z] for n in session.neurons])
        spatial_spread = np.std(coords, axis=0)
        
        return {
            "n_neurons": n_neurons,
            "quality_score_stats": {
                "mean": np.mean(quality_scores) if quality_scores else None,
                "std": np.std(quality_scores) if quality_scores else None,
                "coverage": len(quality_scores) / n_neurons
            },
            "snr_stats": {
                "mean": np.mean(snr_values) if snr_values else None,
                "std": np.std(snr_values) if snr_values else None,
                "coverage": len(snr_values) / n_neurons
            },
            "bf_coverage": bf_coverage,
            "spatial_spread_xyz": spatial_spread.tolist()
        }
    
    def _analyze_trial_quality(self, session: SessionData) -> Dict[str, Any]:
        """Analyze trial-level quality metrics."""
        if not session.trials:
            return {"error": "No trials to analyze"}
        
        durations = [t.duration_frames for t in session.trials]
        frequencies = [t.frequency for t in session.trials if t.frequency is not None]
        conditions = [t.condition for t in session.trials]
        
        return {
            "n_trials": len(session.trials),
            "duration_consistency": {
                "mean": np.mean(durations),
                "std": np.std(durations),
                "cv": np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 0,
                "unique_durations": list(np.unique(durations))
            },
            "frequency_range": {
                "min": min(frequencies) if frequencies else None,
                "max": max(frequencies) if frequencies else None,
                "n_unique": len(set(frequencies))
            },
            "condition_balance": {str(k): v for k, v in pd.Series(conditions).value_counts().items()}
        }
    
    def _analyze_activity_quality(self, session: SessionData) -> Dict[str, Any]:
        """Analyze activity data quality."""
        if not hasattr(session, 'activity_matrix') or session.activity_matrix.size == 0:
            return {"error": "No activity data to analyze"}
        
        # Basic statistics
        data_stats = {
            "shape": session.activity_matrix.shape,
            "data_range": {
                "min": float(np.nanmin(session.activity_matrix)),
                "max": float(np.nanmax(session.activity_matrix)),
                "mean": float(np.nanmean(session.activity_matrix)),
                "std": float(np.nanstd(session.activity_matrix))
            },
            "missing_data": {
                "n_nan": int(np.isnan(session.activity_matrix).sum()),
                "fraction_nan": float(np.isnan(session.activity_matrix).mean())
            }
        }
        
        # Signal quality metrics
        neuron_snr = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            neuron_variance = np.nanvar(session.activity_matrix, axis=1)
        
        for i in range(session.activity_matrix.shape[0]):
            signal = session.activity_matrix[i, :]
            if not np.all(np.isnan(signal)):
                signal_power = np.nanvar(signal)
                # Estimate noise as high-frequency component
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    noise_estimate = np.nanvar(np.diff(signal))
                if noise_estimate > 0:
                    snr = signal_power / noise_estimate
                    neuron_snr.append(snr)
        
        signal_quality = {
            "neuron_variance_stats": {
                "mean": float(np.nanmean(neuron_variance)),
                "std": float(np.nanstd(neuron_variance)),
                "min": float(np.nanmin(neuron_variance)),
                "max": float(np.nanmax(neuron_variance))
            },
            "estimated_snr_stats": {
                "mean": float(np.mean(neuron_snr)) if neuron_snr else None,
                "std": float(np.std(neuron_snr)) if neuron_snr else None
            }
        }
        
        return {**data_stats, "signal_quality": signal_quality}
    
    def _analyze_temporal_quality(self, session: SessionData) -> Dict[str, Any]:
        """Analyze temporal aspects of the data."""
        if not hasattr(session, 'activity_matrix') or session.activity_matrix.size == 0 or not session.trials:
            return {"error": "Insufficient data for temporal analysis"}
        
        # Calculate sampling consistency
        frame_rate_consistency = abs(session.frame_rate - EXPERIMENT_CONFIG.volume_scan_rate_hz)
        
        # Analyze trial timing
        trial_gaps = []
        sorted_trials = sorted(session.trials, key=lambda t: t.start_frame)
        for i in range(len(sorted_trials) - 1):
            gap = sorted_trials[i + 1].start_frame - sorted_trials[i].end_frame
            trial_gaps.append(gap)
        
        return {
            "frame_rate": session.frame_rate,
            "frame_rate_deviation_from_config": frame_rate_consistency,
            "inter_trial_interval_frames": {
                "mean": np.mean(trial_gaps) if trial_gaps else None,
                "std": np.std(trial_gaps) if trial_gaps else None,
                "min": min(trial_gaps) if trial_gaps else None,
                "max": max(trial_gaps) if trial_gaps else None
            },
            "total_duration_frames": session.activity_matrix.shape[1],
            "total_duration_seconds": session.activity_matrix.shape[1] / session.frame_rate if session.frame_rate > 0 else 0
        }
    
    def _calculate_overall_score(self, quality_report: Dict[str, Any]) -> float:
        """Calculate an overall quality score from 0-1."""
        scores = []
        
        # Neuron quality contribution
        neuron_quality = quality_report.get("neuron_quality", {})
        if "quality_score_stats" in neuron_quality and neuron_quality["quality_score_stats"]["mean"] is not None:
            # Normalize to 0-1 if not already
            scores.append(np.clip(neuron_quality["quality_score_stats"]["mean"], 0, 1))
        
        # Activity quality contribution
        activity_quality = quality_report.get("activity_quality", {})
        if "missing_data" in activity_quality:
            missing_fraction = activity_quality["missing_data"]["fraction_nan"]
            scores.append(1.0 - missing_fraction)
        
        # Trial quality contribution
        trial_quality = quality_report.get("trial_quality", {})
        if "duration_consistency" in trial_quality and trial_quality["duration_consistency"]["cv"] is not None:
            cv = trial_quality["duration_consistency"]["cv"]
            # Score is 1 for cv=0, 0.5 for cv=0.5, etc. Caps at 1.
            consistency_score = max(0, 1.0 - cv)
            scores.append(consistency_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _generate_recommendations(self, quality_report: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        # Check overall score
        overall_score = quality_report.get("overall_score", 0)
        if overall_score < 0.7:
            recommendations.append(f"Overall data quality score is low ({overall_score:.2f}). Review warnings and errors.")
        
        # Neuron-specific recommendations
        neuron_quality = quality_report.get("neuron_quality", {})
        if neuron_quality.get("bf_coverage", 0) < 0.5:
            recommendations.append("Low best-frequency coverage (<50%). Consider improving frequency response analysis or check if BF data loaded correctly.")
        
        # Activity-specific recommendations
        activity_quality = quality_report.get("activity_quality", {})
        if activity_quality.get("missing_data", {}).get("fraction_nan", 0) > 0.1:
            recommendations.append("High fraction of missing data (>10%). Check raw data for sentinel values or issues in preprocessing.")
        if activity_quality.get("signal_quality", {}).get("neuron_variance_stats", {}).get("mean", 1) == 0:
            recommendations.append("Average neuron variance is zero. The activity matrix may be empty or constant.")
            
        # Trial-specific recommendations
        trial_quality = quality_report.get("trial_quality", {})
        if trial_quality.get("duration_consistency", {}).get("cv", 0) > 0.1:
            recommendations.append("High variability in trial durations (CV > 0.1). Check `framespertrial` consistency in raw data.")
        
        if not recommendations:
            recommendations.append("Session quality appears good based on automated checks.")
            
        return recommendations


class ComparisonValidator:
    """Validates data consistency across multiple sessions."""
    
    def compare_sessions(self, sessions: List[SessionData]) -> Dict[str, Any]:
        """Compare multiple sessions for consistency."""
        if len(sessions) < 2:
            return {"error": "Need at least 2 sessions for comparison"}
        
        comparison_report = {
            "n_sessions": len(sessions),
            "session_ids": [s.session_id for s in sessions],
            "consistency_checks": {},
            "recommendations": []
        }
        
        # Compare basic parameters
        frame_rates = [s.frame_rate for s in sessions]
        neuron_counts = [len(s.neurons) for s in sessions]
        trial_counts = [len(s.trials) for s in sessions]
        
        # Frame Rate
        fr_std = np.std(frame_rates)
        fr_consistent = fr_std < VALIDATION_CONFIG.frame_rate_tolerance
        comparison_report["consistency_checks"]["frame_rate_consistency"] = {
            "values": frame_rates, "mean": np.mean(frame_rates), "std": fr_std, "is_consistent": bool(fr_consistent)
        }
        if not fr_consistent:
            comparison_report["recommendations"].append("Frame rates vary significantly across sessions.")

        # Neuron Count
        nc_mean, nc_std = np.mean(neuron_counts), np.std(neuron_counts)
        comparison_report["consistency_checks"]["neuron_count_variation"] = {
            "values": neuron_counts, "mean": nc_mean, "std": nc_std, "cv": nc_std / nc_mean if nc_mean > 0 else 0
        }

        # Trial Count
        tc_unique = list(set(trial_counts))
        tc_consistent = len(tc_unique) == 1
        comparison_report["consistency_checks"]["trial_count_consistency"] = {
            "values": trial_counts, "unique_counts": tc_unique, "is_consistent": tc_consistent
        }
        if not tc_consistent:
            comparison_report["recommendations"].append(f"Trial counts are not consistent across all sessions. Found counts: {tc_unique}.")

        return comparison_report


def validate_session_comprehensive(session: SessionData) -> ValidationResult:
    """Comprehensive validation of a single session."""
    validator = DataIntegrityValidator()
    return validator.validate_session(session)


def analyze_session_quality(session: SessionData) -> Dict[str, Any]:
    """Analyze session quality and provide recommendations."""
    analyzer = QualityAnalyzer()
    return analyzer.analyze_session_quality(session)


def compare_sessions(sessions: List[SessionData]) -> Dict[str, Any]:
    """Compare multiple sessions for consistency."""
    validator = ComparisonValidator()
    return validator.compare_sessions(sessions)