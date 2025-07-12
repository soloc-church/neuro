import pytest
import numpy as np
from src.dataio.validators import (
    DataIntegrityValidator, QualityAnalyzer, ComparisonValidator,
    ValidationResult, validate_session_comprehensive, 
    analyze_session_quality, compare_sessions
)
from .data_structures import SessionData, NeuronMetadata, TrialInfo


class TestValidationResult:
    """Test the ValidationResult dataclass."""
    
    def test_validation_result_creation(self):
        """Test creating a ValidationResult."""
        result = ValidationResult(
            is_valid=True,
            warnings=[],
            errors=[],
            metrics={}
        )
        
        assert result.is_valid is True
        assert result.warnings == []
        assert result.errors == []
        assert result.metrics == {}
    
    def test_add_warning(self, caplog):
        """Test adding warnings."""
        result = ValidationResult(True, [], [], {})
        
        result.add_warning("Test warning")
        
        assert len(result.warnings) == 1
        assert result.warnings[0] == "Test warning"
        assert result.is_valid is True  # Warnings don't affect validity
    
    def test_add_error(self, caplog):
        """Test adding errors."""
        result = ValidationResult(True, [], [], {})
        
        result.add_error("Test error")
        
        assert len(result.errors) == 1
        assert result.errors[0] == "Test error"
        assert result.is_valid is False  # Errors affect validity
    
    def test_add_metric(self):
        """Test adding metrics."""
        result = ValidationResult(True, [], [], {})
        
        result.add_metric("test_metric", 42)
        
        assert result.metrics["test_metric"] == 42


class TestDataIntegrityValidator:
    """Test the DataIntegrityValidator class."""
    
    def test_validate_valid_session(self, mock_session_data):
        """Test validation of a valid session."""
        validator = DataIntegrityValidator()
        result = validator.validate_session(mock_session_data)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert "n_neurons" in result.metrics
        assert "n_trials" in result.metrics
    
    def test_validate_invalid_session(self, invalid_session_data):
        """Test validation of an invalid session."""
        validator = DataIntegrityValidator()
        result = validator.validate_session(invalid_session_data)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("Invalid session ID" in error for error in result.errors)
        assert any("Invalid frame rate" in error for error in result.errors)
    
    def test_validate_basic_structure(self, mock_session_data):
        """Test basic structure validation."""
        validator = DataIntegrityValidator()
        result = ValidationResult(True, [], [], {})
        
        validator._validate_basic_structure(mock_session_data, result)
        
        assert result.is_valid is True
        assert "session_id" in result.metrics
        assert "frame_rate" in result.metrics
    
    def test_validate_neurons_with_issues(self):
        """Test neuron validation with problematic data."""
        # Create neurons with various issues
        neurons = [
            # Valid neuron
            NeuronMetadata(
                global_id="valid_neuron",
                session="test_session",
                plane=2,
                local_idx=0,
                x=10.0, y=20.0, z=30.0,
                best_frequency=1000.0,
                quality_score=0.8,
                is_valid=True
            ),
            # Invalid neuron
            NeuronMetadata(
                global_id="invalid_neuron",
                session="test_session",
                plane=3,
                local_idx=1,
                x=50.0, y=60.0, z=70.0,
                is_valid=False
            ),
            # Neuron with quality metrics
            NeuronMetadata(
                global_id="quality_neuron",
                session="test_session",
                plane=4,
                local_idx=2,
                x=80.0, y=90.0, z=100.0,
                snr=5.0
            )
        ]
        
        session = SessionData(
            session_id="test_session",
            experiment_vars={},
            neurons=neurons,
            activity_matrix=np.random.rand(3, 1000),
            trials=[]
        )
        
        validator = DataIntegrityValidator()
        result = validator.validate_session(session)
        
        assert "n_neurons" in result.metrics
        assert "n_invalid_neurons" in result.metrics
        assert "n_bf_neurons" in result.metrics
        assert "n_quality_neurons" in result.metrics
        assert result.metrics["n_invalid_neurons"] == 1
        assert result.metrics["n_bf_neurons"] == 1
        assert result.metrics["n_quality_neurons"] == 2
    
    def test_validate_trials_with_issues(self):
        """Test trial validation with problematic data."""
        # Create trials with issues
        trials = [
            # Valid trial
            TrialInfo(
                trial_idx=0,
                frequency=1000.0,
                level=70.0,
                start_frame=0,
                end_frame=90,
                condition="1000Hz_70dB"
            ),
            # Trial with wrong index
            TrialInfo(
                trial_idx=5,  # Should be 1
                frequency=2000.0,
                level=70.0,
                start_frame=90,
                end_frame=180,
                condition="2000Hz_70dB"
            )
        ]
        
        session = SessionData(
            session_id="test_session",
            experiment_vars={},
            neurons=[],
            activity_matrix=np.array([]),
            trials=trials
        )
        
        validator = DataIntegrityValidator()
        result = validator.validate_session(session)
        
        assert "n_trials" in result.metrics
        assert len(result.warnings) > 0
        assert any("Trial index mismatch" in warning for warning in result.warnings)
    
    def test_validate_activity_matrix_issues(self):
        """Test activity matrix validation with problematic data."""
        # Create activity matrix with issues
        activity_matrix = np.random.rand(5, 1000)
        activity_matrix[0, 0] = np.nan  # Add NaN
        activity_matrix[1, 0] = np.inf  # Add infinity
        activity_matrix[2, :] = 0.0    # Zero variance neuron
        
        neurons = [
            NeuronMetadata(
                global_id=f"test_p2_n{i}",
                session="test_session",
                plane=2,
                local_idx=i,
                x=float(i), y=float(i), z=float(i)
            )
            for i in range(5)
        ]
        
        session = SessionData(
            session_id="test_session",
            experiment_vars={},
            neurons=neurons,
            activity_matrix=activity_matrix,
            trials=[]
        )
        
        validator = DataIntegrityValidator()
        result = validator.validate_session(session)
        
        assert "activity_matrix_shape" in result.metrics
        assert "n_nan_values" in result.metrics
        assert "n_inf_values" in result.metrics
        assert "n_zero_variance_neurons" in result.metrics
        
        assert result.metrics["n_nan_values"] > 0
        assert result.metrics["n_inf_values"] > 0
        assert result.metrics["n_zero_variance_neurons"] > 0


class TestQualityAnalyzer:
    """Test the QualityAnalyzer class."""
    
    def test_analyze_session_quality(self, mock_session_data):
        """Test comprehensive quality analysis."""
        analyzer = QualityAnalyzer()
        quality_report = analyzer.analyze_session_quality(mock_session_data)
        
        assert "overall_score" in quality_report
        assert "neuron_quality" in quality_report
        assert "trial_quality" in quality_report
        assert "activity_quality" in quality_report
        assert "temporal_quality" in quality_report
        assert "recommendations" in quality_report
        
        assert isinstance(quality_report["overall_score"], float)
        assert 0.0 <= quality_report["overall_score"] <= 1.0
        assert isinstance(quality_report["recommendations"], list)
    
    def test_analyze_neuron_quality(self, mock_session_data):
        """Test neuron quality analysis."""
        analyzer = QualityAnalyzer()
        neuron_quality = analyzer._analyze_neuron_quality(mock_session_data)
        
        assert "n_neurons" in neuron_quality
        assert "quality_score_stats" in neuron_quality
        assert "snr_stats" in neuron_quality
        assert "bf_coverage" in neuron_quality
        assert "spatial_spread" in neuron_quality
        
        assert neuron_quality["n_neurons"] == 10
        assert neuron_quality["bf_coverage"] == 1.0  # All mock neurons have BF
    
    def test_analyze_trial_quality(self, mock_session_data):
        """Test trial quality analysis."""
        analyzer = QualityAnalyzer()
        trial_quality = analyzer._analyze_trial_quality(mock_session_data)
        
        assert "n_trials" in trial_quality
        assert "duration_consistency" in trial_quality
        assert "frequency_range" in trial_quality
        assert "condition_balance" in trial_quality
        
        assert trial_quality["n_trials"] == 5
    
    def test_analyze_activity_quality(self, mock_session_data):
        """Test activity quality analysis."""
        analyzer = QualityAnalyzer()
        activity_quality = analyzer._analyze_activity_quality(mock_session_data)
        
        assert "shape" in activity_quality
        assert "data_range" in activity_quality
        assert "missing_data" in activity_quality
        assert "signal_quality" in activity_quality
        
        assert activity_quality["shape"] == (10, 500)
    
    def test_analyze_temporal_quality(self, mock_session_data):
        """Test temporal quality analysis."""
        analyzer = QualityAnalyzer()
        temporal_quality = analyzer._analyze_temporal_quality(mock_session_data)
        
        assert "frame_rate" in temporal_quality
        assert "frame_rate_consistency" in temporal_quality
        assert "trial_timing" in temporal_quality
        assert "total_duration_frames" in temporal_quality
        assert "total_duration_seconds" in temporal_quality
        
        assert temporal_quality["frame_rate"] == 30.0
        assert temporal_quality["total_duration_frames"] == 500
    
    def test_calculate_overall_score_high_quality(self):
        """Test overall score calculation for high-quality data."""
        analyzer = QualityAnalyzer()
        
        # Mock high-quality report
        quality_report = {
            "neuron_quality": {
                "quality_score_stats": {"mean": 0.9}
            },
            "activity_quality": {
                "missing_data": {"fraction_nan": 0.01}
            },
            "trial_quality": {
                "duration_consistency": {"cv": 0.05}
            }
        }
        
        score = analyzer._calculate_overall_score(quality_report)
        assert score > 0.8  # Should be high for good data
    
    def test_calculate_overall_score_low_quality(self):
        """Test overall score calculation for low-quality data."""
        analyzer = QualityAnalyzer()
        
        # Mock low-quality report
        quality_report = {
            "neuron_quality": {
                "quality_score_stats": {"mean": 0.3}
            },
            "activity_quality": {
                "missing_data": {"fraction_nan": 0.5}
            },
            "trial_quality": {
                "duration_consistency": {"cv": 0.5}
            }
        }
        
        score = analyzer._calculate_overall_score(quality_report)
        assert score < 0.5  # Should be low for poor data
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        analyzer = QualityAnalyzer()
        
        # Mock report with issues
        quality_report = {
            "overall_score": 0.5,  # Below threshold
            "neuron_quality": {
                "bf_coverage": 0.3  # Low BF coverage
            },
            "activity_quality": {
                "missing_data": {"fraction_nan": 0.2}  # High missing data
            },
            "trial_quality": {
                "duration_consistency": {"cv": 0.2}  # Inconsistent durations
            }
        }
        
        recommendations = analyzer._generate_recommendations(quality_report)
        
        assert len(recommendations) > 0
        assert any("quality is below recommended threshold" in rec for rec in recommendations)
        assert any("frequency coverage" in rec for rec in recommendations)
        assert any("missing data" in rec for rec in recommendations)
        assert any("Inconsistent trial durations" in rec for rec in recommendations)


class TestComparisonValidator:
    """Test the ComparisonValidator class."""
    
    def test_compare_sessions_consistency(self, mock_session_data):
        """Test session comparison for consistency."""
        # Create multiple similar sessions
        sessions = [mock_session_data]
        
        # Create a second session with slight differences
        session2 = SessionData(
            session_id="mock_session_2",
            experiment_vars={},
            neurons=mock_session_data.neurons.copy(),
            activity_matrix=mock_session_data.activity_matrix.copy(),
            trials=mock_session_data.trials.copy(),
            frame_rate=30.1  # Slightly different frame rate
        )
        sessions.append(session2)
        
        validator = ComparisonValidator()
        comparison = validator.compare_sessions(sessions)
        
        assert "n_sessions" in comparison
        assert "session_ids" in comparison
        assert "consistency_checks" in comparison
        
        assert comparison["n_sessions"] == 2
        assert len(comparison["session_ids"]) == 2
        
        checks = comparison["consistency_checks"]
        assert "frame_rate_consistency" in checks
        assert "neuron_count_variation" in checks
        assert "trial_count_consistency" in checks
    
    def test_compare_sessions_insufficient_data(self):
        """Test comparison with insufficient sessions."""
        validator = ComparisonValidator()
        result = validator.compare_sessions([])
        
        assert "error" in result
        assert "Need at least 2 sessions" in result["error"]


class TestConvenienceFunctions:
    """Test the convenience functions."""
    
    def test_validate_session_comprehensive(self, mock_session_data):
        """Test the comprehensive validation function."""
        result = validate_session_comprehensive(mock_session_data)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
    
    def test_analyze_session_quality_function(self, mock_session_data):
        """Test the quality analysis function."""
        quality_report = analyze_session_quality(mock_session_data)
        
        assert "overall_score" in quality_report
        assert "recommendations" in quality_report
    
    def test_compare_sessions_function(self, mock_session_data):
        """Test the session comparison function."""
        sessions = [mock_session_data, mock_session_data]
        comparison = compare_sessions(sessions)
        
        assert "n_sessions" in comparison
        assert comparison["n_sessions"] == 2


@pytest.mark.integration
class TestValidatorIntegration:
    """Integration tests for validators using real data."""
    
    def test_validate_real_session_180_trials(self, loaded_session_180_trials):
        """Test validation on real 180-trial session."""
        result = validate_session_comprehensive(loaded_session_180_trials)
        
        # Real data should generally be valid
        assert isinstance(result, ValidationResult)
        assert "n_neurons" in result.metrics
        assert "n_trials" in result.metrics
        assert result.metrics["n_trials"] == 180
    
    def test_analyze_real_session_quality_180_trials(self, loaded_session_180_trials):
        """Test quality analysis on real 180-trial session."""
        quality_report = analyze_session_quality(loaded_session_180_trials)
        
        assert isinstance(quality_report["overall_score"], float)
        assert 0.0 <= quality_report["overall_score"] <= 1.0
        
        # Check that all major sections are present
        assert "neuron_quality" in quality_report
        assert "trial_quality" in quality_report
        assert "activity_quality" in quality_report
        assert "temporal_quality" in quality_report
    
    def test_compare_real_sessions(self, loaded_session_180_trials, loaded_session_90_trials):
        """Test comparison of real sessions."""
        sessions = [loaded_session_180_trials, loaded_session_90_trials]
        comparison = compare_sessions(sessions)
        
        assert comparison["n_sessions"] == 2
        assert len(comparison["session_ids"]) == 2
        
        # Trial counts should be different but frame rates should be consistent
        checks = comparison["consistency_checks"]
        assert not checks["trial_count_consistency"]["is_consistent"]
        assert checks["frame_rate_consistency"]["is_consistent"]