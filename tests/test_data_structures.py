import pytest
import numpy as np
from src.dataio.data_structures import (
    SessionData, NeuronMetadata, TrialInfo, 
    ValidationError, DataValidator
)


class TestDataValidator:
    """Test the DataValidator utility class."""
    
    def test_validate_coordinates_valid(self):
        """Test validation of valid coordinates."""
        # Should not raise any exception
        DataValidator.validate_coordinates(0.0, 0.0, 0.0)
        DataValidator.validate_coordinates(100.0, -50.0, 25.0)
    
    def test_validate_coordinates_invalid(self):
        """Test validation of invalid coordinates."""
        with pytest.raises(ValidationError, match="Coordinate x"):
            DataValidator.validate_coordinates(2000.0, 0.0, 0.0)
        
        with pytest.raises(ValidationError, match="Coordinate y"):
            DataValidator.validate_coordinates(0.0, -2000.0, 0.0)
    
    def test_validate_quality_score_valid(self):
        """Test validation of valid quality scores."""
        DataValidator.validate_quality_score(0.5)
        DataValidator.validate_quality_score(1.0)
        DataValidator.validate_quality_score(0.0)
        DataValidator.validate_quality_score(None)
    
    def test_validate_quality_score_invalid(self):
        """Test validation of invalid quality scores."""
        with pytest.raises(ValidationError, match="Quality score"):
            DataValidator.validate_quality_score(1.5)
        
        with pytest.raises(ValidationError, match="Quality score"):
            DataValidator.validate_quality_score(-0.1)
    
    def test_validate_frequency_valid(self):
        """Test validation of valid frequencies."""
        DataValidator.validate_frequency(1000.0)
        DataValidator.validate_frequency(0.1)
    
    def test_validate_frequency_invalid(self):
        """Test validation of invalid frequencies."""
        with pytest.raises(ValidationError, match="Frequency"):
            DataValidator.validate_frequency(0.0)
        
        with pytest.raises(ValidationError, match="Frequency"):
            DataValidator.validate_frequency(-100.0)


class TestNeuronMetadata:
    """Test the NeuronMetadata dataclass."""
    
    def test_valid_neuron_creation(self):
        """Test creation of a valid neuron."""
        neuron = NeuronMetadata(
            global_id="test_p2_n1",
            session="test_session",
            plane=2,
            local_idx=1,
            x=10.0, y=20.0, z=30.0,
            best_frequency=1000.0,
            quality_score=0.8
        )
        
        assert neuron.global_id == "test_p2_n1"
        assert neuron.session == "test_session"
        assert neuron.plane == 2
        assert neuron.local_idx == 1
        assert neuron.best_frequency == 1000.0
        assert neuron.quality_score == 0.8
        assert neuron.is_valid is True
    
    def test_neuron_validation_invalid_coordinates(self):
        """Test that invalid coordinates raise ValidationError."""
        with pytest.raises(ValidationError):
            NeuronMetadata(
                global_id="test_p2_n1",
                session="test_session",
                plane=2,
                local_idx=1,
                x=2000.0, y=20.0, z=30.0  # Invalid x coordinate
            )
    
    def test_neuron_validation_invalid_plane(self):
        """Test that invalid plane numbers raise ValidationError."""
        with pytest.raises(ValidationError):
            NeuronMetadata(
                global_id="test_p8_n1",
                session="test_session",
                plane=8,  # Invalid plane
                local_idx=1,
                x=10.0, y=20.0, z=30.0
            )
    
    def test_neuron_validation_invalid_quality_score(self):
        """Test that invalid quality scores raise ValidationError."""
        with pytest.raises(ValidationError):
            NeuronMetadata(
                global_id="test_p2_n1",
                session="test_session",
                plane=2,
                local_idx=1,
                x=10.0, y=20.0, z=30.0,
                quality_score=1.5  # Invalid quality score
            )
    
    def test_coords_3d_property(self):
        """Test the coords_3d property."""
        neuron = NeuronMetadata(
            global_id="test_p2_n1",
            session="test_session",
            plane=2,
            local_idx=1,
            x=10.0, y=20.0, z=30.0
        )
        
        coords = neuron.coords_3d
        expected = np.array([10.0, 20.0, 30.0])
        np.testing.assert_array_equal(coords, expected)
    
    def test_has_best_frequency_property(self):
        """Test the has_best_frequency property."""
        neuron_with_bf = NeuronMetadata(
            global_id="test_p2_n1",
            session="test_session",
            plane=2,
            local_idx=1,
            x=10.0, y=20.0, z=30.0,
            best_frequency=1000.0
        )
        
        neuron_without_bf = NeuronMetadata(
            global_id="test_p2_n2",
            session="test_session",
            plane=2,
            local_idx=2,
            x=10.0, y=20.0, z=30.0
        )
        
        assert neuron_with_bf.has_best_frequency is True
        assert neuron_without_bf.has_best_frequency is False
    
    def test_has_quality_metrics_property(self):
        """Test the has_quality_metrics property."""
        neuron_with_quality = NeuronMetadata(
            global_id="test_p2_n1",
            session="test_session",
            plane=2,
            local_idx=1,
            x=10.0, y=20.0, z=30.0,
            quality_score=0.8
        )
        
        neuron_with_snr = NeuronMetadata(
            global_id="test_p2_n2",
            session="test_session",
            plane=2,
            local_idx=2,
            x=10.0, y=20.0, z=30.0,
            snr=5.0
        )
        
        neuron_without_quality = NeuronMetadata(
            global_id="test_p2_n3",
            session="test_session",
            plane=2,
            local_idx=3,
            x=10.0, y=20.0, z=30.0
        )
        
        assert neuron_with_quality.has_quality_metrics is True
        assert neuron_with_snr.has_quality_metrics is True
        assert neuron_without_quality.has_quality_metrics is False
    
    def test_distance_to_method(self):
        """Test the distance_to method."""
        neuron1 = NeuronMetadata(
            global_id="test_p2_n1",
            session="test_session",
            plane=2,
            local_idx=1,
            x=0.0, y=0.0, z=0.0
        )
        
        neuron2 = NeuronMetadata(
            global_id="test_p2_n2",
            session="test_session",
            plane=2,
            local_idx=2,
            x=3.0, y=4.0, z=0.0
        )
        
        distance = neuron1.distance_to(neuron2)
        expected_distance = 5.0  # 3-4-5 triangle
        assert abs(distance - expected_distance) < 1e-6
    
    def test_to_dict_method(self):
        """Test the to_dict method."""
        neuron = NeuronMetadata(
            global_id="test_p2_n1",
            session="test_session",
            plane=2,
            local_idx=1,
            x=10.0, y=20.0, z=30.0,
            best_frequency=1000.0,
            quality_score=0.8
        )
        
        neuron_dict = neuron.to_dict()
        
        assert neuron_dict['global_id'] == "test_p2_n1"
        assert neuron_dict['session'] == "test_session"
        assert neuron_dict['plane'] == 2
        assert neuron_dict['best_frequency'] == 1000.0
        assert neuron_dict['quality_score'] == 0.8


class TestTrialInfo:
    """Test the TrialInfo dataclass."""
    
    def test_valid_trial_creation(self):
        """Test creation of a valid trial."""
        trial = TrialInfo(
            trial_idx=0,
            frequency=1000.0,
            level=70.0,
            start_frame=0,
            end_frame=90,
            condition="1000Hz_70dB"
        )
        
        assert trial.trial_idx == 0
        assert trial.frequency == 1000.0
        assert trial.level == 70.0
        assert trial.start_frame == 0
        assert trial.end_frame == 90
        assert trial.condition == "1000Hz_70dB"
    
    def test_trial_validation_invalid_trial_idx(self):
        """Test that negative trial indices raise ValidationError."""
        with pytest.raises(ValidationError):
            TrialInfo(
                trial_idx=-1,  # Invalid negative index
                frequency=1000.0,
                level=70.0,
                start_frame=0,
                end_frame=90,
                condition="1000Hz_70dB"
            )
    
    def test_trial_validation_invalid_frequency(self):
        """Test that invalid frequencies raise ValidationError."""
        with pytest.raises(ValidationError):
            TrialInfo(
                trial_idx=0,
                frequency=0.0,  # Invalid frequency
                level=70.0,
                start_frame=0,
                end_frame=90,
                condition="0Hz_70dB"
            )
    
    def test_trial_validation_invalid_frames(self):
        """Test that invalid frame ranges raise ValidationError."""
        with pytest.raises(ValidationError):
            TrialInfo(
                trial_idx=0,
                frequency=1000.0,
                level=70.0,
                start_frame=90,
                end_frame=0,  # End before start
                condition="1000Hz_70dB"
            )
    
    def test_duration_frames_property(self):
        """Test the duration_frames property."""
        trial = TrialInfo(
            trial_idx=0,
            frequency=1000.0,
            level=70.0,
            start_frame=100,
            end_frame=190,
            condition="1000Hz_70dB"
        )
        
        assert trial.duration_frames == 90
    
    def test_to_dict_method(self):
        """Test the to_dict method."""
        trial = TrialInfo(
            trial_idx=0,
            frequency=1000.0,
            level=70.0,
            start_frame=0,
            end_frame=90,
            condition="1000Hz_70dB"
        )
        
        trial_dict = trial.to_dict()
        
        assert trial_dict['trial_idx'] == 0
        assert trial_dict['frequency'] == 1000.0
        assert trial_dict['level'] == 70.0
        assert trial_dict['duration_frames'] == 90


class TestSessionData:
    """Test the SessionData class."""
    
    def test_session_data_methods(self, loaded_session_180_trials: SessionData):
        """
        Tests the convenience methods on the SessionData object using real data.
        This implicitly tests the data structures with real-world values.
        """
        session = loaded_session_180_trials
        assert session.session_id == "031020_367n_100um20st_FRA"

        # Test get_quality_neurons()
        quality_neurons = session.get_quality_neurons(min_quality=0.0) # Get all
        assert len(quality_neurons) == len(session.neurons)
        
        # Test get_bf_groups() - Best Frequency
        bf_groups = session.get_bf_groups()
        assert isinstance(bf_groups, dict)
        if bf_groups: # Check only if data is present
          assert all(isinstance(freq, float) for freq in bf_groups.keys())
          assert all(isinstance(ids, list) for ids in bf_groups.values())
          
          # Check that the number of neurons in groups matches neurons with a BF
          total_grouped_neurons = sum(len(ids) for ids in bf_groups.values())
          total_bf_neurons = sum(1 for n in session.neurons if n.has_best_frequency)
          assert total_grouped_neurons == total_bf_neurons

        # Test get_spatial_neighbors()
        if len(session.neurons) > 1:
            first_neuron_id = session.neurons[0].global_id
            neighbors = session.get_spatial_neighbors(first_neuron_id, radius=100.0)
            assert isinstance(neighbors, list)
            # The target neuron should not be in its own neighbor list
            assert first_neuron_id not in neighbors
    
    def test_session_validation_empty_data(self):
        """Test session validation with empty data."""
        with pytest.warns(UserWarning):
            session = SessionData(
                session_id="empty_session",
                experiment_vars={},
                neurons=[],
                activity_matrix=np.array([]),
                trials=[]
            )
    
    def test_session_validation_mismatched_dimensions(self):
        """Test that mismatched dimensions raise ValidationError."""
        neurons = [
            NeuronMetadata(
                global_id="test_p2_n1",
                session="test_session",
                plane=2,
                local_idx=1,
                x=10.0, y=20.0, z=30.0
            )
        ]
        
        # Activity matrix with wrong number of neurons
        activity_matrix = np.random.rand(2, 1000)  # 2 neurons but only 1 in metadata
        
        with pytest.raises(ValidationError):
            SessionData(
                session_id="test_session",
                experiment_vars={},
                neurons=neurons,
                activity_matrix=activity_matrix,
                trials=[]
            )
    
    def test_get_neuron_by_id(self, mock_session_data):
        """Test getting neuron by ID."""
        session = mock_session_data
        
        # Test existing neuron
        neuron = session.get_neuron_by_id("mock_session_p2_n0")
        assert neuron is not None
        assert neuron.global_id == "mock_session_p2_n0"
        
        # Test non-existing neuron
        neuron = session.get_neuron_by_id("nonexistent")
        assert neuron is None
    
    def test_get_neurons_by_plane(self, mock_session_data):
        """Test getting neurons by plane."""
        session = mock_session_data
        
        plane_2_neurons = session.get_neurons_by_plane(2)
        assert len(plane_2_neurons) == 10  # All neurons are in plane 2
        assert all(n.plane == 2 for n in plane_2_neurons)
        
        plane_3_neurons = session.get_neurons_by_plane(3)
        assert len(plane_3_neurons) == 0  # No neurons in plane 3
    
    def test_get_trial_by_condition(self, mock_session_data):
        """Test getting trials by condition."""
        session = mock_session_data
        
        condition_trials = session.get_trial_by_condition("1000Hz_70dB")
        assert len(condition_trials) == 1
        assert condition_trials[0].condition == "1000Hz_70dB"
        
        nonexistent_trials = session.get_trial_by_condition("nonexistent")
        assert len(nonexistent_trials) == 0
    
    def test_get_activity_for_neuron(self, mock_session_data):
        """Test getting activity for a specific neuron."""
        session = mock_session_data
        
        # Test existing neuron
        activity = session.get_activity_for_neuron("mock_session_p2_n0")
        assert activity is not None
        assert activity.shape == (500,)  # Time points
        
        # Test non-existing neuron
        activity = session.get_activity_for_neuron("nonexistent")
        assert activity is None
    
    def test_get_trial_responses(self, mock_session_data):
        """Test getting trial responses."""
        session = mock_session_data
        
        # Test specific trials
        responses = session.get_trial_responses([0, 1])
        assert len(responses) == 2
        assert "trial_0" in responses
        assert "trial_1" in responses
        assert responses["trial_0"].shape == (10, 100)  # 10 neurons, 100 frames per trial
        
        # Test all trials
        all_responses = session.get_trial_responses()
        assert len(all_responses) == 5  # All 5 trials
    
    def test_compute_summary_stats(self, mock_session_data):
        """Test computing summary statistics."""
        session = mock_session_data
        
        stats = session.compute_summary_stats()
        
        assert stats['session_id'] == "mock_session"
        assert stats['n_neurons'] == 10
        assert stats['n_trials'] == 5
        assert stats['n_planes'] == 1
        assert stats['frame_rate'] == 30.0
        assert stats['duration_frames'] == 500
        assert 'bf_coverage' in stats
        assert 'trial_stats' in stats
    
    def test_bf_groups_with_tolerance(self, mock_session_data):
        """Test BF grouping with tolerance."""
        session = mock_session_data
        
        # Test exact grouping (no tolerance)
        exact_groups = session.get_bf_groups(tolerance_hz=0.0)
        assert len(exact_groups) == 10  # Each neuron has unique frequency
        
        # Test with tolerance
        tolerance_groups = session.get_bf_groups(tolerance_hz=150.0)
        assert len(tolerance_groups) <= 10  # Should group some frequencies
    
    def test_get_quality_neurons_with_snr(self):
        """Test getting quality neurons with SNR filtering."""
        neurons = [
            NeuronMetadata(
                global_id=f"test_p2_n{i}",
                session="test_session",
                plane=2,
                local_idx=i,
                x=float(i), y=float(i), z=float(i),
                quality_score=0.8,
                snr=float(i)  # SNR varies from 0 to 4
            )
            for i in range(5)
        ]
        
        session = SessionData(
            session_id="test_session",
            experiment_vars={},
            neurons=neurons,
            activity_matrix=np.random.rand(5, 1000),
            trials=[]
        )
        
        # Test with quality threshold only
        quality_neurons = session.get_quality_neurons(min_quality=0.7)
        assert len(quality_neurons) == 5  # All neurons pass quality
        
        # Test with SNR threshold
        snr_filtered = session.get_quality_neurons(min_quality=0.7, min_snr=2.0)
        assert len(snr_filtered) == 3  # Only neurons with SNR >= 2.0
    
    def test_caching_behavior(self, mock_session_data):
        """Test that caching works correctly."""
        session = mock_session_data
        
        # First call should compute and cache
        result1 = session.get_quality_neurons(min_quality=0.5, use_cache=True)
        
        # Second call should use cache
        result2 = session.get_quality_neurons(min_quality=0.5, use_cache=True)
        
        assert result1 == result2
        
        # Different parameters should recompute
        result3 = session.get_quality_neurons(min_quality=0.9, use_cache=True)
        
        # Result3 might be different due to different threshold
        assert isinstance(result3, list)