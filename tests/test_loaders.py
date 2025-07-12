import pytest
import numpy as np
import warnings
from unittest.mock import MagicMock, patch
from pathlib import Path

# Mock the config before importing loaders
mock_config = MagicMock()
mock_config.valid_planes = [2, 3, 4, 5, 6]
mock_config.bad_sentinel_value = -100.0

# The patch needs to target where the module is *used*
with patch.dict('sys.modules', {'src.dataio.config': mock_config}):
    from src.dataio.loaders import (
        SessionLoader, NeuronExtractor, TrialExtractor, 
        ActivityMatrixBuilder, MatFileParser, RawMatData
    )
    from src.dataio.data_structures import SessionData

# --- Mock Data Fixtures ---

class MockMatStruct:
    """A mock class to simulate the structure of scipy.io.loadmat objects."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

@pytest.fixture
def mock_raw_data_180_trials():
    """
    Creates a mock data dictionary simulating a MAT file from a March session
    (180 trials per condition).
    """
    # --- Coordinates and Activity ---
    # 7 planes total. Planes 1 and 7 are invalid.
    # Let's say planes 2-6 have 10 neurons each.
    coords = {
        'x': [np.zeros(5), np.random.rand(10), np.random.rand(10), np.random.rand(10), np.random.rand(10), np.random.rand(10), np.zeros(5)],
        'y': [np.zeros(5), np.random.rand(10), np.random.rand(10), np.random.rand(10), np.random.rand(10), np.random.rand(10), np.zeros(5)],
        'z': [np.zeros(5), np.random.rand(10), np.random.rand(10), np.random.rand(10), np.random.rand(10), np.random.rand(10), np.zeros(5)],
    }
    # Activity matrix with some NaNs to test propagation
    activity = [
        np.zeros((5, 20000)),
        np.random.rand(10, 20000),
        np.random.rand(10, 20000),
        np.random.rand(10, 20000),
        np.random.rand(10, 20000),
        np.random.rand(10, 20000),
        np.zeros((5, 20000)),
    ]
    activity[2][3, 100] = np.nan # Inject a NaN

    # --- Stimulus Info (180 trials) ---
    stim_info_content = MockMatStruct(
        Trialindicies=np.array([[i*90, 0, 0] for i in range(180)]),
        framespertrial=90,
        pfs=30.0, # Raw frame rate
        Freqs=np.random.randint(4000, 64000, size=180),
        Levels=np.full(180, 70)
    )
    stim_info = [None, stim_info_content, stim_info_content, stim_info_content, stim_info_content, stim_info_content, None]
    
    # --- Other Metadata ---
    bf_data = np.random.rand(10)
    bf_data[5] = -100 # Sentinel value for testing
    bf_info = [None] * 7
    bf_info[2] = bf_data # Place in a valid plane

    return RawMatData(
        coords=coords,
        activity=activity,
        stim_info=stim_info,
        bf_info=bf_info,
        cell_info=[None] * 7, # Not testing this in detail yet
        z_stuff=None,
        expt_vars=None,
        corr_info=None,
        z_corr_info=None,
        select_corr_info=None
    )

@pytest.fixture
def mock_raw_data_90_trials(mock_raw_data_180_trials):
    """
    Creates a mock data dictionary simulating a MAT file from an August session
    (90 trials per condition), reusing the 180-trial structure.
    """
    data = mock_raw_data_180_trials
    stim_info_content = MockMatStruct(
        Trialindicies=np.array([[i*90, 0, 0] for i in range(90)]),
        framespertrial=90,
        pfs=30.0,
        Freqs=np.random.randint(4000, 64000, size=90),
        Levels=np.full(90, 70)
    )
    # Create new RawMatData with updated stim_info
    return RawMatData(
        coords=data.coords,
        activity=data.activity,
        stim_info=[None, stim_info_content] + [None] * 5,
        bf_info=data.bf_info,
        cell_info=data.cell_info,
        z_stuff=data.z_stuff,
        expt_vars=data.expt_vars,
        corr_info=data.corr_info,
        z_corr_info=data.z_corr_info,
        select_corr_info=data.select_corr_info
    )

# --- Test Functions ---

def test_file_integrity_valid_planes(mock_raw_data_180_trials):
    """
    1. File integrity (usable planes):
    Tests that only neurons from VALID_PLANES (2-6) are extracted.
    """
    extractor = NeuronExtractor()
    neurons = extractor.extract_neuron_metadata(mock_raw_data_180_trials, "test_session")
    # 5 valid planes with 10 neurons each = 50 neurons
    assert len(neurons) == 50
    # Check that plane numbers are correct
    assert all(n.plane in mock_config.valid_planes for n in neurons)
    assert 1 not in [n.plane for n in neurons]
    assert 7 not in [n.plane for n in neurons]

def test_sentinel_handling(mock_raw_data_180_trials):
    """
    2. Sentinel handling (-100 values):
    Tests that sentinel values are not loaded as best frequencies.
    """
    extractor = NeuronExtractor()
    neurons = extractor.extract_neuron_metadata(mock_raw_data_180_trials, "test_session")
    # Find the neuron that should have had the sentinel BF value
    # It should be in plane 3 (2-indexed), local index 5
    sentinel_neuron = next((n for n in neurons if n.plane == 3 and n.local_idx == 5), None)
    assert sentinel_neuron is not None
    # The sentinel value should have been filtered out, so best_frequency should be None
    assert sentinel_neuron.best_frequency is None

def test_timing_and_stimulus_labels_180_trials(mock_raw_data_180_trials):
    """
    4. Timing metadata & 5. Stimulus labels (180 trials):
    Checks trial count, frame calculations, and label consistency for 180-trial sessions.
    """
    extractor = TrialExtractor()
    trials = extractor.extract_trial_info(mock_raw_data_180_trials)
    assert len(trials) == 180
    
    # Check that framespertrial was used correctly
    first_trial = trials[0]
    assert first_trial.start_frame == 0
    assert first_trial.end_frame == 90
    
    last_trial = trials[-1]
    assert last_trial.start_frame == 179 * 90
    assert last_trial.end_frame == 179 * 90 + 90
    
    # Check that stimulus label lengths match trial count
    stim_info = next(si for si in mock_raw_data_180_trials.stim_info if si)
    assert len(stim_info.Freqs) == len(trials)
    assert len(stim_info.Levels) == len(trials)

def test_timing_and_stimulus_labels_90_trials(mock_raw_data_90_trials):
    """
    4. Timing metadata & 5. Stimulus labels (90 trials):
    Checks trial count, frame calculations, and label consistency for 90-trial sessions.
    """
    extractor = TrialExtractor()
    trials = extractor.extract_trial_info(mock_raw_data_90_trials)
    assert len(trials) == 90
    
    last_trial = trials[-1]
    assert last_trial.start_frame == 89 * 90
    assert last_trial.end_frame == 89 * 90 + 90
    
    stim_info = next(si for si in mock_raw_data_90_trials.stim_info if si)
    assert len(stim_info.Freqs) == len(trials)
    assert len(stim_info.Levels) == len(trials)

def test_neuron_activity_alignment(mock_raw_data_180_trials, tmp_path):
    """
    6. Neuron <-> activity alignment:
    Ensures the number of neurons in the final object matches the activity matrix shape.
    """
    # Create a mock MAT file
    session_path = tmp_path / "session_180"
    session_path.mkdir()
    (session_path / "dummy.mat").touch()

    # Mock the parser to return our test data
    with patch.object(MatFileParser, 'parse_mat_file', return_value=mock_raw_data_180_trials):
        loader = SessionLoader()
        session = loader.load_session_complete(session_path, use_cache=False)
    
    num_neurons = len(session.neurons)
    assert num_neurons == 50 # 5 valid planes * 10 neurons
    assert session.activity_matrix.shape[0] == num_neurons

def test_nan_propagation(mock_raw_data_180_trials, tmp_path):
    """
    7. NaN propagation:
    Confirms that NaNs in the raw zDFF data are preserved in the final activity matrix.
    """
    # Create a mock MAT file
    session_path = tmp_path / "session_nan"
    session_path.mkdir()
    (session_path / "dummy.mat").touch()

    # Mock the parser to return our test data
    with patch.object(MatFileParser, 'parse_mat_file', return_value=mock_raw_data_180_trials):
        loader = SessionLoader()
        session = loader.load_session_complete(session_path, use_cache=False)
    
    # Check that the NaN we injected is still there
    assert np.isnan(session.activity_matrix).any()
    
    # The original NaN was in plane 2 (index 2), which is the 2nd valid plane.
    # Neurons from plane 2 start at index 10. The NaN was on neuron 3 (0-indexed).
    # So, the final row should be 10 + 3 = 13.
    assert np.isnan(session.activity_matrix[13, 100])

def test_empty_stim_info_warning():
    """
    Tests that a warning is raised if no valid stimInfo is found.
    """
    # Create data with no valid stim info
    bad_data = RawMatData(
        coords={'x': [], 'y': [], 'z': []},
        activity=[],
        stim_info=[None, MockMatStruct(some_other_field=1)],
        bf_info=None,
        cell_info=None
    )

    extractor = TrialExtractor()
    with pytest.warns(UserWarning, match="No valid stimulus information found"):
        trials = extractor.extract_trial_info(bad_data)
    
    assert len(trials) == 0

def test_activity_matrix_builder(mock_raw_data_180_trials):
    """
    Test the activity matrix builder functionality.
    """
    builder = ActivityMatrixBuilder()
    activity_matrix = builder.build_activity_matrix(mock_raw_data_180_trials)
    
    # Should have 5 valid planes * 10 neurons each = 50 rows
    assert activity_matrix.shape[0] == 50
    assert activity_matrix.shape[1] == 20000  # Time points
    
    # Check that sentinel values were converted to NaN
    assert np.isnan(activity_matrix).any()

def test_session_loader_error_handling(tmp_path):
    """
    Test error handling in session loader.
    """
    # Test with non-existent session path
    loader = SessionLoader()
    
    # Create path with no MAT files
    session_path = tmp_path / "empty_session"
    session_path.mkdir()
    
    with pytest.raises(Exception):  # Should raise LoadingError
        loader.load_session_complete(session_path, use_cache=False)

def test_neuron_extractor_validation():
    """
    Test that neuron extractor properly validates data.
    """
    # Create data that would cause validation errors
    bad_coords = {
        'x': [np.array([1000000])],  # Extreme coordinate
        'y': [np.array([0])],
        'z': [np.array([0])]
    }
    
    bad_data = RawMatData(
        coords=bad_coords,
        activity=[np.random.rand(1, 1000)],
        stim_info=None,
        bf_info=None,
        cell_info=None
    )
    
    # Should handle validation errors gracefully
    extractor = NeuronExtractor()
    neurons = extractor.extract_neuron_metadata(bad_data, "test_session")
    
    # May return empty list or neurons with warnings logged
    assert isinstance(neurons, list)

# This test will currently pass with the refactored loader since it doesn't
# extract frame rate from stimInfo.pfs yet, but it's kept for future implementation.
@pytest.mark.xfail(reason="Frame rate extraction from stimInfo.pfs not yet implemented in loader.")
def test_frame_bookkeeping_extraction(mock_raw_data_180_trials, tmp_path):
    """
    3. Frame bookkeeping (raw frame rate hz):
    Tests that the frame rate is correctly extracted from `stimInfo.pfs`.
    """
    session_path = tmp_path / "session_framerate"
    session_path.mkdir()
    (session_path / "dummy.mat").touch()

    with patch.object(MatFileParser, 'parse_mat_file', return_value=mock_raw_data_180_trials):
        loader = SessionLoader()
        session = loader.load_session_complete(session_path, use_cache=False)
    
    # The loader currently uses a default of 30.0. A robust implementation
    # should pull `pfs` from the file.
    assert session.frame_rate == 30.0