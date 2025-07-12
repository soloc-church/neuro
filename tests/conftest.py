import pytest
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataio.loaders import SessionLoader
from src.dataio.config import FILESYSTEM_CONFIG
from src.dataio.validators import validate_session_comprehensive, analyze_session_quality

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Fixture to get the project root directory."""
    return Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    """Fixture to get the data directory."""
    data_path = project_root / FILESYSTEM_CONFIG.data_dir
    if not data_path.exists():
        pytest.skip(f"Data directory not found at {data_path}. Skipping real data tests.")
    return data_path

@pytest.fixture(scope="session")
def cache_dir(project_root: Path) -> Path:
    """Fixture to get the cache directory."""
    cache_path = project_root / FILESYSTEM_CONFIG.cache_dir
    cache_path.mkdir(exist_ok=True)
    return cache_path

@pytest.fixture(scope="session")
def session_180_trials_path(data_dir: Path) -> Path:
    """Fixture for a session path known to have 180 trials."""
    # Based on your description, this is a "March" session.
    path = data_dir / "031020_367n_100um20st_FRA"
    if not path.exists():
        pytest.skip("180-trial session '031020_367n_100um20st_FRA' not found.")
    return path
    
@pytest.fixture(scope="session")
def session_90_trials_path(data_dir: Path) -> Path:
    """Fixture for a session path known to have 90 trials."""
    # Based on your description, this is an "August" session.
    path = data_dir / "081820_355n"
    if not path.exists():
        pytest.skip("90-trial session '081820_355n' not found.")
    return path

# --- SessionData Fixtures ---
# These fixtures load the data once per session, making tests run much faster.

@pytest.fixture(scope="session")
def session_loader():
    """Fixture to provide a session loader instance."""
    return SessionLoader(validate=True)

@pytest.fixture(scope="session")
def loaded_session_180_trials(session_180_trials_path: Path, session_loader: SessionLoader):
    """Loads a full SessionData object for the 180-trial session."""
    return session_loader.load_session_complete(session_180_trials_path, use_cache=True)
    
@pytest.fixture(scope="session")
def loaded_session_90_trials(session_90_trials_path: Path, session_loader: SessionLoader):
    """Loads a full SessionData object for the 90-trial session."""
    return session_loader.load_session_complete(session_90_trials_path, use_cache=True)

# --- Validation Fixtures ---

@pytest.fixture
def validation_result_180(loaded_session_180_trials):
    """Fixture providing validation results for 180-trial session."""
    return validate_session_comprehensive(loaded_session_180_trials)

@pytest.fixture
def validation_result_90(loaded_session_90_trials):
    """Fixture providing validation results for 90-trial session."""
    return validate_session_comprehensive(loaded_session_90_trials)

@pytest.fixture
def quality_analysis_180(loaded_session_180_trials):
    """Fixture providing quality analysis for 180-trial session."""
    return analyze_session_quality(loaded_session_180_trials)

@pytest.fixture
def quality_analysis_90(loaded_session_90_trials):
    """Fixture providing quality analysis for 90-trial session."""
    return analyze_session_quality(loaded_session_90_trials)

# --- Test Configuration ---

@pytest.fixture
def test_session_paths(data_dir: Path):
    """Fixture providing all available test session paths."""
    if not data_dir.exists():
        return []
    
    # Find all session directories
    session_paths = []
    for path in data_dir.iterdir():
        if path.is_dir():
            # Check if it contains a MAT file
            mat_files = list(path.glob("*allPlanesVariables*.mat"))
            if mat_files:
                session_paths.append(path)
    
    return session_paths

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Fixture providing a temporary cache directory for tests."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return cache_dir

# --- Helper Functions for Tests ---

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "validation: marks tests as validation tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid or "loaded_session" in str(item.fixturenames):
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if "comprehensive" in item.name or "quality_analysis" in str(item.fixturenames):
            item.add_marker(pytest.mark.slow)
        
        # Mark validation tests
        if "validation" in item.name or "validate" in item.name:
            item.add_marker(pytest.mark.validation)

# --- Utility Fixtures ---

@pytest.fixture
def mock_session_data():
    """Fixture providing a minimal mock SessionData object for unit tests."""
    from src.dataio.data_structures import SessionData, NeuronMetadata, TrialInfo
    import numpy as np
    
    # Create mock neurons
    neurons = [
        NeuronMetadata(
            global_id=f"mock_session_p2_n{i}",
            session="mock_session",
            plane=2,
            local_idx=i,
            x=float(i), y=float(i), z=float(i),
            best_frequency=1000.0 + i * 100,
            quality_score=0.8
        )
        for i in range(10)
    ]
    
    # Create mock trials
    trials = [
        TrialInfo(
            trial_idx=i,
            frequency=1000.0 + i * 100,
            level=70.0,
            start_frame=i * 100,
            end_frame=(i + 1) * 100,
            condition=f"{1000 + i * 100}Hz_70dB"
        )
        for i in range(5)
    ]
    
    # Create mock activity matrix
    activity_matrix = np.random.rand(10, 500)
    
    return SessionData(
        session_id="mock_session",
        experiment_vars={},
        neurons=neurons,
        activity_matrix=activity_matrix,
        trials=trials
    )

@pytest.fixture
def invalid_session_data():
    """Fixture providing invalid SessionData for testing validation."""
    from src.dataio.data_structures import SessionData
    import numpy as np
    
    # Create session with inconsistent data
    return SessionData(
        session_id="",  # Invalid empty ID
        experiment_vars={},
        neurons=[],  # No neurons
        activity_matrix=np.array([]),  # Empty activity matrix
        trials=[],  # No trials
        frame_rate=-1.0  # Invalid frame rate
    )