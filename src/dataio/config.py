from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import os


def find_project_root() -> Path:
    """
    Automatically find the project root directory by looking for common markers.
    Looks for a directory containing 'data' folder or specific files.
    """
    current = Path.cwd()
    
    # Try current directory first
    if (current / "data").exists():
        return current
    
    # Walk up the directory tree looking for project root markers
    for parent in [current] + list(current.parents):
        # Check for data directory
        if (parent / "data").exists():
            return parent
        # Check for other common project markers
        if any((parent / marker).exists() for marker in [
            "setup.py", "pyproject.toml", "requirements.txt", ".git", 
            "src", "README.md", "README.txt"
        ]):
            # If we found project markers, check if data dir exists
            if (parent / "data").exists():
                return parent
    
    # If no project root found, use current directory and let user configure
    return current


def get_data_directory() -> Path:
    """
    Get the data directory, trying multiple common locations.
    """
    # Check environment variable first
    if "NEUROSCIENCE_DATA_DIR" in os.environ:
        data_dir = Path(os.environ["NEUROSCIENCE_DATA_DIR"])
        if data_dir.exists():
            return data_dir
    
    # Try project root
    project_root = find_project_root()
    data_dir = project_root / "data"
    if data_dir.exists():
        return data_dir
    
    # Try common alternative locations
    alternatives = [
        Path.cwd() / "data",
        Path.home() / "data" / "neuroscience",
        Path("/Volumes/My Passport for Mac/church app/neuroscience-nexus/data"),  # User's specific path
        Path.cwd().parent / "data",
    ]
    
    for alt in alternatives:
        if alt.exists():
            return alt
    
    # Return the expected location even if it doesn't exist (user will get clear error)
    return project_root / "data"


@dataclass(frozen=True)
class DataLoadingConfig:
    """Configuration for data loading parameters."""
    # Per the README.txt, data from planes 2 through 6 should be used.
    # This corresponds to 1-based MATLAB plane numbers.
    valid_planes: List[int] = None
    
    # Sentinel value found in the raw zDFF data that represents missing or
    # invalid data points. This value will be replaced with NaN upon loading.
    bad_sentinel_value: float = -100.0
    
    def __post_init__(self):
        if self.valid_planes is None:
            # MATLAB planes 2-6 correspond to Python indices 1-5
            # (MATLAB uses 1-based indexing, Python uses 0-based)
            # So MATLAB plane 2 = Python index 1, etc.
            object.__setattr__(self, 'valid_planes', list(range(1, 6)))

@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for experimental parameters."""
    # The overall rate at which the microscope scans the entire volume (all 7 planes).
    # This is the primary "master clock" for the experiment, used for calculating
    # total duration and for aligning stimulus timings.
    volume_scan_rate_hz: float = 30.0
    
    # Expected trial counts for different session types
    expected_trial_counts: List[int] = None
    
    # Quality thresholds
    min_quality_score: float = 0.7
    min_snr: float = 2.0
    
    def __post_init__(self):
        if self.expected_trial_counts is None:
            object.__setattr__(self, 'expected_trial_counts', [90, 180])


@dataclass(frozen=True)
class FilesystemConfig:
    """Configuration for filesystem paths."""
    # The directory where the session data folders are located.
    data_dir: Path = None
    cache_dir: Path = None
    output_dir: Path = None
    
    # File patterns
    mat_file_pattern: str = "*allPlanesVariables*.mat"
    cache_file_suffix: str = "_complete.pkl"
    
    def __post_init__(self):
        if self.data_dir is None:
            data_dir = get_data_directory()
            object.__setattr__(self, 'data_dir', data_dir)
        
        if self.cache_dir is None:
            cache_dir = self.data_dir.parent / "cache"
            object.__setattr__(self, 'cache_dir', cache_dir)
            
        if self.output_dir is None:
            output_dir = self.data_dir.parent / "outputs"
            object.__setattr__(self, 'output_dir', output_dir)


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for data validation."""
    # Neuron count validation
    min_neurons_per_session: int = 50
    max_neurons_per_session: int = 10000
    
    # Coordinate validation
    min_coordinate_value: float = -1000.0
    max_coordinate_value: float = 1000.0
    
    # Frame rate validation
    frame_rate_tolerance: float = 1.0


# Global configuration instances
DATA_CONFIG = DataLoadingConfig()
EXPERIMENT_CONFIG = ExperimentConfig()
FILESYSTEM_CONFIG = FilesystemConfig()
VALIDATION_CONFIG = ValidationConfig()

# Backward compatibility - maintain original constants
VALID_PLANES = DATA_CONFIG.valid_planes
BAD_SENTINEL_VALUE = DATA_CONFIG.bad_sentinel_value
VOLUME_SCAN_RATE_HZ = EXPERIMENT_CONFIG.volume_scan_rate_hz
NUM_PLANES_USED = len(VALID_PLANES)
DATA_DIR = FILESYSTEM_CONFIG.data_dir


def set_data_directory(path: str | Path) -> None:
    """
    Override the default data directory.
    Useful for testing or when data is in a non-standard location.
    """
    global FILESYSTEM_CONFIG
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {path}")
    
    # Create new config with updated path
    object.__setattr__(FILESYSTEM_CONFIG, 'data_dir', path)
    object.__setattr__(FILESYSTEM_CONFIG, 'cache_dir', path.parent / "cache")
    object.__setattr__(FILESYSTEM_CONFIG, 'output_dir', path.parent / "outputs")
    
    # Update global constant
    globals()['DATA_DIR'] = path


def get_config_summary() -> Dict[str, Any]:
    """Get a summary of all configuration settings."""
    return {
        "data_loading": {
            "valid_planes": DATA_CONFIG.valid_planes,
            "bad_sentinel_value": DATA_CONFIG.bad_sentinel_value,
        },
        "experiment": {
            "volume_scan_rate_hz": EXPERIMENT_CONFIG.volume_scan_rate_hz,
            "expected_trial_counts": EXPERIMENT_CONFIG.expected_trial_counts,
            "min_quality_score": EXPERIMENT_CONFIG.min_quality_score,
        },
        "filesystem": {
            "data_dir": str(FILESYSTEM_CONFIG.data_dir),
            "cache_dir": str(FILESYSTEM_CONFIG.cache_dir),
            "output_dir": str(FILESYSTEM_CONFIG.output_dir),
            "data_dir_exists": FILESYSTEM_CONFIG.data_dir.exists(),
        },
        "validation": {
            "min_neurons_per_session": VALIDATION_CONFIG.min_neurons_per_session,
            "max_neurons_per_session": VALIDATION_CONFIG.max_neurons_per_session,
        }
    }


def verify_data_structure() -> Dict[str, Any]:
    """
    Verify that the expected data structure exists and return a report.
    """
    data_dir = FILESYSTEM_CONFIG.data_dir
    
    report = {
        "data_dir": str(data_dir),
        "data_dir_exists": data_dir.exists(),
        "sessions_found": [],
        "sessions_with_mat_files": [],
        "issues": []
    }
    
    if not data_dir.exists():
        report["issues"].append(f"Data directory does not exist: {data_dir}")
        return report
    
    # Look for session directories
    for item in data_dir.iterdir():
        if item.is_dir():
            report["sessions_found"].append(item.name)
            
            # Check for MAT files
            mat_files = list(item.glob(FILESYSTEM_CONFIG.mat_file_pattern))
            if mat_files:
                report["sessions_with_mat_files"].append({
                    "session": item.name,
                    "mat_files": [f.name for f in mat_files]
                })
            else:
                report["issues"].append(f"No MAT files found in session: {item.name}")
    
    if not report["sessions_found"]:
        report["issues"].append("No session directories found in data directory")
    
    return report


# Print configuration info when module is imported
def _print_config_info():
    """Print configuration information for debugging."""
    config = get_config_summary()
    print(f"📁 Data directory: {config['filesystem']['data_dir']}")
    print(f"✓ Data directory exists: {config['filesystem']['data_dir_exists']}")
    
    if not config['filesystem']['data_dir_exists']:
        print("⚠️  Data directory not found. You may need to:")
        print("   1. Set the NEUROSCIENCE_DATA_DIR environment variable")
        print("   2. Run from the project root directory")
        print("   3. Use set_data_directory() to specify the correct path")


# Print info when imported (can be disabled by setting environment variable)
if os.environ.get("NEUROSCIENCE_QUIET_CONFIG") != "1":
    _print_config_info()