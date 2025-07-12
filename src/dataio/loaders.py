# loaders.py

import scipy.io as sio
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
from tqdm import tqdm
import joblib
import logging
from dataclasses import dataclass

from .data_structures import SessionData, NeuronMetadata, TrialInfo, ValidationError
from .config          import DATA_CONFIG, EXPERIMENT_CONFIG, FILESYSTEM_CONFIG, VALIDATION_CONFIG


# Set up logging
logger = logging.getLogger(__name__)


class LoadingError(Exception):
    """Custom exception for data loading errors."""
    pass


class MatFileValidator:
    """Validator for MAT file structure and content."""
    
    REQUIRED_FIELDS = ['allxc', 'allyc', 'allzc', 'zDFF', 'stimInfo']
    OPTIONAL_FIELDS = ['BFinfo', 'CellInfo', 'CorrInfo', 'allZCorrInfo', 
                      'selectZCorrInfo', 'zStuff', 'exptVars']
    
    @classmethod
    def validate_mat_structure(cls, mat_data: Dict[str, Any]) -> None:
        """Validate that MAT file has required structure."""
        logger.debug("--- Starting MAT file structure validation ---")
        missing_fields = []
        logger.debug(f"Checking for required fields: {cls.REQUIRED_FIELDS}")
        for field in cls.REQUIRED_FIELDS:
            if field not in mat_data:
                missing_fields.append(field)
        
        if missing_fields:
            error_msg = f"Missing required fields in MAT file: {missing_fields}"
            logger.error(error_msg)
            raise LoadingError(error_msg)
        logger.debug("All required fields are present.")
    
    @classmethod
    def validate_plane_consistency(cls, mat_data: Dict[str, Any]) -> None:
        """Validate that all arrays have consistent plane structure."""
        logger.debug("--- Starting MAT file plane consistency validation ---")
        coord_fields = ['allxc', 'allyc', 'allzc']
        fields_to_check = coord_fields + ['zDFF', 'stimInfo']
        plane_counts = {}
        
        logger.debug(f"Checking for consistent plane counts across: {fields_to_check}")
        for field in fields_to_check:
            if field in mat_data:
                data = mat_data[field]
                if hasattr(data, '__len__'):
                    plane_counts[field] = len(data)
                    logger.debug(f"Field '{field}' has length (plane count): {len(data)}")
                else:
                    logger.warning(f"Field '{field}' is present but does not have a length. Cannot check for plane consistency.")
        
        unique_counts = set(plane_counts.values())
        logger.debug(f"Found unique plane counts: {unique_counts}")
        if len(unique_counts) > 1:
            error_msg = f"Inconsistent plane counts across fields: {plane_counts}"
            logger.error(error_msg)
            raise LoadingError(error_msg)
        logger.debug("Plane counts are consistent across all checked fields.")


@dataclass
class RawMatData:
    """Container for raw MAT file data with validation."""
    coords: Dict[str, Any]
    activity: Any
    stim_info: Any
    bf_info: Optional[Any] = None
    cell_info: Optional[Any] = None
    corr_info: Optional[Any] = None
    z_corr_info: Optional[Any] = None
    select_corr_info: Optional[Any] = None
    z_stuff: Optional[Any] = None
    expt_vars: Optional[Any] = None
    raw_mat: Optional[Dict[str, Any]] = None


class MatFileParser:
    """Handles parsing of MAT files with comprehensive error handling."""
    
    def __init__(self, validate: bool = True):
        self.validate = validate
    
    def parse_mat_file(self, mat_path: Path) -> RawMatData:
        """Extract ALL useful data from MAT file with validation."""
        logger.info(f"--- Parsing MAT file: {mat_path.name} ---")
        
        try:
            logger.debug(f"Attempting to load {mat_path} with scipy.io.loadmat")
            mat = sio.loadmat(
                mat_path, 
                squeeze_me=True, 
                struct_as_record=False,
                mat_dtype=True
            )
            logger.info(f"Successfully loaded MAT file. Top-level keys: {list(mat.keys())}")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load MAT file {mat_path}. Error: {str(e)}")
            raise LoadingError(f"Failed to load MAT file {mat_path}: {str(e)}")
        
        if self.validate:
            logger.info("Performing validation on MAT file structure.")
            MatFileValidator.validate_mat_structure(mat)
            MatFileValidator.validate_plane_consistency(mat)
            logger.info("MAT file structure validation passed.")
        else:
            logger.warning("Skipping MAT file validation.")
        
        try:
            logger.debug("Extracting data from MAT structure into RawMatData object.")
            raw_data = RawMatData(
                coords={
                    'x': mat.get('allxc'), 
                    'y': mat.get('allyc'), 
                    'z': mat.get('allzc')
                },
                activity=mat.get('zDFF'),
                stim_info=mat.get('stimInfo'),
                bf_info=mat.get('BFinfo'),
                cell_info=mat.get('CellInfo'),
                corr_info=mat.get('CorrInfo'),
                z_corr_info=mat.get('allZCorrInfo'),
                select_corr_info=mat.get('selectZCorrInfo'),
                z_stuff=mat.get('zStuff'),
                expt_vars=mat.get('exptVars'),
                raw_mat=mat if logger.isEnabledFor(logging.DEBUG) else None
            )
            logger.debug("Successfully created RawMatData object.")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to extract data from MAT file into RawMatData object. Error: {str(e)}")
            raise LoadingError(f"Failed to extract data from MAT file: {str(e)}")
        
        return raw_data


class NeuronExtractor:
    """Handles extraction of neuron metadata from raw MAT data."""
    
    def __init__(self, config: Optional[object] = None):
        self.config = config or DATA_CONFIG
    
    def extract_neuron_metadata(self, raw_data: RawMatData, session_id: str) -> List[NeuronMetadata]:
        """Extract comprehensive neuron metadata including BF and quality."""
        logger.info(f"--- Starting Neuron Extraction for session {session_id} ---")
        logger.debug(f"Using valid planes from config: {self.config.valid_planes}")
        neurons = []
        
        if raw_data.coords['x'] is None or not hasattr(raw_data.coords['x'], '__len__'):
            logger.error(f"Coordinate data ('allxc') is missing or not iterable for session {session_id}. Cannot extract neurons.")
            return []
        
        num_planes_in_file = len(raw_data.coords['x'])
        logger.info(f"Found data for {num_planes_in_file} planes in the file.")

        for plane_idx in range(num_planes_in_file):
            plane_num = plane_idx + 1
            logger.debug(f"Processing plane {plane_num} (python index {plane_idx}).")
            if plane_num not in self.config.valid_planes:
                logger.info(f"Skipping plane {plane_num} as it is not in the configured valid_planes list.")
                continue
            
            try:
                logger.debug(f"Extracting neurons for plane {plane_num}.")
                plane_neurons = self._extract_plane_neurons(
                    raw_data, plane_idx, session_id
                )
                logger.info(f"Extracted {len(plane_neurons)} neurons from plane {plane_num}.")
                neurons.extend(plane_neurons)
            except Exception as e:
                logger.error(f"Failed to extract neurons from plane {plane_num} due to an unexpected error: {str(e)}", exc_info=True)
                continue
        
        logger.info(f"--- Neuron Extraction Complete for {session_id}. Total neurons extracted: {len(neurons)} ---")
        return neurons
    
    def _extract_plane_neurons(self, raw_data: RawMatData, plane_idx: int, 
                              session_id: str) -> List[NeuronMetadata]:
        """Extract neurons from a single plane."""
        x_coords = raw_data.coords['x'][plane_idx]
        y_coords = raw_data.coords['y'][plane_idx]
        z_coords = raw_data.coords['z'][plane_idx]
        
        # Handle nested array structure and detect truly empty planes
        if hasattr(x_coords, '__len__') and len(x_coords) > 0:
            # Extract the actual coordinate arrays (handle MATLAB cell structure)
            if hasattr(x_coords[0], '__len__'):
                x_coords = x_coords[0]
                y_coords = y_coords[0] 
                z_coords = z_coords[0]
        
        # Check if plane is truly empty using array size
        if not hasattr(x_coords, '__len__') or np.size(x_coords) == 0:
            logger.info(f"Plane {plane_idx+1} is empty (no neurons).")
            return []
        
        # Handle scalar coordinates by converting to single-element arrays
        if not hasattr(x_coords, '__len__'):
            logger.warning(f"Coordinate data for plane {plane_idx+1} is a scalar. Converting to a single-element list.")
            x_coords, y_coords, z_coords = [x_coords], [y_coords], [z_coords]
        
        n_neurons = len(x_coords)
        if n_neurons == 0:
            logger.info(f"No neurons found in plane {plane_idx+1}.")
            return []
        
        bf_data = self._safe_get_plane_data(raw_data.bf_info, plane_idx)
        cell_data = self._safe_get_plane_data(raw_data.cell_info, plane_idx)
        
        neurons = []
        for idx in range(n_neurons):
            try:
                neuron = self._create_neuron_metadata(
                    session_id, plane_idx, idx, n_neurons,
                    x_coords, y_coords, z_coords,
                    bf_data, cell_data
                )
                neurons.append(neuron)
            except ValidationError as e:
                logger.warning(f"Skipping invalid neuron {session_id}_p{plane_idx+1}_n{idx} due to validation error: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error creating metadata for neuron {session_id}_p{plane_idx+1}_n{idx}: {str(e)}", exc_info=True)
        
        return neurons

    def _safe_get_plane_data(self, data_array: Any, plane_idx: int) -> Optional[Any]:
        """Safely extract data for a specific plane."""
        if data_array is None:
            return None
        try:
            if hasattr(data_array, '__len__') and len(data_array) > plane_idx:
                return data_array[plane_idx]
            elif not hasattr(data_array, '__len__'):
                return data_array
            else:
                logger.warning(f"Data array has length {len(data_array)} but tried to access index {plane_idx}. Returning None.")
                return None
        except (IndexError, TypeError) as e:
            logger.warning(f"Could not safely get data for plane index {plane_idx}. Error: {e}")
            return None
    
    def _coerce_bf_value(self, bf_item: Any) -> float:
        """
        Safely coerce a best-frequency (BF) value from various MAT file formats.
        Handles mat_struct objects (with a 'bf' field) and standard scalars.
        """
        if hasattr(bf_item, '__dict__') and hasattr(bf_item, 'bf'):
            return float(getattr(bf_item, 'bf', np.nan))
        try:
            return float(bf_item)
        except (ValueError, TypeError):
            logger.debug(f"Could not coerce BF item to float. Item: {bf_item}")
            return np.nan

    def _create_neuron_metadata(self, session_id: str, plane_idx: int, idx: int, 
                            n_neurons: int, x_coords: List, y_coords: List, z_coords: List,
                            bf_data: Optional[Any], cell_data: Optional[Any]) -> NeuronMetadata:
        """Create NeuronMetadata object with error handling."""
        x = float(x_coords[idx] if n_neurons > 1 else x_coords[0])
        y = float(y_coords[idx] if n_neurons > 1 else y_coords[0])
        z = float(z_coords[idx] if n_neurons > 1 else z_coords[0])
        
        neuron = NeuronMetadata(
            global_id=f"{session_id}_p{plane_idx+1}_n{idx}",
            session=session_id, plane=plane_idx + 1, local_idx=idx,
            x=x, y=y, z=z
        )
        
        # Extract best frequency - FIX: Access the BFval field from the struct
        if bf_data is not None:
            try:
                # Check if bf_data is a MATLAB struct with BFval field
                if hasattr(bf_data, 'BFval'):
                    # Access the BFval array from the struct
                    bf_values = bf_data.BFval
                    logger.debug(f"Found BFval field with shape: {getattr(bf_values, 'shape', 'unknown')}")
                    
                    # Now extract the value for this neuron
                    if hasattr(bf_values, '__len__') and len(bf_values) > idx:
                        bf_value = float(bf_values[idx])
                    elif hasattr(bf_values, '__len__') and len(bf_values) == 1:
                        # Single value for all neurons in this plane
                        bf_value = float(bf_values[0])
                    else:
                        # Scalar value
                        bf_value = float(bf_values)
                    
                    # Validate and assign
                    if not np.isnan(bf_value) and bf_value != self.config.bad_sentinel_value:
                        neuron.best_frequency = bf_value
                        logger.debug(f"Set best_frequency={bf_value} for neuron {neuron.global_id}")
                    else:
                        logger.debug(f"Neuron {neuron.global_id} has sentinel or NaN value for best_frequency.")
                        
                else:
                    # bf_data might be a direct array (not a struct)
                    logger.debug(f"BF data is not a struct with BFval field. Type: {type(bf_data)}")
                    # Try the original approach as fallback
                    item_to_coerce = bf_data[idx] if hasattr(bf_data, '__len__') and len(bf_data) > idx else bf_data
                    bf_value = self._coerce_bf_value(item_to_coerce)
                    
                    if not np.isnan(bf_value) and bf_value != self.config.bad_sentinel_value:
                        neuron.best_frequency = bf_value
                    else:
                        logger.debug(f"Neuron {neuron.global_id} has sentinel or NaN value for best_frequency.")
                        
            except (TypeError, IndexError, AttributeError) as e:
                logger.warning(f"Could not extract best_frequency for neuron {neuron.global_id}. Error: {e}")

        if cell_data is not None:
            self._extract_quality_metrics(neuron, cell_data, idx)
        
        return neuron
    def _extract_quality_metrics(self, neuron: NeuronMetadata, cell_data: Any, idx: int) -> None:
        """Extract quality metrics from cell data."""
        metrics = ['quality', 'roi_size', 'snr']
        for metric in metrics:
            try:
                if hasattr(cell_data, metric):
                    data = getattr(cell_data, metric)
                    if hasattr(data, '__len__') and len(data) > idx:
                        value = float(data[idx])
                        attr_name = "quality_score" if metric == 'quality' else metric
                        setattr(neuron, attr_name, value)
                else:
                    logger.debug(f"Metric '{metric}' not found in cell_data for neuron {neuron.global_id}")
            except (ValueError, TypeError, IndexError, AttributeError) as e:
                logger.warning(f"Could not extract metric '{metric}' for neuron {neuron.global_id}. Error: {e}")


class TrialExtractor:
    """Handles extraction of trial information from raw MAT data."""

    def extract_trial_info(self, raw_data: RawMatData) -> List[TrialInfo]:
        """Extract detailed trial information with validation."""
        logger.info(f"--- Starting Trial Extraction ---")
        
        # Find first valid stim_info
        stim_info = self._find_valid_stim_info(raw_data.stim_info)

        if stim_info is None:
            logger.error("FATAL: No valid stimulus information structure (stim_info) could be found in any plane. Trial extraction aborted.")
            warnings.warn("No valid stimulus information found")
            return []

        try:
            logger.debug("Parsing trials from the found valid stim_info object.")
            return self._parse_trials_from_stim_info(stim_info)
        except Exception as e:
            logger.error(f"Failed to parse trial information from stim_info object: {str(e)}", exc_info=True)
            return []

    def _find_valid_stim_info(self, stim_info_array: Any) -> Optional[Any]:
        """
        Find the first valid stimulus info object.
        Note: Per the README, stimInfo[0] is always empty and actual data starts from stimInfo[1]
        """
        logger.debug("--- Searching for a valid stim_info structure ---")
        if stim_info_array is None:
            logger.warning("stim_info_array is None. Cannot find a valid structure.")
            return None

        if not hasattr(stim_info_array, '__len__'):
            logger.warning("stim_info_array is not an array, treating it as a single struct.")
            if hasattr(stim_info_array, 'Trialindicies') and getattr(stim_info_array.Trialindicies, "shape", (0,))[0] > 0:
                 logger.info("Found valid stim_info in single struct.")
                 return stim_info_array
            logger.warning("Single stim_info struct is invalid or empty.")
            return None

        logger.debug(f"stim_info_array has {len(stim_info_array)} elements (planes). Iterating to find a valid one...")
        
        # Start from index 0 but expect plane 0 to be empty per README
        for idx in range(len(stim_info_array)):
            logger.debug(f"Checking stim_info at index {idx}...")
            si = stim_info_array[idx]
            if si is None:
                logger.debug(f"Index {idx} is None. Skipping.")
                continue
            
            if not hasattr(si, "Trialindicies"):
                logger.debug(f"Index {idx} has no 'Trialindicies' attribute. Skipping.")
                continue

            trial_indices = getattr(si, 'Trialindicies')
            if not hasattr(trial_indices, 'shape'):
                logger.debug(f"Index {idx} has 'Trialindicies' but it has no 'shape' attribute. Skipping.")
                continue

            logger.debug(f"Index {idx} has 'Trialindicies' with shape: {trial_indices.shape}")

            # Valid structure should have non-empty Trialindicies array
            if len(trial_indices.shape) == 2 and trial_indices.shape[0] > 0:
                logger.info(f"SUCCESS: Found valid stim_info at index {idx} with Trialindicies shape {trial_indices.shape}. Using this for trial extraction.")
                return si
            elif len(trial_indices.shape) == 1 and trial_indices.shape[0] > 0:
                logger.warning(f"Found 1D Trialindicies at index {idx}. This might be an issue. Attempting to use it anyway.")
                return si
            else:
                 logger.debug(f"Index {idx} has 'Trialindicies' but it is empty or has an invalid shape. Skipping.")

        logger.error("--- Search Complete: No valid stimulus info found in any plane of the stim_info_array. ---")
        return None

    def _parse_trials_from_stim_info(self, stim_info: Any) -> List[TrialInfo]:
        """Parse trials from stimulus info object."""
        logger.info("--- Parsing individual trials from stim_info ---")
        required_fields = ['Trialindicies', 'framespertrial']
        for field in required_fields:
            if not hasattr(stim_info, field):
                error_msg = f"stim_info object is missing required field: '{field}'"
                logger.error(error_msg)
                raise LoadingError(error_msg)
        
        trial_indices = stim_info.Trialindicies
        frames_per_trial = int(stim_info.framespertrial)
        
        logger.info(f"Found Trialindicies shape: {trial_indices.shape}. Expecting {trial_indices.shape[0]} trials.")
        logger.info(f"Frames per trial: {frames_per_trial}")
        
        # Get frequency and level arrays with fallback handling
        freqs = getattr(stim_info, 'Freqs', None)
        levels = getattr(stim_info, 'Levels', None)
        if levels is None:
            levels = getattr(stim_info, 'Amplitudes', None)  # Fallback for older files

        # Validate metadata consistency
        n_expected_trials = trial_indices.shape[0]
        self._validate_metadata_consistency(freqs, levels, n_expected_trials)

        trials = []
        
        # Log sample Trialindicies data for debugging
        logger.debug(f"Sample Trialindicies data (first 5 rows):")
        for i in range(min(5, n_expected_trials)):
            logger.debug(f"  Row {i}: {trial_indices[i]}")
        
        for idx in range(n_expected_trials):
            try:
                trial_data = trial_indices[idx]
                logger.debug(f"Parsing trial index {idx}. Raw Trialindicies row: {trial_data}")

                # Calculate actual frame ranges sequentially
                # NOTE: Trialindicies contains condition indices, NOT frame numbers!
                start_frame = idx * frames_per_trial
                end_frame = start_frame + frames_per_trial
                
                logger.debug(f"Trial {idx}: Calculated frames {start_frame}-{end_frame} (duration: {frames_per_trial})")

                # Extract frequency with bounds checking
                frequency = 1.0  # default fallback
                if freqs is not None and hasattr(freqs, '__len__') and len(freqs) > idx:
                    frequency = float(freqs[idx])
                
                if frequency <= 0:
                    logger.warning(f"Invalid frequency {frequency} for trial {idx}, using default 1.0")
                    frequency = 1.0

                # Extract level with bounds checking
                level_val = 0.0  # default fallback
                if levels is not None and hasattr(levels, '__len__') and len(levels) > idx:
                    level_val = float(levels[idx])

                # Generate condition label
                condition = self._generate_condition_label(freqs, levels, idx)

                trial = TrialInfo(
                    trial_idx=idx,
                    frequency=frequency,
                    level=level_val,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    condition=condition
                )
                
                logger.debug(f"Successfully created trial {idx}: {trial.frequency}Hz, frames {start_frame}-{end_frame}")
                trials.append(trial)
                
            except ValidationError as e:
                logger.error(f"SKIPPING trial {idx} due to a ValidationError: {e}")
                continue
            except (ValueError, IndexError, TypeError) as e:
                logger.error(f"SKIPPING trial {idx} due to a parsing error: {e}", exc_info=True)
                continue

        logger.info(f"--- Trial Parsing Complete. Successfully parsed {len(trials)} trials out of {n_expected_trials} expected. ---")
        
        if len(trials) == 0:
            logger.error("CRITICAL: No valid trials were created! Check the frame calculation logic.")
        
        return trials

    def _validate_metadata_consistency(self, freqs: Optional[Any], levels: Optional[Any], 
                                     n_trials: int) -> None:
        """Validate consistency between different metadata fields."""
        logger.debug("--- Validating metadata consistency ---")
        
        if freqs is not None and hasattr(freqs, '__len__'):
            if len(freqs) != n_trials:
                logger.warning(f"Frequency array length ({len(freqs)}) doesn't match trial count ({n_trials})")
        
        if levels is not None and hasattr(levels, '__len__'):
            if len(levels) != n_trials:
                logger.warning(f"Level array length ({len(levels)}) doesn't match trial count ({n_trials})")

    def _generate_condition_label(self, freqs: Optional[Any], levels: Optional[Any],
                                 idx: int) -> str:
        """Generate a condition label for the trial."""
        if freqs is not None and hasattr(freqs, '__len__') and len(freqs) > idx \
        and levels is not None and hasattr(levels, '__len__') and len(levels) > idx:
            try:
                return f"{int(freqs[idx])}Hz_{int(levels[idx])}dB"
            except (ValueError, IndexError, TypeError):
                logger.debug(f"Could not form condition label for trial {idx} due to non-numeric values.")
                pass
        return "unknown"


class CorrelationExtractor:
    """Handles extraction of correlation matrices from raw MAT data."""
    
    def extract_correlations(self, raw_data: RawMatData) -> Tuple[Optional[np.ndarray], 
                                                                Optional[np.ndarray], 
                                                                Dict[str, Any]]:
        """Extract pre-computed correlation matrices."""
        logger.info("--- Starting Correlation Matrix Extraction ---")
        signal_corr = None
        noise_corr = None
        metadata = {}
        
        # Define the order of sources to check for correlation data
        corr_sources = [
            ('corr_info', raw_data.corr_info),
            ('z_corr_info', raw_data.z_corr_info),
            ('select_corr_info', raw_data.select_corr_info)
        ]
        
        logger.debug(f"Checking for correlation data in sources: {[s[0] for s in corr_sources]}")
        
        for source_name, corr_data in corr_sources:
            if corr_data is None:
                logger.debug(f"Source '{source_name}' is not present in the MAT file.")
                continue
            
            logger.info(f"Found potential correlation data in source: '{source_name}'. Attempting to extract.")
            try:
                signal, noise, meta = self._extract_from_source(corr_data, source_name)
                
                # We only take the first valid matrix found for each type
                if signal_corr is None and signal is not None:
                    logger.info(f"Extracted 'signal_correlation' from '{source_name}' with shape {signal.shape}.")
                    signal_corr = signal
                if noise_corr is None and noise is not None:
                    logger.info(f"Extracted 'noise_correlation' from '{source_name}' with shape {noise.shape}.")
                    noise_corr = noise
                    
                metadata.update(meta)
            except Exception as e:
                logger.error(f"Failed to extract correlations from source '{source_name}': {str(e)}", exc_info=True)
                continue
        
        if signal_corr is None:
            logger.warning("No signal correlation matrix was found in any source.")
        if noise_corr is None:
            logger.warning("No noise correlation matrix was found in any source.")

        logger.info("--- Correlation Matrix Extraction Complete ---")
        return signal_corr, noise_corr, metadata
    
    def _extract_from_source(self, corr_data: Any, source_name: str) -> Tuple[Optional[np.ndarray], 
                                                                             Optional[np.ndarray], 
                                                                             Dict[str, Any]]:
        """Extract correlations from a specific source."""
        signal_corr = None
        noise_corr = None
        metadata = {}
        
        # Check for signal correlation
        if hasattr(corr_data, 'signal_correlation'):
            logger.debug(f"Found 'signal_correlation' attribute in '{source_name}'.")
            signal_corr = np.array(corr_data.signal_correlation)
            metadata['signal_source'] = source_name
        
        # Check for noise correlation
        if hasattr(corr_data, 'noise_correlation'):
            logger.debug(f"Found 'noise_correlation' attribute in '{source_name}'.")
            noise_corr = np.array(corr_data.noise_correlation)
            metadata['noise_source'] = source_name
        
        # Extract additional metadata if it exists
        if hasattr(corr_data, 'method'):
            method_val = str(corr_data.method)
            metadata['correlation_method'] = method_val
            logger.debug(f"Found correlation method: {method_val}")
        if hasattr(corr_data, 'params'):
            params_val = dict(corr_data.params)
            metadata['correlation_params'] = params_val
            logger.debug(f"Found correlation params: {params_val}")
        
        return signal_corr, noise_corr, metadata


class ActivityMatrixBuilder:
    """Handles construction of the activity matrix from raw data."""
    
    def __init__(self, config: Optional[object] = None):
        self.config = config or DATA_CONFIG
    
    def build_activity_matrix(self, raw_data: RawMatData) -> np.ndarray:
        """Build activity matrix from valid planes, ensuring time axis consistency."""
        activity_planes = []
        
        if raw_data.activity is None or not hasattr(raw_data.activity, '__len__'):
            logger.warning("Activity data ('zDFF') is missing or not in the expected array format.")
            return np.array([])

        # Ensure consistent time axis across planes
        for plane_idx in range(len(raw_data.activity)):
            plane_num = plane_idx + 1
            if plane_num not in self.config.valid_planes:
                continue
            
            plane_activity = raw_data.activity[plane_idx]
            
            # Check if this is actually a nested structure (like coordinates)
            # or already a 2D array (like zDFF typically is)
            if hasattr(plane_activity, 'shape') and hasattr(plane_activity, 'ndim'):
                # It's already a numpy array
                if plane_activity.ndim == 2:
                    # Already 2D, use as is
                    pass
                elif plane_activity.ndim == 1:
                    # 1D array - could be empty or single neuron
                    if plane_activity.size == 0:
                        logger.debug(f"Plane {plane_num} activity is empty. Skipping.")
                        continue
                    else:
                        plane_activity = plane_activity.reshape(1, -1)
                elif plane_activity.ndim == 0:
                    # Scalar - skip
                    logger.debug(f"Plane {plane_num} activity is scalar. Skipping.")
                    continue
            else:
                # Handle nested structure (like coordinate arrays)
                if hasattr(plane_activity, '__len__') and len(plane_activity) > 0:
                    # Check if it's a nested array by looking at the first element
                    first_elem = plane_activity[0]
                    if hasattr(first_elem, 'shape') and hasattr(first_elem, 'ndim'):
                        # First element is an array, so this is nested
                        if first_elem.ndim >= 2:
                            # Extract the nested array
                            plane_activity = first_elem
                        else:
                            logger.debug(f"Unexpected nested structure in plane {plane_num}. Skipping.")
                            continue
                    else:
                        # Not nested, convert to numpy array if needed
                        plane_activity = np.array(plane_activity)
            
            # Skip empty planes using array size
            if plane_activity is None or np.size(plane_activity) == 0:
                logger.debug(f"Plane {plane_num} activity is empty. Skipping.")
                continue
            
            # Final check - ensure it's 2D
            if plane_activity.ndim == 1:
                plane_activity = plane_activity.reshape(1, -1)
            elif plane_activity.ndim != 2:
                logger.warning(f"Plane {plane_num} has unexpected dimensionality: {plane_activity.ndim}D. Skipping.")
                continue
            
            cleaned_activity = self._clean_activity_data(plane_activity)
            activity_planes.append(cleaned_activity)
        
        if not activity_planes:
            logger.warning("No valid activity data found in any of the configured valid planes.")
            return np.array([])

        # Check for temporal consistency before stacking
        first_plane_timepoints = activity_planes[0].shape[1]
        for i, plane_activity in enumerate(activity_planes[1:], 1):
            if plane_activity.shape[1] != first_plane_timepoints:
                error_msg = (
                    f"Inconsistent number of timepoints across planes. "
                    f"Plane 0 has {first_plane_timepoints} timepoints, but "
                    f"plane {i} has {plane_activity.shape[1]} timepoints. "
                    f"Cannot stack to form a valid activity matrix."
                )
                logger.error(error_msg)
                raise LoadingError(error_msg)
        
        logger.info(f"All {len(activity_planes)} valid planes have a consistent time axis of {first_plane_timepoints} points.")
        
        return np.vstack(activity_planes)

    def _clean_activity_data(self, activity_data: np.ndarray) -> np.ndarray:
        """Clean activity data by replacing sentinel values and invalid data with NaN."""
        cleaned = activity_data.astype(float).copy()
        
        # Track all cleaning operations
        cleaning_stats = {
            'sentinel_values': 0,
            'extreme_negative': 0,
            'inf_values': 0,
            'extreme_positive': 0
        }
        
        # 1. Replace known sentinel value (-100)
        sentinel_mask = cleaned == self.config.bad_sentinel_value
        cleaning_stats['sentinel_values'] = np.sum(sentinel_mask)
        cleaned[sentinel_mask] = np.nan
        
        # 2. Replace other common sentinel values
        # Common placeholders: -32768 (int16 min), -2147483648 (int32 min), -9999, etc.
        extreme_negative_mask = cleaned < -1000  # Anything below -1000 is likely invalid for dF/F
        cleaning_stats['extreme_negative'] = np.sum(extreme_negative_mask) - cleaning_stats['sentinel_values']
        cleaned[extreme_negative_mask] = np.nan
        
        # 3. Replace Inf values
        inf_mask = np.isinf(cleaned)
        cleaning_stats['inf_values'] = np.sum(inf_mask)
        cleaned[inf_mask] = np.nan
        
        # 4. Replace unrealistic positive values
        # dF/F values above 1000% (10x) are extremely unlikely
        extreme_positive_mask = cleaned > 1000
        cleaning_stats['extreme_positive'] = np.sum(extreme_positive_mask)
        cleaned[extreme_positive_mask] = np.nan
        
        # Log cleaning summary
        total_cleaned = sum(cleaning_stats.values())
        if total_cleaned > 0:
            logger.info(f"Cleaned {total_cleaned} invalid values from activity data:")
            for reason, count in cleaning_stats.items():
                if count > 0:
                    logger.info(f"  - {reason}: {count}")
        
        # Additional validation: warn if too many values were cleaned
        total_values = cleaned.size
        cleaned_percentage = (total_cleaned / total_values) * 100
        if cleaned_percentage > 10:
            logger.warning(f"Cleaned {cleaned_percentage:.1f}% of activity data - this seems high!")
        
        return cleaned


class SessionDataValidator:
    """Validates consistency between trials and activity data."""
    
    def validate_trials_activity_alignment(self, trials: List[TrialInfo], 
                                         activity_matrix: np.ndarray) -> Dict[str, Any]:
        """Validate that trial definitions align with activity data length."""
        logger.info("--- Validating trial-activity alignment ---")
        
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'trimmed_trials': False,
            'original_trial_count': len(trials),
            'valid_trial_count': len(trials)
        }
        
        if len(trials) == 0:
            validation_results['warnings'].append("No trials to validate")
            return validation_results
        
        if activity_matrix.size == 0:
            validation_results['errors'].append("Activity matrix is empty")
            validation_results['is_valid'] = False
            return validation_results
        
        total_frames = activity_matrix.shape[1]
        logger.info(f"Activity matrix has {total_frames} frames")
        
        # Check for trials extending beyond data
        invalid_trials = []
        for trial in trials:
            if trial.end_frame > total_frames:
                invalid_trials.append(trial.trial_idx)
        
        if invalid_trials:
            validation_results['errors'].append(
                f"Trials {invalid_trials} extend beyond activity matrix ({total_frames} frames)"
            )
            validation_results['is_valid'] = False
        
        # Check expected vs actual frame counts
        if trials:
            frames_per_trial = trials[0].duration_frames
            expected_total = len(trials) * frames_per_trial
            
            logger.info(f"Expected total frames: {expected_total}, Actual: {total_frames}")
            
            if expected_total > total_frames:
                # Calculate how many complete trials we can support
                max_complete_trials = total_frames // frames_per_trial
                validation_results['warnings'].append(
                    f"Data truncated: only {max_complete_trials} complete trials out of {len(trials)} expected"
                )
                validation_results['valid_trial_count'] = max_complete_trials
                validation_results['trimmed_trials'] = True
            elif total_frames > expected_total:
                excess_frames = total_frames - expected_total
                validation_results['warnings'].append(
                    f"Extra {excess_frames} frames beyond expected trials"
                )
        
        return validation_results


class SessionLoader:
    """Main class for loading complete sessions."""
    
    def __init__(self, validate: bool = True):
        self.validate = validate
        self.parser = MatFileParser(validate=validate)
        self.neuron_extractor = NeuronExtractor()
        self.trial_extractor = TrialExtractor()
        self.correlation_extractor = CorrelationExtractor()
        self.activity_builder = ActivityMatrixBuilder()
        self.data_validator = SessionDataValidator()
    
    def load_session_complete(self, session_path: Path, use_cache: bool = True) -> SessionData:
        """Load complete session with all metadata and validation."""
        cache_path = FILESYSTEM_CONFIG.cache_dir / f"{session_path.name}{FILESYSTEM_CONFIG.cache_file_suffix}"
        
        if use_cache and cache_path.exists():
            logger.info(f"Loading from cache: {session_path.name}")
            return joblib.load(cache_path)
        
        # Find MAT file
        mat_files = list(session_path.glob(FILESYSTEM_CONFIG.mat_file_pattern))
        if not mat_files:
            # Fallback for testing dummy mat files
            mat_files = list(session_path.glob("*.mat"))
        
        if not mat_files:
            raise LoadingError(f"No MAT file found in {session_path}")
        
        try:
            # Parse raw data
            raw_data = self.parser.parse_mat_file(mat_files[0])
            
            # Extract components
            neurons = self.neuron_extractor.extract_neuron_metadata(raw_data, session_path.name)
            trials = self.trial_extractor.extract_trial_info(raw_data)
            signal_corr, noise_corr, corr_meta = self.correlation_extractor.extract_correlations(raw_data)
            activity_matrix = self.activity_builder.build_activity_matrix(raw_data)
            
            # Validate trial-activity alignment and trim if necessary
            validation_results = self.data_validator.validate_trials_activity_alignment(
                trials, activity_matrix
            )
            
            if validation_results['trimmed_trials']:
                valid_count = validation_results['valid_trial_count']
                trials = trials[:valid_count]
                logger.warning(f"Trimmed trials to {valid_count} to match available data")
            
            # Extract preprocessing parameters and experiment variables
            preproc_params = self._extract_preprocessing_params(raw_data.z_stuff)
            expt_vars = self._extract_experiment_vars(raw_data.expt_vars)
            
            # Create session object
            session = SessionData(
                session_id=session_path.name,
                experiment_vars=expt_vars,
                neurons=neurons,
                activity_matrix=activity_matrix,
                trials=trials,
                signal_correlations=signal_corr,
                noise_correlations=noise_corr,
                correlation_metadata=corr_meta,
                preprocessing_params=preproc_params,
                frame_rate=EXPERIMENT_CONFIG.volume_scan_rate_hz
            )
            
            # Cache for future use
            if use_cache:
                cache_path.parent.mkdir(exist_ok=True)
                joblib.dump(session, cache_path)
                logger.info(f"Cached to {cache_path}")
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to load session {session_path.name}: {str(e)}")
            raise LoadingError(f"Failed to load session {session_path.name}: {str(e)}")
    
    def _extract_preprocessing_params(self, z_stuff: Optional[Any]) -> Dict[str, Any]:
        """Extract preprocessing parameters."""
        params = {}
        if z_stuff is not None and hasattr(z_stuff, '__dict__'):
            params = {k: v for k, v in z_stuff.__dict__.items() 
                     if not k.startswith('_')}
        return params
    
    def _extract_experiment_vars(self, expt_vars: Optional[Any]) -> Dict[str, Any]:
        """Extract experiment variables."""
        vars_dict = {}
        if expt_vars is not None and hasattr(expt_vars, '__dict__'):
            vars_dict = {k: v for k, v in expt_vars.__dict__.items() 
                        if not k.startswith('_')}
        return vars_dict


class SessionValidator:
    """Validates loaded session data against expected values."""
    
    def __init__(self, config: Optional[object] = None):
        self.config = config or VALIDATION_CONFIG
    
    def validate_against_paper(self, session: SessionData) -> Dict[str, bool]:
        """Validate session against expected values from paper."""
        checks = {}
        
        # Check neuron counts are reasonable
        n_neurons = len(session.neurons)
        checks['neuron_count_valid'] = (
            self.config.min_neurons_per_session <= n_neurons <= self.config.max_neurons_per_session
        )
        
        # Check if we have BF data (mentioned in paper)
        bf_neurons = sum(1 for n in session.neurons if n.has_best_frequency)
        checks['has_bf_data'] = bf_neurons > 0
        
        # Check if we have pre-computed correlations
        checks['has_signal_corr'] = session.signal_correlations is not None
        checks['has_noise_corr'] = session.noise_correlations is not None
        
        # Check trial structure
        checks['trial_structure_valid'] = len(session.trials) in EXPERIMENT_CONFIG.expected_trial_counts
        
        # Check frame rate
        checks['frame_rate_valid'] = abs(
            session.frame_rate - EXPERIMENT_CONFIG.volume_scan_rate_hz
        ) < self.config.frame_rate_tolerance
        
        return checks


# Convenience functions for backward compatibility
def load_session_complete(session_path: Path, use_cache: bool = True) -> SessionData:
    """Load complete session with all metadata."""
    loader = SessionLoader()
    return loader.load_session_complete(session_path, use_cache)


def validate_against_paper(session: SessionData) -> Dict[str, bool]:
    """Validate session against expected values from paper."""
    validator = SessionValidator()
    return validator.validate_against_paper(session)