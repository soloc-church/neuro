import pytest
import numpy as np
import networkx as nx
import copy


from src.dataio.data_structures import SessionData
from src.dataio.processors import (
    ResponseProcessor,
    NetworkAnalyzer,
    TemporalAnalyzer,
    DimensionalityReducer,
    ResponseCharacteristics,
    NetworkMetrics,
    TemporalDynamics
)

# Mark all tests in this file as integration tests since they use real data
pytestmark = pytest.mark.integration


class TestResponseProcessor:
    """Tests for the ResponseProcessor class using real session data."""

    def test_analyze_trial_responses_structure_and_types(self, loaded_session_180_trials: SessionData):
        """
        Tests that analyze_trial_responses runs and returns a correctly structured
        dictionary of ResponseCharacteristics objects.
        """
        if not loaded_session_180_trials.trials:
            pytest.skip("Session has no trials, cannot test response processing.")

        processor = ResponseProcessor(frame_rate=loaded_session_180_trials.frame_rate)
        response_chars = processor.analyze_trial_responses(loaded_session_180_trials)

        assert isinstance(response_chars, dict)
        # Should have results for each valid neuron
        num_valid_neurons = sum(1 for n in loaded_session_180_trials.neurons if n.is_valid)
        assert len(response_chars) == num_valid_neurons

        # Check the output for one neuron
        first_neuron_id = next(iter(response_chars))
        first_neuron_char = response_chars[first_neuron_id]

        assert isinstance(first_neuron_char, ResponseCharacteristics)
        assert first_neuron_char.neuron_id == first_neuron_id
        assert isinstance(first_neuron_char.peak_response, float)
        assert isinstance(first_neuron_char.response_latency, float)
        assert isinstance(first_neuron_char.signal_to_noise, float)
        assert -1.0 <= first_neuron_char.response_reliability <= 1.0

    def test_compute_frequency_tuning_curves(self, loaded_session_180_trials: SessionData):
        """
        Tests that frequency tuning curves can be computed and have the correct structure.
        """
        if not loaded_session_180_trials.trials:
            pytest.skip("Session has no trials, cannot test frequency tuning.")

        processor = ResponseProcessor(frame_rate=loaded_session_180_trials.frame_rate)
        tuning_curves = processor.compute_frequency_tuning(loaded_session_180_trials)

        assert isinstance(tuning_curves, dict)
        assert len(tuning_curves) > 0  # Should find tuning for at least some neurons

        # Inspect the tuning curve for one neuron
        first_neuron_id = next(iter(tuning_curves))
        curve = tuning_curves[first_neuron_id]
        assert isinstance(curve, dict)

        # Keys should be frequencies (floats), values should be response magnitudes (floats)
        assert all(isinstance(k, float) for k in curve.keys())
        assert all(isinstance(v, float) for v in curve.values())


class TestNetworkAnalyzer:
    """Tests for the NetworkAnalyzer class using real session data."""

    @pytest.fixture(scope="class")
    def functional_network(self, loaded_session_180_trials: SessionData) -> nx.Graph:
        """Fixture to build a functional network once for all tests in this class."""
        # Use a lower threshold for testing to ensure the graph is not empty
        analyzer = NetworkAnalyzer(correlation_threshold=0.1)
        # Build network from raw activity, as pre-computed correlations may not exist
        network = analyzer.build_functional_network(loaded_session_180_trials, use_signal_correlations=False)
        return network

    def test_build_functional_network(self, functional_network: nx.Graph, loaded_session_180_trials: SessionData):
        """
        Tests the construction of a functional network graph.
        """
        assert isinstance(functional_network, nx.Graph)
        assert functional_network.number_of_nodes() == len(loaded_session_180_trials.neurons)
        assert functional_network.number_of_edges() > 0 # Expect some connections

        # Check that node attributes were correctly assigned from NeuronMetadata
        a_neuron_id = loaded_session_180_trials.neurons[0].global_id
        a_neuron_metadata = loaded_session_180_trials.get_neuron_by_id(a_neuron_id)

        assert functional_network.nodes[a_neuron_id]['plane'] == a_neuron_metadata.plane
        assert functional_network.nodes[a_neuron_id]['x'] == a_neuron_metadata.x

    def test_analyze_network_properties(self, functional_network: nx.Graph):
        """
        Tests the calculation of network-level metrics.
        """
        if functional_network.number_of_nodes() == 0:
            pytest.skip("Network is empty, cannot analyze properties.")

        analyzer = NetworkAnalyzer()
        metrics = analyzer.analyze_network_properties(functional_network)

        assert isinstance(metrics, NetworkMetrics)
        assert isinstance(metrics.clustering_coefficient, float)
        assert isinstance(metrics.path_length, float)
        assert isinstance(metrics.modularity, float)
        assert 0.0 <= metrics.clustering_coefficient <= 1.0

    def test_identify_network_hubs(self, functional_network: nx.Graph):
        """
        Tests that network hub identification returns a list of neuron IDs.
        """
        if functional_network.number_of_nodes() == 0:
            pytest.skip("Network is empty, cannot identify hubs.")
        
        analyzer = NetworkAnalyzer()
        hubs = analyzer.identify_network_hubs(functional_network, method='degree')
        
        assert isinstance(hubs, list)
        assert len(hubs) > 0
        assert isinstance(hubs[0], str) # Should be neuron global_ids


@pytest.mark.slow
class TestTemporalAnalyzer:
    """
    Tests for the TemporalAnalyzer class. These can be slow due to correlation calculations.
    """

    def test_analyze_temporal_dynamics(self, loaded_session_180_trials: SessionData):
        """
        Tests the main entry point for temporal analysis, checking output structure.
        """
        analyzer = TemporalAnalyzer(frame_rate=loaded_session_180_trials.frame_rate)
        
        # Create a deep copy to avoid modifying the original session fixture
        session_subset = copy.deepcopy(loaded_session_180_trials)
        # Analyze a small subset of neurons to speed up the test
        session_subset.neurons = session_subset.neurons[:5]
        # Also slice the activity matrix to match the neuron subset
        session_subset.activity_matrix = session_subset.activity_matrix[:5, :]
        
        dynamics = analyzer.analyze_temporal_dynamics(session_subset, max_lag=50)

        assert isinstance(dynamics, dict)
        assert len(dynamics) > 0

        first_neuron_id = next(iter(dynamics))
        neuron_dynamics = dynamics[first_neuron_id]

        assert isinstance(neuron_dynamics, TemporalDynamics)
        assert isinstance(neuron_dynamics.autocorrelation, np.ndarray)
        # Check that autocorrelation peaks at lag 0
        assert np.isclose(neuron_dynamics.autocorrelation[50], 1.0)
        assert isinstance(neuron_dynamics.spectral_power, dict)
        assert 'low' in neuron_dynamics.spectral_power


class TestDimensionalityReducer:
    """Tests for the DimensionalityReducer class."""

    def test_pca_fit_transform_and_getters(self, loaded_session_180_trials: SessionData):
        """
        Tests fitting PCA, transforming data, and retrieving components.
        """
        n_components = 5
        reducer = DimensionalityReducer(method='pca')
        
        # NOTE: This test now runs on the original, unmodified fixture because the
        #       pollution from TestTemporalAnalyzer has been fixed.
        transformed_data = reducer.fit_transform(loaded_session_180_trials, n_components=n_components)
        
        assert isinstance(transformed_data, np.ndarray)
        assert transformed_data.shape == (len(loaded_session_180_trials.neurons), n_components)
        
        # Test getters
        explained_variance = reducer.get_explained_variance()
        assert isinstance(explained_variance, np.ndarray)
        assert len(explained_variance) == n_components
        assert all(0 <= v <= 1 for v in explained_variance)
        
        components = reducer.get_components()
        assert isinstance(components, np.ndarray)
        
        # The components' shape is (n_components, n_features).
        # The features are the timepoints from the activity matrix.
        n_features = loaded_session_180_trials.activity_matrix.shape[1]
        assert components.shape == (n_components, n_features)