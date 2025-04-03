import os
import numpy as np
from NeuronSpikeConfig import L1_Neuron, L2_Neuron, L3_Neuron
from NeuroDataConfig import NeuroDataLoader

class ResponseCalculator:
    """Class for calculating neural response patterns based on upstream connections"""
    
    def __init__(self, config):
        """
        Initialize response calculator with configuration
        
        Args:
            config: Configuration object containing analysis parameters
        """
        self.config = config
        self.data_loader = NeuroDataLoader(config)

    def calculate_total_response(self, simulation_array, neuron_id):
        """
        Calculate total response for a given neuron considering all upstream connections
        
        Args:
            simulation_array: 3D array (height × width × time) of stimulus values
            neuron_id: ID of the target neuron
            
        Returns:
            float: Maximum normalized response value (tanh-transformed)
        """
        # Get upstream connections and their weights
        upstream_neurons, upstream_weights = self.data_loader.get_upstream_neurons(neuron_id)
        TIME_STEPS = simulation_array.shape[2]
        
        # Initialize response tracking arrays
        upstream_total_response = np.zeros(TIME_STEPS)
        upstream_total_inhibition = np.zeros(TIME_STEPS)
        upstream_total_activation = np.zeros(TIME_STEPS)

        # Process each upstream neuron
        for upstream_neuron_id, weight in zip(upstream_neurons, upstream_weights):
            # Get neuron type and skip if unknown
            upstream_neuron_type = self.data_loader.get_neurons_types(upstream_neuron_id)
            if upstream_neuron_type == 'Unknown':
                continue

            # Load connection matrices for this upstream neuron
            upstream_hex_dir = f'./result/NeuronMatrix/{upstream_neuron_type}'
            upstream_l1_matrix, upstream_l2_matrix, upstream_l3_matrix = (
                self._get_matrix_from_neuron_id(upstream_neuron_id, upstream_hex_dir))
                
            if any(matrix is None for matrix in (upstream_l1_matrix, upstream_l2_matrix, upstream_l3_matrix)):
                continue

            # Initialize neuron models
            upstream_l1_neuron = L1_Neuron()
            upstream_l2_neuron = L2_Neuron()
            upstream_l3_neuron = L3_Neuron()

            # Calculate responses for each neuron type
            upstream_l1_response_matrix, _ = self._process_neuron_response(
                simulation_array, upstream_l1_neuron, upstream_l1_matrix)
            upstream_l2_response_matrix, _ = self._process_neuron_response(
                simulation_array, upstream_l2_neuron, upstream_l2_matrix)
            upstream_l3_response_matrix, upstream_l3_response_tanh = self._process_L3_neuron_response(
                simulation_array, upstream_l3_neuron, upstream_l3_matrix)

            # Sum responses across spatial dimensions
            upstream_l1_response = np.sum(upstream_l1_response_matrix, axis=(0, 1))
            upstream_l2_response = np.sum(upstream_l2_response_matrix, axis=(0, 1))
            upstream_l3_response = np.sum(upstream_l3_response_matrix, axis=(0, 1))

            # Calculate combined response with inhibition dynamics
            total_response = np.zeros(TIME_STEPS)
            total_response_inhibit = np.zeros(TIME_STEPS)
            inhibition_sum = 0
            total_response_activate = np.zeros(TIME_STEPS)

            for t in range(TIME_STEPS):
                # Combine responses with L3 modulation
                response = (
                    (upstream_l1_response[t]) * (1 - upstream_l3_response_tanh[t]) +
                    (upstream_l2_response[t]) * (1 + upstream_l3_response_tanh[t]) +
                    upstream_l3_response[t]
                )
                
                # Separate activation and inhibition
                if response < 0:
                    inhibition_sum -= response
                else:
                    total_response_activate[t] = response
                    
                total_response_inhibit[t] = inhibition_sum
                inhibition_sum = inhibition_sum * np.exp(-1 / 25)  # Inhibition decay

            # Apply weights and accumulate across upstream neurons
            for t in range(TIME_STEPS):
                response = total_response_activate[t] * (1 - np.tanh(total_response_inhibit[t])) * weight
                if response > 0:
                    upstream_total_activation[t] += response
                else:
                    upstream_total_inhibition[t] -= response

        # Calculate final response with global inhibition
        upstream_inhibition_sum = 0
        for t in range(TIME_STEPS):
            upstream_inhibition_sum += upstream_total_inhibition[t]
            upstream_total_response[t] = upstream_total_activation[t] * (
                1 - np.tanh(upstream_inhibition_sum))
            upstream_inhibition_sum = upstream_inhibition_sum * np.exp(-1 / 25)  # Global inhibition decay

        return np.max(np.tanh(upstream_total_response))

    def _get_matrix_from_neuron_id(self, neuron_id, hex_dir):
        """
        Load connection matrices for a neuron from L1/L2/L3 directories
        
        Args:
            neuron_id: ID of the neuron to load
            hex_dir: Base directory containing the matrix files
            
        Returns:
            tuple: (L1_matrix, L2_matrix, L3_matrix) - any may be None if not found
        """
        l1_matrix = l2_matrix = l3_matrix = None
        
        # Try to load each matrix file if it exists
        l1_matrix_filename = os.path.join(hex_dir, 'L1', f'{neuron_id}.txt')
        if os.path.exists(l1_matrix_filename):
            l1_matrix = np.loadtxt(l1_matrix_filename, delimiter=',')

        l2_matrix_filename = os.path.join(hex_dir, 'L2', f'{neuron_id}.txt')
        if os.path.exists(l2_matrix_filename):
            l2_matrix = np.loadtxt(l2_matrix_filename, delimiter=',')

        l3_matrix_filename = os.path.join(hex_dir, 'L3', f'{neuron_id}.txt')
        if os.path.exists(l3_matrix_filename):
            l3_matrix = np.loadtxt(l3_matrix_filename, delimiter=',')
        
        return l1_matrix, l2_matrix, l3_matrix

    def _process_neuron_response(self, stimulus_matrix, neuron_instance, hex_matrix):
        """
        Calculate response matrix for a neuron over time
        
        Args:
            stimulus_matrix: 3D array of stimulus values (height × width × time)
            neuron_instance: Initialized neuron model (L1_Neuron or L2_Neuron)
            hex_matrix: Connection weight matrix for this neuron
            
        Returns:
            tuple: (response_matrix, None) - response matrix matches stimulus dimensions
        """
        neuron_instance.response = np.zeros_like(hex_matrix)
        response_matrix = np.zeros_like(stimulus_matrix, dtype=float)
        
        # Calculate response at each time step
        for time in range(1, stimulus_matrix.shape[2]):
            neuron_instance.update_time_step(
                stimulus_matrix[:, :, time - 1],
                stimulus_matrix[:, :, time],
                hex_matrix
            )
            response_matrix[:, :, time] = neuron_instance.response
            
        return response_matrix, None

    def _process_L3_neuron_response(self, stimulus_matrix, neuron_instance, hex_matrix):
        """
        Calculate response matrix for an L3 neuron over time including tanh modulation
        
        Args:
            stimulus_matrix: 3D array of stimulus values (height × width × time)
            neuron_instance: Initialized L3_Neuron model
            hex_matrix: Connection weight matrix for this neuron
            
        Returns:
            tuple: (response_matrix, tanh_matrix) - both response and modulation values
        """
        neuron_instance.response = np.zeros_like(hex_matrix)
        tanh_matrix = np.zeros(stimulus_matrix.shape[2])
        response_matrix = np.zeros_like(stimulus_matrix)
        
        # Calculate response at each time step
        for time in range(1, stimulus_matrix.shape[2]):
            tanh_matrix[time] = neuron_instance.update_time_step(
                stimulus_matrix[:, :, time - 1],
                stimulus_matrix[:, :, time],
                hex_matrix
            )
            response_matrix[:, :, time] = neuron_instance.response
            
        return response_matrix, tanh_matrix
