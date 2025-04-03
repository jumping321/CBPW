import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class NeuronSpikeDelay:
    """Base class for modeling neuron response with temporal delay characteristics"""
    
    def __init__(self, neuron_type='default', tau=20):
        """
        Initialize neuron response model
        
        Args:
            neuron_type (str): Type identifier for the neuron
            tau (float): Time constant for response decay (in milliseconds)
        """
        self.neuron_type = neuron_type
        self.tau = tau  # Membrane time constant
        self.response = np.zeros((30, 35))  # Response matrix (30x35 grid)
        
    def decay_response(self):
        """
        Calculate stimulus decay over time using exponential decay
        
        The decay follows the formula: response = response * exp(-1/tau)
        This models the natural decay of neural response over time
        """
        self.response = self.response * np.exp(-1 / self.tau)
        
    def update_time_step(self, old_stimulus, new_stimulus, weight_matrix):
        """
        Complete time step update including decay and stimulus processing
        
        Args:
            old_stimulus: Stimulus values from previous time step
            new_stimulus: Current stimulus values
            weight_matrix: Connection weight matrix (30x35)
            
        Note:
            External calls should use this method rather than calling
            decay_response and process_stimulus separately
        """
        self.decay_response()  # Apply temporal decay
        self.process_stimulus(old_stimulus, new_stimulus, weight_matrix)  # Process new stimulus


class L1_Neuron(NeuronSpikeDelay):
    """Model for L1 neurons responding to brightness increases"""
    
    def __init__(self, tau=5):
        """
        Initialize L1 neuron with specific response parameters
        
        Args:
            tau (float): Time constant (default 5ms for fast response)
        """
        super().__init__(neuron_type='L1', tau=tau)
        # Positive response parameters (brightness increase)
        self.a_positive = 1.60  # Amplitude factor
        self.b_positive = 0.38  # Nonlinear exponent
        
        # Negative response parameters (brightness decrease)
        self.a_negative = -0.48
        self.b_negative = 0.58

    def process_stimulus(self, old_stimulus, new_stimulus, weight_matrix):
        """
        Process stimulus changes with ON response (sensitive to brightness increases)
        
        Args:
            old_stimulus: Previous stimulus values (30x35)
            new_stimulus: Current stimulus values (30x35)
            weight_matrix: Connection weights (30x35)
        """
        # Calculate brightness change (0-100 scale)
        brightness_increase = (new_stimulus - old_stimulus) * 100 / 255
        
        # Apply nonlinear response function
        response = np.where(
            brightness_increase >= 0,
            self.a_positive * brightness_increase**self.b_positive,  # Positive response
            self.a_negative * (np.abs(brightness_increase))**self.b_negative  # Negative response
        )
        
        # Apply weights and add to current response
        activated = response * weight_matrix
        self.response += activated
        
    def update_time_step(self, old_stimulus, new_stimulus, weight_matrix):
        """Complete time step update for L1 neuron"""
        super().update_time_step(old_stimulus, new_stimulus, weight_matrix)


class L2_Neuron(NeuronSpikeDelay):
    """Model for L2 neurons responding to brightness decreases"""
    
    def __init__(self, tau=15):
        """
        Initialize L2 neuron with specific response parameters
        
        Args:
            tau (float): Time constant (default 15ms for medium response)
        """
        super().__init__(neuron_type='L2', tau=tau)
        # Positive response parameters (brightness decrease)
        self.a_positive = 1.84
        self.b_positive = 0.43
        
        # Negative response parameters (brightness increase)
        self.a_negative = -0.54
        self.b_negative = 0.58

    def process_stimulus(self, old_stimulus, new_stimulus, weight_matrix):
        """
        Process stimulus changes with OFF response (sensitive to brightness decreases)
        
        Args:
            old_stimulus: Previous stimulus values (30x35)
            new_stimulus: Current stimulus values (30x35)
            weight_matrix: Connection weights (30x35)
        """
        # Calculate brightness change (0-100 scale)
        brightness_decrease = (old_stimulus - new_stimulus) * 100 / 255
        
        # Apply nonlinear response function
        response = np.where(
            brightness_decrease >= 0,
            self.a_positive * brightness_decrease**self.b_positive,  # Positive response
            self.a_negative * (np.abs(brightness_decrease))**self.b_negative  # Negative response
        )
        
        # Apply weights and add to current response
        activated = response * weight_matrix
        self.response += activated
    
    def update_time_step(self, old_stimulus, new_stimulus, weight_matrix):
        """Complete time step update for L2 neuron"""
        super().update_time_step(old_stimulus, new_stimulus, weight_matrix)


class L3_Neuron(NeuronSpikeDelay):
    """Model for L3 neurons with sustained response characteristics"""
    
    def __init__(self, tau=100, max_response_time=100):
        """
        Initialize L3 neuron with slow response and adaptation
        
        Args:
            tau (float): Time constant (default 100ms for slow response)
            max_response_time (int): Maximum time window for response history (in ms)
        """
        super().__init__(neuron_type='L3', tau=tau)
        # Response parameters
        self.a = 91.32  # Response scaling factor
        self.b = 10.82  # Response offset
        
        # State variables
        self.activated_value = 0  # Current activation level
        self.voltage = 0  # Membrane potential
        self.response_history = []  # History of response values
        self.max_response_time = max_response_time  # History window size

    def process_stimulus(self, old_stimulus, new_stimulus, weight_matrix):
        """
        Calculate activation value based on current brightness
        
        Args:
            old_stimulus: Previous stimulus values (unused for L3)
            new_stimulus: Current stimulus values (30x35)
            weight_matrix: Connection weights (30x35)
        """
        brightness = new_stimulus
        self.activated_value = self.a / (brightness + self.b)  # Hyperbolic response

    def update_time_step(self, old_stimulus, new_stimulus, weight_matrix):
        """
        Complete time step update with adaptation mechanism
        
        Returns:
            float: Tanh-normalized response relative to mean activity
        """
        # Process current stimulus
        self.process_stimulus(old_stimulus, new_stimulus, weight_matrix)

        # Update membrane potential using Euler integration
        delta = (self.activated_value - self.voltage) / self.tau
        self.voltage += delta

        # Update response matrix
        self.response = self.voltage * weight_matrix
        
        # Record response history for adaptation
        self.response_history.append(np.sum(self.response, axis=(0, 1)))

        # Maintain fixed history window
        if len(self.response_history) > self.max_response_time:
            self.response_history.pop(0)
            
        # Calculate normalized response relative to mean
        avg_response = np.mean(self.response_history)
        l3_response_tanh = np.tanh(20*(np.sum(self.response, axis=(0, 1))-avg_response))
        
        return l3_response_tanh
