import pandas as pd
import warnings
import os
import numpy as np
from tqdm import tqdm
from typing import Set, List, Dict, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)

class NeuroConfig:
    """Configuration parameters for neural connection analysis"""
    DATA_DIR = "./data"
    RESULT_ROOT = "./result/visual_neuron_ORF"
    
    CLASSIFICATION_FILE = "classification.txt"
    CONNECTIONS_FILE = "connections.txt"
    SYNAPSES_FILE = "synapses.txt"
    COLUMN_ASSIGNMENT_FILE = "column_assignment.txt"
    VISUAL_NEURONS_FILE = "visual_neuron_types.txt"
    
    TARGET_NEURON_TYPE = "LC11"
    SOURCE_NEURON_TYPE = "L2"
    TARGET_COLUMN = "cell_type"
    SOURCE_COLUMN = "cell_type"
    SIDE_FILTER = "right"
    MAX_PATH_LENGTH = 10           # Maximum path depth
    MIN_WEIGHT = 1e-5             # Minimum weight threshold
    
class NeuroDataLoader:
    """Neuron data loader"""
    
    def __init__(self, config: NeuroConfig):
        """Initialize data loader
        
        Args:
            config: Neuron configuration object
        """
        self.cfg = config
        self._load_data()
        self._prepare_neuron_ids()
    
    def _load_data(self):
        """Load data files"""
        self.classification = pd.read_csv(os.path.join(self.cfg.DATA_DIR, self.cfg.CLASSIFICATION_FILE))
        self.connections = pd.read_csv(os.path.join(self.cfg.DATA_DIR, self.cfg.CONNECTIONS_FILE))
        self.synapses = pd.read_csv(os.path.join(self.cfg.DATA_DIR, self.cfg.SYNAPSES_FILE))
        self.column_df = pd.read_csv(os.path.join(self.cfg.DATA_DIR, self.cfg.COLUMN_ASSIGNMENT_FILE), dtype={'root_id': str})
        self.visual_neurons = pd.read_csv(os.path.join(self.cfg.DATA_DIR, self.cfg.VISUAL_NEURONS_FILE))
        self.valid_neurons = set(self.visual_neurons[self.visual_neurons['side'] == 'right']['root_id'])
        
    
    def _get_target_ids(self) -> Set[str]:
        """Get target neuron ID set
        
        Returns:
            Set of target neuron IDs
        """
        return set(self.classification[
            (self.classification[self.cfg.TARGET_COLUMN] == self.cfg.TARGET_NEURON_TYPE) & 
            (self.classification['side'] == self.cfg.SIDE_FILTER)
        ]['root_id'])
    
    def get_coordinates(self, root_id: str) -> Tuple[float, float]:
        """Get neuron coordinates
        
        Args:
            root_id: Neuron ID
            
        Returns:
            Coordinate tuple (x, y), returns None if not found
        """
        result = self.column_df[self.column_df['root_id'] == root_id][['x', 'y']].values
        return result[0] if len(result) > 0 else None
    
    @staticmethod
    def hex_to_cartesian(p: float, q: float) -> Tuple[float, float]:
        """Convert hexagonal coordinates to Cartesian coordinates
        
        Args:
            p: x coordinate
            q: y coordinate
            
        Returns:
            Cartesian coordinate tuple (x, y)
        """
        x = 2 * p + 1 if q % 2 == 1 else 2 * p
        y = q / 2
        return x, y
    
    def _prepare_neuron_ids(self):
        # Source neurons (L2)
        self.source_ids = self._filter_neuron_ids(
            column=self.cfg.SOURCE_COLUMN,
            value=self.cfg.SOURCE_NEURON_TYPE,
            side=self.cfg.SIDE_FILTER
        )
        
        # Target neurons (LC11)
        self.target_ids = self._filter_neuron_ids(
            column=self.cfg.TARGET_COLUMN,
            value=self.cfg.TARGET_NEURON_TYPE,
            side=self.cfg.SIDE_FILTER
        )
        
    def _filter_neuron_ids(self, column: str, value: str, side: str) -> Set[int]:
        """Filter method with validation"""
        if column not in self.classification.columns:
            raise KeyError(f"Column {column} does not exist in classification data")
            
        query = f"{column} == {repr(value)} & side == {repr(side)}"
        try:
            filtered = self.classification.query(query)
            return set(filtered["root_id"])
        except pd.errors.UndefinedVariableError:
            print(f"Invalid filter condition: {query}")
            return set()

    def get_upstream_neurons(self, neuron_id: str) -> Tuple[Set[str], List[float]]:
        """Get upstream neuron IDs and their weights for a given neuron ID
        
        Args:
            neuron_id: Neuron ID
            
        Returns:
            Tuple containing set of upstream neuron IDs and list of their weights
        """
        # Find connections related to the given neuron ID
        connections = self.connections[self.connections['post_root_id'] == neuron_id]
        upstream_neurons = set(connections['pre_root_id'])
        
        # Filter upstream neurons to ensure they exist in visual_neuron_types
        valid_neurons = set(self.visual_neurons['root_id'])
        upstream_neurons = upstream_neurons.intersection(valid_neurons)

        # Calculate weights for each upstream neuron
        weights = []
        synapse_series = self.synapses.set_index('root_id')['input synapses']

        for uid in upstream_neurons:
            # Get connections related to current upstream neuron
            relevant_connections = connections[connections['pre_root_id'] == uid]
            if not relevant_connections.empty:
                nt_type = relevant_connections['nt_type'].values
                syn_count = relevant_connections['syn_count'].abs().values
                sign = np.where(np.isin(nt_type, ["GABA", "GLUT"]), -1, 1)

                # Calculate weight
                if uid in synapse_series.index:
                    synapse_count = synapse_series[uid]
                    weight = np.tanh(np.sum(syn_count * sign) / synapse_count) if syn_count.any() else 0
                    weights.append(weight)
                else:
                    weights.append(0)  # Weight is 0 if no input synapses found
            else:
                weights.append(0)  # Weight is 0 if no relevant connections found

        return upstream_neurons, weights  # Return upstream neurons and their weights
    
    def get_neurons_types(self, neuron_id: str) -> List[str]:
        """Get neuron type for a given neuron ID
        
        Args:
            neuron_id: Neuron ID
            
        Returns:
            Neuron type string
        """
        # Get type of upstream neuron
        filtered_types = self.visual_neurons.loc[self.visual_neurons['root_id'] == neuron_id, 'type'].fillna('')

        if not filtered_types.empty:
            neuron_type = filtered_types.values[0]  # Access first element
        else:
            neuron_type = 'Unknown'  # Default value if not found
        return neuron_type
