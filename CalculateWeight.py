import os
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
from collections import defaultdict
from NeuroDataConfig import NeuroConfig, NeuroDataLoader
import pandas as pd 
from typing import Set

def check_png_files(output_dir):
    """Check if any PNG files exist in the output directory
    
    Args:
        output_dir: Directory path to check
        
    Returns:
        bool: True if PNG files exist, False otherwise
    """
    if os.path.exists(output_dir):
        # Check all files in the directory
        for file in os.listdir(output_dir):
            if file.endswith(".png"):
                return True  # PNG file found
        return False  # No PNG files found
    else:
        return False  # Directory doesn't exist
    
class NeuroAnalyzer:
    """Neural connection analyzer for processing and mapping neuron connections"""
    
    def __init__(self, config: NeuroConfig):
        """Initialize analyzer with configuration
        
        Args:
            config: NeuroConfig object containing analysis parameters
        """
        self.cfg = config
        self.data_loader = NeuroDataLoader(config)
        self.output_dir = os.path.join(
            self.cfg.RESULT_ROOT,
            f"{self.cfg.SOURCE_NEURON_TYPE}_to_{self.cfg.TARGET_NEURON_TYPE}/"
        )
        self.l1_ids = self.get_neuron_ids_by_target_type("L1")
        self.mi1_ids = self.get_neuron_ids_by_target_type("Mi1")

    def build_connection_map(self) -> Dict[int, List[Tuple[int, float]]]:
        """Build optimized connection map using vectorized operations
        
        Returns:
            Dictionary mapping pre-synaptic neuron IDs to lists of 
            (post-synaptic ID, weight) tuples
        """
        connection_map = {}
        synapse_series = self.data_loader.synapses.set_index('root_id')['input synapses']
        valid_neurons = self.data_loader.valid_neurons.union(self.data_loader.target_ids)
        
        # Pre-filter valid connections
        valid_conn = self.data_loader.connections[
            (self.data_loader.connections['pre_root_id'].isin(valid_neurons)) &
            (self.data_loader.connections['post_root_id'].isin(valid_neurons))
        ]
        
        # Batch process connections using vectorized operations
        for pre_id, group in tqdm(valid_conn.groupby('pre_root_id'), desc="Building connection map"):
            post_ids = group['post_root_id'].values
            nt_types = group['nt_type'].values
            syn_counts = group['syn_count'].abs().values
            
            # Vectorized input synapse count calculation
            input_synapses = synapse_series.reindex(post_ids, fill_value=0).values
            input_synapses[input_synapses == 0] = 1e-6  # Prevent division by zero
            
            # Calculate sign weights
            sign = np.where(np.isin(nt_types, ["GABA"]), -1, 1)
            
            # Special handling for L1 connections
            is_l1 = (pre_id in self.l1_ids)  # Check if pre_id is L1
            is_l1_mi1 = is_l1  # L1 condition
            sign[is_l1_mi1] = 1  # Force positive for L1 even if GABA
            
            weights = np.tanh(syn_counts * sign / input_synapses)
            
            # Filter valid connections
            mask = np.abs(weights) >= self.cfg.MIN_WEIGHT
            filtered = [(post_id, float(w)) for post_id, w in zip(post_ids[mask], weights[mask])]
            
            if filtered:
                connection_map[pre_id] = filtered
        
        return connection_map

    def analyze_connections(self, connection_map: Dict[int, List[Tuple[int, float]]], parent_pbar=None):
        """Analyze neural connections using depth-first search
        
        Args:
            connection_map: Pre-built connection map dictionary
            parent_pbar: Parent progress bar for nested progress tracking
        """
        if check_png_files(self.output_dir):
            return True
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        target_weights = defaultdict(lambda: defaultdict(float))
        target_ids = set(self.data_loader.target_ids)  # Convert to set for faster lookup
        
        position = 1 if parent_pbar is not None else 0
        with tqdm(total=len(self.data_loader.source_ids), 
             desc=f"  └─ Calculating weights {self.cfg.SOURCE_NEURON_TYPE}—{self.cfg.TARGET_NEURON_TYPE}", 
             ncols=80,
             position=position,
             leave=False) as pbar:
            
            for source_id in self.data_loader.source_ids:
                weights = defaultdict(float)
                stack = [(source_id, 1.0, 0, {source_id})]  # Using set for visited nodes
                
                while stack:
                    current_id, current_weight, depth, visited = stack.pop()
                    
                    # Termination conditions
                    if depth > self.cfg.MAX_PATH_LENGTH or abs(current_weight) < self.cfg.MIN_WEIGHT:
                        continue
                    
                    # Reached target node
                    if current_id in target_ids:
                        weights[current_id] += current_weight
                        continue
                    
                    # Process connections
                    for post_id, conn_weight in connection_map.get(current_id, []):
                        if post_id not in visited:
                            new_weight = current_weight * conn_weight
                            new_visited = visited.copy()
                            new_visited.add(post_id)
                            stack.append((post_id, new_weight, depth+1, new_visited))
                
                # Record valid results
                for target_id, weight in weights.items():
                    target_weights[target_id][source_id] = weight
                pbar.update(1)
        
        # Batch save results
        for target_id, sources in target_weights.items():
            self._save_results(target_id, sources)

    def _save_results(self, target_id: int, weights: Dict[int, float]):
        """Save connection weights to output file
        
        Args:
            target_id: Target neuron ID
            weights: Dictionary of source IDs and their weights
        """
        output_path = os.path.join(self.output_dir, f"{target_id}.txt")
        with open(output_path, 'w') as f:
            for src, w in weights.items():
                f.write(f"{src}: {w:.6f}\n")
    
    def get_neuron_ids_by_target_type(self, target_type: str) -> Set[str]:
        """Get neuron IDs for specified type from visual_neuron_types.txt
        
        Args:
            target_type: Neuron type to filter for
            
        Returns:
            Set of root IDs matching the target type
        """
        visual_neurons_path = os.path.join(self.cfg.DATA_DIR, self.cfg.VISUAL_NEURONS_FILE)
        df = pd.read_csv(visual_neurons_path)
        
        # Filter IDs by target neuron type
        target_neuron_ids = set(df[
            (df['type'] == target_type) & 
            (df['side'] == self.cfg.SIDE_FILTER)
        ]['root_id'])
        
        return target_neuron_ids
