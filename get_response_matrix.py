import os
import sys
from NeuroDataConfig import NeuroConfig
from GetNeuroMatrix import GetNeuroMatrix
import pandas as pd
from typing import Set
from multiprocessing import Pool, cpu_count

def get_neuron_ids_by_target_type(config: NeuroConfig) -> Set[str]:
    """Get a set of neuron IDs for the target neuron type from visual_neuron_types.txt"""
    visual_neurons_path = os.path.join(config.DATA_DIR, config.VISUAL_NEURONS_FILE)
    df = pd.read_csv(visual_neurons_path)
    
    # Filter IDs based on target neuron type and side
    target_neuron_ids = set(df[
        (df['type'] == config.TARGET_NEURON_TYPE) & 
        (df['side'] == config.SIDE_FILTER)
    ]['root_id'])
    
    print(f"Number of IDs for target neuron type '{config.TARGET_NEURON_TYPE}': {len(target_neuron_ids)}")
    return target_neuron_ids

def process_neuron_type(neuron_type: str):
    """Process a single neuron type task"""
    # Create new config instance for each iteration
    current_config = NeuroConfig()
    current_config.TARGET_NEURON_TYPE = neuron_type
    # Get neuron IDs for current type
    neuron_ids = list(get_neuron_ids_by_target_type(current_config))
    
    if not neuron_ids:
        print(f"Warning: No matching neuron IDs found for type '{neuron_type}', skipping.")
        return
    
    # Generate and visualize connection matrix
    neuro_matrix = GetNeuroMatrix(current_config)
    neuro_matrix.visualize_distribution(neuron_ids)

def main():
    # Load temp config to get file paths
    temp_config = NeuroConfig()
    visual_neurons_path = os.path.join(temp_config.DATA_DIR, temp_config.VISUAL_NEURONS_FILE)
    df = pd.read_csv(visual_neurons_path)
    # Get all unique neuron types
    all_neuron_types = df['type'].unique().tolist()
    # Define neuron types to exclude
    types_to_remove = {"L1", "L2", "L3", "R1-6"}
    # Set max number of processes (up to CPU count or 20)
    num_processes = min(cpu_count(), 20)
    # Filter out unwanted types
    filtered_neuron_types = [neuron_type for neuron_type in all_neuron_types if neuron_type not in types_to_remove]

    # Process in parallel using multiprocessing pool
    with Pool(processes=num_processes) as pool:
        pool.map(process_neuron_type, filtered_neuron_types)

if __name__ == "__main__":
    main()
