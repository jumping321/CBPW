from NeuroDataConfig import NeuroConfig
from NeuronSpikeConfig import L1_Neuron, L2_Neuron, L3_Neuron
from RasterMotionSimulation import RasterMotionSimulation
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import csv

def get_target_neuron_ids(config: NeuroConfig) -> set:
    """Retrieve neuron IDs of specified type from visual_neuron_types.txt"""
    df = pd.read_csv(os.path.join(config.DATA_DIR, config.VISUAL_NEURONS_FILE))
    target_ids = set(df[
        (df['type'] == config.TARGET_NEURON_TYPE) & 
        (df['side'] == config.SIDE_FILTER)
    ]['root_id'])
    print(f"Found {len(target_ids)} {config.TARGET_NEURON_TYPE} neurons")
    return target_ids

def load_connectivity_matrices(neuron_id, matrix_dir):
    """Load L1/L2/L3 connectivity matrices for a neuron"""
    matrices = []
    for layer in ['L1', 'L2', 'L3']:
        file_path = f"{matrix_dir}/{layer}/{neuron_id}.txt"
        matrices.append(np.loadtxt(file_path, delimiter=',') if os.path.exists(file_path) else None)
    return tuple(matrices)

def calculate_neural_response(stimulus, neuron, weight_matrix):
    """Compute neuron's temporal response to stimulus"""
    neuron.response = np.zeros_like(weight_matrix)
    response = np.zeros_like(stimulus)
    
    for t in range(1, stimulus.shape[2]):
        neuron.update_time_step(
            stimulus[:,:,t-1],
            stimulus[:,:,t],
            weight_matrix
        )
        response[:,:,t] = neuron.response
    
    return response

def analyze_directional_tuning():
    """Main analysis pipeline for directional tuning properties"""
    # Define neuron classes and their light conditions
    NEURON_CLASSES = {
        'T4a': 'light',
        'T4b': 'light',
        'T4c': 'light',
        'T4d': 'light',
        'T5a': 'dark',
        'T5b': 'dark',
        'T5c': 'dark',
        'T5d': 'dark'
    }
    
    # Simulation parameters
    TIME_STEPS = 1000
    DIRECTIONS = range(0, 360, 90)  # Cardinal directions
    STIMULUS_PARAMS = {
        'light': {'speed': 20, 'edge_type': 'light'},
        'dark': {'speed': 20, 'edge_type': 'dark'}
    }
    COMMON_PARAMS = {
        'bar_width': 10,
        'bar_height': 200,
        'sigma': 1.5,
        'frame_rate': 100,
        'start_height': 18,
        'target_size': (35, 30),
        'pause_time': 3
    }

    for target_class, light_condition in NEURON_CLASSES.items():
        print(f"\nAnalyzing {target_class} ({light_condition})")
        
        # Initialize configuration
        config = NeuroConfig()
        config.TARGET_NEURON_TYPE = target_class
        
        # Get neuron IDs
        neuron_ids = list(get_target_neuron_ids(config))
        if not neuron_ids:
            continue
            
        # Initialize data storage
        max_responses = {d: [] for d in DIRECTIONS}
        time_series_data = {
            'l1': [], 'l2': [], 'l3': [],
            'l3_tanh': [], 'total': []
        }

        # Process each neuron
        for neuron_id in tqdm(neuron_ids, desc=f"Processing {target_class}"):
            # Load connectivity matrices
            matrix_dir = f'./result/NeuronMatrix/{target_class}'
            l1_mat, l2_mat, l3_mat = load_connectivity_matrices(neuron_id, matrix_dir)
            if None in (l1_mat, l2_mat, l3_mat):
                continue

            # Initialize neurons
            l1_neuron = L1_Neuron()
            l2_neuron = L2_Neuron()
            l3_neuron = L3_Neuron()

            # Test each direction
            direction_responses = []
            for direction in DIRECTIONS:
                # Generate stimulus
                sim = RasterMotionSimulation(
                    motion_direction=direction,
                    dark_bar_width=COMMON_PARAMS['bar_width'],
                    dark_bar_gap=10,
                    speed=STIMULUS_PARAMS[light_condition]['speed'],
                    sigma=COMMON_PARAMS['sigma'],
                    total_time=TIME_STEPS/COMMON_PARAMS['frame_rate'],
                    frame_rate=COMMON_PARAMS['frame_rate'],
                    initial_pause_time=COMMON_PARAMS['pause_time'],
                    final_pause_time=3
                )
                stimulus = sim.generate_stimulus()

                # Calculate layer responses
                l1_resp = calculate_neural_response(stimulus, l1_neuron, l1_mat)
                l2_resp = calculate_neural_response(stimulus, l2_neuron, l2_mat)
                l3_resp, l3_tanh = process_L3_neuron_response(stimulus, l3_neuron, l3_mat)

                # Sum responses spatially
                l1_sum = np.sum(l1_resp, axis=(0,1))
                l2_sum = np.sum(l2_resp, axis=(0,1))
                l3_sum = np.sum(l3_resp, axis=(0,1))

                # Combine responses
                total_response = np.zeros(TIME_STEPS)
                for t in range(TIME_STEPS):
                    total_response[t] = (
                        l1_sum[t] * (1 - l3_tanh[t]) + 
                        l2_sum[t] * (1 + l3_tanh[t]) + 
                        l3_sum[t]
                    )

                # Store results
                direction_responses.append((direction, total_response))
                max_responses[direction].append(np.max(total_response))

            # Store time series data for preferred direction
            pref_dir, pref_response = max(direction_responses, key=lambda x: np.max(x[1]))
            time_series_data['l1'].append([neuron_id] + l1_sum.tolist())
            time_series_data['l2'].append([neuron_id] + l2_sum.tolist())
            time_series_data['l3'].append([neuron_id] + l3_sum.tolist())
            time_series_data['l3_tanh'].append([neuron_id] + l3_tanh.tolist())
            time_series_data['total'].append([neuron_id] + pref_response.tolist())

        # Save results
        save_results(target_class, max_responses, time_series_data)

def save_results(neuron_class, max_responses, time_series_data):
    """Save analysis results to CSV files"""
    # Save max direction responses
    max_dir_path = f'./result/MaxResponses/{neuron_class}/'
    os.makedirs(max_dir_path, exist_ok=True)
    
    with open(f"{max_dir_path}{neuron_class}_max_responses.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['direction', 'max_response'])
        for direction, responses in max_responses.items():
            for response in responses:
                writer.writerow([direction, response])

    # Save time series data
    time_dir_path = f'./result/TimeResponses/{neuron_class}/'
    os.makedirs(time_dir_path, exist_ok=True)
    
    for response_type, data in time_series_data.items():
        with open(f"{time_dir_path}{neuron_class}_{response_type}.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['neuron_id'] + [f't_{t}' for t in range(len(data[0])-1)])
            writer.writerows(data)

if __name__ == '__main__':
    analyze_directional_tuning()
