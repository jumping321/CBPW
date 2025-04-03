from NeuroDataConfig import NeuroConfig, NeuroDataLoader
from typing import Set
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from NeuronSpikeConfig import L1_Neuron, L2_Neuron, L3_Neuron
from tqdm import tqdm
import cv2
import seaborn as sns

def plot_neuron_responses(total_response, total_activation, total_inhibition, output_path, frame_rate):
    """Plot neuron response components (total, activation, inhibition)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_palette("muted")
    
    ax.plot(total_response, label='Total Response', linewidth=2)
    ax.plot(total_activation, label='Activation', linewidth=2, color='green')
    ax.plot(total_inhibition, label='Inhibition', linewidth=2, color='red')

    max_val = np.max(total_response)
    min_val = np.min(total_response)
    ax.axhline(max_val, color='blue', linestyle='--', label=f'Max: {max_val:.2f}')
    ax.axhline(min_val, color='orange', linestyle='--', label=f'Min: {min_val:.2f}')

    ax.set_title('Neural Response Components', fontsize=12)
    ax.set_ylabel('Response Amplitude', fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=9)

    ticks = np.arange(0, len(total_response), frame_rate)
    plt.xticks(ticks=ticks, labels=np.arange(len(ticks)), rotation=45)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def get_target_neuron_ids(config: NeuroConfig) -> Set[str]:
    """Retrieve neuron IDs of specified type from visual_neuron_types.txt"""
    df = pd.read_csv(os.path.join(config.DATA_DIR, config.VISUAL_NEURONS_FILE))
    target_ids = set(df[
        (df['type'] == config.TARGET_NEURON_TYPE) & 
        (df['side'] == config.SIDE_FILTER)
    ]['root_id'])
    print(f"Found {len(target_ids)} IDs for neuron type '{config.TARGET_NEURON_TYPE}'")
    return target_ids

def load_neuron_matrices(neuron_id, matrix_dir):
    """Load L1/L2/L3 connection matrices for given neuron"""
    matrices = []
    for layer in ['L1', 'L2', 'L3']:
        file_path = f"{matrix_dir}/{layer}/{neuron_id}.txt"
        matrices.append(np.loadtxt(file_path, delimiter=',') if os.path.exists(file_path) else None)
    return tuple(matrices)

def calculate_response(stimulus, neuron, weight_matrix):
    """Calculate neuron response over time"""
    neuron.response = np.zeros_like(weight_matrix)
    response = np.zeros_like(stimulus)
    
    for t in range(1, stimulus.shape[2]):
        if isinstance(neuron, L3_Neuron):
            tanh_val = neuron.update_time_step(stimulus[:,:,t-1], stimulus[:,:,t], weight_matrix)
        else:
            neuron.update_time_step(stimulus[:,:,t-1], stimulus[:,:,t], weight_matrix)
        response[:,:,t] = neuron.response
    
    return (response, tanh_val) if isinstance(neuron, L3_Neuron) else (response, None)

def generate_stimulus(direction, time_steps, light_condition='light'):
    """Create stimulus matrix from downsampled images"""
    stimulus = np.zeros((30, 35, time_steps))
    for t in range(time_steps):
        img_path = f'./downsampled_frames/20250224_150858/{light_condition}/{direction}deg/downsampled_{t:04d}.png'
        stimulus[:,:,t] = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return stimulus

def analyze_neuron_responses(neuron_type, matrix_dir, params):
    """Generate and save response data for specified neuron type"""
    time_steps = int(params['total_time'] * params['frame_rate'])
    neuron_ids = get_target_neuron_ids(params['config'])
    results = []

    for neuron_id in tqdm(neuron_ids, desc=f"Analyzing {neuron_type}"):
        l1_mat, l2_mat, l3_mat = load_neuron_matrices(neuron_id, matrix_dir)
        if None in (l1_mat, l2_mat, l3_mat):
            continue

        stimulus = generate_stimulus(params['direction'], time_steps)
        
        # Initialize neurons
        l1 = L1_Neuron()
        l2 = L2_Neuron()
        l3 = L3_Neuron()

        # Calculate responses
        l1_resp, _ = calculate_response(stimulus, l1, l1_mat)
        l2_resp, _ = calculate_response(stimulus, l2, l2_mat)
        l3_resp, l3_tanh = calculate_response(stimulus, l3, l3_mat)

        # Combine responses
        combined = np.zeros(time_steps)
        for t in range(time_steps):
            combined[t] = (np.sum(l1_resp[:,:,t]) * (1 - l3_tanh[t]) +
                          np.sum(l2_resp[:,:,t]) * (1 + l3_tanh[t]) +
                          np.sum(l3_resp[:,:,t]))
        
        results.append([neuron_id] + list(combined))

    # Save results
    os.makedirs(f'./result/NeuronPulse/{neuron_type}_light', exist_ok=True)
    with open(f'./result/NeuronPulse/{neuron_type}_light/responses.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['neuron_id'] + [f't_{t}' for t in range(time_steps)])
        writer.writerows(results)

if __name__ == '__main__':
    analysis_params = {
        'config': NeuroConfig(),
        'stimulus_width': 2,
        'stimulus_speed': 100,
        'frame_rate': 1000,
        'total_time': 10,
        'pause_time': 3,
        'direction': 42
    }

    for neuron_type in ['Tm1', 'Tm2', 'Tm3', 'Mi1', 'Mi9', 'Tm9', 'Tm4', 'Mi4', 'C3']:
        analysis_params['config'].TARGET_NEURON_TYPE = neuron_type
        analyze_neuron_responses(neuron_type, f'./result/NeuronMatrix/{neuron_type}', analysis_params)
