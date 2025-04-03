from NeuroDataConfig import NeuroConfig, NeuroDataLoader
from NeuronSpikeConfig import L1_Neuron, L2_Neuron, L3_Neuron, OUT_Neuron
from DotMotionSimulation import BarMotionSimulation
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

def plot_neural_responses(total_response, activation, inhibition, output_path, frame_rate):
    """Visualize neural response components with max/min indicators"""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_palette("muted")
    
    # Plot response components
    ax.plot(total_response, label='Total Response', linewidth=2)
    ax.plot(activation, label='Activation', linewidth=2, color='green')
    ax.plot(inhibition, label='Inhibition', linewidth=2, color='red')

    # Add max/min reference lines
    max_val, min_val = np.max(total_response), np.min(total_response)
    ax.axhline(max_val, color='blue', linestyle='--', label=f'Max: {max_val:.2f}')
    ax.axhline(min_val, color='orange', linestyle='--', label=f'Min: {min_val:.2f}')

    # Configure plot aesthetics
    ax.set_title('Neural Response Dynamics', fontsize=12)
    ax.set_ylabel('Response Amplitude', fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=9)

    # Convert x-axis to seconds
    ticks = np.arange(0, len(total_response), frame_rate)
    plt.xticks(ticks=ticks, labels=np.arange(len(ticks)), rotation=45)
    plt.xlabel('Time (seconds)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def get_target_neurons(config: NeuroConfig) -> set:
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

def analyze_stimulus_height_effect(config):
    """Main analysis pipeline for stimulus height effect on neural response"""
    neuron_ids = list(get_target_neurons(config))
    matrix_dir = f'./result/NeuronMatrix/{config.TARGET_NEURON_TYPE}'
    stimulus_params = {
        'width': 10,
        'speed': 100,
        'frame_rate': 100,
        'total_time': 10,
        'pause_time': 5,
        'height_range': range(5, 40, 5),
        'edge_type': 'dark'
    }

    for neuron_id in tqdm(neuron_ids, desc="Processing neurons"):
        # Load connectivity matrices
        l1_mat, l2_mat, l3_mat = load_connectivity_matrices(neuron_id, matrix_dir)
        if None in (l1_mat, l2_mat, l3_mat):
            continue

        # Determine stimulus position based on receptive field
        centroid_x = (30 - np.mean(np.argwhere(l2_mat > 0)[:,1])) * 3.6 if np.any(l2_mat > 0) else None
        if centroid_x is None:
            continue

        # Prepare output directory
        output_dir = f'./result/NeuronPulse/{config.TARGET_NEURON_TYPE}/{neuron_id}/'
        os.makedirs(output_dir, exist_ok=True)

        # Initialize response storage
        height_responses = np.zeros(len(stimulus_params['height_range']))

        for i, height in enumerate(stimulus_params['height_range']):
            # Generate moving bar stimulus
            sim = BarMotionSimulation(
                bar_width=stimulus_params['width'],
                bar_height=height,
                speed=stimulus_params['speed'],
                edge_type=stimulus_params['edge_type'],
                frame_rate=stimulus_params['frame_rate'],
                start_height=centroid_x,
                initial_pause_time=stimulus_params['pause_time']
            )
            stimulus = sim.generate_stimulus(stimulus_params['total_time'])

            # Calculate layer responses
            l1_resp = calculate_neural_response(stimulus, L1_Neuron(), l1_mat)
            l2_resp = calculate_neural_response(stimulus, L2_Neuron(), l2_mat)
            l3_resp = calculate_neural_response(stimulus, L3_Neuron(), l3_mat)

            # Combine responses
            time_steps = stimulus.shape[2]
            activation = np.zeros(time_steps)
            inhibition = np.zeros(time_steps)
            total_response = np.zeros(time_steps)
            out_neuron = OUT_Neuron()

            for t in range(time_steps):
                activation[t] = np.sum(np.maximum(0, [
                    np.sum(l1_resp[:,:,t]),
                    np.sum(l2_resp[:,:,t])
                ]))
                inhibition[t] = np.sum(np.minimum(0, [
                    np.sum(l1_resp[:,:,t]),
                    np.sum(l2_resp[:,:,t])
                ]))
                out_neuron.update_time_step(activation[t], inhibition[t])
                total_response[t] = out_neuron.response

            # Store and visualize results
            height_responses[i] = np.max(total_response)
            plot_neural_responses(
                total_response, activation, inhibition,
                f"{output_dir}{neuron_id}_{stimulus_params['width']}_{height}.png",
                stimulus_params['frame_rate']
            )

        # Plot height response curve
        plt.figure(figsize=(10, 6))
        sns.set_palette("muted")
        plt.plot(stimulus_params['height_range'], height_responses, marker='o')
        plt.title('Stimulus Height Tuning Curve', fontsize=12)
        plt.ylabel('Peak Response', fontsize=10)
        plt.xlabel('Stimulus Height', fontsize=12)
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        plt.savefig(f"{output_dir}height_tuning.png", dpi=300)
        plt.close()

if __name__ == '__main__':
    config = NeuroConfig()
    config.TARGET_NEURON_TYPE = 'T2'
    analyze_stimulus_height_effect(config)
