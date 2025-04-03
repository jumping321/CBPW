import os
import sys
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from NeuroDataConfig import NeuroConfig
from ResponseCalculator import ResponseCalculator
from DotMotionSimulation import DotMotionSimulation
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Set
import pandas as pd

def get_neuron_ids_by_target_type(config: NeuroConfig) -> Set[str]:
    """Get set of neuron IDs for target neuron type from visual_neuron_types.txt"""
    visual_neurons_path = os.path.join(config.DATA_DIR, config.VISUAL_NEURONS_FILE)
    df = pd.read_csv(visual_neurons_path)
    
    # Filter IDs based on target neuron type and side
    target_neuron_ids = set(df[
        (df['type'] == config.TARGET_NEURON_TYPE) & 
        (df['side'] == config.SIDE_FILTER)
    ]['root_id'])
    
    print(f"Number of IDs for target neuron type '{config.TARGET_NEURON_TYPE}': {len(target_neuron_ids)}")
    return target_neuron_ids  # Return set of target neuron IDs

def get_centroid_and_heights(matrix):
    # Find indices of elements greater than zero
    indices = np.argwhere(matrix > 0)

    # Calculate centroid
    if matrix is None:
        print("Error: The input matrix is None.")
        return None, None
    if indices.size > 0:
        centroid_x = np.mean(indices[:, 1])  # Calculate x-axis (column) coordinate average
        centroid_y = np.mean(indices[:, 0])  # Calculate y-axis (row) coordinate average
        return centroid_x, centroid_y
    else:
        print("No non-zero elements found in the matrix.")
        return None,None
    
def plot_contour(stimulus_width_range, stimulus_height_range, response_matrix, output_path):
    """Plot equidistant stimulus size response contour map (red-black color scheme)"""
    plt.figure(figsize=(10, 8))
    
    # Set professional journal style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

    # Create position indices
    x_pos = np.arange(len(stimulus_width_range))
    y_pos = np.arange(len(stimulus_height_range))
    
    # Generate grid
    X, Y = np.meshgrid(x_pos, y_pos)
    
    # Custom red-black color palette
    colors = ["black", "maroon", "red", "white"]  # From black to red
    cmap = LinearSegmentedColormap.from_list("RedBlack", colors)
    
    # Plot contour
    contour = plt.contourf(X, Y, response_matrix, levels=20, cmap=cmap)
    
    # Add colorbar
    cbar = plt.colorbar(contour, shrink=0.8)
    cbar.set_label('Response Amplitude', fontsize=14)
    
    # Set axis labels and ticks
    plt.xticks(x_pos, labels=[f"{w}" for w in stimulus_width_range], rotation=45)
    plt.yticks(y_pos, labels=[f"{h}" for h in stimulus_height_range])
    
    plt.xlabel('Stimulus Width (degrees)', fontsize=14)
    plt.ylabel('Stimulus Height (degrees)', fontsize=14)
    plt.title('Size Tuning Contour', fontsize=16, pad=20)
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Optimize layout
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_neuron(neuron_id, config, hex_dir, stimulus_height_range, stimulus_width_range, output_dir):
    """Process response calculation for a single neuron"""
    response_calculator = ResponseCalculator(config)
    l1_matrix, l2_matrix, l3_matrix = response_calculator._get_matrix_from_neuron_id(neuron_id, hex_dir)
    if any(matrix is None for matrix in (l1_matrix, l2_matrix, l3_matrix)):
        return None

    centroid_x, centroid_y = get_centroid_and_heights(l2_matrix)
    if centroid_y is None:
        return None

    # Initialize response matrix for current neuron
    neuron_response_matrix = np.zeros((len(stimulus_height_range), len(stimulus_width_range)))

    # Calculate total iterations
    total_iterations = len(stimulus_height_range) * len(stimulus_width_range)

    # Iterate through all size combinations
    with tqdm(total=total_iterations) as pbar:
        for i, height in enumerate(stimulus_height_range):
            for j, width in enumerate(stimulus_width_range):
                # Generate stimulus
                sim = DotMotionSimulation(
                    bar_width=width,
                    bar_height=height,
                    speed=20,
                    sigma=1.5,
                    edge_type='dark',
                    frame_rate=100,
                    start_height=centroid_y,
                    target_size=(35, 30),
                    initial_pause_time=2
                )
                simulation_array = sim.show_simulation(total_time=15, neuron_id=neuron_id)

                # Calculate total response
                total_response = response_calculator.calculate_total_response(simulation_array, neuron_id)
                neuron_response_matrix[i, j] = total_response
                pbar.update(1)

    # Save response matrix for current neuron
    contour_path = os.path.join(output_dir, f'{neuron_id}.png')
    plot_contour(stimulus_width_range, stimulus_height_range, neuron_response_matrix, contour_path)
    data_path = os.path.join(output_dir, f'{neuron_id}.npy')
    np.save(data_path, neuron_response_matrix)

    return neuron_response_matrix

def process_neuron_type(neuron_type, num_processes):
    """Process task for a single neuron type"""
    config = NeuroConfig()
    config.TARGET_NEURON_TYPE = neuron_type
    neuron_ids = list(get_neuron_ids_by_target_type(config))
    hex_dir = f'./result/NeuronMatrix/{config.TARGET_NEURON_TYPE}'
    stimulus_height_range = range(5, 126, 5)
    stimulus_width_range = range(5, 108, 5)
    output_dir = f'./result/NeuronPulse2/{config.TARGET_NEURON_TYPE}_hotmap/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each neuron in parallel using multiprocessing
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(
            process_neuron,
            [(neuron_id, config, hex_dir, stimulus_height_range, stimulus_width_range, output_dir) for neuron_id in neuron_ids]
        )

    # Calculate average response matrix
    response_matrices = [result for result in results if result is not None]
    if response_matrices:
        average_response_matrix = np.mean(response_matrices, axis=0)
        average_contour_path = os.path.join(output_dir, f'average_{neuron_type}.png')
        plot_contour(stimulus_width_range, stimulus_height_range, average_response_matrix, average_contour_path)
        average_data_path = os.path.join(output_dir, f'average_{neuron_type}.npy')
        np.save(average_data_path, average_response_matrix)

if __name__ == '__main__':
    neuron_types = ["LC11", "LC4", "LC12", "LC15", "LC18", "LC21", "LC25", "LPLC1", "LPLC2", "LC6", "LC9", "LC13", "LC16",]
    num_processes = min(cpu_count(), 20)
    for neuron_type in neuron_types:
        process_neuron_type(neuron_type, num_processes)
