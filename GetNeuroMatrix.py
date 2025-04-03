import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from tqdm import tqdm
import os
import warnings
from matplotlib.colors import LinearSegmentedColormap, Normalize
import math
from NeuroDataConfig import NeuroConfig, NeuroDataLoader
import colorsys
from collections import defaultdict
from typing import List

# Configure matplotlib settings
plt.rcParams['font.family'] = 'Times New Roman'
warnings.filterwarnings("ignore", category=FutureWarning)

class GetNeuroMatrix:
    """Class for processing and visualizing neuron connection matrices"""
    
    def __init__(self, config: NeuroConfig):
        """Initialize with configuration
        
        Args:
            config: NeuroConfig object containing analysis parameters
        """
        self.cfg = config
        self.data_loader = NeuroDataLoader(config)
        
        # Directory paths for different neuron layers
        self.L1_dir = './result/visual_neuron_ORF/L1'
        self.L2_dir = './result/visual_neuron_ORF/L2'
        self.L3_dir = './result/visual_neuron_ORF/L3'
        
        # Output directories for images and data
        self.image_output_dir = f'./result/NeuronMatrix/{config.TARGET_NEURON_TYPE}/'
        self.data_output_dir = f'./result/NeuronMatrix/{config.TARGET_NEURON_TYPE}/'
        
        self.max_val = 1  # Maximum value for normalization

    def _process_neuron_data(self, neuron_id):
        """Process data for a single neuron
        
        Args:
            neuron_id: ID of the neuron to process
            
        Returns:
            Dictionary containing coordinate-value pairs for each layer
        """
        file_paths = {
            "L1": self.find_neuron_id_file(neuron_id, self.L1_dir),
            "L2": self.find_neuron_id_file(neuron_id, self.L2_dir),
            "L3": self.find_neuron_id_file(neuron_id, self.L3_dir)
        }

        # Check if all required files exist
        if any(path is None for path in file_paths.values()):
            return {}

        coords_dict = defaultdict(dict)  # Dictionary to store coordinate-value pairs

        # Process each layer's file
        for layer, file_path in file_paths.items():
            if os.path.exists(file_path):
                self._process_layer_file(file_path, layer, coords_dict)

        return coords_dict

    def _process_layer_file(self, file_path, layer, coords_dict):
        """Process data file for a single layer
        
        Args:
            file_path: Path to the data file
            layer: Name of the layer being processed
            coords_dict: Dictionary to store the results
        """
        p_values = self.data_loader.column_df.iloc[:, 4]  # Get p column
        q_values = self.data_loader.column_df.iloc[:, 5]  # Get q column
        
        # Initialize all coordinates with zero values
        for p, q in zip(p_values, q_values):
            cartesian_coord = self.data_loader.hex_to_cartesian(p, q)
            coords_dict[layer][cartesian_coord] = 0
            
        # Read actual values from file
        with open(file_path) as f:
            for line in f:
                parts = line.strip().split(': ')
                if len(parts) == 2:
                    root_id, value = parts[0], float(parts[1])
                    coord = self.data_loader.get_coordinates(root_id)
                    if coord is None:
                        continue
                    p, q = coord
                    cart_coord = self.data_loader.hex_to_cartesian(p, q)
                    coords_dict[layer][cart_coord] = value

    def _calculate_color(self, value):
        """Calculate color for visualization with modern tech aesthetic
        
        Args:
            value: Input value to determine color
            
        Returns:
            RGBA tuple representing the color
        """
        # Common parameter calculations
        abs_value = abs(value)
        normalized = (abs_value / self.max_val) ​** 0.5 if self.max_val != 0 else 0  # Non-linear enhancement for small values
                
        if abs(value) < 10e-5:
            # Neutral color: light silver gray (RGB: 235,235,235)
            return (0.92, 0.92, 0.92, 1)
        elif value >= 10e-5:
            # Positive values (red/orange spectrum)
            hue = 15 * (1 - normalized)  # Dynamic hue variation
            saturation = 0.05 + 0.95 * normalized  # Increasing saturation
            brightness = 1 - normalized  # Decreasing brightness
            rgb = colorsys.hsv_to_rgb(hue/360, saturation, brightness)
            return (rgb[0], rgb[1], rgb[2], 1)
        else:
            # Negative values (blue spectrum)
            hue = 195 + 45 * normalized  # Dynamic hue variation
            saturation = 0.05 + 0.95 * normalized  # Increasing saturation
            brightness = 1 - normalized  # Decreasing brightness
            rgb = colorsys.hsv_to_rgb(hue/360, saturation, brightness)
            return (rgb[0], rgb[1], rgb[2], 1)

    def visualize_distribution(self, neuron_ids: List[str]):
        """Visualize and save neuron connection matrices
        
        Args:
            neuron_ids: List of neuron IDs to process
        """
        n_neurons = len(neuron_ids)
        if n_neurons == 0:
            return
            
        hex_marker = self.create_hexagon_marker()
        
        # Skip if output already exists
        if os.path.exists(self.data_output_dir):
            return
            
        for idx, neuron_id in tqdm(enumerate(neuron_ids), desc="Processing neurons", dynamic_ncols=True, total=n_neurons):
            coords_dict = self._process_neuron_data(neuron_id)
            if not coords_dict:
                print(f"No coordinates found for neuron {neuron_id}. Skipping.")
                continue

            # Process each layer
            for layer in ['L1', 'L2', 'L3']:
                if layer in coords_dict:
                    points = np.array(list(coords_dict[layer].keys()))
                    values = np.array(list(coords_dict[layer].values()))

                    # Calculate coordinate ranges
                    if points.size == 0:
                        print(f"No points found for {layer} in neuron {neuron_id}. Skipping.")
                        continue
                        
                    min_x, max_x = points[:, 0].min(), points[:, 0].max()
                    min_y, max_y = points[:, 1].min(), points[:, 1].max()
                    matrix_shape = (30, 35)  # Fixed matrix size

                    # Calculate colors
                    colors = [self._calculate_color(v) for v in values]

                    # Create scatter plot
                    plt.figure(figsize=(7, 6))
                    ax = plt.gca()
                    sc = ax.scatter(points[:, 0], points[:, 1], c=colors, s=200, marker=hex_marker, edgecolors='none')
                    plt.title(f'{self.cfg.TARGET_NEURON_TYPE}:{neuron_id}: {layer} Coordinates')
                    plt.tight_layout()
                    ax.axis('off')  # Hide axes

                    # Add colorbar
                    norm = plt.Normalize(vmin=-1, vmax=1)
                    sm = plt.cm.ScalarMappable(
                        cmap=LinearSegmentedColormap.from_list(
                            "neurohex", 
                            [self._calculate_color(v) for v in np.linspace(-1, 1, 256)]
                        ), 
                        norm=norm
                    )
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax)
                    cbar.set_label('Value')

                    # Save image
                    output_dir = os.path.join(self.image_output_dir, f'{layer}/')
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = os.path.join(output_dir, f'{neuron_id}.png')
                    plt.savefig(output_filename, dpi=300)
                    plt.close()

                    # Create and save data matrix
                    value_matrix = np.zeros(matrix_shape)
                    data_filename = os.path.join(output_dir, f'{neuron_id}.txt')

                    # Fill matrix with values
                    for point, value in zip(points, values):
                        if abs(value) < 10e-5:
                            value = 0
                        row = int(max_y - point[1])  # Invert y-axis for matrix
                        col = int(point[0] - min_x)
                        if 0 <= row < matrix_shape[0] and 0 <= col < matrix_shape[1]:
                            value_matrix[row, col] = value

                    # Save matrix to file
                    np.savetxt(data_filename, value_matrix, delimiter=',', fmt='%.6f')
    
    def find_neuron_id_file(self, target_id: str, base_dir: str) -> str:
        """Find neuron data file in directory structure
        
        Args:
            target_id: Neuron ID to search for
            base_dir: Directory to search in
            
        Returns:
            Path to the found file, or None if not found
        """
        target_filename = f"{target_id}.txt"
        for root, dirs, files in os.walk(base_dir):
            if target_filename in files:
                return os.path.join(root, target_filename)
        return None
    
    def create_hexagon_marker(self, size=1, orientation=0):
        """Create hexagon marker for scatter plots
        
        Args:
            size: Size of the marker
            orientation: Rotation angle in degrees
            
        Returns:
            Numpy array defining the hexagon shape
        """
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.radians(orientation)
        return np.column_stack([np.cos(angles), np.sin(angles)]) * size
