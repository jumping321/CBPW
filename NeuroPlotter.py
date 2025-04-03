import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from NeuroDataConfig import NeuroConfig, NeuroDataLoader
from tqdm import tqdm

class ORFPlotter:
    """Neuron plotting utility class"""
    
    def __init__(self, config: NeuroConfig):
        """Initialize plotting utility
        
        Args:
            config: Neuron configuration object
        """
        self.cfg = config
        self.data_loader = NeuroDataLoader(config)
    
    @staticmethod
    def save_high_res_plot(fig, filename: str, dpi: int = 600):
        """Save high resolution plot image
        
        Args:
            fig: matplotlib figure object
            filename: Output filename
            dpi: Resolution (default 600)
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    
    @staticmethod
    def set_plot_style(ax, title: str = None, fontsize: int = 100):
        """Configure plot styling
        
        Args:
            ax: matplotlib axes object
            title: Plot title (optional)
            fontsize: Font size for title (default 100)
        """
        if title:
            ax.text(-20, -20, title, fontsize=fontsize, fontname='Times New Roman', 
                   ha='left', va='bottom', color="black")
        
        # Set axis limits and ticks
        ax.set_xlim(-20.5, 20.5)
        ax.set_ylim(-20.5, 20.5)
        ax.set_xticks(np.linspace(-20.5, 20.5, 5))
        ax.set_yticks(np.linspace(-20.5, 20.5, 5))
        
        # Configure grid and appearance
        ax.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=4)
        ax.set_aspect('equal')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        # Customize spines
        for spine in ax.spines.values():
            spine.set_linewidth(10)
            spine.set_color("black")
    
    @staticmethod
    def create_figure(figsize: Tuple[int, int] = (12, 12)):
        """Create figure with specified size
        
        Args:
            figsize: Figure dimensions (default 12x12 inches)
            
        Returns:
            Tuple of (figure, axes) objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('white')
        return fig, ax
    
    def get_color_value(self, value: float, max_value: float) -> str:
        """Get RGBA color based on normalized value
        
        Args:
            value: Input value (positive or negative)
            max_value: Maximum absolute value for normalization
            
        Returns:
            RGBA tuple with appropriate color and transparency
        """
        normalized_value = min(abs(value) / max_value, 1.0)  # Normalize and clamp to [0, 1] range
        
        if value > 0:
            return (1, 0.11, 0.18, normalized_value)  # Red for positive values
        else:
            return (0.1, 0.4, 1, normalized_value)  # Blue for negative values
        
    def plot_receptive_field(self, parent_pbar=None):
        """Plot combined receptive field for all target neurons
        
        Args:
            parent_pbar: Parent progress bar (optional) for nested progress tracking
        """
        # Create output directory path
        output_dir = os.path.join(self.cfg.RESULT_ROOT, 
                                f"{self.cfg.SOURCE_NEURON_TYPE}_to_{self.cfg.TARGET_NEURON_TYPE}/")
        
        # Create figure
        fig, ax = self.create_figure()
        
        # Preload and cache coordinate data
        valid_root_ids = set(self.data_loader.column_df['root_id'].values)
        coordinates_cache = {
            row['root_id']: self.data_loader.hex_to_cartesian(row['x'], row['y'])
            for _, row in self.data_loader.column_df.iterrows()
        }
        
        # Initialize storage for accumulated values
        accumulated_values = {}
        max_positive = 0
        max_negative = 0
        
        # Configure progress bar
        position = 1 if parent_pbar is not None else 0
        
        with tqdm(total=len(self.data_loader.source_ids), 
             desc="  └─ Translating centroids", 
             ncols=80,
             position=position,
             leave=False) as pbar:
            
            # Process each target neuron
            for target_id in self.data_loader.target_ids:
                file_path = os.path.join(output_dir, f'{target_id}.txt')
                if not os.path.exists(file_path):
                    continue
                    
                pos_coords = []
                neg_coords = []
                
                # Read data file in one operation
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                
                # Batch process data lines
                for line in lines:
                    parts = line.strip().split(': ')
                    if len(parts) != 2:
                        continue
                        
                    neuron_id, value_str = parts
                    if neuron_id not in valid_root_ids:
                        continue
                        
                    value = float(value_str)
                    coordinates = coordinates_cache.get(neuron_id)
                    if coordinates is None:
                        continue
                        
                    x_cartesian, y_cartesian = coordinates
                    if value > 0:
                        pos_coords.append((x_cartesian, y_cartesian, value))
                        max_positive = max(max_positive, value)
                    elif value < 0:
                        neg_coords.append((x_cartesian, y_cartesian, value))
                        max_negative = max(max_negative, abs(value))
                
                # Process positive coordinates in batch
                if pos_coords:
                    pos_array = np.array(pos_coords)
                    weights = pos_array[:, 2]
                    pos_centroid = np.average(pos_array[:, :2], weights=weights, axis=0)
                    translation_vector = -pos_centroid
                    
                    # Batch translate all coordinates
                    all_coords = np.vstack([pos_coords, neg_coords]) if neg_coords else pos_array
                    translated_coords = all_coords.copy()
                    translated_coords[:, :2] += translation_vector
                    
                    # Update accumulated values
                    for x, y, v in translated_coords:
                        key = (x, y)
                        accumulated_values[key] = accumulated_values.get(key, 0) + v
                pbar.update(1)
        
        # Normalize values for plotting
        max_positive = max_positive if max_positive != 0 else 1
        max_negative = max_negative if max_negative != 0 else 1
        
        # Convert accumulated values to numpy arrays for efficient processing
        points = np.array(list(accumulated_values.keys()))
        values = np.array(list(accumulated_values.values()))
        
        # Plot points with appropriate colors
        if len(points) > 0:
            colors = [self.get_color_value(v, max_positive if v > 0 else max_negative) for v in values]
            ax.scatter(points[:, 0], points[:, 1], c=colors, s=20)
        
        # Configure plot styling
        self.set_plot_style(ax, f"{self.cfg.SOURCE_NEURON_TYPE} to {self.cfg.TARGET_NEURON_TYPE}")
        plt.tight_layout()
        
        # Save output image
        output_filename = os.path.join(output_dir, 
            f"{self.cfg.SOURCE_NEURON_TYPE}_to_{self.cfg.TARGET_NEURON_TYPE}_combined_receptive_field.png")
        self.save_high_res_plot(fig, output_filename)
        plt.close(fig)
        
    def save_centroids_to_file(self, output_dir):
        """Save translated centroids for each target neuron to text files
        
        Args:
            output_dir: Directory to save centroid files
        """
        output_dir = os.path.join(self.cfg.RESULT_ROOT, 
                                f"{self.cfg.SOURCE_NEURON_TYPE}_centroids/")
        
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        centroid_filename = os.path.join(output_dir, f'{self.cfg.TARGET_NEURON_TYPE}_centroid.txt')
        
        # Process each target neuron
        for target_id in self.data_loader.target_ids:
            file_path = os.path.join(output_dir, f'{target_id}.txt')
            if not os.path.exists(file_path):
                continue

            pos_coords = []
            
            # Read data file in one operation
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Batch process data lines
            for line in lines:
                parts = line.strip().split(': ')
                if len(parts) != 2:
                    continue

                neuron_id, value_str = parts
                value = float(value_str)

                if value > 0:  # Only process positive values
                    pos_coords.append((neuron_id, value))

            # Calculate and save centroid if positive coordinates exist
            if pos_coords:
                pos_array = np.array(pos_coords)
                weights = pos_array[:, 1].astype(float)  # Extract weights
                # Calculate weighted centroid
                pos_centroid = np.average(pos_array[:, 0], weights=weights, axis=0)

                # Save centroid to file
                with open(centroid_filename, 'w') as centroid_file:
                    centroid_file.write(f"{target_id}:{pos_centroid[0]},{pos_centroid[1]}\n")
