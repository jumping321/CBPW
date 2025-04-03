import numpy as np
from PIL import Image, ImageFilter
import os
import math
from datetime import datetime
from tqdm import tqdm

class BarMotionSimulation:
    """Simulates moving bar stimuli for visual neuroscience experiments"""
    
    def __init__(self, width=126, height=108, bar_width=10, bar_height=30, speed=8, 
                 frame_rate=100, sigma=1, bar_gap=5, edge_type='dark', 
                 start_height=None, target_size=(35, 30), initial_pause_time=5):
        """
        Initialize bar motion simulation parameters
        
        Args:
            width (int): Original stimulus width in pixels
            height (int): Original stimulus height in pixels
            bar_width (int): Bar width along motion direction (pixels)
            bar_height (int): Bar height perpendicular to motion (pixels)
            speed (float): Bar movement speed (pixels/second)
            frame_rate (int): Simulation frame rate (Hz)
            sigma (float): Gaussian blur standard deviation
            bar_gap (int): Spacing between bars (pixels)
            edge_type (str): 'dark' or 'light' edge stimulus
            start_height (float): Initial bar center position (degrees)
            target_size (tuple): Downsampled dimensions (width, height)
            initial_pause_time (float): Initial static period (seconds)
        """
        # Stimulus geometry parameters
        self.width = width
        self.height = height
        self.bar_width = bar_width
        self.bar_height = bar_height
        self.speed = speed
        self.sigma = sigma
        self.bar_gap = bar_gap
        self.frame_rate = frame_rate
        self.edge_type = edge_type
        self.start_height = start_height * 3.6 if start_height else None  # Convert to pixels
        self.target_size = target_size
        self.initial_pause_time = initial_pause_time

        # Motion direction (fixed horizontal)
        self.dx = 1  # x-direction component
        self.dy = 0  # y-direction component

        # Pre-calculate projection limits
        self._calculate_projection_limits()

        # Movement calculations
        self.pixel_per_frame = self.speed / self.frame_rate
        self.initial_offset = self.max_proj + self.bar_width
        self.total_distance = (self.initial_offset - self.min_proj) + self.bar_width

    def _calculate_projection_limits(self):
        """Calculate min/max projections of image corners onto motion vector"""
        corners = [
            (0, 0),
            (self.width-1, 0),
            (0, self.height-1),
            (self.width-1, self.height-1)
        ]
        projections = [x * self.dx + y * self.dy for x, y in corners]
        self.max_proj = max(projections)
        self.min_proj = min(projections)

    def generate_image(self, time_step):
        """Generate single frame of moving bar stimulus"""
        # Create blank canvas
        img = np.zeros((self.height, self.width), dtype=np.uint8)
        if self.edge_type == 'dark':
            img.fill(255)

        # Return static image during initial pause
        if time_step < self.initial_pause_time * self.frame_rate:
            img_pil = Image.fromarray(img)
            return np.array(img_pil.resize(self.target_size, Image.Resampling.LANCZOS))

        # Generate coordinate grids
        x_grid, y_grid = np.meshgrid(np.arange(self.width), np.arange(self.height))

        # Calculate current bar position
        elapsed_time = time_step - self.initial_pause_time * self.frame_rate
        current_offset = self.initial_offset - elapsed_time * self.pixel_per_frame

        # Create motion direction mask
        motion_proj = x_grid * self.dx + y_grid * self.dy
        mask_motion = (motion_proj <= current_offset) & \
                      (motion_proj > current_offset - self.bar_width)

        # Create perpendicular direction mask
        if self.start_height is not None:
            center_proj = self.start_height
            perp_proj = y_grid
            mask_height = np.abs(perp_proj - center_proj) < self.bar_height / 2
        else:
            center_proj = self.height / 2
            perp_proj = y_grid
            mask_height = np.abs(perp_proj - center_proj) < self.bar_height / 2

        # Apply bar stimulus
        bar_value = 0 if self.edge_type == 'dark' else 255
        img[np.logical_and(mask_motion, mask_height)] = bar_value

        # Apply image processing
        img_pil = Image.fromarray(img)
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(self.sigma))
        img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
        img_pil = img_pil.resize(self.target_size, Image.Resampling.LANCZOS)
        
        return np.array(img_pil)

    def generate_stimulus(self, total_time=3, neuron_id=None):
        """
        Generate complete stimulus sequence and save as GIF
        
        Args:
            total_time (float): Total simulation duration (seconds)
            neuron_id (str): Optional neuron ID for file organization
            
        Returns:
            np.ndarray: Stimulus array (height, width, frames)
        """
        timestamp = datetime.now().strftime("%H%M%S")
        output_dir = f'./output/BarMotion/{datetime.now().strftime("%Y%m%d")}/{self.edge_type}/{neuron_id or ""}/'
        os.makedirs(output_dir, exist_ok=True)

        total_frames = int(total_time * self.frame_rate)
        frames = []
        
        for t in tqdm(range(total_frames), desc="Generating frames", dynamic_ncols=True):
            frames.append(self.generate_image(t))

        # Convert to 3D array (height, width, time)
        stimulus_array = np.stack(frames, axis=-1).astype(np.float32)

        # Save as GIF
        gif_path = f"{output_dir}/{timestamp}.gif"
        Image.fromarray(frames[0]).save(
            gif_path,
            save_all=True,
            append_images=[Image.fromarray(frame) for frame in frames[1:]],
            duration=1000 // self.frame_rate,  # ms per frame
            loop=0
        )

        return stimulus_array
