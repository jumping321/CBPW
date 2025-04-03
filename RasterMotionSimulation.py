import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import os
import math
from datetime import datetime
from tqdm import tqdm
import imageio

# Get current timestamp
datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

class RasterMotionSimulation:
    def __init__(self, width=126, height=108, motion_direction=0, 
                 dark_bar_width=10, dark_bar_gap=5, speed=8, 
                 sigma=1, total_time=3, frame_rate=100, 
                 initial_pause_time=1, final_pause_time=1, 
                 target_size=(35, 30)):
        """
        Initialize raster motion simulation parameters.

        Parameters:
        - width: Image width (pixels)
        - height: Image height (pixels)
        - motion_direction: Movement direction (degrees, 0 is horizontal right)
        - dark_bar_width: Dark bar width (pixels)
        - dark_bar_gap: Dark bar gap (pixels)
        - speed: Movement speed (pixels/second)
        - sigma: Standard deviation for Gaussian blur
        - total_time: Total simulation time (seconds)
        - frame_rate: Frame rate (frames/second)
        - initial_pause_time: Initial pause time (seconds)
        - final_pause_time: Final pause time (seconds)
        - target_size: Downsampling target size (width, height)
        """
        self.width = width
        self.height = height
        self.motion_direction = motion_direction  # Movement direction (degrees)
        self.dark_bar_width = dark_bar_width      # Dark bar width
        self.dark_bar_gap = dark_bar_gap          # Dark bar gap
        self.speed = speed                        # Movement speed (pixels/second)
        self.sigma = sigma                        # Gaussian blur standard deviation
        self.total_time = total_time              # Total simulation time (seconds)
        self.frame_rate = frame_rate              # Frame rate (frames/second)
        self.initial_pause_time = initial_pause_time  # Initial pause time (seconds)
        self.final_pause_time = final_pause_time      # Final pause time (seconds)
        self.target_size = target_size                # Downsampling target size

        # Calculate movement per frame
        self.pixel_per_frame = self.speed / self.frame_rate

        # Calculate trigonometric values for motion direction
        theta = math.radians(self.motion_direction)
        self.dx = math.cos(theta)  # x-direction component
        self.dy = -math.sin(theta)  # y-direction component (negative because image y-axis points down)

    def generate_image(self, time_step):
        """
        Generate single frame image.

        Parameters:
        - time_step: Current time step (frame number)

        Returns:
        - img: Generated image (NumPy array)
        """
        # Create empty image
        img = np.zeros((self.height, self.width), dtype=np.uint8)
        img.fill(255)  # Initialize with white background

        # Calculate current offset
        current_offset = time_step * self.pixel_per_frame

        # Generate coordinate grid
        x_grid, y_grid = np.meshgrid(np.arange(self.width), np.arange(self.height))

        # Calculate motion direction projection
        motion_proj = x_grid * self.dx + y_grid * self.dy

        # Create raster mask
        motion_phase = (motion_proj - current_offset) % (self.dark_bar_width + self.dark_bar_gap)
        mask_motion = motion_phase < self.dark_bar_width

        # Set dark bar regions to black
        img[mask_motion] = 0

        # Apply Gaussian blur
        img_pil = Image.fromarray(img)
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(self.sigma))

        return np.array(img_pil)

    def show_simulation(self):
        """
        Generate GIF animation of raster motion and return downsampled simulation array (35, 30, t).

        Returns:
        - simulation_array: Downsampled simulation array (35, 30, t)
        """
        # Set output directory
        output_dir = f'./output/RasterMotion/{datetime_str}/{self.motion_direction}deg'
        os.makedirs(output_dir, exist_ok=True)

        # Calculate total frames
        total_frames = int(self.total_time * self.frame_rate)
        initial_pause_frames = int(self.initial_pause_time * self.frame_rate)
        final_pause_frames = int(self.final_pause_time * self.frame_rate)

        # Generate animation frames
        frames = []
        for t in tqdm(range(total_frames), desc="Generating frames", dynamic_ncols=True, leave=False):
            if t < initial_pause_frames:
                # Initial pause phase: show static raster
                img = self.generate_image(0)  # Use frame 0 raster
            elif t >= total_frames - final_pause_frames:
                # Final pause phase: show last frame
                img = frames[-1]  # Use last frame
            else:
                # Movement phase: generate raster image
                img = self.generate_image(t - initial_pause_frames)
            
            # Downsample
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize(self.target_size, Image.Resampling.LANCZOS)
            frames.append(np.array(img_pil))

        # Convert frame list to NumPy array (35, 30, t)
        simulation_array = np.stack(frames, axis=-1).astype(np.float32)

        # Save GIF
        gif_path = f"{output_dir}/raster_motion_{self.motion_direction}deg.gif"
        imageio.mimsave(gif_path, frames, duration=1000 / self.frame_rate)

        # print(f"GIF saved to: {gif_path}")

        # Return downsampled simulation array
        return simulation_array
