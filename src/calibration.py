
# ==========================================
# KNIFE-EDGE SCHLIEREN CALIBRATION
# ==========================================


# --- Libraries ---

from pathlib import Path
from utils import load_config, load_image, crop_image

import numpy as np
import matplotlib.pyplot as plt
import tifffile
import yaml
import cv2


# --- Load all calibration images ---

def load_calibration_images(config):

    # Read settings from config.yaml
    frames = config["calibration"]["frames"]
    roi = config["calibration"]["roi"]
    jpg_gamma = config.get("images", {}).get("jpg_gamma", 2.2)

    # Create empty lists
    positions = []
    images = []

    # Loop through each calibration frame
    for frame in frames:

        # Get blade position and image path
        blade_mm = frame["blade_mm"]
        image_path = frame["file"]

        # Load and crop the image
        image = load_image(image_path, jpg_gamma=jpg_gamma)
        cropped = crop_image(image, roi)

        # Store results
        positions.append(blade_mm)
        images.append(cropped)

        print(f"Loaded: {image_path} | Blade: {blade_mm} mm")

    # Convert lists to NumPy arrays
    positions = np.array(positions, dtype=np.float64)
    images = np.stack(images, axis=0)

    # Sort images by knife-edge position
    order = np.argsort(positions)

    positions = positions[order]
    images = images[order]

    return positions, images



# --- Main program ---

def main():

    print("Starting knife-edge calibration...")

    # Load configuration
    config = load_config()

    # Load all calibration images
    positions, images = load_calibration_images(config)

    # Print results
    print("\nCalibration images loaded successfully!")

    print(f"Blade positions: {positions}")
    print(f"Image stack shape: {images.shape}")

    # Calculate average brightness of each image
    mean_brightness = images.mean(axis=(1, 2))

    for position, brightness in zip(positions, mean_brightness):
        print(
            f"Blade {position:8.1f} mm | "
            f"Mean brightness: {brightness:.6f}"
        )


if __name__ == "__main__":
    main()
