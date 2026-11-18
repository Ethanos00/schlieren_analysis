
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

# --- Plot average calibration brightnesses ---

def plot_calibration_curve(positions, images):

    # calculate mean brightness of each image
    mean_brightness = images.mean(axis=(1, 2))

    # create figure
    plt.figure(figsize = (8,5))

    # plot brightness against position
    plt.plot(
        positions,
        mean_brightness,
        "o-",
        color="red",
        markersize=3,
        label="Average ROI brightness"
    )

    plt.xlabel("Knife-edge position (mm)")
    plt.ylabel("Average image brightness")
    plt.title("Calibration Image Brightness Curve")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()


# --- Display all calibration images ---

def display_calibration_images(positions, images):

    # Use a shared brightness scale for all images
    display_min = np.percentile(images, 1)
    display_max = np.percentile(images, 99)

    # Prevent invalid display limits
    if display_max <= display_min:
        display_max = display_min + 1e-6

    # Create one subplot per calibration image
    n_images = len(positions)

    fig, axes = plt.subplots(
        1, n_images,
        figsize=(4 * n_images, 5),
        squeeze=False
    )

    # Display images in blade-position order
    for i in range(n_images):

        ax = axes[0, i]

        im = ax.imshow(
            images[i],
            cmap="gray",
            vmin=display_min,
            vmax=display_max
        )

        ax.set_title(f"Blade: {positions[i]:.3f} mm")
        ax.axis("off")

    # Shared brightness colorbar
    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        label="Linear Brightness",
        shrink=0.7
    )

    plt.show()

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

    
    # Display all cropped calibration images
    display_calibration_images(positions, images)

    # Plot brightness versus knife-edge position
    plot_calibration_curve(positions, images)

if __name__ == "__main__":
    main()
