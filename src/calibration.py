
# ==========================================
# KNIFE-EDGE SCHLIEREN CALIBRATION
# ==========================================

# TAKES CALIBRATION IMAGES, CALIBRATES FOR 50% / ALLOWS US TO USE 50% BRIGHTNESS REFERENCE MEASUREMENT
# RETURNS .npz FILE WITH REFERENCE_IMAGE, REFERENCE_POSITION, BRIGHTNESS CHANGE PER MM PER PIXEL
# FITTED BRIGHTNESS CURVE, BOOLEAN ARRAY WITH USABLE PIXELS, ALL KNIFE_EDGE POSITIONS USED, RMSE, RELATIVE ERROR, REGION OF INTEREST
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


# --- Linear fit of average calibration brightness ---

def fit_average_calibration(positions, images):

    # Calculate mean intensity of each calibration image
    mean_brightness = images.mean(axis=(1, 2))

    # Fit brightness = slope * position + intercept
    slope, intercept = np.polyfit(
        positions,
        mean_brightness,
        1
    )

    # Predicted brightness at measured positions
    fitted_brightness = slope * positions + intercept

    # Calculate goodness of fit (R squared)
    residuals = mean_brightness - fitted_brightness

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum(
        (mean_brightness - mean_brightness.mean()) ** 2
    )

    r_squared = (
        1 - ss_res / ss_tot
        if ss_tot > 0 else np.nan
    )

    # Display measured data and fitted line
    plt.figure(figsize=(8, 5))

    plt.scatter(
        positions,
        mean_brightness,
        color="blue",
        label="Measured brightness"
    )

    fit_positions = np.linspace(
        positions.min(),
        positions.max(),
        200
    )

    plt.plot(
        fit_positions,
        slope * fit_positions + intercept,
        color="red",
        label=f"Linear fit (R² = {r_squared:.4f})"
    )

    plt.xlabel("Knife-edge position (mm)")
    plt.ylabel("Mean linear brightness")
    plt.title("Average Knife-Edge Calibration Fit")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return slope, intercept, r_squared



# --- Linear fit of average calibration brightness ---

def fit_average_calibration(positions, images):

    # Calculate mean intensity of each calibration image
    mean_brightness = images.mean(axis=(1, 2))

    # Fit brightness = slope * position + intercept
    slope, intercept = np.polyfit(
        positions,
        mean_brightness,
        1
    )

    # Predicted brightness at measured positions
    fitted_brightness = slope * positions + intercept

    # Calculate goodness of fit (R squared)
    residuals = mean_brightness - fitted_brightness

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum(
        (mean_brightness - mean_brightness.mean()) ** 2
    )

    r_squared = (
        1 - ss_res / ss_tot
        if ss_tot > 0 else np.nan
    )

    # Display measured data and fitted line
    plt.figure(figsize=(8, 5))

    plt.scatter(
        positions,
        mean_brightness,
        color="blue",
        label="Measured brightness"
    )

    fit_positions = np.linspace(
        positions.min(),
        positions.max(),
        200
    )

    plt.plot(
        fit_positions,
        slope * fit_positions + intercept,
        color="red",
        label=f"Linear fit (R² = {r_squared:.4f})"
    )

    plt.xlabel("Knife-edge position (mm)")
    plt.ylabel("Mean linear brightness")
    plt.title("Average Knife-Edge Calibration Fit")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return slope, intercept, r_squared



# --- Fit knife-edge calibration for every pixel ---

def fit_pixel_calibration(positions, images):

    # Center blade positions around their mean
    x = positions - positions.mean()

    # Calculate denominator for linear regression
    denominator = np.sum(x ** 2)

    if denominator == 0:
        raise ValueError("Calibration requires different blade positions.")

    # Mean brightness of each pixel across all frames
    mean_image = images.mean(axis=0)

    # Compute slope for every pixel simultaneously
    slope = np.tensordot(
        x,
        images,
        axes=(0, 0)
    ) / denominator

    # Calculate intercept for every pixel
    intercept = mean_image - slope * positions.mean()

    # Calculate predicted brightness for every frame
    predicted = (
        positions[:, None, None] * slope[None, :, :]
        + intercept[None, :, :]
    )

    # Calculate fitting residuals
    residuals = images - predicted

    # Root mean squared fitting error per pixel
    rmse = np.sqrt(
        np.mean(residuals ** 2, axis=0)
    )

    # Measured intensity variation per pixel
    intensity_span = (
        images.max(axis=0) - images.min(axis=0)
    )

    # Relative fitting error
    relative_error = rmse / np.maximum(
        intensity_span, 1e-10
    )

    return slope, intercept, rmse, intensity_span, relative_error



# --- Display calibration quality maps ---

def plot_pixel_calibration(slope, relative_error):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Display calibration slope
    im1 = axes[0].imshow(slope, cmap="coolwarm")
    axes[0].set_title("Calibration Slope (brightness/mm)")
    fig.colorbar(im1, ax=axes[0])

    # Display relative fitting error
    im2 = axes[1].imshow(
        relative_error,
        cmap="magma",
        vmin=0,
        vmax=0.1
    )

    axes[1].set_title("Relative Linear Fit Error")
    fig.colorbar(im2, ax=axes[1])

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# --- Identify usable calibration pixels ---

def find_valid_pixels(intensity_span, relative_error, config):

    settings = config["calibration"]

    min_span = settings.get("minimum_intensity_span", 0.001)
    max_error = settings.get("max_relative_fit_error", 0.05)

    # Require sufficient brightness variation
    valid_span = intensity_span >= min_span

    # Require a sufficiently accurate linear fit
    valid_fit = relative_error <= max_error

    # Pixel must satisfy both requirements
    valid = valid_span & valid_fit

    print(f"Usable pixels: {100 * valid.mean():.2f}%")

    return valid


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

    
    # --- Pixel-by-pixel calibration ---

    slope_map, intercept_map, rmse, span, relative_error = (
        fit_pixel_calibration(positions, images)
    )

    # Determine usable pixels
    valid = find_valid_pixels(
        span,
        relative_error,
        config
    )

    # Display results
    plot_pixel_calibration(slope_map, relative_error)


    # ==========================================
    # SELECT REFERENCE IMAGE AND SAVE CALIBRATION
    # ==========================================

    # --- Find reference closest to 50% brightness ---

    # Average brightness of each calibration image
    mean_brightness = images.mean(axis=(1, 2))

    # Midpoint of the measured brightness range
    target_brightness = (
        mean_brightness.min() + mean_brightness.max()
    ) / 2

    # Find image closest to this midpoint
    reference_index = np.argmin(
        np.abs(mean_brightness - target_brightness)
    )

    # Extract reference brightness array
    reference_image = images[reference_index].copy()

    # Record corresponding knife-edge position
    reference_position_mm = positions[reference_index]

    print("\n--- Reference Image Selection ---")
    print(f"Target brightness: {target_brightness:.6f}")
    print(f"Selected blade position: {reference_position_mm:.3f} mm")
    print(f"Reference image shape: {reference_image.shape}")


    # --- Save all calibration data ---


    # --- Set output folder outside src ---

    # Directory containing calibrate.py
    src_dir = Path(__file__).resolve().parent

    # Project root is one level above src
    project_dir = src_dir.parent

    # Save to schlieren_capstone/output/calibration
    output_dir = project_dir / "output" / "calibration"

    # Create folder if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "calibration.npz",

        # Reference image
        reference_image=reference_image,
        reference_position_mm=reference_position_mm,

        # Pixel-by-pixel calibration
        slope=slope_map,
        intercept=intercept_map,
        valid_pixels=valid,

        # Calibration diagnostics
        positions_mm=positions,
        rmse=rmse,
        intensity_span=span,
        relative_error=relative_error,

        # Region of interest
        roi=np.array([
            config["calibration"]["roi"][key]
            for key in ("x", "y", "width", "height")
        ])
    )

    print("\nCalibration saved successfully!")
    print(f"File: {output_dir / 'calibration.npz'}")


    print("Calibration saved successfully!")


if __name__ == "__main__":
    main()
