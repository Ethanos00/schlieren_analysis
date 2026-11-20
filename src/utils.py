# ======================
# SHARED UTILITY FUNCS
# ======================


from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tifffile
import yaml
import cv2


# --- Load configuration ---

def load_config(config_path="config.yaml"):

    # Get directory containing utils.py
    src_dir = Path(__file__).resolve().parent

    # Locate config.yaml inside src/
    config_path = src_dir / config_path

    # Read configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


# --- Load JPG or TIFF image ---

def load_image(image_path, jpg_gamma=2.2):
# Read ONE image
    # Identify the file extension
    extension = Path(image_path).suffix.lower()

    # Read the image based on its format
    if extension in (".jpg", ".jpeg"):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            raise FileNotFoundError(image_path)

        # OpenCV loads color images as BGR
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    elif extension in (".tif", ".tiff"):
        image = tifffile.imread(image_path)

    else:
        raise ValueError(f"Unsupported image format: {extension}")

    # Normalize integer brightness values
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    elif image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0

    else:
        raise ValueError(f"Unsupported image type: {image.dtype}")

    # Approximate linearization for JPG files
    if extension in (".jpg", ".jpeg"):
        image = image ** jpg_gamma

    # Convert RGB images to grayscale
    if image.ndim == 3 and image.shape[2] in (3, 4):
        image = (
            0.2126 * image[:, :, 0]
            + 0.7152 * image[:, :, 1]
            + 0.0722 * image[:, :, 2]
        )

    elif image.ndim != 2:
        raise ValueError(f"Unsupported image dimensions: {image.shape}")

    # returns array with brightness values (images[200,100] would return brightnes at that location)
    return image


# --- Crop image using user-defined ROI ---

def crop_image(image, roi):
# crop ONE image

    # Read rectangle coordinates
    x = roi["x"]
    y = roi["y"]
    width = roi["width"]
    height = roi["height"]

    # Get image dimensions
    image_height, image_width = image.shape

    # Validate ROI
    if (
        x < 0 or y < 0
        or width <= 0 or height <= 0
        or x + width > image_width
        or y + height > image_height
    ):
        raise ValueError(
            f"ROI is outside image boundaries: {roi}"
        )

    # Crop image using NumPy slicing
    cropped = image[
        y:y + height,
        x:x + width
    ]

    return cropped

