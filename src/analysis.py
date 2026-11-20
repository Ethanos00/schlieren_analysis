# ==========================================
# QUANTITATIVE SCHLIEREN ANALYSIS
# ==========================================
# Run from any directory:
#   python src/analysis.py
#   python src/analysis.py --config src/config.yaml
# Assumes analysis.py and config.yaml are in project_root/src/.
# Relative paths in config.yaml are resolved from the PROJECT ROOT.

from pathlib import Path
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

from utils import load_image, crop_image


def derive_gradient(
    heated: np.ndarray,
    reference: np.ndarray,
    slope: np.ndarray,                 # dI / d(blade position in mm)
    valid: np.ndarray,
    f_eff_m: float,
    optical_depth_m: float,
    n_ambient: float,
    smooth_sigma_px: float,
    angle_sign: int = 1,
    min_abs_slope: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert brightness difference to spot shift (m) and dn/dy (1/m).

    Assumes the camera's intensity response is locally linear around the
    selected reference blade position. A positive blade shift is assumed to
    be the negative of the optical spot shift. Set angle_sign with a known
    optical deflection test (not by guessing the desired temperature sign).
    """
    if not (heated.shape == reference.shape == slope.shape == valid.shape):
        raise ValueError("Heated, reference, slope, and valid masks must have identical shapes")
    if f_eff_m <= 0 or optical_depth_m <= 0:
        raise ValueError("f_eff_m and optical_depth_m must be positive")
    if angle_sign not in (-1, 1):
        raise ValueError("angle_sign must be +1 or -1")

    safe = np.isfinite(slope) & (np.abs(slope) > min_abs_slope)
    delta_I = heated.astype(np.float64) - reference.astype(np.float64)
    delta_spot_m = np.full(reference.shape, np.nan, dtype=np.float64)

    # slope is brightness/mm; delta_I / slope gives equivalent blade shift in mm.
    # Convert mm to m, and flip sign to describe the optical spot movement.
    delta_spot_m[safe] = -1e-3 * delta_I[safe] / slope[safe]

    theta = angle_sign * delta_spot_m / f_eff_m
    grad = n_ambient * theta / optical_depth_m

    # Normalized convolution avoids spreading NaN values into valid pixels.
    if smooth_sigma_px > 0:
        finite = safe & np.isfinite(grad)
        weights = gaussian_filter(finite.astype(float), smooth_sigma_px)
        summed = gaussian_filter(np.where(finite, grad, 0.0), smooth_sigma_px)
        grad = np.where(weights > 0.1, summed / np.maximum(weights, 1e-9), np.nan)

    # Never report values at pixels that failed calibration.
    grad[~safe] = np.nan
    return delta_I, delta_spot_m, grad


def integrate_from_top(
    dn_dy: np.ndarray,
    meters_per_pixel: float,
    n_ambient: float,
    ref_top_rows: int,
) -> np.ndarray:
    """Integrate dn/dy downward, treating pixel row as negative physical y."""
    H, W = dn_dy.shape
    if not 1 <= ref_top_rows < H:
        raise ValueError("ref_top_rows must be between 1 and H-1")
    if meters_per_pixel <= 0:
        raise ValueError("meters_per_pixel must be positive")
    n_out = np.full((H, W), np.nan, dtype=np.float64)
    n_out[ref_top_rows - 1] = n_ambient

    for row in range(ref_top_rows, H):
        a = dn_dy[row - 1]
        b = dn_dy[row]
        prev = n_out[row - 1]
        valid_step = np.isfinite(prev) & np.isfinite(a) & np.isfinite(b)
        n_out[row] = np.where(
            valid_step,
            prev - 0.5 * (a + b) * meters_per_pixel,
            np.nan,
        )

    # These rows are assumed ambient boundary conditions, not measurements.
    n_out[:ref_top_rows] = n_ambient
    return n_out


def index_to_temperature(
    n: np.ndarray,
    pressure_pa: float,
    K_m3_kg: float,
    R_j_kgK: float,
) -> np.ndarray:
    """Gladstone-Dale + ideal-gas-law temperature estimate in degrees C."""
    denom = R_j_kgK * (n - 1.0)
    T_kelvin = np.divide(
        K_m3_kg * pressure_pa,
        denom,
        out=np.full_like(n, np.nan, dtype=np.float64),
        where=np.isfinite(denom) & (denom > 0),
    )
    return T_kelvin - 273.15


def resolve_path(path_value: str, project_dir: Path) -> Path:
    """Interpret relative paths from the project root (parent of src/)."""
    p = Path(path_value)
    return p if p.is_absolute() else (project_dir / p).resolve()


def show_and_save_maps(label, reference, heated, delta_I, grad, index, temperature, output_dir):
    """Six quick diagnostic plots; maps remain in numerical units."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panels = [
        (reference, "Reference intensity", "gray"),
        (heated, "Heated intensity", "gray"),
        (delta_I, "Brightness change ΔI", "coolwarm"),
        (grad, "dn/dy (1/m)", "coolwarm"),
        (index, "Refractive index n", "viridis"),
        (temperature, "Estimated temperature (°C)", "inferno"),
    ]
    for ax, (data, title, cmap) in zip(axes.flat, panels):
        vals = data[np.isfinite(data)]
        if vals.size:
            lo, hi = np.percentile(vals, [1, 99])
            if hi <= lo:
                hi = lo + 1e-9
            im = ax.imshow(data, cmap=cmap, vmin=lo, vmax=hi)
            fig.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, "No valid pixels", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"Schlieren analysis: {label}")
    fig.savefig(output_dir / f"{label}_maps.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze heated schlieren images")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    # Script and configuration are both inside project_root/src/.
    src_dir = Path(__file__).resolve().parent
    project_dir = src_dir.parent

    # An explicit --config path is relative to the terminal's current folder.
    # Without --config, always read src/config.yaml.
    config_path = args.config.resolve() if args.config else src_dir / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    # Do NOT replace project_dir with config_path.parent: that would make
    # output/calibration and output/analysis resolve inside src/.

    cal_path = resolve_path(config.get("calibration", {}).get(
        "results_file", "output/calibration/calibration.npz"), project_dir)
    with np.load(cal_path, allow_pickle=False) as cal:
        required = ("reference_image", "reference_position_mm", "slope", "valid_pixels", "roi")
        missing = [key for key in required if key not in cal.files]
        if missing:
            raise KeyError(f"Missing calibration arrays: {missing}")
        reference = cal["reference_image"].copy()
        reference_position_mm = float(cal["reference_position_mm"])
        slope = cal["slope"].copy()
        valid = cal["valid_pixels"].copy()
        saved_roi = tuple(int(v) for v in cal["roi"])

    # Use the ROI stored in the calibration file, not a potentially changed YAML ROI.
    roi = dict(zip(("x", "y", "width", "height"), saved_roi))
    print(f"Loaded calibration: {cal_path}")
    print(f"Reference blade position: {reference_position_mm:.3f} mm")
    print(f"Calibration ROI: {roi}")

    valid_fraction = float(np.mean(valid))
    print(f"Calibration-valid pixels: {100 * valid_fraction:.5f}%")
    if valid_fraction < 0.05:
        warnings.warn("Fewer than 5% of calibration pixels are valid. Temperature reconstruction "
                      "will be mostly unavailable; validate calibration before interpreting maps.")

    geom = config["geometry"]
    phys = config["physics"]
    analysis = config["analysis"]
    image_settings = config["images"]

    P = float(phys["pressure_pa"])
    T0_K = float(phys["ambient_temp_C"]) + 273.15
    K = float(phys["K_m3_per_kg"])
    R = float(phys["R_air_j_per_kgK"])
    n0 = 1.0 + K * P / (R * T0_K)

    output_dir = resolve_path(config.get("output", {}).get("analysis_dir", "output/analysis"), project_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jpg_gamma = float(image_settings.get("jpg_gamma", 2.2))

    for entry in image_settings["heated"]:
        label = str(entry["label"])
        if not label.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Heated image labels must consist of letters, digits, - or _")
        path = resolve_path(entry["file"], src_dir)
        heated = crop_image(load_image(path, jpg_gamma=jpg_gamma), roi)
        if heated.shape != reference.shape:
            raise ValueError(f"Heated image shape {heated.shape} != calibration {reference.shape}")

        delta_I, delta_spot_m, gradient = derive_gradient(
            heated, reference, slope, valid,
            f_eff_m=float(geom["f_eff_m"]),
            optical_depth_m=float(geom["optical_depth_m"]),
            n_ambient=n0,
            smooth_sigma_px=float(analysis.get("smooth_sigma_px", 0.0)),
            angle_sign=int(geom.get("angle_sign", 1)),
            min_abs_slope=float(analysis.get("min_abs_slope_per_mm", 1e-10)),
        )
        n_field = integrate_from_top(
            gradient,
            meters_per_pixel=float(geom["meters_per_pixel"]),
            n_ambient=n0,
            ref_top_rows=int(analysis["ref_top_rows"]),
        )
        temp_C = index_to_temperature(n_field, P, K, R)

        # Save full numerical outputs without costly compression.
        output_path = output_dir / f"{label}_fields.npz"
        np.savez(
            output_path, delta_I=delta_I, delta_spot_m=delta_spot_m,
            dn_dy=gradient, refractive_index=n_field,
            temperature_C=temp_C, valid_pixels=valid,
            roi=np.array(saved_roi), reference_position_mm=reference_position_mm,
            n_ambient=n0,
        )
        show_and_save_maps(label, reference, heated, delta_I, gradient, n_field, temp_C, output_dir)
        finite_temperature = np.isfinite(temp_C)
        print(f"[{label}] Saved: {output_path}")
        print(f"[{label}] Finite reconstructed temperature pixels (including assumed ambient top rows): "
              f"{100 * finite_temperature.mean():.4f}%")
        if entry.get("heater_reference_C") is not None:
            print(f"[{label}] Heater reference: {entry['heater_reference_C']} °C "
                  "(metadata only; not used in reconstruction)")


if __name__ == "__main__":
    main()
