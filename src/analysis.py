"""Quantitative schlieren analysis from calibration.npz.

Place as project/src/analysis.py alongside config.yaml and utils.py.
Run: python analysis.py, or python main.py analysis.

NOTE: One knife edge measures one component of the path-integrated
refractive-index gradient. The temperature reconstruction depends on
calibration, geometric assumptions, and a known ambient boundary.
"""
from pathlib import Path
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

from utils import load_image, crop_image


def path_from(value, base):
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def describe(name, array):
    vals = np.asarray(array)
    finite = vals[np.isfinite(vals)]
    if not finite.size:
        print(f"  {name}: no finite pixels")
        return
    p01, p50, p99 = np.percentile(finite, [1, 50, 99])
    print(f"  {name}: finite={100 * finite.size / vals.size:.3f}% "
          f"p01={p01:.5g} median={p50:.5g} p99={p99:.5g}")


def load_calibration(npz_path, yaml_roi, yaml_reference_mm=None):
    """Read required arrays; fall back to YAML if optional ROI is missing."""
    with np.load(npz_path, allow_pickle=False) as cal:
        print(f"Calibration: {npz_path}")
        print(f"Available NPZ arrays: {', '.join(cal.files)}")
        necessary = ('reference_image', 'slope')
        missing = [name for name in necessary if name not in cal.files]
        if missing:
            raise KeyError(f"Calibration file lacks {missing}. Re-run calibrate.py.")
        reference = np.asarray(cal['reference_image'], dtype=np.float32).copy()
        slope = np.asarray(cal['slope'], dtype=np.float32).copy()
        valid = (np.asarray(cal['valid_pixels'], dtype=bool).copy()
                 if 'valid_pixels' in cal.files else np.ones(reference.shape, dtype=bool))
        if 'roi' in cal.files:
            xywh = tuple(int(v) for v in cal['roi'].ravel())
        elif yaml_roi is not None:
            print('NPZ has no roi; using calibration.roi from config.yaml')
            xywh = tuple(int(yaml_roi[k]) for k in ('x', 'y', 'width', 'height'))
        else:
            raise ValueError('No ROI in NPZ or config.yaml; provide calibration.roi.')
        blade_mm = (float(cal['reference_position_mm'])
                    if 'reference_position_mm' in cal.files else yaml_reference_mm)
    if reference.ndim != 2 or reference.shape != slope.shape or reference.shape != valid.shape:
        raise ValueError('reference_image, slope, and valid_pixels must be matching 2D arrays.')
    if len(xywh) != 4 or reference.shape != (xywh[3], xywh[2]):
        raise ValueError(f'Calibration shape {reference.shape} differs from ROI (x,y,w,h) {xywh}.')
    return reference, slope, valid, dict(zip(('x', 'y', 'width', 'height'), xywh)), blade_mm


def derive_gradient(heated, reference, slope, saved_valid, n_ambient, geom, settings):
    """Intensity difference -> equivalent spot shift -> path-averaged dn/dy."""
    if not (heated.shape == reference.shape == slope.shape == saved_valid.shape):
        raise ValueError('Heated image and calibration array shapes differ.')
    f = float(geom['f_eff_m'])
    L = float(geom['optical_depth_m'])
    sign = int(geom.get('angle_sign', 1))
    if f <= 0 or L <= 0 or sign not in (-1, 1):
        raise ValueError('f_eff_m, optical_depth_m must be positive; angle_sign must be ±1.')

    delta_I = heated.astype(np.float32) - reference
    min_slope = float(settings.get('min_abs_slope_per_mm', 1e-4))
    min_light = float(settings.get('min_reference_intensity', 0.002))
    use_mask = bool(settings.get('use_calibration_mask', True))
    safe = (np.isfinite(delta_I) & np.isfinite(slope)
            & (np.abs(slope) >= min_slope) & (reference >= min_light))
    if use_mask:
        safe &= saved_valid
    else:
        warnings.warn('Calibration validity mask disabled: DIAGNOSTIC MODE, not validated temperatures.')

    # A shift of 1 mm blade position = 1e-3 m. Optical spot shift has opposite sign.
    spot_m = np.full(reference.shape, np.nan, dtype=np.float32)
    spot_m[safe] = -1e-3 * delta_I[safe] / slope[safe]
    gradient = (sign * n_ambient / (f * L)) * spot_m

    sigma = float(settings.get('smooth_sigma_px', 0))
    if sigma > 0:
        weights = gaussian_filter(safe.astype(np.float32), sigma)
        numerator = gaussian_filter(np.where(safe, gradient, 0), sigma)
        gradient = np.where(weights > 0.5, numerator / np.maximum(weights, 1e-9), np.nan)
        gradient[~safe] = np.nan
    return delta_I, spot_m, gradient, safe


def integrate_from_bottom(dn_dy, meters_per_pixel, n_ambient, ambient_row):
    """Integrate each image column UPWARD from a bottom ambient row.

    This setup assumes the recorded image is vertically inverted, so physical
    upward corresponds to INCREASING image row. Therefore, moving upward in
    the displayed image (decreasing row) gives dy_phys = -meters_per_pixel.

    The input dn_dy must use the physical vertical-axis sign convention;
    verify geometry.angle_sign separately against the knife-edge orientation.
    Invalid gradient pixels interrupt the integration path in that column.
    """
    h, w = dn_dy.shape
    if meters_per_pixel <= 0 or not (0 <= ambient_row < h):
        raise ValueError('Invalid meters_per_pixel or ambient_row.')

    out = np.full((h, w), np.nan, dtype=np.float64)

    # Set n = n_ambient only where the chosen starting row is measurable.
    anchor_ok = np.isfinite(dn_dy[ambient_row, :])
    out[ambient_row, anchor_ok] = n_ambient

    # Every column is processed simultaneously, pixel by pixel from bottom to top.
    for row in range(ambient_row - 1, -1, -1):
        good = (np.isfinite(out[row + 1, :])
                & np.isfinite(dn_dy[row + 1, :])
                & np.isfinite(dn_dy[row, :]))
        out[row, good] = out[row + 1, good] - (meters_per_pixel / 2) * (
            dn_dy[row + 1, good] + dn_dy[row, good])

    # Rows below the starting boundary remain NaN: not reconstructed.
    return out


def index_to_temperature(n_field, pressure_pa, K_m3_per_kg, R_air_j_per_kgK):
    denom = R_air_j_per_kgK * (n_field - 1.0)
    out = np.full(n_field.shape, np.nan, dtype=np.float64)
    np.divide(K_m3_per_kg * pressure_pa, denom, out=out,
              where=np.isfinite(denom) & (denom > 0))
    return out - 273.15


def plot_maps(label, reference, heated, delta_I, spot_m, gradient, delta_n, temperature, out_dir):
    panels = [
        (reference, 'Reference brightness', 'gray', False),
        (heated, 'Heated brightness', 'gray', False),
        (delta_I, 'Brightness difference', 'coolwarm', True),
        (spot_m * 1e6, 'Spot shift (µm)', 'coolwarm', True),
        (gradient, 'dn/dy (1/m)', 'coolwarm', True),
        (delta_n, 'n − n_ambient', 'coolwarm', True),
        (temperature, 'Estimated temperature (°C)', 'inferno', False),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    for ax, item in zip(axes.flat, panels):
        data, title, cmap, signed = item
        finite = np.asarray(data)[np.isfinite(data)]
        ax.set_title(title)
        ax.axis('off')
        if not finite.size:
            ax.text(.5, .5, 'No valid pixels', ha='center', va='center', transform=ax.transAxes)
            continue
        if signed:
            limit = float(np.percentile(np.abs(finite), 95))
            limit = max(limit, 1e-12)
            lo, hi = -limit, limit
        else:
            lo, hi = np.percentile(finite, (1, 99))
            if hi <= lo:
                hi = lo + 1e-9
        im = ax.imshow(data, cmap=cmap, vmin=lo, vmax=hi)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes.flat[-1].axis('off')
    fig.suptitle(f'Schlieren diagnostics — {label} (display scaled to percentiles)')
    fig.savefig(out_dir / f'{label}_maps.png', dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Analyze heated schlieren images')
    parser.add_argument('--config', type=Path, default=None)
    args = parser.parse_args()
    src_dir = Path(__file__).resolve().parent
    project_dir = src_dir.parent
    config_path = args.config.resolve() if args.config else src_dir / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    cal_settings = config.get('calibration', {})
    # Calibration and output paths are project-root relative.
    cal_path = path_from(cal_settings.get('results_file', 'output/calibration/calibration.npz'), project_dir)
    reference, slope, saved_valid, roi, blade_mm = load_calibration(
        cal_path, cal_settings.get('roi'), cal_settings.get('reference_blade_mm'))
    print(f'ROI: {roi}; reference blade: {blade_mm} mm')
    print(f'Saved calibration mask accepts {100 * saved_valid.mean():.5f}% of pixels')

    geom = config['geometry']
    phys = config['physics']
    settings = config.get('analysis', {})
    imgs = config['images']
    pressure = float(phys['pressure_pa'])
    temp_ambient_K = float(phys['ambient_temp_C']) + 273.15
    K = float(phys['K_m3_per_kg'])
    R = float(phys['R_air_j_per_kgK'])
    n0 = 1 + K * pressure / (R * temp_ambient_K)
    ambient_row = int(settings['ambient_row'])  # Bottom ambient row in cropped-image coordinates
    out_dir = path_from(config.get('output', {}).get('analysis_dir', 'output/analysis'), project_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for entry in imgs['heated']:
        label = str(entry['label'])
        if not label or not all(ch.isalnum() or ch in '-_' for ch in label):
            raise ValueError(f'Unsafe heated image label {label!r}')
        # Match calibrate.py: image paths are relative to src/ directory.
        img_path = path_from(entry['file'], src_dir)
        print(f'\nAnalyzing {label}: {img_path}')
        if not img_path.is_file():
            raise FileNotFoundError(img_path)
        heated = crop_image(load_image(img_path, jpg_gamma=float(imgs.get('jpg_gamma', 2.2))), roi)
        delta_I, spot_m, gradient, safe = derive_gradient(
            heated, reference, slope, saved_valid, n0, geom, settings)
        n_field = integrate_from_bottom(gradient, float(geom['meters_per_pixel']), n0, ambient_row)
        temp_C = index_to_temperature(n_field, pressure, K, R)

        print('Reconstruction diagnostics:')
        describe('ΔI', delta_I)
        describe('Optical spot shift (µm)', spot_m * 1e6)
        describe('dn/dy (1/m)', gradient)
        describe('n − n_ambient', n_field - n0)
        describe('Temperature (°C)', temp_C)
        print(f'  Analysis-safe mask: {100 * safe.mean():.4f}%')
        if blade_mm is not None and cal_settings.get('reference_blade_mm') is not None:
            if not np.isclose(float(cal_settings['reference_blade_mm']), blade_mm, atol=1e-6):
                warnings.warn('YAML reference_blade_mm differs from saved NPZ reference blade. '
                              'Heated image MUST match the NPZ blade position.')

        # Store primary results as float32 to reduce disk size, without compression.
        outfile = out_dir / f'{label}_fields.npz'
        np.savez(outfile, delta_I=delta_I.astype(np.float32),
                 delta_spot_m=spot_m.astype(np.float32),
                 dn_dy=gradient.astype(np.float32),
                 refractive_index=n_field.astype(np.float64),
                 delta_n=(n_field - n0).astype(np.float64),
                 temperature_C=temp_C.astype(np.float32),
                 valid_pixels=safe, n_ambient=n0,
                 roi=np.array([roi[k] for k in ('x','y','width','height')]),
                 reference_position_mm=np.nan if blade_mm is None else blade_mm)
        plot_maps(label, reference, heated, delta_I, spot_m, gradient,
                  n_field - n0, temp_C, out_dir)
        print(f'Saved {outfile} and {out_dir / (label + "_maps.png")}')
    print('\nAnalysis complete. Do not interpret temperatures without validating calibration and geometry.')


if __name__ == '__main__':
    main()
