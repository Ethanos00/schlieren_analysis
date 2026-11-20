from __future__ import annotations
import argparse
import csv
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from utils import load_config, load_image, crop_image
from analysis import derive_gradient, integrate_from_bottom, index_to_temperature


def resolve_path(p: str, config_dir: Path) -> str:
    """Resolve relative paths in config.yaml relative to config.yaml's directory."""
    path = Path(p)
    return str(path if path.is_absolute() else (config_dir / path).resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="src/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent

    # 1. Load config.yaml via your team's shared utils
    config = load_config(str(config_path))
    roi = config["calibration"]["roi"]
    jpg_gamma = config.get("images", {}).get("jpg_gamma", 2.2)
    phys = config["physics"]
    geo = config["geometry"]

    # 2. Set up input/output paths (matches calibration.py's folder structure)
    src_dir = Path(__file__).resolve().parent
    project_dir = src_dir.parent
    cal_path = project_dir / "output" / "calibration" / "calibration.npz"
    out_dir = project_dir / "output" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load calibration data saved by calibration.py
    cal = np.load(cal_path)
    ref_pos_mm = float(cal["reference_position_mm"])
    slope = cal["slope"]                  # brightness / mm
    positions_mm = cal["positions_mm"]    # calibrated blade positions in mm

    # Load dedicated heater-OFF reference image from config.yaml (or fall back to calibration reference)
    if config.get("images", {}).get("reference"):
        ref_path = resolve_path(config["images"]["reference"], config_dir)
        ref_full = load_image(ref_path, jpg_gamma=jpg_gamma)
        reference = crop_image(ref_full, roi)
    else:
        reference = cal["reference_image"]

    # 4. Compute ambient refractive index n_0 from room temperature and pressure
    n_ambient = 1.0 + (
        phys["K_m3_per_kg"] * phys["pressure_pa"]
        / (phys["R_air_j_per_kgK"] * (phys["ambient_temp_C"] + 273.15))
    )
    print(f"Ambient refractive index n_0 = {n_ambient:.9f}")

    # Support either 'ref_bottom_rows' or legacy 'ref_top_rows' in config.yaml
    ref_bottom_rows = int(geo.get("ref_bottom_rows", geo.get("ref_top_rows", 4)))

    conditions = config["images"]["heated"]
    if not conditions:
        raise ValueError("Provide at least one heated image in config.yaml.")

    results = []
    summary = []

    # 5. Run analysis on each heated image
    for cond in conditions:
        label = cond["label"]
        heated_path = resolve_path(cond["file"], config_dir)
        heated_full = load_image(heated_path, jpg_gamma=jpg_gamma)
        heated = crop_image(heated_full, roi)

        if heated.shape != reference.shape:
            raise ValueError(
                f"Shape mismatch for '{label}': heated {heated.shape} vs reference {reference.shape}"
            )

        # Step 1: Brightness change -> light-spot shift (m) -> dn/dy (1/m)
        delta_I, delta_spot_m, dn_dy = derive_gradient(
            heated=heated,
            reference=reference,
            slope=slope,
            f_eff_m=geo["f_eff_m"],
            optical_depth_m=geo["optical_depth_m"],
            n_ambient=n_ambient,
            smooth_sigma_px=float(geo.get("smooth_sigma_px", 0.0)),
        )

        # Reject extrapolation beyond the measured knife-edge calibration sweep (mm)
        s_equiv_mm = ref_pos_mm - (delta_spot_m * 1e3)
        inside = (s_equiv_mm >= positions_mm.min()) & (s_equiv_mm <= positions_mm.max())
        dn_dy[~inside] = np.nan

        # Step 2: Integrate upward from bottom ambient rows -> refractive index n(x, y)
        n_field = integrate_from_bottom(
            dn_dy=dn_dy,
            meters_per_pixel=geo["meters_per_pixel"],
            n_ambient=n_ambient,
            ref_bottom_rows=ref_bottom_rows,
        )

        # Step 3: Refractive index -> air temperature T(x, y) in Celsius
        T_celsius = index_to_temperature(
            n=n_field,
            pressure_pa=phys["pressure_pa"],
            K_m3_kg=phys["K_m3_per_kg"],
            R_j_kgK=phys["R_air_j_per_kgK"],
        )

        # Optional: mask unphysical temperature outliers
        if phys.get("plausible_temp_C"):
            lo, hi = phys["plausible_temp_C"]
            T_celsius[(T_celsius < lo) | (T_celsius > hi)] = np.nan

        # Direct vertical temperature gradient dT/dy via chain rule
        grad_T = -(phys["K_m3_per_kg"] * phys["pressure_pa"]) / (
            phys["R_air_j_per_kgK"] * (n_field - 1.0) ** 2
        ) * dn_dy
        grad_T[~np.isfinite(T_celsius)] = np.nan

        results.append((cond, heated, delta_I, dn_dy, n_field, T_celsius, grad_T))

        # Save all 2D field arrays for this condition
        np.savez_compressed(
            out_dir / f"{label}_fields.npz",
            brightness=heated,
            brightness_change=delta_I,
            dn_dy_per_m=dn_dy,
            n=n_field,
            temperature_C=T_celsius,
            dT_dy_K_per_m=grad_T,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median_T = f"{np.nanmedian(T_celsius):.3f}" if np.isfinite(T_celsius).any() else ""
            summary.append({
                "label": label,
                "heater_reference_C": cond.get("heater_reference_C", ""),
                "valid_gradient_fraction": f"{np.mean(np.isfinite(dn_dy)):.3f}",
                "median_reconstructed_air_C": median_T,
            })

    # 6. Write summary CSV
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    # 7. Plot comparison grid: heated, delta I, dn/dy, (n - n_0), T, dT/dy
    panels = [
        (1, "Heated image", None),
        (2, "Δ brightness", "coolwarm"),
        (3, "∂n/∂y (m⁻¹)", "coolwarm"),
        (4, "n − n₀", "coolwarm"),
        (5, "Air temperature (°C)", "inferno"),
        (6, "∂T/∂y (K/m)", "coolwarm"),
    ]
    fig, axes = plt.subplots(
        len(results), len(panels),
        figsize=(18, max(3.3, 3.1 * len(results))),
        squeeze=False,
        constrained_layout=True,
    )

    for i, (cond, heated, dI, g, n_field, T_celsius, grad_T) in enumerate(results):
        values = [None, heated, dI, g, n_field - n_ambient, T_celsius, grad_T]
        for j, (idx, title, cmap) in enumerate(panels):
            ax = axes[i, j]
            data = values[idx]
            if idx in (2, 3, 4, 6):
                finite = data[np.isfinite(data)]
                lim = np.nanpercentile(np.abs(finite), 98) if finite.size else 1.0
                lim = max(lim, 1e-15)
                im = ax.imshow(data, cmap=cmap, vmin=-lim, vmax=lim)
            elif idx == 5:
                im = ax.imshow(data, cmap=cmap)
            else:
                im = ax.imshow(data, cmap="gray", vmin=0, vmax=1)

            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=11)
            if j == 0:
                ax.set_ylabel(cond["label"], fontsize=11)
            if j > 0:
                fig.colorbar(im, ax=ax, fraction=0.045, pad=0.015)

    fig.suptitle("Knife-edge schlieren: brightness → dn/dy → n → air temperature", fontsize=14)
    grid_path = out_dir / "comparison_grid.png"
    fig.savefig(grid_path, dpi=170)
    plt.show()

    print(f"Wrote {grid_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
