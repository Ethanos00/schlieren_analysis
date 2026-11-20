import numpy as np
from scipy.ndimage import gaussian_filter


def derive_gradient(
    heated: np.ndarray,           # [2D float array, (H, W)] Grayscale image (0.0 to 1.0) with heat source ON
    reference: np.ndarray,        # [2D float array, (H, W)] Baseline grayscale image (0.0 to 1.0) with heat source OFF
    slope: np.ndarray,            # [2D float array, (H, W)] Pixelwise calibration slope: dI / ds_blade (intensity / mm)
    f_eff_m: float,               # [scalar, meters] Effective focal length of the schlieren mirror/lens before the knife edge
    optical_depth_m: float,       # [scalar, meters] Length (L) of the heat plate along the optical beam axis
    n_ambient: float,             # [scalar, dimensionless] Refractive index (n_0) of undisturbed room air (~1.00027)
    smooth_sigma_px: float        # [scalar, pixels] Gaussian blur radius to reduce pixel noise before integrating (0 = off)
) -> tuple[
    np.ndarray,                   # delta_I:      [2D float array, (H, W)] Raw brightness change (heated - reference)
    np.ndarray,                   # delta_spot_m: [2D float array, (H, W)] Inferred light-spot shift at knife edge (meters)
    np.ndarray                    # grad:         [2D float array, (H, W)] Vertical refractive index gradient dn/dy (1 / meters)
]:
    # We know that theta_y ~= (L/n_0) * dn/dy
    # Since n_0~=1 and theta_y = \Delta s_diff/f_eff = (-\Delta s_blade)/f_eff
    # Where f_eff is focal length, \Delta s_diff is the amount the light bends
    # \Delta s_diff = -\Delta s_blade its analagous
    
    # This is either plus or minus 1. Put +1 which assumes that increasing the knife edge calibration
    # from 0 to +n would be moving the knife UP into the beam
    angle_sign = +1

    # Finds the brightness difference
    delta_I = heated - reference

    # Finds the diffracted displacement (1e-3 converts mm from calibration slope into meters)
    # Ignore valid mask; protect only against division by zero if slope == 0
    safe_slope = np.where(np.abs(slope) > 1e-12, slope, np.nan)
    delta_spot_m = (1e-3) * (-delta_I / safe_slope)

    # Small angle approximation (deflection angle theta_y = delta_spot / f_eff)
    theta = angle_sign * delta_spot_m / f_eff_m

    # theta_y = (L / n_0) * (dn/dy)
    grad = n_ambient * theta / optical_depth_m

    # NaN safe Gaussian smoothing (normalized convolution) so NaNs don't bleed into valid neighbors
    if smooth_sigma_px > 0:
        finite_mask = np.isfinite(grad).astype(float)
        weights = gaussian_filter(finite_mask, smooth_sigma_px)
        gf = gaussian_filter(np.nan_to_num(grad, nan=0.0), smooth_sigma_px)
        grad = np.where(weights > 0.1, gf / np.maximum(weights, 1e-9), np.nan)

    return delta_I, delta_spot_m, grad


def integrate_from_bottom(
    dn_dy: np.ndarray,        # [2D float array, (H, W)] Vertical gradient from Step 1 (1 / meters)
    meters_per_pixel: float,  # [scalar, meters/px] Physical height of 1 pixel at the test section
    n_ambient: float,         # [scalar, dimensionless] Refractive index of undisturbed room air (~1.00027)
    ref_bottom_rows: int      # [scalar, int] Number of bottom rows assumed to be ambient air (e.g. 4)
) -> np.ndarray:              # returns n_out: [2D float array, (H, W)] Refractive index field
    H, W = dn_dy.shape
    if ref_bottom_rows >= H:
        raise ValueError("ref_bottom_rows must be less than image height")

    # Replace NaNs with 0.0 during integration so dark border pixels outside the circular mirror
    # keep n = n_ambient instead of turning the entire column into NaN
    g_clean = np.nan_to_num(dn_dy, nan=0.0)

    n_out = np.full((H, W), np.nan, dtype=float)
    anchor_row = H - ref_bottom_rows
    n_out[anchor_row:] = n_ambient

    # March upward row-by-row across all W columns simultaneously
    for row in range(anchor_row - 1, -1, -1):
        a = g_clean[row + 1]
        b = g_clean[row]
        n_out[row] = n_out[row + 1] + 0.5 * (a + b) * meters_per_pixel

    # Mask out pixels outside the valid circular mirror region after integrating
    n_out[~np.isfinite(dn_dy)] = np.nan

    return n_out


def index_to_temperature(
    n: np.ndarray,           # [2D float array, (H, W)] Reconstructed refractive index from Step 2
    pressure_pa: float,      # [scalar, Pascals] Room atmospheric pressure (~101325.0 Pa at sea level)
    K_m3_kg: float,          # [scalar, m^3/kg] Gladstone-Dale constant for air (~2.26e-4)
    R_j_kgK: float           # [scalar, J/(kg*K)] Specific gas constant for dry air (~287.05)
) -> np.ndarray:             # returns T_celsius: [2D float array, (H, W)] Air temperature in Celsius

    # we know that from the Gladstone Dale Relation: n-1 = Kp, K is a constant
    # we recover air density p = (n - 1)/K.
    # We can then find temperature from T = P/(pR), where P is atmospheric pressure and R is a constant

    # 1. Compute denominator R * (n - 1)
    denom = R_j_kgK * (n - 1.0)

    # 2. Safely divide K * P by denom; if noise ever makes n <= 1.0 (denom <= 0), fill with NaN
    T_kelvin = np.divide(
        K_m3_kg * pressure_pa,
        denom,
        out=np.full_like(n, np.nan),
        where=denom > 0
    )

    # 3. Convert Kelvin to Celsius
    return T_kelvin - 273.15
