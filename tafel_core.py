"""
Core Tafel-fitting logic — copied unchanged from the original batch script.
Only the tkinter folder picker, batch-folder loop and save-to-disk steps were
removed; the Streamlit app (app.py) handles input/output instead.
Settings below are module globals; app.py overrides them from the sidebar.
"""

from pathlib import Path
import io
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

warnings.filterwarnings("ignore")

# Electrode area in cm². Use 1.0 if current is already current density.
ELECTRODE_AREA_CM2 = 1.0

# -------------------------------------------------------------------------
# Automatic active-region constraints
# -------------------------------------------------------------------------

# Minimum number of data points in each selected Tafel branch.
MIN_POINTS_PER_BRANCH = 6

# Minimum potential span of a selected candidate window, V.
MIN_POTENTIAL_SPAN_V = 0.020

# Minimum log10(|j|) span of a selected candidate window, decades.
MIN_LOG_CURRENT_SPAN_DECADES = 0.30

# Minimum linearity of a selected candidate window.
# Increase for stricter selection; reduce for noisier data.
MIN_R2_FOR_ACTIVE_REGION = 0.940

# Exclude noisy points around observed Ecorr.
ECORR_EXCLUSION_WINDOW_V = 0.010

# Maximum number of points in a candidate window.
# Use None to allow all available points.
MAX_POINTS_PER_WINDOW = 80

# Minimum absolute current density used for fitting, A/cm².
MIN_ABS_CURRENT_DENSITY_A_CM2 = 1e-10

# -------------------------------------------------------------------------
# Ecorr anchoring
# -------------------------------------------------------------------------

# Maximum allowed difference between fitted Tafel-intersection Ecorr and
# observed zero-crossing Ecorr. Units: V.
MAX_ECORR_SHIFT_V = 0.015

# Penalty strength for Ecorr mismatch. Larger values force the selected
# anodic/cathodic pair to extrapolate closer to observed Ecorr.
ECORR_MATCH_WEIGHT = 1000.0

# If no candidate pair meets MAX_ECORR_SHIFT_V, select the best available
# pair and flag it in the Excel/CSV summary.
ALLOW_ECORR_FALLBACK = True

# -------------------------------------------------------------------------
# Plot settings
# -------------------------------------------------------------------------

SHOW_FULL_CURVE_IN_BACKGROUND = True
PNG_DPI = 300


# =============================================================================
# FILE READING
# =============================================================================

COLUMN_SIGNATURES = [
    (r"we.*potential", r"we.*current", "A"),
    (r"ewe", r"i/ma", "mA"),
    (r"ewe", r"/ma", "mA"),
    (r"^vf$", r"^im$", "A"),
    (r"potential/v", r"current/a", "A"),
    (r"e/v", r"i/a", "A"),
    (
        r"potential|volt|^e$|e \(v\)|e_v",
        r"current|amps|^i$|i \(a\)|i_a",
        "A",
    ),
    (r"potential|volt|^e$", r"current.*ma|ima", "mA"),
]

UNIT_HINTS = {
    r"\(a\)|_a$|/a$": 1.0,
    r"\(ma\)|_ma$|/ma$": 1e-3,
    r"\(ua\)|_ua$|/ua$": 1e-6,
    r"a/cm": 1.0,
    r"ma/cm": 1e-3,
}


def auto_detect_columns(dataframe: pd.DataFrame):
    """Detect potential/current columns and current conversion factor to A."""
    lower_names = {
        column: str(column).lower().strip()
        for column in dataframe.columns
    }

    numeric_columns = [
        column
        for column in dataframe.columns
        if pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    for potential_pattern, current_pattern, unit in COLUMN_SIGNATURES:
        potential_matches = [
            column
            for column, lower_name in lower_names.items()
            if re.search(potential_pattern, lower_name)
            and column in numeric_columns
        ]

        current_matches = [
            column
            for column, lower_name in lower_names.items()
            if re.search(current_pattern, lower_name)
            and column in numeric_columns
            and column not in potential_matches
        ]

        if potential_matches and current_matches:
            potential_column = sorted(
                potential_matches,
                key=lambda column: (
                    0 if "we" in str(column).lower() else 1
                ),
            )[0]

            current_column = current_matches[0]
            conversion_factor = 1e-3 if unit == "mA" else 1.0

            for pattern, factor in UNIT_HINTS.items():
                if re.search(pattern, lower_names[current_column]):
                    conversion_factor = factor
                    break

            return potential_column, current_column, conversion_factor

    if len(numeric_columns) >= 2:
        return numeric_columns[0], numeric_columns[1], 1.0

    raise ValueError(
        "Potential/current columns could not be detected. "
        "At least two numeric columns are required."
    )


def load_lsv_file(file_path: Path) -> pd.DataFrame:
    """Read Excel, CSV, TXT, tab-, comma-, semicolon-, or space-separated data."""
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)

    raw_bytes = file_path.read_bytes()
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()

    skip_rows = 0

    for index, line in enumerate(lines):
        values = re.split(r"[,;\t ]+", line.strip())

        numeric_count = sum(
            1
            for value in values
            if re.match(r"^-?[\d.eE+]+$", value)
        )

        if numeric_count >= 2:
            previous_is_header = (
                index > 0
                and not any(
                    re.match(r"^-?[\d.eE+]+$", value)
                    for value in re.split(
                        r"[,;\t ]+",
                        lines[index - 1].strip(),
                    )
                )
            )

            skip_rows = max(0, index - 1) if previous_is_header else index
            break

    for separator in ["\t", ";", ",", r"\s+"]:
        try:
            dataframe = pd.read_csv(
                io.StringIO(text),
                sep=separator,
                skiprows=skip_rows,
                engine="python",
            )

            if dataframe.shape[1] >= 2 and dataframe.shape[0] > 5:
                return dataframe.dropna(axis=1, how="all")

        except Exception:
            continue

    raise ValueError(f"Could not parse input file: {file_path.name}")


# =============================================================================
# ELECTROCHEMICAL HELPERS
# =============================================================================

def estimate_noise_floor(
    current_density: np.ndarray,
    k: int = 5,
) -> float:
    """Estimate current-density noise floor from the smallest |j| values."""
    absolute_current = np.abs(current_density)
    absolute_current = absolute_current[np.isfinite(absolute_current)]

    if len(absolute_current) == 0:
        return MIN_ABS_CURRENT_DENSITY_A_CM2

    k = max(1, min(k, len(absolute_current)))

    estimated_floor = float(
        np.mean(np.sort(absolute_current)[:k])
    )

    return max(
        estimated_floor,
        MIN_ABS_CURRENT_DENSITY_A_CM2,
    )


def estimate_ecorr_from_zero_crossing(
    potential: np.ndarray,
    current_density: np.ndarray,
) -> float:
    """
    Estimate observed Ecorr from the measured current zero crossing.

    If multiple sign changes occur, select the crossing with the smallest
    neighboring absolute current values. If no sign change exists, use the
    potential at minimum absolute current.
    """
    potential = np.asarray(potential, dtype=float)
    current_density = np.asarray(current_density, dtype=float)

    valid = np.isfinite(potential) & np.isfinite(current_density)

    potential = potential[valid]
    current_density = current_density[valid]

    if len(potential) < 2:
        raise ValueError(
            "At least two valid points are required to estimate Ecorr."
        )

    crossings = []

    for index in range(len(potential) - 1):
        potential_1 = potential[index]
        potential_2 = potential[index + 1]

        current_1 = current_density[index]
        current_2 = current_density[index + 1]

        if current_1 == 0:
            crossings.append(
                (potential_1, abs(current_1))
            )
            continue

        if current_1 * current_2 < 0:
            ecorr = (
                potential_1
                + (0.0 - current_1)
                * (potential_2 - potential_1)
                / (current_2 - current_1)
            )

            crossing_quality = abs(current_1) + abs(current_2)

            crossings.append(
                (ecorr, crossing_quality)
            )

    if crossings:
        crossings.sort(key=lambda item: item[1])
        return float(crossings[0][0])

    closest_index = int(
        np.argmin(np.abs(current_density))
    )

    return float(potential[closest_index])


def calculate_window_score(
    log_current: np.ndarray,
    potential: np.ndarray,
    slope: float,
    r_squared: float,
) -> float:
    """
    Score one candidate Tafel window.

    The candidate must satisfy all configured constraints. Higher score is
    better and rewards R², current span, potential span, and point count.
    """
    potential_span = float(np.ptp(potential))
    log_current_span = float(np.ptp(log_current))
    number_of_points = len(log_current)

    if potential_span < MIN_POTENTIAL_SPAN_V:
        return -np.inf

    if log_current_span < MIN_LOG_CURRENT_SPAN_DECADES:
        return -np.inf

    if r_squared < MIN_R2_FOR_ACTIVE_REGION:
        return -np.inf

    if abs(slope) < 0.010:
        return -np.inf

    return (
        1000.0 * r_squared
        + 80.0 * log_current_span
        + 30.0 * potential_span
        + 0.25 * number_of_points
    )


# =============================================================================
# AUTOMATIC ACTIVE-REGION SELECTION
# =============================================================================

def generate_tafel_window_candidates(
    potential: np.ndarray,
    current_density: np.ndarray,
    branch_name: str,
    ecorr_observed: float,
) -> list[dict]:
    """
    Generate all valid linear Tafel-window candidates for one branch.

    Existing constraints are applied:
    - correct current sign;
    - Ecorr exclusion zone;
    - noise floor;
    - minimum points;
    - minimum potential span;
    - minimum log-current span;
    - minimum R².
    """
    potential = np.asarray(potential, dtype=float)
    current_density = np.asarray(current_density, dtype=float)

    noise_floor = estimate_noise_floor(current_density)

    valid = (
        np.isfinite(potential)
        & np.isfinite(current_density)
        & (np.abs(current_density) >= noise_floor)
    )

    if branch_name.lower() == "anodic":
        branch_mask = (
            valid
            & (current_density > 0)
            & (
                potential
                > ecorr_observed + ECORR_EXCLUSION_WINDOW_V
            )
        )

    elif branch_name.lower() == "cathodic":
        branch_mask = (
            valid
            & (current_density < 0)
            & (
                potential
                < ecorr_observed - ECORR_EXCLUSION_WINDOW_V
            )
        )

    else:
        raise ValueError(
            "branch_name must be either 'anodic' or 'cathodic'."
        )

    branch_potential = potential[branch_mask]
    branch_current = current_density[branch_mask]

    if len(branch_potential) < MIN_POINTS_PER_BRANCH:
        raise ValueError(
            f"Not enough valid {branch_name} points after Ecorr exclusion."
        )

    order = np.argsort(branch_potential)

    branch_potential = branch_potential[order]
    branch_current = branch_current[order]

    log_current_all = np.log10(np.abs(branch_current))
    total_points = len(branch_potential)

    maximum_window = (
        total_points
        if MAX_POINTS_PER_WINDOW is None
        else min(MAX_POINTS_PER_WINDOW, total_points)
    )

    candidates = []

    for window_size in range(
        MIN_POINTS_PER_BRANCH,
        maximum_window + 1,
    ):
        for start_index in range(
            0,
            total_points - window_size + 1,
        ):
            end_index = start_index + window_size

            x_window = log_current_all[
                start_index:end_index
            ]

            y_window = branch_potential[
                start_index:end_index
            ]

            regression = linregress(
                x_window,
                y_window,
            )

            r_squared = regression.rvalue ** 2

            linearity_score = calculate_window_score(
                x_window,
                y_window,
                regression.slope,
                r_squared,
            )

            if not np.isfinite(linearity_score):
                continue

            predicted_potential = (
                regression.intercept
                + regression.slope * x_window
            )

            candidates.append(
                {
                    "branch": branch_name,
                    "linearity_score": linearity_score,
                    "potential": y_window,
                    "current": branch_current[
                        start_index:end_index
                    ],
                    "log_current": x_window,
                    "predicted_potential": predicted_potential,
                    "residual": y_window - predicted_potential,
                    "slope_v_dec": regression.slope,
                    "intercept_v": regression.intercept,
                    "r_squared": r_squared,
                    "number_of_points": window_size,
                    "potential_min_v": float(
                        np.min(y_window)
                    ),
                    "potential_max_v": float(
                        np.max(y_window)
                    ),
                    "log_current_min": float(
                        np.min(x_window)
                    ),
                    "log_current_max": float(
                        np.max(x_window)
                    ),
                }
            )

    if not candidates:
        raise ValueError(
            f"No suitable linear {branch_name} Tafel region was detected. "
            "Try reducing MIN_R2_FOR_ACTIVE_REGION or "
            "MIN_LOG_CURRENT_SPAN_DECADES."
        )

    return candidates


def select_ecorr_anchored_tafel_pair(
    anodic_candidates: list[dict],
    cathodic_candidates: list[dict],
    ecorr_observed: float,
) -> dict:
    """
    Select the best anodic/cathodic candidate pair.

    The pair must be linear on both branches and is scored using:
    - anodic linearity score;
    - cathodic linearity score;
    - a quadratic penalty for fitted Ecorr differing from observed Ecorr.

    First, only pairs within MAX_ECORR_SHIFT_V are accepted. If none exist
    and ALLOW_ECORR_FALLBACK is True, the best available pair is returned
    with `ecorr_constraint_met = False`.
    """
    best_pair = None
    best_fallback_pair = None

    for anodic_fit in anodic_candidates:
        for cathodic_fit in cathodic_candidates:
            slope_anodic = anodic_fit["slope_v_dec"]
            intercept_anodic = anodic_fit["intercept_v"]

            slope_cathodic = cathodic_fit["slope_v_dec"]
            intercept_cathodic = cathodic_fit["intercept_v"]

            denominator = slope_anodic - slope_cathodic

            if abs(denominator) < 1e-15:
                continue

            log_icorr = (
                intercept_cathodic - intercept_anodic
            ) / denominator

            ecorr_fitted = (
                intercept_anodic
                + slope_anodic * log_icorr
            )

            if not np.isfinite(ecorr_fitted):
                continue

            if not np.isfinite(log_icorr):
                continue

            ecorr_shift_v = abs(
                ecorr_fitted - ecorr_observed
            )

            normalized_shift = (
                ecorr_shift_v
                / max(MAX_ECORR_SHIFT_V, 1e-12)
            )

            pair_score = (
                anodic_fit["linearity_score"]
                + cathodic_fit["linearity_score"]
                - ECORR_MATCH_WEIGHT * normalized_shift ** 2
            )

            pair = {
                "anodic_fit": anodic_fit,
                "cathodic_fit": cathodic_fit,
                "ecorr_fitted_v": ecorr_fitted,
                "log_icorr": log_icorr,
                "icorr_a_cm2": 10 ** log_icorr,
                "ecorr_shift_v": ecorr_shift_v,
                "pair_score": pair_score,
            }

            if (
                best_fallback_pair is None
                or pair_score > best_fallback_pair["pair_score"]
            ):
                best_fallback_pair = pair

            if ecorr_shift_v <= MAX_ECORR_SHIFT_V:
                if (
                    best_pair is None
                    or pair_score > best_pair["pair_score"]
                ):
                    best_pair = pair

    if best_pair is not None:
        best_pair["ecorr_constraint_met"] = True
        return best_pair

    if ALLOW_ECORR_FALLBACK and best_fallback_pair is not None:
        best_fallback_pair["ecorr_constraint_met"] = False
        return best_fallback_pair

    raise ValueError(
        "No anodic/cathodic Tafel-window pair produced a fitted Ecorr "
        f"within ±{MAX_ECORR_SHIFT_V * 1000:.1f} mV of observed Ecorr."
    )


def fit_active_tafel_region(
    potential: np.ndarray,
    current_density: np.ndarray,
) -> dict:
    """
    Detect active anodic/cathodic regions and select the pair whose
    extrapolated Tafel intersection is closest to observed Ecorr.
    """
    potential = np.asarray(potential, dtype=float)
    current_density = np.asarray(current_density, dtype=float)

    valid = np.isfinite(potential) & np.isfinite(current_density)

    potential = potential[valid]
    current_density = current_density[valid]

    if len(potential) < 2 * MIN_POINTS_PER_BRANCH:
        raise ValueError(
            "Too few valid data points for automatic active-region fitting."
        )

    noise_floor = estimate_noise_floor(current_density)

    ecorr_observed = estimate_ecorr_from_zero_crossing(
        potential,
        current_density,
    )

    anodic_candidates = generate_tafel_window_candidates(
        potential,
        current_density,
        branch_name="anodic",
        ecorr_observed=ecorr_observed,
    )

    cathodic_candidates = generate_tafel_window_candidates(
        potential,
        current_density,
        branch_name="cathodic",
        ecorr_observed=ecorr_observed,
    )

    selected_pair = select_ecorr_anchored_tafel_pair(
        anodic_candidates,
        cathodic_candidates,
        ecorr_observed,
    )

    anodic_fit = selected_pair["anodic_fit"]
    cathodic_fit = selected_pair["cathodic_fit"]

    selected_mask = np.zeros(len(potential), dtype=bool)

    for selected_potential in (
        anodic_fit["potential"],
        cathodic_fit["potential"],
    ):
        for value in selected_potential:
            selected_mask |= np.isclose(
                potential,
                value,
                rtol=0.0,
                atol=1e-12,
            )

    return {
        "noise_floor": noise_floor,

        # Observed Ecorr from current zero crossing.
        "ecorr_estimated_v": ecorr_observed,

        # Fitted Ecorr from selected Tafel-line intersection.
        "ecorr_v": selected_pair["ecorr_fitted_v"],

        "ecorr_shift_v": selected_pair["ecorr_shift_v"],
        "ecorr_constraint_met": selected_pair[
            "ecorr_constraint_met"
        ],
        "pair_score": selected_pair["pair_score"],

        "log_icorr": selected_pair["log_icorr"],
        "icorr_a_cm2": selected_pair["icorr_a_cm2"],

        "anodic_fit": anodic_fit,
        "cathodic_fit": cathodic_fit,
        "active_mask": selected_mask,

        "anodic_potential": anodic_fit["potential"],
        "anodic_current": anodic_fit["current"],
        "cathodic_potential": cathodic_fit["potential"],
        "cathodic_current": cathodic_fit["current"],

        "x_anodic": anodic_fit["log_current"],
        "x_cathodic": cathodic_fit["log_current"],

        "predicted_anodic": anodic_fit[
            "predicted_potential"
        ],
        "predicted_cathodic": cathodic_fit[
            "predicted_potential"
        ],

        "residual_anodic": anodic_fit["residual"],
        "residual_cathodic": cathodic_fit["residual"],

        "slope_anodic_v_dec": anodic_fit["slope_v_dec"],
        "slope_cathodic_v_dec": cathodic_fit["slope_v_dec"],

        "intercept_anodic_v": anodic_fit["intercept_v"],
        "intercept_cathodic_v": cathodic_fit["intercept_v"],

        "beta_anodic_mv_dec": (
            abs(anodic_fit["slope_v_dec"]) * 1000
        ),
        "beta_cathodic_mv_dec": (
            abs(cathodic_fit["slope_v_dec"]) * 1000
        ),

        "r2_anodic": anodic_fit["r_squared"],
        "r2_cathodic": cathodic_fit["r_squared"],
    }


# =============================================================================
# PLOTTING
# =============================================================================

def save_active_tafel_plot(
    folder: Path,
    potential: np.ndarray,
    current_density: np.ndarray,
    fit: dict,
):
    """Build an Ecorr-anchored automatic active-region Tafel plot."""
    figure, axis = plt.subplots(
        figsize=(10, 7),
        dpi=PNG_DPI,
    )

    log_abs_current = np.log10(
        np.maximum(
            np.abs(current_density),
            fit["noise_floor"],
        )
    )

    if SHOW_FULL_CURVE_IN_BACKGROUND:
        axis.plot(
            potential,
            log_abs_current,
            color="#A0A0A0",
            linewidth=1.0,
            alpha=0.60,
            label="Full measured curve",
            zorder=1,
        )

    anodic_region = fit["anodic_fit"]
    cathodic_region = fit["cathodic_fit"]

    axis.axvspan(
        anodic_region["potential_min_v"],
        anodic_region["potential_max_v"],
        color="#D62728",
        alpha=0.10,
        label="Auto-selected anodic region",
        zorder=0,
    )

    axis.axvspan(
        cathodic_region["potential_min_v"],
        cathodic_region["potential_max_v"],
        color="#1F77B4",
        alpha=0.10,
        label="Auto-selected cathodic region",
        zorder=0,
    )

    axis.scatter(
        fit["anodic_potential"],
        fit["x_anodic"],
        color="#D62728",
        s=30,
        label="Anodic fit data",
        zorder=3,
    )

    axis.scatter(
        fit["cathodic_potential"],
        fit["x_cathodic"],
        color="#1F77B4",
        s=30,
        label="Cathodic fit data",
        zorder=3,
    )

    anodic_order = np.argsort(fit["x_anodic"])
    cathodic_order = np.argsort(fit["x_cathodic"])

    axis.plot(
        fit["predicted_anodic"][anodic_order],
        fit["x_anodic"][anodic_order],
        color="#8B0000",
        linewidth=2.5,
        label="Anodic Tafel fit",
        zorder=4,
    )

    axis.plot(
        fit["predicted_cathodic"][cathodic_order],
        fit["x_cathodic"][cathodic_order],
        color="#003F8C",
        linewidth=2.5,
        label="Cathodic Tafel fit",
        zorder=4,
    )

    # Observed Ecorr from measured zero crossing.
    axis.axvline(
        fit["ecorr_estimated_v"],
        color="#666666",
        linestyle=":",
        linewidth=1.5,
        alpha=0.9,
        label="Ecorr: observed zero crossing",
    )

    # Fitted Ecorr from Tafel-line intersection.
    axis.axvline(
        fit["ecorr_v"],
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.9,
        label="Ecorr: Tafel intersection",
    )

    # This marker is exactly at (Ecorr fitted, log10(icorr)).
    axis.scatter(
        [fit["ecorr_v"]],
        [fit["log_icorr"]],
        marker="X",
        s=160,
        color="black",
        label="Ecorr / icorr",
        zorder=5,
    )

    axis.set_xlabel("Potential / V")
    axis.set_ylabel(
        r"$\log_{10}(|j|)$ / log(A cm$^{-2}$)"
    )
    axis.set_title(
        f"Ecorr-Anchored Automatic Active-Region Tafel Fit — {folder.name}"
    )

    axis.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.5,
    )

    axis.legend(
        loc="best",
        fontsize=8,
    )

    ecorr_shift_mv_signed = (
        fit["ecorr_v"] - fit["ecorr_estimated_v"]
    ) * 1000

    ecorr_status = (
        "Ecorr constraint: met"
        if fit["ecorr_constraint_met"]
        else "Ecorr constraint: fallback used"
    )

    annotation = (
        f"Ecorr observed = {fit['ecorr_estimated_v']:.5f} V\n"
        f"Ecorr fitted = {fit['ecorr_v']:.5f} V\n"
        f"ΔEcorr = {ecorr_shift_mv_signed:+.1f} mV\n"
        f"{ecorr_status}\n"
        f"icorr = {fit['icorr_a_cm2']:.3e} A/cm²\n"
        f"βa = {fit['beta_anodic_mv_dec']:.1f} mV/dec "
        f"(R² = {fit['r2_anodic']:.4f})\n"
        f"βc = {fit['beta_cathodic_mv_dec']:.1f} mV/dec "
        f"(R² = {fit['r2_cathodic']:.4f})"
    )

    axis.text(
        0.02,
        0.98,
        annotation,
        transform=axis.transAxes,
        verticalalignment="top",
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.92,
        },
    )

    figure.tight_layout()

    # Streamlit: return the figure instead of saving it to a folder.
    return figure

# =============================================================================
# OUTPUT TABLES
# =============================================================================

def create_result_row(
    folder: Path,
    source_file: Path,
    potential_column,
    current_column,
    current_factor: float,
    fit: dict,
) -> dict:
    """Create one summary row for one point folder."""
    return {
        "Folder": folder.name,
        "Source file": source_file.name,
        "Status": (
            "Success"
            if fit["ecorr_constraint_met"]
            else "Success with Ecorr fallback"
        ),
        "Potential column": str(potential_column),
        "Current column": str(current_column),
        "Current conversion factor to A": current_factor,
        "Electrode area (cm²)": ELECTRODE_AREA_CM2,
        "Noise floor (A/cm²)": fit["noise_floor"],

        "Ecorr observed zero crossing (V)": (
            fit["ecorr_estimated_v"]
        ),
        "Ecorr fitted Tafel intersection (V)": fit["ecorr_v"],
        "Ecorr difference fitted - observed (mV)": (
            (fit["ecorr_v"] - fit["ecorr_estimated_v"]) * 1000
        ),
        "Ecorr constraint met": fit["ecorr_constraint_met"],
        "Tafel-pair score": fit["pair_score"],

        "icorr (A/cm²)": fit["icorr_a_cm2"],
        "log10(icorr / A cm⁻²)": fit["log_icorr"],

        "Anodic active region min (V)": (
            fit["anodic_fit"]["potential_min_v"]
        ),
        "Anodic active region max (V)": (
            fit["anodic_fit"]["potential_max_v"]
        ),
        "Anodic active region log-current span (dec)": (
            fit["anodic_fit"]["log_current_max"]
            - fit["anodic_fit"]["log_current_min"]
        ),
        "Anodic fit points": (
            fit["anodic_fit"]["number_of_points"]
        ),
        "Anodic Tafel slope βa (mV/dec)": (
            fit["beta_anodic_mv_dec"]
        ),
        "Anodic slope (V/dec)": (
            fit["slope_anodic_v_dec"]
        ),
        "Anodic R²": fit["r2_anodic"],

        "Cathodic active region min (V)": (
            fit["cathodic_fit"]["potential_min_v"]
        ),
        "Cathodic active region max (V)": (
            fit["cathodic_fit"]["potential_max_v"]
        ),
        "Cathodic active region log-current span (dec)": (
            fit["cathodic_fit"]["log_current_max"]
            - fit["cathodic_fit"]["log_current_min"]
        ),
        "Cathodic fit points": (
            fit["cathodic_fit"]["number_of_points"]
        ),
        "Cathodic Tafel slope βc (mV/dec)": (
            fit["beta_cathodic_mv_dec"]
        ),
        "Cathodic slope (V/dec)": (
            fit["slope_cathodic_v_dec"]
        ),
        "Cathodic R²": fit["r2_cathodic"],
    }


# =============================================================================
# GAMRY .DTA READER (added for the Streamlit app)
# =============================================================================

def load_gamry_dta(text: str):
    """
    Parse a Gamry Framework .DTA file.

    Returns (dataframe of the CURVE table, header dict of key -> value).
    The CURVE table's column names (Vf, Im, ...) are kept so that
    auto_detect_columns() picks Vf / Im exactly as for exported data.
    """
    lines = text.splitlines()
    header = {}
    table_start = None

    for index, line in enumerate(lines):
        parts = line.split("\t")
        if len(parts) >= 3 and parts[1] == "TABLE" and parts[0].upper().startswith("CURVE"):
            table_start = index
            break
        if len(parts) >= 3:
            header[parts[0].strip()] = parts[2].strip()

    if table_start is None:
        raise ValueError("No CURVE TABLE block found — is this a Gamry .DTA file?")

    column_names = [c.strip() for c in lines[table_start + 1].split("\t")]
    # line table_start + 2 holds the units row
    rows = []
    for line in lines[table_start + 3:]:
        parts = line.split("\t")
        if len(parts) < len(column_names) or parts[0].strip() != "":
            break  # end of table (next block or blank line)
        rows.append(parts[: len(column_names)])

    dataframe = pd.DataFrame(rows, columns=column_names)
    dataframe = dataframe.loc[:, [c for c in dataframe.columns if c]]
    for column in dataframe.columns:
        converted = pd.to_numeric(dataframe[column], errors="coerce")
        if converted.notna().mean() > 0.9:
            dataframe[column] = converted

    return dataframe, header


# =============================================================================
# FAST (VECTORISED) EQUIVALENTS — same maths, same constraints, same ordering
# =============================================================================
# The original functions loop in pure Python (one scipy.linregress per window,
# then every anodic x cathodic pair). For Gamry files with thousands of points
# that takes minutes. These versions compute the identical quantities with
# NumPy arrays, keep the same candidate order, and break ties the same way
# (first-encountered wins), so the selected fit is the same.

from numpy.lib.stride_tricks import sliding_window_view


def generate_tafel_window_candidates_fast(
    potential, current_density, branch_name, ecorr_observed
):
    potential = np.asarray(potential, dtype=float)
    current_density = np.asarray(current_density, dtype=float)

    noise_floor = estimate_noise_floor(current_density)

    valid = (
        np.isfinite(potential)
        & np.isfinite(current_density)
        & (np.abs(current_density) >= noise_floor)
    )

    if branch_name.lower() == "anodic":
        branch_mask = (
            valid
            & (current_density > 0)
            & (potential > ecorr_observed + ECORR_EXCLUSION_WINDOW_V)
        )
    elif branch_name.lower() == "cathodic":
        branch_mask = (
            valid
            & (current_density < 0)
            & (potential < ecorr_observed - ECORR_EXCLUSION_WINDOW_V)
        )
    else:
        raise ValueError("branch_name must be either 'anodic' or 'cathodic'.")

    branch_potential = potential[branch_mask]
    branch_current = current_density[branch_mask]

    if len(branch_potential) < MIN_POINTS_PER_BRANCH:
        raise ValueError(
            f"Not enough valid {branch_name} points after Ecorr exclusion."
        )

    order = np.argsort(branch_potential)
    branch_potential = branch_potential[order]
    branch_current = branch_current[order]

    log_current_all = np.log10(np.abs(branch_current))
    total_points = len(branch_potential)

    maximum_window = (
        total_points
        if MAX_POINTS_PER_WINDOW is None
        else min(MAX_POINTS_PER_WINDOW, total_points)
    )

    candidates = []

    for window_size in range(MIN_POINTS_PER_BRANCH, maximum_window + 1):
        X = sliding_window_view(log_current_all, window_size)
        Y = sliding_window_view(branch_potential, window_size)

        # Same as scipy.stats.linregress (np.cov with bias=1)
        x_mean = X.mean(axis=1)
        y_mean = Y.mean(axis=1)
        dx = X - x_mean[:, None]
        dy = Y - y_mean[:, None]
        ssxm = (dx * dx).mean(axis=1)
        ssym = (dy * dy).mean(axis=1)
        ssxym = (dx * dy).mean(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            slope = ssxym / ssxm
            intercept = y_mean - slope * x_mean
            r = ssxym / np.sqrt(ssxm * ssym)
        r = np.where((ssxm == 0) | (ssym == 0), 0.0, np.clip(r, -1.0, 1.0))
        r_squared = r ** 2

        potential_span = Y.max(axis=1) - Y.min(axis=1)
        log_current_span = X.max(axis=1) - X.min(axis=1)

        ok = (
            (ssxm > 0)
            & ~(potential_span < MIN_POTENTIAL_SPAN_V)
            & ~(log_current_span < MIN_LOG_CURRENT_SPAN_DECADES)
            & ~(r_squared < MIN_R2_FOR_ACTIVE_REGION)
            & ~(np.abs(slope) < 0.010)
        )

        score = (
            1000.0 * r_squared
            + 80.0 * log_current_span
            + 30.0 * potential_span
            + 0.25 * window_size
        )
        ok &= np.isfinite(score)

        for start_index in np.flatnonzero(ok):
            end_index = start_index + window_size
            x_window = log_current_all[start_index:end_index]
            y_window = branch_potential[start_index:end_index]
            s = float(slope[start_index])
            b = float(intercept[start_index])
            predicted_potential = b + s * x_window

            candidates.append(
                {
                    "branch": branch_name,
                    "linearity_score": float(score[start_index]),
                    "potential": y_window,
                    "current": branch_current[start_index:end_index],
                    "log_current": x_window,
                    "predicted_potential": predicted_potential,
                    "residual": y_window - predicted_potential,
                    "slope_v_dec": s,
                    "intercept_v": b,
                    "r_squared": float(r_squared[start_index]),
                    "number_of_points": window_size,
                    "potential_min_v": float(np.min(y_window)),
                    "potential_max_v": float(np.max(y_window)),
                    "log_current_min": float(np.min(x_window)),
                    "log_current_max": float(np.max(x_window)),
                }
            )

    if not candidates:
        raise ValueError(
            f"No suitable linear {branch_name} Tafel region was detected. "
            "Try reducing MIN_R2_FOR_ACTIVE_REGION or "
            "MIN_LOG_CURRENT_SPAN_DECADES."
        )

    return candidates


def select_ecorr_anchored_tafel_pair_fast(
    anodic_candidates, cathodic_candidates, ecorr_observed
):
    sa = np.array([c["slope_v_dec"] for c in anodic_candidates])
    ia = np.array([c["intercept_v"] for c in anodic_candidates])
    la = np.array([c["linearity_score"] for c in anodic_candidates])
    sc = np.array([c["slope_v_dec"] for c in cathodic_candidates])
    ic = np.array([c["intercept_v"] for c in cathodic_candidates])
    lc = np.array([c["linearity_score"] for c in cathodic_candidates])

    n_c = len(sc)
    chunk = max(1, int(4_000_000 // max(n_c, 1)))

    best = None           # (score, i, j) within MAX_ECORR_SHIFT_V
    best_fallback = None  # (score, i, j) any valid pair

    for i0 in range(0, len(sa), chunk):
        i1 = min(i0 + chunk, len(sa))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            denominator = sa[i0:i1, None] - sc[None, :]
            log_icorr = (ic[None, :] - ia[i0:i1, None]) / denominator
            ecorr_fitted = ia[i0:i1, None] + sa[i0:i1, None] * log_icorr
            shift = np.abs(ecorr_fitted - ecorr_observed)
            normalized = shift / max(MAX_ECORR_SHIFT_V, 1e-12)
            score = (
                la[i0:i1, None] + lc[None, :]
                - ECORR_MATCH_WEIGHT * normalized ** 2
            )
        valid = (
            (np.abs(denominator) >= 1e-15)
            & np.isfinite(ecorr_fitted)
            & np.isfinite(log_icorr)
        )

        s_all = np.where(valid, score, -np.inf)
        k = int(np.argmax(s_all))
        if valid.flat[k] and (best_fallback is None or s_all.flat[k] > best_fallback[0]):
            best_fallback = (s_all.flat[k], i0 + k // n_c, k % n_c)

        s_in = np.where(valid & (shift <= MAX_ECORR_SHIFT_V), score, -np.inf)
        k = int(np.argmax(s_in))
        if np.isfinite(s_in.flat[k]) and (best is None or s_in.flat[k] > best[0]):
            best = (s_in.flat[k], i0 + k // n_c, k % n_c)

    def build(i, j):
        a = anodic_candidates[i]
        c = cathodic_candidates[j]
        log_icorr = (c["intercept_v"] - a["intercept_v"]) / (a["slope_v_dec"] - c["slope_v_dec"])
        ecorr_fitted = a["intercept_v"] + a["slope_v_dec"] * log_icorr
        shift = abs(ecorr_fitted - ecorr_observed)
        return {
            "anodic_fit": a,
            "cathodic_fit": c,
            "ecorr_fitted_v": ecorr_fitted,
            "log_icorr": log_icorr,
            "icorr_a_cm2": 10 ** log_icorr,
            "ecorr_shift_v": shift,
            "pair_score": a["linearity_score"] + c["linearity_score"]
            - ECORR_MATCH_WEIGHT * (shift / max(MAX_ECORR_SHIFT_V, 1e-12)) ** 2,
        }

    if best is not None:
        pair = build(best[1], best[2])
        pair["ecorr_constraint_met"] = True
        return pair

    if ALLOW_ECORR_FALLBACK and best_fallback is not None:
        pair = build(best_fallback[1], best_fallback[2])
        pair["ecorr_constraint_met"] = False
        return pair

    raise ValueError(
        "No anodic/cathodic Tafel-window pair produced a fitted Ecorr "
        f"within ±{MAX_ECORR_SHIFT_V * 1000:.1f} mV of observed Ecorr."
    )


_ORIGINAL_GENERATE = generate_tafel_window_candidates
_ORIGINAL_SELECT = select_ecorr_anchored_tafel_pair


def set_fast_mode(enabled: bool) -> None:
    """Switch fit_active_tafel_region between original and vectorised helpers."""
    global generate_tafel_window_candidates, select_ecorr_anchored_tafel_pair
    if enabled:
        generate_tafel_window_candidates = generate_tafel_window_candidates_fast
        select_ecorr_anchored_tafel_pair = select_ecorr_anchored_tafel_pair_fast
    else:
        generate_tafel_window_candidates = _ORIGINAL_GENERATE
        select_ecorr_anchored_tafel_pair = _ORIGINAL_SELECT


# =============================================================================
# OUTPUT FILES — copied unchanged from the original script
# =============================================================================
ASCII_FOLDER = Path('.')  # set by app.py to a temporary output folder

def save_individual_results(
    folder: Path,
    potential: np.ndarray,
    current_density: np.ndarray,
    result_row: dict,
    fit: dict,
) -> None:
    """Save individual CSV and Excel result files."""
    result_table = pd.DataFrame(
        {
            "Parameter": list(result_row.keys()),
            "Value": list(result_row.values()),
        }
    )

    full_data = pd.DataFrame(
        {
            "Potential (V)": potential,
            "Current density (A/cm²)": current_density,
            "log10(abs(current density))": np.log10(
                np.maximum(
                    np.abs(current_density),
                    fit["noise_floor"],
                )
            ),
            "In auto-selected active region": fit["active_mask"],
        }
    )

    anodic_fit_data = pd.DataFrame(
        {
            "Potential measured (V)": fit["anodic_potential"],
            "log10(j) measured": fit["x_anodic"],
            "Potential fitted (V)": fit["predicted_anodic"],
            "Residual (V)": fit["residual_anodic"],
        }
    )

    cathodic_fit_data = pd.DataFrame(
        {
            "Potential measured (V)": fit["cathodic_potential"],
            "log10(abs(j)) measured": fit["x_cathodic"],
            "Potential fitted (V)": fit["predicted_cathodic"],
            "Residual (V)": fit["residual_cathodic"],
        }
    )

    result_table.to_csv(
        folder / "tafel_active_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    full_data.to_csv(
        folder / "tafel_active_data.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with pd.ExcelWriter(
        folder / "tafel_active_results.xlsx",
        engine="openpyxl",
    ) as writer:
        result_table.to_excel(
            writer,
            sheet_name="Fit Results",
            index=False,
        )

        full_data.to_excel(
            writer,
            sheet_name="All Data",
            index=False,
        )

        anodic_fit_data.to_excel(
            writer,
            sheet_name="Anodic Fit",
            index=False,
        )

        cathodic_fit_data.to_excel(
            writer,
            sheet_name="Cathodic Fit",
            index=False,
        )



def save_summary(summary_rows: list[dict]) -> None:
    """Save combined Excel and CSV summary files in the ASCII folder."""
    summary = pd.DataFrame(summary_rows)

    if "Folder" in summary.columns:
        remaining_columns = [
            column
            for column in summary.columns
            if column != "Folder"
        ]

        summary = summary[
            ["Folder"] + remaining_columns
        ]

    excel_path = ASCII_FOLDER / "Tafel Summary.xlsx"
    csv_path = ASCII_FOLDER / "Tafel Summary.csv"

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:
        summary.to_excel(
            writer,
            sheet_name="Tafel Summary",
            index=False,
        )

        worksheet = writer.sheets["Tafel Summary"]
        worksheet.freeze_panes = "A2"

        for column_cells in worksheet.columns:
            column_letter = column_cells[0].column_letter

            maximum_length = max(
                len(str(cell.value))
                if cell.value is not None
                else 0
                for cell in column_cells
            )

            worksheet.column_dimensions[
                column_letter
            ].width = min(
                maximum_length + 2,
                45,
            )

    summary.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nSummary Excel: {excel_path}")
    print(f"Summary CSV:   {csv_path}")
