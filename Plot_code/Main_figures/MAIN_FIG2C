#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot open-ocean RW convergence changes for two radial zones."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter


# ============================================================
# 1. Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = (
    PROJECT_ROOT
    / "Data"
    / "Processed"
    / "ERA5"
    / "Convergence"
    / "Bootstrap"
    / "CONVERGENCE_LAYERS_1982_2024_MSWEP30_BOOTSTRAP_LONG.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "Results" / "Main_figures" / "FIG2"

SPATIAL_TYPE = "OPEN_OCEAN"
VERTICAL_PRODUCT = "500_PB"
DISTANCE_ZONES = ["R000_200", "R200_500"]
DISTANCE_LABELS = ["0–200 km", "200–500 km"]
GROUP_CODE = "LE_5TH"
GROUP_COLOR = "#56A1BB"

FIGURE_DPI = 600
FIGURE_SIZE = (8.0, 10.0)
PLOT_FONT_SIZE = 30

# The displayed axis uses a 1e-6 multiplier.
Y_AXIS_MIN = -4.0e-6
Y_AXIS_MAX = 1.0e-6
Y_TICK_INTERVAL = 1.0e-6

BAR_WIDTH = 0.34
BAR_EDGE_WIDTH = 2.6
ERROR_LINE_WIDTH = 2.4
ERROR_CAP_SIZE = 8.0

GRID_COLOR = "#9E9E9E"
GRID_ALPHA = 0.55
GRID_LINE_WIDTH = 1.0

NUMERIC_COLUMNS = [
    "COMMON_VALID_WINDOW_COUNT",
    "MEAN_CHANGE",
    "BOOTSTRAP_CI_LOWER",
    "BOOTSTRAP_CI_UPPER",
]

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": PLOT_FONT_SIZE,
        "axes.labelsize": PLOT_FONT_SIZE,
        "xtick.labelsize": PLOT_FONT_SIZE,
        "ytick.labelsize": PLOT_FONT_SIZE,
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "axes.edgecolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
    }
)


# ============================================================
# 2. Input validation
# ============================================================

def read_and_validate_table():
    """Read and validate the two selected OPEN_OCEAN RW rows."""
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Input table does not exist: {INPUT_CSV}")

    print("=" * 96)
    print("Reading FUHE 500-PB convergence bootstrap table")
    print("=" * 96)
    print(f"Input file: {INPUT_CSV}")

    frame = pd.read_csv(INPUT_CSV, low_memory=False)
    required_columns = [
        "SPATIAL_TYPE",
        "VERTICAL_PRODUCT",
        "INTENSITY_GROUP_CODE",
        "DISTANCE_ZONE",
        *NUMERIC_COLUMNS,
    ]
    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        raise KeyError(f"Input table is missing required columns: {missing}")

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[NUMERIC_COLUMNS] = frame[NUMERIC_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )

    selected = frame.loc[
        frame["SPATIAL_TYPE"].eq(SPATIAL_TYPE)
        & frame["VERTICAL_PRODUCT"].eq(VERTICAL_PRODUCT)
        & frame["INTENSITY_GROUP_CODE"].eq(GROUP_CODE)
        & frame["DISTANCE_ZONE"].isin(DISTANCE_ZONES)
    ].copy()

    if len(selected) != len(DISTANCE_ZONES):
        raise ValueError(
            f"Incomplete plotting rows: found={len(selected)}, "
            f"expected={len(DISTANCE_ZONES)}"
        )

    duplicate = selected.duplicated(
        ["SPATIAL_TYPE", "VERTICAL_PRODUCT", "INTENSITY_GROUP_CODE", "DISTANCE_ZONE"],
        keep=False,
    )
    if duplicate.any():
        examples = selected.loc[
            duplicate,
            ["SPATIAL_TYPE", "VERTICAL_PRODUCT", "INTENSITY_GROUP_CODE", "DISTANCE_ZONE"],
        ].to_dict("records")
        raise ValueError(f"Duplicate plotting combinations: {examples}")

    selected = (
        selected.set_index("DISTANCE_ZONE")
        .reindex(DISTANCE_ZONES)
        .reset_index()
    )

    if selected[NUMERIC_COLUMNS].isna().any(axis=None):
        raise ValueError("Missing or invalid values exist in the plotting rows")

    lower = selected["BOOTSTRAP_CI_LOWER"].to_numpy(dtype=float)
    mean = selected["MEAN_CHANGE"].to_numpy(dtype=float)
    upper = selected["BOOTSTRAP_CI_UPPER"].to_numpy(dtype=float)

    if np.any(lower > upper):
        raise ValueError("A bootstrap lower bound exceeds its upper bound")
    if np.any(lower > mean) or np.any(mean > upper):
        raise ValueError(
            "A mean lies outside its bootstrap confidence interval"
        )

    if lower.min() < Y_AXIS_MIN or upper.max() > Y_AXIS_MAX:
        raise ValueError(
            "The requested fixed Y-axis would clip a confidence interval: "
            f"data range={lower.min():.6e} to {upper.max():.6e}, "
            f"axis range={Y_AXIS_MIN:.6e} to {Y_AXIS_MAX:.6e}"
        )

    count_values = selected["COMMON_VALID_WINDOW_COUNT"].to_numpy(float)
    if np.any(count_values < 0) or not np.isclose(
        count_values, np.round(count_values)
    ).all():
        raise ValueError("Invalid COMMON_VALID_WINDOW_COUNT values were found")

    selected["COMMON_VALID_WINDOW_COUNT"] = np.round(
        count_values
    ).astype(np.int64)

    print(f"Input rows: {len(frame):,}")
    print(f"Selected plotting rows: {len(selected)}")
    print("[PASS] Plotting-data validation completed")
    return selected


# ============================================================
# 3. Gradient bar
# ============================================================

def draw_gradient_bar(axis, x_position, value, width, end_color):
    """Draw one white-at-zero to blue-at-value gradient bar."""
    bar = axis.bar(
        x_position,
        value,
        width=width,
        facecolor="none",
        edgecolor="#000000",
        linewidth=BAR_EDGE_WIDTH,
        zorder=3,
    )[0]

    if np.isclose(value, 0.0, atol=1.0e-20):
        return

    bottom = min(0.0, value)
    top = max(0.0, value)
    colors = (
        ["#FFFFFF", end_color]
        if value > 0
        else [end_color, "#FFFFFF"]
    )
    cmap = LinearSegmentedColormap.from_list(
        f"gradient_{x_position}",
        colors,
        N=512,
    )
    gradient = np.linspace(0.0, 1.0, 1024).reshape(-1, 1)
    image = axis.imshow(
        gradient,
        extent=[
            x_position - width / 2.0,
            x_position + width / 2.0,
            bottom,
            top,
        ],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="bicubic",
        zorder=2,
    )
    image.set_clip_path(bar)


# ============================================================
# 4. Plot
# ============================================================

def plot_figure(plot_frame):
    """Create FIG2C with the established two-bar layout."""
    x = np.arange(len(DISTANCE_ZONES), dtype=float)
    mean = plot_frame["MEAN_CHANGE"].to_numpy(dtype=float)
    lower = plot_frame["BOOTSTRAP_CI_LOWER"].to_numpy(dtype=float)
    upper = plot_frame["BOOTSTRAP_CI_UPPER"].to_numpy(dtype=float)

    fig, axis = plt.subplots(figsize=FIGURE_SIZE)

    for x_position, value in zip(x, mean):
        draw_gradient_bar(
            axis=axis,
            x_position=x_position,
            value=value,
            width=BAR_WIDTH,
            end_color=GROUP_COLOR,
        )

    y_error = np.vstack((mean - lower, upper - mean))
    axis.errorbar(
        x,
        mean,
        yerr=y_error,
        fmt="none",
        ecolor="#000000",
        elinewidth=ERROR_LINE_WIDTH,
        capsize=ERROR_CAP_SIZE,
        capthick=ERROR_LINE_WIDTH,
        zorder=5,
    )

    axis.axhline(
        0.0,
        color="#000000",
        linestyle="--",
        linewidth=1.6,
        zorder=6,
    )

    axis.set_xticks(x)
    axis.set_xlim(-0.5, len(DISTANCE_ZONES) - 0.5)
    axis.set_xticklabels(
        DISTANCE_LABELS,
        fontsize=PLOT_FONT_SIZE,
        rotation=0,
        ha="center",
        color="#000000",
    )

    axis.set_xlabel("")
    axis.set_ylabel(
        r"Change in convergence (s$^{-1}$)",
        fontsize=PLOT_FONT_SIZE,
        fontfamily="Arial",
        color="#000000",
        labelpad=10,
    )

    axis.tick_params(
        axis="x",
        labelsize=PLOT_FONT_SIZE,
        pad=8,
        length=10,
        width=2.0,
        colors="#000000",
        direction="out",
    )
    axis.tick_params(
        axis="y",
        labelsize=PLOT_FONT_SIZE,
        length=10,
        width=2.0,
        colors="#000000",
        direction="out",
    )

    axis.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)
    axis.set_yticks(
        np.arange(
            Y_AXIS_MIN,
            Y_AXIS_MAX + Y_TICK_INTERVAL * 0.5,
            Y_TICK_INTERVAL,
        )
    )

    formatter = ScalarFormatter(useMathText=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    formatter.set_useOffset(False)
    axis.yaxis.set_major_formatter(formatter)

    offset_text = axis.yaxis.get_offset_text()
    offset_text.set_fontsize(PLOT_FONT_SIZE)
    offset_text.set_fontfamily("Arial")
    offset_text.set_color("#000000")
    offset_text.set_y(1.025)
    offset_text.set_verticalalignment("bottom")

    axis.set_axisbelow(True)
    axis.grid(
        axis="y",
        visible=True,
        linestyle="--",
        color=GRID_COLOR,
        alpha=GRID_ALPHA,
        linewidth=GRID_LINE_WIDTH,
    )
    axis.grid(axis="x", visible=False)

    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(2.8)

    fig.subplots_adjust(
        left=0.25,
        right=0.98,
        bottom=0.14,
        top=0.90,
    )

    output_path = OUTPUT_DIR / "FIG2C.png"
    fig.savefig(
        output_path,
        dpi=FIGURE_DPI,
        facecolor="white",
    )
    plt.close(fig)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Figure output is missing or empty: {output_path}")

    print(f"Saved figure: {output_path}")
    return output_path


# ============================================================
# 5. Main program
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_frame = read_and_validate_table()

    print("\n" + "=" * 96)
    print("Creating OPEN_OCEAN RW convergence FIG2C")
    print("=" * 96)

    plot_figure(plot_frame)

    print("\n" + "=" * 96)
    print("FIG2C completed successfully")
    print("=" * 96)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
