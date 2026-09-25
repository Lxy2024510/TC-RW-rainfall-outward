#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot the nearshore RW change in convergence for five radial bands."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


INPUT_CSV = Path(
    str(PROJECT_ROOT / "Data" / "Processed" / "MSWEP_CONVERGENCE_LAYERS_BOOTSTRAP_RESULTS") + "/" +
    "CONVERGENCE_LAYERS_1982_2024_MSWEP30_BOOTSTRAP_LONG.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "Results" / "Extended_figures" / "EX_FIG5"
OUTPUT_PNG = OUTPUT_DIR / "EX_FIG5C.png"

SPATIAL_TYPE = "NEARSHORE"
VERTICAL_PRODUCT = "500_PB"
RW_GROUP_CODE = "LE_5TH"

DISTANCE_ZONES = [
    "R000_100",
    "R100_200",
    "R200_300",
    "R300_400",
    "R400_500",
]
DISTANCE_LABELS = [
    "0–100 km",
    "100–200 km",
    "200–300 km",
    "300–400 km",
    "400–500 km",
]

RW_COLOR = "#D5A0B6"
FONT_SIZE = 32
FIGURE_SIZE = (6.4, 12.8)
FIGURE_DPI = 600

Y_MIN = -7.0e-6
Y_MAX = 1.0e-6
Y_STEP = 2.0e-6
Y_TICKS = np.array([-6.0, -4.0, -2.0, 0.0]) * 1.0e-6


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.cursive": ["Arial"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.cal": "Arial",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
    }
)


def read_rw_data() -> pd.DataFrame:
    """Read and validate the five nearshore RW plotting records."""
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Input table does not exist: {INPUT_CSV}")

    frame = pd.read_csv(INPUT_CSV, low_memory=False)
    required = {
        "SPATIAL_TYPE",
        "VERTICAL_PRODUCT",
        "INTENSITY_GROUP_CODE",
        "DISTANCE_ZONE",
        "MEAN_CHANGE",
        "BOOTSTRAP_CI_LOWER",
        "BOOTSTRAP_CI_UPPER",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Input table is missing required columns: {missing}")

    frame = frame.loc[
        frame["SPATIAL_TYPE"].eq(SPATIAL_TYPE)
        & frame["VERTICAL_PRODUCT"].eq(VERTICAL_PRODUCT)
        & frame["INTENSITY_GROUP_CODE"].eq(RW_GROUP_CODE)
        & frame["DISTANCE_ZONE"].isin(DISTANCE_ZONES)
    ].copy()

    numeric_columns = [
        "MEAN_CHANGE",
        "BOOTSTRAP_CI_LOWER",
        "BOOTSTRAP_CI_UPPER",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[numeric_columns] = frame[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    if frame["DISTANCE_ZONE"].duplicated().any():
        raise ValueError("Duplicate nearshore RW distance-zone records were found.")

    frame = frame.set_index("DISTANCE_ZONE").reindex(DISTANCE_ZONES)
    if frame[numeric_columns].isna().any(axis=None):
        raise ValueError("The five nearshore RW records are incomplete or invalid.")

    if (frame["BOOTSTRAP_CI_LOWER"] > frame["BOOTSTRAP_CI_UPPER"]).any():
        raise ValueError("A bootstrap lower bound exceeds its upper bound.")

    return frame


def format_scaled_y_tick(value: float, _position: int) -> str:
    """Format scaled ticks with the proper Unicode minus sign."""
    scaled = value / 1.0e-6
    if np.isclose(scaled, 0.0):
        return "0"
    if scaled < 0.0:
        return f"\N{MINUS SIGN}{abs(scaled):g}"
    return f"{scaled:g}"


def plot_figure(frame: pd.DataFrame) -> None:
    """Create the formal nearshore RW convergence figure."""
    x = np.arange(len(DISTANCE_ZONES), dtype=float)
    mean = frame["MEAN_CHANGE"].to_numpy(dtype=float)
    lower = frame["BOOTSTRAP_CI_LOWER"].to_numpy(dtype=float)
    upper = frame["BOOTSTRAP_CI_UPPER"].to_numpy(dtype=float)

    fig, axis = plt.subplots(figsize=FIGURE_SIZE)

    axis.fill_between(
        x, lower, upper, color=RW_COLOR, alpha=0.18, linewidth=0.0, zorder=2
    )
    axis.plot(
        x,
        mean,
        color=RW_COLOR,
        linewidth=4.0,
        marker="o",
        markersize=10.0,
        zorder=3,
    )

    # Auxiliary dashed lines follow the established FIG5 style.
    for value in Y_TICKS:
        if np.isclose(value, 0.0):
            continue
        axis.axhline(
            value,
            color="#C4C4C4",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            zorder=0,
        )
    axis.axhline(
        0.0,
        color="#333333",
        linestyle="--",
        linewidth=1.4,
        zorder=1,
    )

    axis.set_xlim(-0.2, len(x) - 0.8)
    axis.set_ylim(Y_MIN, Y_MAX)
    # Use the regular two-unit sequence: -6, -4, -2, and 0.
    axis.set_yticks(Y_TICKS)
    axis.yaxis.set_major_formatter(FuncFormatter(format_scaled_y_tick))

    axis.set_xticks(x)
    axis.set_xticklabels(
        DISTANCE_LABELS,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    axis.set_xlabel("")
    axis.set_ylabel(
        r"Change in convergence (s$^{-1}$)",
        labelpad=16,
    )

    axis.tick_params(
        axis="x",
        pad=8,
        length=10,
        width=2.0,
        direction="out",
        colors="#000000",
    )
    axis.tick_params(
        axis="y",
        length=10,
        width=2.0,
        direction="out",
        colors="#000000",
    )

    for spine in axis.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(2.0)

    # Draw the multiplier manually so its distance from the top axis is stable.
    axis.text(
        0.0,
        1.010,
        "1e\N{MINUS SIGN}6",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_SIZE,
        color="#000000",
        clip_on=False,
    )

    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.25, top=0.96)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved figure: {OUTPUT_PNG}")
    print(f"Spatial category: {SPATIAL_TYPE}")
    print("Intensity group: RW (LE_5TH)")
    print("Y-axis: -7 to 1 × 10^-6 s^-1; ticks = -6, -4, -2, 0")


def main() -> None:
    plot_figure(read_rw_data())


if __name__ == "__main__":
    main()
