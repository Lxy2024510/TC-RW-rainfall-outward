#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot closed-ring 500-PB open-ocean RW, Steady, and RI mass flux."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


INPUT_CSV = Path(
    str(PROJECT_ROOT / "Data" / "Processed" / "ERA5" / "MASS_FLUX_BOOTSTRAP_RESULTS_CLOSED_RINGS") + "/" +
    "RADIAL_MASS_FLUX_RW_STEADY_RI_1982_2024_"
    "MSWEP30_BOOTSTRAP_LONG.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "Results" / "Extended_figures" / "EX_FIG3"
OUTPUT_PNG = OUTPUT_DIR / "EX_FIG3D.png"

PRODUCT = "500_PB"
COMPONENTS = ["INNER_BOUNDARY", "NET_INFLUX", "OUTER_BOUNDARY"]
METRIC_LABELS = ["200 km", "200–500 km\nnet flux", "500 km"]
SERIES = [
    ("RW", "RW", "#4F9DB8"),
    ("STEADY", "Steady", "#D8D8D8"),
    ("RI", "RI", "#D5A0B6"),
]

FIGURE_SIZE = (6.4, 14.4)
FIGURE_DPI = 600
FONT_SIZE = 32
LEGEND_FONT_SIZE = 30
Y_MAX = 2.0e9
Y_TICK_STEP = 1.0e9

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
        "legend.fontsize": LEGEND_FONT_SIZE,
    }
)


def format_scaled_tick(value, _position):
    if np.isclose(value, 0.0, atol=1.0):
        return "0"
    scaled = value / 1.0e9
    if scaled < 0:
        return f"\N{MINUS SIGN}{abs(scaled):g}"
    return f"{scaled:g}"


def read_data():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {INPUT_CSV}")
    frame = pd.read_csv(INPUT_CSV, low_memory=False)
    required = {
        "SPATIAL_TYPE", "INTENSITY_GROUP", "PRODUCT", "COMPONENT",
        "OBSERVED_MEAN", "BOOTSTRAP_CI_LOWER", "BOOTSTRAP_CI_UPPER",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Input CSV is missing columns: {missing}")
    frame = frame.loc[
        frame["SPATIAL_TYPE"].eq("OPEN_OCEAN")
        & frame["PRODUCT"].eq(PRODUCT)
        & frame["INTENSITY_GROUP"].isin([item[0] for item in SERIES])
        & frame["COMPONENT"].isin(COMPONENTS)
    ].copy()
    numeric = ["OBSERVED_MEAN", "BOOTSTRAP_CI_LOWER", "BOOTSTRAP_CI_UPPER"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if len(frame) != 9 or frame.duplicated(["INTENSITY_GROUP", "COMPONENT"]).any():
        raise ValueError(f"Expected nine unique plotting rows, found {len(frame)}")
    if frame[numeric].isna().any(axis=None):
        raise ValueError("Selected plotting values are incomplete")
    return frame


def ordered(frame, group):
    result = frame.loc[frame["INTENSITY_GROUP"].eq(group)].set_index(
        "COMPONENT"
    ).reindex(COMPONENTS)
    if result[["OBSERVED_MEAN", "BOOTSTRAP_CI_LOWER", "BOOTSTRAP_CI_UPPER"]].isna().any(axis=None):
        raise ValueError(f"Incomplete data for {group}")
    return result


def main():
    frame = read_data()
    x = np.arange(3, dtype=float)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    plotted = []

    for group, label, color in SERIES:
        data = ordered(frame, group)
        plotted.append(data)
        mean = data["OBSERVED_MEAN"].to_numpy(dtype=float)
        lower_ci = data["BOOTSTRAP_CI_LOWER"].to_numpy(dtype=float)
        upper_ci = data["BOOTSTRAP_CI_UPPER"].to_numpy(dtype=float)
        axis.fill_between(
            x, lower_ci, upper_ci,
            color=color, alpha=0.18, linewidth=0,
        )
        axis.plot(
            x, mean, color=color, linewidth=4.0,
            marker="o", markersize=10.0, label=label,
        )

    axis.legend(
        loc="lower right", bbox_to_anchor=(1.04, 1.025), ncol=1,
        frameon=False, fontsize=LEGEND_FONT_SIZE, handlelength=1.6,
        handletextpad=0.55, labelspacing=0.18, borderaxespad=0.0,
    )
    axis.axhline(0.0, color="#333333", linestyle="--", linewidth=1.4)
    axis.set_xticks(x)
    axis.set_xticklabels(
        METRIC_LABELS, rotation=45, ha="right",
        rotation_mode="anchor", multialignment="center",
    )
    axis.set_xlim(-0.1, 2.1)
    lower = min(
        0.0,
        min(float(item["BOOTSTRAP_CI_LOWER"].min()) for item in plotted),
    )
    y_min = lower - (Y_MAX - lower) * 0.12
    axis.set_ylim(y_min, Y_MAX)
    first_tick = np.ceil(y_min / Y_TICK_STEP) * Y_TICK_STEP
    axis.set_yticks(np.arange(first_tick, Y_MAX + 1.0, Y_TICK_STEP))
    axis.yaxis.set_major_formatter(FuncFormatter(format_scaled_tick))
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color="#C4C4C4", linestyle="--", linewidth=1.2, alpha=0.85)
    axis.xaxis.grid(False)
    axis.set_xlabel("")
    axis.set_ylabel(r"Change in mass flux (kg s$^{-1}$)", labelpad=16)
    axis.tick_params(axis="x", pad=8, length=10, width=2.0, direction="out")
    axis.tick_params(axis="y", length=10, width=2.0, direction="out")
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(2.0)
    axis.text(
        0.0, 1.010, "1e9", transform=axis.transAxes,
        ha="left", va="bottom", fontsize=FONT_SIZE, clip_on=False,
    )

    figure.subplots_adjust(
        left=0.13, right=0.98, bottom=3.20 / 14.40, top=12.288 / 14.40
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PNG, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved figure: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
