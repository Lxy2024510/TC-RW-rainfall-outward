#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data" / "Figure_data" / "Main_figures" / "FIG2"
FIG_DIR = PROJECT_ROOT / "Results" / "Main_figures" / "FIG2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = DATA_DIR / (
    "radial_mass_flux_500_to_local_bottom_"
    "EXP_minus_CTRL_16h_mean_summary_clean.csv"
)
OUT_FIG = FIG_DIR / "FIG2D.png"

VALUE_COLUMN = (
    "interface_total_inward_mass_transport_16h_mean_"
    "EXP_minus_CTRL_kg_s"
)


# ============================================================
# 2. Figure configuration
# ============================================================

HEIGHT_SCALE = 5.63 / 7.03
WIDTH_SCALE = 7.10 / 7.50
FIGSIZE = (8.0 * WIDTH_SCALE, 7.5 * HEIGHT_SCALE)
DPI = 300

AXES_LEFT = 0.19
AXES_RIGHT = 0.98
AXES_BOTTOM = 0.20
AXES_TOP = 0.94

X_MIN = 0.89
X_MAX = 1.11
POSITIONS = [0.93, 1.00, 1.07]

BOX_WIDTH = 0.0225
BOX_LINEWIDTH = 1.8
WHISKER_LINEWIDTH = 1.8
CAP_LINEWIDTH = 1.8
CAP_WIDTH_RATIO = 0.20
MEDIAN_LINEWIDTH = 2.2
ZERO_LINEWIDTH = 1.6

GRID_LINEWIDTH = 1.0
GRID_ALPHA = 0.55
GRID_COLOR = "#9E9E9E"

BOX_COLOR = "#56A1B8"

YMIN = -7.0e9
YMAX = 3.0e9
YTICKS = [-6.0e9, -4.0e9, -2.0e9, 0.0, 2.0e9]

BASE_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 18

plt.rcParams.update(
    {
        "font.family": "Arial",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "axes.edgecolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "axes.linewidth": 1.8,
        "grid.linewidth": GRID_LINEWIDTH,
        "lines.linewidth": 1.8,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "xtick.major.size": 5.0,
        "ytick.major.size": 5.0,
    }
)


# ============================================================
# 3. Data helpers
# ============================================================

def finite_array(values, name):
    """Return finite values and reject an empty array."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError(f"{name} has no valid values.")
    return array


def box_stats(values, label):
    """Calculate P5, P25, median, P75 and P95 box statistics."""
    values = finite_array(values, label)
    p5, p25, p50, p75, p95 = np.percentile(
        values,
        [5, 25, 50, 75, 95],
    )
    print(
        f"{label}: n={values.size}, "
        f"P5={p5:.8e}, P25={p25:.8e}, P50={p50:.8e}, "
        f"P75={p75:.8e}, P95={p95:.8e}"
    )
    return {
        "label": label,
        "whislo": float(p5),
        "q1": float(p25),
        "med": float(p50),
        "q3": float(p75),
        "whishi": float(p95),
        "fliers": [],
    }


# ============================================================
# 4. Main program
# ============================================================

def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError(CSV_FILE)

    data = pd.read_csv(CSV_FILE)
    data.columns = [str(column).strip() for column in data.columns]

    required_columns = [
        "initial_strength",
        "interface_radius_km",
        VALUE_COLUMN,
    ]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise KeyError(
            f"Missing columns: {missing_columns}; "
            f"available columns: {data.columns.tolist()}"
        )

    for column in required_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=required_columns).copy()
    data = data[data["initial_strength"].between(42, 51)]
    data["initial_strength"] = data["initial_strength"].astype(int)

    radius_200 = (
        data[np.isclose(data["interface_radius_km"], 200.0)][
            ["initial_strength", VALUE_COLUMN]
        ]
        .rename(columns={VALUE_COLUMN: "value_200"})
    )
    radius_500 = (
        data[np.isclose(data["interface_radius_km"], 500.0)][
            ["initial_strength", VALUE_COLUMN]
        ]
        .rename(columns={VALUE_COLUMN: "value_500"})
    )

    aligned = pd.merge(
        radius_200,
        radius_500,
        on="initial_strength",
        how="inner",
    ).sort_values("initial_strength")

    if aligned.empty:
        raise RuntimeError("The merged 200-km and 500-km data are empty.")

    values_200 = finite_array(
        aligned["value_200"].to_numpy(),
        "200 km",
    )
    values_net = finite_array(
        aligned["value_500"].to_numpy()
        - aligned["value_200"].to_numpy(),
        "200–500 km net flux",
    )
    values_500 = finite_array(
        aligned["value_500"].to_numpy(),
        "500 km",
    )

    groups = [values_200, values_net, values_500]
    labels = [
        "200 km",
        "200–500 km\nnet flux",
        "500 km",
    ]
    statistics = [
        box_stats(values, label)
        for values, label in zip(groups, labels)
    ]

    figure = plt.figure(figsize=FIGSIZE)
    axis = figure.add_axes(
        [
            AXES_LEFT,
            AXES_BOTTOM,
            AXES_RIGHT - AXES_LEFT,
            AXES_TOP - AXES_BOTTOM,
        ]
    )
    # Preserve the previously reduced frame height while shrinking only the
    # frame width from a nominal 7.50 units to 7.10 units.
    axis.set_box_aspect(HEIGHT_SCALE / WIDTH_SCALE)

    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(1.8)

    boxplot = axis.bxp(
        statistics,
        positions=POSITIONS,
        widths=BOX_WIDTH,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        showcaps=True,
        manage_ticks=False,
    )

    for patch in boxplot["boxes"]:
        patch.set_facecolor(BOX_COLOR)
        patch.set_edgecolor("#000000")
        patch.set_linewidth(BOX_LINEWIDTH)

    for whisker in boxplot["whiskers"]:
        whisker.set_color("#000000")
        whisker.set_linewidth(WHISKER_LINEWIDTH)

    for median in boxplot["medians"]:
        median.set_color("#000000")
        median.set_linewidth(MEDIAN_LINEWIDTH)

    for cap in boxplot["caps"]:
        cap.set_visible(True)
        cap.set_color("#000000")
        cap.set_linewidth(CAP_LINEWIDTH)
        cap_x = np.asarray(cap.get_xdata(), dtype=float)
        cap_center = float(np.mean(cap_x))
        cap_half_width = BOX_WIDTH * CAP_WIDTH_RATIO / 2.0
        cap.set_xdata(
            [
                cap_center - cap_half_width,
                cap_center + cap_half_width,
            ]
        )

    axis.axhline(
        0.0,
        color="#000000",
        linestyle="--",
        linewidth=ZERO_LINEWIDTH,
        zorder=1,
    )

    axis.set_xlim(X_MIN, X_MAX)
    axis.set_xticks(POSITIONS)
    axis.set_xticklabels(
        labels,
        fontsize=TICK_FONT_SIZE,
        fontfamily="Arial",
        color="#000000",
        rotation=8,
    )
    axis.set_ylabel(
        r"Change in mass flux (kg s$^{-1}$)",
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontfamily="Arial",
        color="#000000",
    )

    axis.set_ylim(YMIN, YMAX)
    axis.set_yticks(YTICKS)
    axis.set_yticklabels(
        ["−6", "−4", "−2", "0", "2"],
        fontsize=TICK_FONT_SIZE,
        fontfamily="Arial",
        color="#000000",
    )

    # Draw the scientific multiplier manually so its font exactly matches
    # the remaining figure text instead of using Matplotlib's offset text.
    axis.text(
        0.0,
        1.015,
        "1e9",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=TICK_FONT_SIZE,
        fontfamily="Arial",
        color="#000000",
        clip_on=False,
    )

    axis.tick_params(
        axis="both",
        labelsize=TICK_FONT_SIZE,
        colors="#000000",
    )

    axis.grid(
        True,
        axis="y",
        linestyle="--",
        color=GRID_COLOR,
        linewidth=GRID_LINEWIDTH,
        alpha=GRID_ALPHA,
    )

    figure.savefig(
        OUT_FIG,
        format="png",
        dpi=DPI,
        facecolor="white",
        edgecolor="white",
    )
    plt.close(figure)

    print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    main()
