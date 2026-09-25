#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot the EXP-minus-CTRL convergence boxplots for EX_FIG7E."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. Paths
# ============================================================

DATA_DIR = Path(PROJECT_ROOT / "Data" / "Figure_data" / "Extended_figures")
FIG_DIR = PROJECT_ROOT / "Results" / "Extended_figures" / "EX_FIG7"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EXP_CSV = DATA_DIR / "convergence_500_to_local_bottom_EXP_16h_mean_summary.csv"
CTRL_CSV = DATA_DIR / "convergence_500_to_local_bottom_CTRL_16h_mean_summary.csv"
OUT_FIG = FIG_DIR / "EX_FIG7E.png"


# ============================================================
# 2. Figure configuration copied from the FIG2D reference
# ============================================================

HEIGHT_SCALE = 5.63 / 7.03
WIDTH_SCALE = 7.10 / 7.50
HORIZONTAL_COMPRESSION = 9.67 / 13.83
FIGSIZE = (
    8.0 * WIDTH_SCALE * HORIZONTAL_COMPRESSION,
    7.5 * HEIGHT_SCALE,
)
DPI = 300

AXES_LEFT = 0.19
AXES_RIGHT = 0.98
AXES_BOTTOM = 0.10
AXES_TOP = 0.94

X_MIN = 0.89
X_MAX = 1.11
POSITIONS = [0.95, 1.05]

# Compensate for the narrower frame so the rendered box width remains equal
# to the preceding version.
BOX_WIDTH = 0.0225 / HORIZONTAL_COMPRESSION
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

YMIN = -10.0e-6
YMAX = 2.0e-6
YTICKS = np.array([-8.0, -4.0, 0.0]) * 1.0e-6

BASE_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 20

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
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError(f"{name} has no valid values.")
    return array


def box_stats(values, label):
    values = finite_array(values, label)
    p5, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
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


def find_mean_column(frame):
    preferred_keywords = [
        "mean_convergence",
        "weighted_mean",
        "time_mean",
        "mean_value",
        "mean",
    ]
    columns = [str(column).strip() for column in frame.columns]

    for keyword in preferred_keywords:
        for column in columns:
            if keyword.lower() in column.lower():
                return column

    excluded = {
        "initial_strength",
        "n_hours",
        "valid_hour_count",
        "sample_count_used",
        "valid_grid_count",
    }
    numeric_candidates = [
        column
        for column in columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if len(numeric_candidates) == 1:
        return numeric_candidates[0]
    raise ValueError(
        "Cannot identify the convergence mean column. "
        f"Columns={columns}; numeric candidates={numeric_candidates}"
    )


def standardize(frame, name):
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in ["initial_strength", "region"]:
        if column not in frame.columns:
            raise KeyError(f"{name} is missing column: {column}")

    mean_column = find_mean_column(frame)
    frame["initial_strength"] = pd.to_numeric(
        frame["initial_strength"], errors="coerce"
    )
    frame["region"] = frame["region"].astype(str).str.strip()
    frame[mean_column] = pd.to_numeric(frame[mean_column], errors="coerce")
    frame = frame.dropna(subset=["initial_strength", "region", mean_column])
    frame = frame.loc[frame["initial_strength"].between(42, 51)].copy()
    frame["initial_strength"] = frame["initial_strength"].astype(int)
    return frame[["initial_strength", "region", mean_column]].rename(
        columns={mean_column: "mean_value"}
    )


def format_scaled_tick(value):
    scaled = value / 1.0e-6
    if np.isclose(scaled, 0.0, atol=1.0e-12):
        return "0"
    if scaled < 0.0:
        return f"\N{MINUS SIGN}{abs(scaled):g}"
    return f"{scaled:g}"


# ============================================================
# 4. Main program
# ============================================================

def main():
    if not EXP_CSV.is_file():
        raise FileNotFoundError(EXP_CSV)
    if not CTRL_CSV.is_file():
        raise FileNotFoundError(CTRL_CSV)

    exp = standardize(pd.read_csv(EXP_CSV), "EXP")
    ctrl = standardize(pd.read_csv(CTRL_CSV), "CTRL")
    merged = pd.merge(
        exp,
        ctrl,
        on=["initial_strength", "region"],
        how="inner",
        suffixes=("_EXP", "_CTRL"),
    )
    if merged.empty:
        raise RuntimeError("The EXP/CTRL merged table is empty.")

    merged["EXP_minus_CTRL"] = (
        merged["mean_value_EXP"] - merged["mean_value_CTRL"]
    )

    region_codes = ["inner_0_200km", "outer_200_500km"]
    labels = ["0\u2013200 km", "200\u2013500 km"]
    groups = [
        finite_array(
            merged.loc[merged["region"].eq(region), "EXP_minus_CTRL"],
            region,
        )
        for region in region_codes
    ]
    statistics = [
        box_stats(values, label) for values, label in zip(groups, labels)
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
    axis.set_box_aspect(
        HEIGHT_SCALE / (WIDTH_SCALE * HORIZONTAL_COMPRESSION)
    )

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
            [cap_center - cap_half_width, cap_center + cap_half_width]
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
        linespacing=1.15,
    )
    axis.set_ylabel(
        r"Change in convergence (s$^{-1}$)",
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontfamily="Arial",
        color="#000000",
    )

    axis.set_ylim(YMIN, YMAX)
    axis.set_yticks(YTICKS)
    axis.set_yticklabels(
        [format_scaled_tick(value) for value in YTICKS],
        fontsize=TICK_FONT_SIZE,
        fontfamily="Arial",
        color="#000000",
    )

    axis.text(
        0.0,
        1.015,
        "1e\N{MINUS SIGN}6",
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
