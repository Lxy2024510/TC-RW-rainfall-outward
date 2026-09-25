#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(PROJECT_ROOT / "Data" / "Figure_data" / "Extended_figures")
FIG_DIR = PROJECT_ROOT / "Results" / "Extended_figures" / "EX_FIG7"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = DATA_DIR / "omega500_time_mean_by_strength_CTRL_EXP_summary_clean_v2.csv"
OUT_FIG = FIG_DIR / "EX_FIG7F.png"

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

YMIN = -5.9e-1
YMAX = 1.0e-1
YINTERVAL = 2.0e-1
YTICKS = np.array([-4.0, -2.0, 0.0]) * 1.0e-1

BASE_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 20

plt.rcParams.update({
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
})


def finite_array(values, name):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f"{name}没有有效值。")
    return arr


def box_stats(values, label):
    values = finite_array(values, label)
    p5, p25, p50, p75, p95 = np.percentile(
        values, [5, 25, 50, 75, 95]
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


def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError(CSV_FILE)

    df = pd.read_csv(CSV_FILE)
    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "initial_strength",
        "region",
        "field_type",
        "mean_omega_Pa_s",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"缺少列：{missing}；现有列：{df.columns.tolist()}")

    df["region"] = df["region"].astype(str).str.strip()
    df["field_type"] = df["field_type"].astype(str).str.strip()
    df["initial_strength"] = pd.to_numeric(
        df["initial_strength"], errors="coerce"
    )
    df["mean_omega_Pa_s"] = pd.to_numeric(
        df["mean_omega_Pa_s"], errors="coerce"
    )

    df = df[
        (df["field_type"] == "EXP-CTRL")
        & df["initial_strength"].between(42, 51)
    ].copy()

    # WRF omega: 正值向下。保持此前最终图的定义：
    # plotted value = -(omega_EXP - omega_CTRL)
    sign_factor = -1.0

    groups = []
    for region in ["inner_0_200km", "outer_200_500km"]:
        sub = df[df["region"] == region].sort_values("initial_strength")
        groups.append(
            finite_array(
                sign_factor * sub["mean_omega_Pa_s"].to_numpy(),
                region,
            )
        )

    labels = ["0–200 km", "200–500 km"]
    stats = [
        box_stats(values, label)
        for values, label in zip(groups, labels)
    ]

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([
        AXES_LEFT,
        AXES_BOTTOM,
        AXES_RIGHT - AXES_LEFT,
        AXES_TOP - AXES_BOTTOM,
    ])
    ax.set_box_aspect(
        HEIGHT_SCALE / (WIDTH_SCALE * HORIZONTAL_COMPRESSION)
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(1.8)

    bp = ax.bxp(
        stats,
        positions=POSITIONS,
        widths=BOX_WIDTH,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        showcaps=True,
        manage_ticks=False,
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(BOX_COLOR)
        patch.set_edgecolor("#000000")
        patch.set_linewidth(BOX_LINEWIDTH)

    for whisker in bp["whiskers"]:
        whisker.set_color("#000000")
        whisker.set_linewidth(WHISKER_LINEWIDTH)

    for median in bp["medians"]:
        median.set_color("#000000")
        median.set_linewidth(MEDIAN_LINEWIDTH)

    for cap in bp["caps"]:
        cap.set_visible(True)
        cap.set_color("#000000")
        cap.set_linewidth(CAP_LINEWIDTH)
        cap_x = np.asarray(cap.get_xdata(), dtype=float)
        cap_center = float(np.mean(cap_x))
        cap_half_width = BOX_WIDTH * CAP_WIDTH_RATIO / 2.0
        cap.set_xdata([
            cap_center - cap_half_width,
            cap_center + cap_half_width,
        ])

    ax.axhline(
        0.0,
        color="#000000",
        linestyle="--",
        linewidth=ZERO_LINEWIDTH,
        zorder=1,
    )

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xticks(POSITIONS)
    ax.set_xticklabels(
        labels,
        fontsize=TICK_FONT_SIZE,
        fontfamily="Arial",
        color="#000000",
        linespacing=1.15,
    )
    ax.set_ylabel(
        r"Change in 500-hPa vertical velocity (Pa s$^{-1}$)",
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontfamily="Arial",
        color="#000000",
    )

    ax.set_ylim(YMIN, YMAX)
    ax.set_yticks(YTICKS)
    ax.set_yticklabels(
        [
            "0"
            if np.isclose(value, 0.0, atol=1.0e-12)
            else (
                f"\N{MINUS SIGN}{abs(value / 1.0e-1):g}"
                if value < 0.0
                else f"{value / 1.0e-1:g}"
            )
            for value in YTICKS
        ],
        fontsize=TICK_FONT_SIZE,
        fontfamily="Arial",
        color="#000000",
    )

    ax.text(
        0.0,
        1.015,
        "1e\N{MINUS SIGN}1",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=TICK_FONT_SIZE,
        fontfamily="Arial",
        color="#000000",
        clip_on=False,
    )

    ax.tick_params(
        axis="both",
        labelsize=TICK_FONT_SIZE,
        colors="#000000",
    )

    ax.grid(
        True,
        axis="y",
        linestyle="--",
        color=GRID_COLOR,
        linewidth=GRID_LINEWIDTH,
        alpha=GRID_ALPHA,
    )

    fig.savefig(
        OUT_FIG,
        format="png",
        dpi=DPI,
        facecolor="white",
        edgecolor="white",
    )
    plt.close(fig)
    print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    main()
