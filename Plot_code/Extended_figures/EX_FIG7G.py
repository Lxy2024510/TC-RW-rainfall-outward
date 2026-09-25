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

CSV_FILE = DATA_DIR / "delta_mean_by_initial_strength_EXP_only.csv"
OUT_FIG = FIG_DIR / "EX_FIG7G.png"

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
MEDIAN_LINEWIDTH = 2.2
ZERO_LINEWIDTH = 1.6
GRID_LINEWIDTH = 1.0
GRID_ALPHA = 0.55
CAP_WIDTH_RATIO = 0.20

BOX_COLOR = "#56A1B8"

YMIN = -14.0e3
YMAX = 1.0e3
YINTERVAL = 5.0e3
YTICKS = np.array([-10.0, -5.0, 0.0]) * 1.0e3

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
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.linewidth": 1.8,
    "grid.linewidth": GRID_LINEWIDTH,
    "lines.linewidth": 1.8,
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "xtick.major.width": 1.8,
    "ytick.major.width": 1.8,
    "xtick.major.size": 6.0,
    "ytick.major.size": 6.0,
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
        "case",
        "region",
        "mean_delta_area_km2",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"缺少列：{missing}；现有列：{df.columns.tolist()}")

    df["case"] = df["case"].astype(str).str.strip()
    df["region"] = df["region"].astype(str).str.strip()
    df["initial_strength"] = pd.to_numeric(
        df["initial_strength"], errors="coerce"
    )
    df["mean_delta_area_km2"] = pd.to_numeric(
        df["mean_delta_area_km2"], errors="coerce"
    )

    df = df[
        (df["case"] == "EXP")
        & df["initial_strength"].between(42, 51)
    ].copy()

    groups = []
    for region in ["inner_0_200km", "outer_200_500km"]:
        sub = df[df["region"] == region].sort_values("initial_strength")
        groups.append(
            finite_array(
                sub["mean_delta_area_km2"].to_numpy(),
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
        patch.set_edgecolor("black")
        patch.set_linewidth(BOX_LINEWIDTH)

    for whisker in bp["whiskers"]:
        whisker.set_color("black")
        whisker.set_linewidth(WHISKER_LINEWIDTH)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(MEDIAN_LINEWIDTH)

    for cap in bp.get("caps", []):
        cap.set_color("black")
        cap.set_linewidth(WHISKER_LINEWIDTH)
        xdata = np.asarray(cap.get_xdata(), dtype=float)
        if xdata.size == 2:
            center = float(np.mean(xdata))
            half_width = BOX_WIDTH * CAP_WIDTH_RATIO / 2.0
            cap.set_xdata([center - half_width, center + half_width])

    ax.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=ZERO_LINEWIDTH,
        zorder=1,
    )

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xticks(POSITIONS)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Change in heavy-rainfall area (km$^2$)")

    ax.set_ylim(YMIN, YMAX)
    ax.set_yticks(YTICKS)
    ax.set_yticklabels([
        f"{int(value / 1.0e3):d}".replace("-", "\N{MINUS SIGN}")
        for value in YTICKS
    ])
    ax.text(
        0.0,
        1.012,
        "1e3",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=20,
        color="black",
        clip_on=False,
    )

    ax.grid(
        True,
        axis="y",
        linestyle="--",
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
