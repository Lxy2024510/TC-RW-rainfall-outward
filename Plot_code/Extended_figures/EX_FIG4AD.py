#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Export the original RW 3-D pies separately as EX_FIG4A and EX_FIG4D."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Wedge
from matplotlib.transforms import Affine2D


INPUT_CSV = Path(
    str(PROJECT_ROOT / "Data" / "Processed" / "MSWEP_RH600_24H_RESULTS") + "/" +
    "RW_DIFF_RH600_BOOTSTRAP_COMPARISON/"
    "ALL_RW_SAMPLES_WITH_CHANGE_GROUPS.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "Results" / "Extended_figures" / "EX_FIG4"
OUTPUT_A = OUTPUT_DIR / "EX_FIG4A.png"
OUTPUT_D = OUTPUT_DIR / "EX_FIG4D.png"

AREA_COLUMN = "OUTER_AREA_CHANGE_GROUP"
DISTANCE_COLUMN = "MSWEP_DISTANCE_CHANGE_GROUP"
ET_COUNT_COLUMN = "COUNT_ET_24H"

PANEL_CONFIG = [
    ("DISTANCE_INCREASE", OUTPUT_A),
    ("DISTANCE_DECREASE", OUTPUT_D),
]

AREA_ORDER = ["AREA_INCREASE", "AREA_DECREASE", "AREA_NO_CHANGE"]
AREA_LABELS = {
    "AREA_INCREASE": r"$\Delta\mathit{Outer\ area} > 0$",
    "AREA_DECREASE": r"$\Delta\mathit{Outer\ area} < 0$",
    "AREA_NO_CHANGE": "Invalid",
}
AREA_COLORS = {
    "AREA_INCREASE": "#D8A1B9",
    "AREA_DECREASE": "#56A1BB",
    "AREA_NO_CHANGE": "#D9D9D9",
}

# The original combined canvas was 8.0 × 9.6 inches. Each original panel
# occupied 43% of its height (= 4.128 inches). A half-height 8.0 × 4.8 canvas
# with an 86%-high axes preserves that exact physical plotting size.
FIGSIZE = (8.0, 4.8)
AXES_POSITION = [0.055, 0.07, 0.89, 0.86]
FIGSIZE_WITH_LEGEND = (8.0, 6.0)
FIGURE_DPI = 600
BASE_FONT_SIZE = 15
SLICE_LABEL_FONT_SIZE = 15
GROUP_TOTAL_FONT_SIZE = 15
LEGEND_FONT_SIZE = 15

PIE_VERTICAL_SCALE = 0.62
PIE_RADIUS = 1.00
PIE_DEPTH = 0.20
SIDE_LAYERS = 28
START_ANGLE = 90.0
EDGE_COLOR = "white"
EDGE_WIDTH = 1.3


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.cursive": ["Arial"],
        "font.size": BASE_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "text.color": "#000000",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.cal": "Arial",
    }
)


def save_figure_atomic(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    try:
        figure.savefig(
            temporary,
            format="png",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
            facecolor="white",
        )
        if not temporary.exists() or temporary.stat().st_size == 0:
            raise RuntimeError(f"Temporary figure is missing or empty: {temporary}")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def read_composition():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {INPUT_CSV}")

    frame = pd.read_csv(
        INPUT_CSV,
        usecols=[AREA_COLUMN, DISTANCE_COLUMN, ET_COUNT_COLUMN],
        low_memory=False,
    )
    if frame.empty:
        raise RuntimeError("Input RW sample table is empty")

    frame[ET_COUNT_COLUMN] = pd.to_numeric(frame[ET_COUNT_COLUMN], errors="coerce")
    if frame[[AREA_COLUMN, DISTANCE_COLUMN, ET_COUNT_COLUMN]].isna().any().any():
        raise ValueError("Input contains missing or invalid required values")

    original_count = len(frame)
    removed_et_count = int(frame[ET_COUNT_COLUMN].gt(0).sum())
    frame = frame.loc[frame[ET_COUNT_COLUMN].eq(0)].copy()
    if len(frame) != 5992:
        raise RuntimeError(f"Expected 5992 non-ET RW samples, found {len(frame)}")

    distance_order = [item[0] for item in PANEL_CONFIG]
    unexpected_area = set(frame[AREA_COLUMN]).difference(AREA_ORDER)
    unexpected_distance = set(frame[DISTANCE_COLUMN]).difference(distance_order)
    if unexpected_area:
        raise ValueError(f"Unexpected area groups: {sorted(unexpected_area)}")
    if unexpected_distance:
        raise ValueError(f"Unexpected distance groups: {sorted(unexpected_distance)}")

    counts = pd.crosstab(frame[DISTANCE_COLUMN], frame[AREA_COLUMN]).reindex(
        index=distance_order,
        columns=AREA_ORDER,
        fill_value=0,
    )
    totals = counts.sum(axis=1)
    percentages = counts.div(totals, axis=0).mul(100.0)

    if int(totals.sum()) != len(frame):
        raise RuntimeError("Pie groups do not cover the full RW sample")
    if not np.allclose(percentages.sum(axis=1), 100.0):
        raise RuntimeError("Percentages do not sum to 100%")

    return (
        counts,
        totals,
        percentages,
        original_count,
        removed_et_count,
    )


def darken_color(color: str, factor: float):
    rgb = np.asarray(to_rgb(color), dtype=float)
    return tuple(np.clip(rgb * factor, 0.0, 1.0))


def calculate_angles(values, start_angle=90.0):
    values = np.asarray(values, dtype=float)
    boundaries = start_angle - np.r_[0.0, np.cumsum(values / values.sum()) * 360.0]
    return [
        (boundaries[index + 1], boundaries[index])
        for index in range(values.size)
    ]


def add_elliptical_wedge(
    axis,
    theta1,
    theta2,
    center_y,
    facecolor,
    edgecolor,
    linewidth,
    zorder,
):
    wedge = Wedge(
        (0.0, center_y),
        PIE_RADIUS,
        theta1,
        theta2,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    wedge.set_transform(Affine2D().scale(1.0, PIE_VERTICAL_SCALE) + axis.transData)
    wedge.set_zorder(zorder)
    axis.add_patch(wedge)


def draw_3d_pie(axis, counts, percentages, total):
    values = np.asarray(counts, dtype=float)
    percent_values = np.asarray(percentages, dtype=float)
    colors = [AREA_COLORS[group] for group in AREA_ORDER]
    angles = calculate_angles(values, START_ANGLE)

    for layer_index, center_y in enumerate(
        np.linspace(-PIE_DEPTH, 0.0, SIDE_LAYERS, endpoint=False)
    ):
        progress = layer_index / max(SIDE_LAYERS - 1, 1)
        shade_factor = 0.58 + 0.20 * progress
        for (theta1, theta2), color in zip(angles, colors):
            add_elliptical_wedge(
                axis,
                theta1,
                theta2,
                center_y,
                darken_color(color, shade_factor),
                "none",
                0.0,
                1.0 + layer_index * 0.001,
            )

    for (theta1, theta2), color in zip(angles, colors):
        add_elliptical_wedge(
            axis,
            theta1,
            theta2,
            0.0,
            color,
            EDGE_COLOR,
            EDGE_WIDTH,
            3.0,
        )

    for (theta1, theta2), percent in zip(angles, percent_values):
        middle_angle = np.deg2rad((theta1 + theta2) / 2.0)
        label_radius = 0.52
        label_x = label_radius * np.cos(middle_angle)
        label_y = label_radius * np.sin(middle_angle) * PIE_VERTICAL_SCALE

        if percent < 7.0:
            outside_radius = 1.13
            text_x = outside_radius * np.cos(middle_angle)
            text_y = outside_radius * np.sin(middle_angle) * PIE_VERTICAL_SCALE
            axis.annotate(
                f"{percent:.1f}%",
                xy=(label_x, label_y),
                xytext=(text_x, text_y),
                ha="left" if text_x >= 0 else "right",
                va="center",
                fontsize=SLICE_LABEL_FONT_SIZE,
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#000000",
                    "linewidth": 1.1,
                },
                zorder=5.0,
            )
        else:
            axis.text(
                label_x,
                label_y,
                f"{percent:.1f}%",
                ha="center",
                va="center",
                fontsize=SLICE_LABEL_FONT_SIZE,
                color="#000000",
                zorder=5.0,
            )

    axis.text(
        0.0,
        -0.82,
        f"$N$ = {int(total):,}",
        ha="center",
        va="top",
        fontsize=GROUP_TOTAL_FONT_SIZE,
        color="#000000",
        zorder=6.0,
    )
    axis.set_xlim(-1.25, 1.25)
    axis.set_ylim(-1.02, 0.78)
    axis.set_aspect("equal")
    axis.axis("off")


def create_panel_figure(
    counts,
    totals,
    percentages,
    distance_group,
    show_legend=False,
):
    if show_legend:
        # Preserve the original physical pie-axes dimensions while adding
        # canvas space above them for the three-row legend.
        original_width, original_height = FIGSIZE
        _, legend_height = FIGSIZE_WITH_LEGEND
        original_axes_bottom_in = AXES_POSITION[1] * original_height
        original_axes_height_in = AXES_POSITION[3] * original_height
        legend_axes_position = [
            AXES_POSITION[0],
            original_axes_bottom_in / legend_height,
            AXES_POSITION[2],
            original_axes_height_in / legend_height,
        ]
        figure = plt.figure(figsize=FIGSIZE_WITH_LEGEND)
        axis = figure.add_axes(legend_axes_position)
    else:
        figure = plt.figure(figsize=FIGSIZE)
        axis = figure.add_axes(AXES_POSITION)
    draw_3d_pie(
        axis,
        counts.loc[distance_group, AREA_ORDER].to_numpy(dtype=int),
        percentages.loc[distance_group, AREA_ORDER].to_numpy(dtype=float),
        totals.loc[distance_group],
    )

    if show_legend:
        handles = [
            Patch(
                facecolor=AREA_COLORS[group],
                edgecolor="white",
                linewidth=1.0,
                label=AREA_LABELS[group],
            )
            for group in AREA_ORDER
        ]
        axis.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.50, 0.97),
            ncol=1,
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
            handlelength=1.15,
            handletextpad=0.5,
            labelspacing=0.35,
            borderaxespad=0.0,
        )
    return figure


def main():
    (
        counts,
        totals,
        percentages,
        original_count,
        removed_et_count,
    ) = read_composition()

    for distance_group, output_path in PANEL_CONFIG:
        figure = create_panel_figure(
            counts,
            totals,
            percentages,
            distance_group,
            show_legend=(distance_group == "DISTANCE_INCREASE"),
        )
        try:
            save_figure_atomic(figure, output_path)
        finally:
            plt.close(figure)
        print(f"Saved {distance_group}: {output_path}")

    print(f"Original RW samples: {original_count:,}")
    print(f"Removed ET samples: {removed_et_count:,}")
    print(f"Non-ET RW samples: {int(totals.sum()):,}")


if __name__ == "__main__":
    main()
