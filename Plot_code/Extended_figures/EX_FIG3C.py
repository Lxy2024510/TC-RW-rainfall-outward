#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot open-ocean RW, zero-change, and RI FUHE convergence changes.

The script follows the supplied reference plotting logic: it creates only the
OPEN_OCEAN radial profile, keeps values in their original s^-1 units, and uses
Matplotlib scientific notation on the y axis. The profile shows the mean
24-hour convergence change and its percentile-bootstrap 95% confidence
interval for five non-overlapping 100-km radial zones.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import matplotlib

# Use a non-interactive backend on headless Linux servers.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


# ============================================================================
# 1. Paths and output configuration
# ============================================================================

INPUT_CSV = Path(
    str(PROJECT_ROOT / "Data" / "Processed" / "MSWEP_CONVERGENCE_LAYERS_BOOTSTRAP_RESULTS") + "/" +
    "CONVERGENCE_LAYERS_1982_2024_MSWEP30_BOOTSTRAP_LONG.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "Results" / "Extended_figures" / "EX_FIG3"

SPATIAL_TYPE = "OPEN_OCEAN"
VERTICAL_PRODUCT = "500_PB"

FIGURE_DPI = 600
FIGURE_SIZE = (6.4, 12.8)
LEGEND_FIGURE_SIZE = (6.4, 1.5)
TABLE_FIGURE_SIZE = (13.0, 11.0)

PLOT_FONT_SIZE = 32
TABLE_FONT_SIZE = 12


# ============================================================================
# 2. Radial-zone and intensity-group configuration
# ============================================================================

# Only the five mutually exclusive 100-km annuli are plotted. The composite
# 0-200 km and 200-500 km zones remain available in the RUN4 result table.
DISTANCE_ZONES = [
    "R000_100",
    "R100_200",
    "R200_300",
    "R300_400",
    "R400_500",
]

DISTANCE_LABELS = [
    "0\u2013100 km",
    "100\u2013200 km",
    "200\u2013300 km",
    "300\u2013400 km",
    "400\u2013500 km",
]

# RW uses the fixed <= -30 kt group, 0 uses the exact-zero group, and RI uses
# the fixed >= 30 kt group.
GROUPS = [
    ("LE_5TH", "RW", "#4F9DB8"),
    ("EQ_50TH", "0", "#D8D8D8"),
    ("GE_95TH", "RI", "#D5A0B6"),
]

NUMERIC_COLUMNS = [
    "COMMON_VALID_WINDOW_COUNT",
    "MEAN_CHANGE",
    "BOOTSTRAP_CI_LOWER",
    "BOOTSTRAP_CI_UPPER",
]


# ============================================================================
# 3. Matplotlib style
# ============================================================================

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
        "font.size": PLOT_FONT_SIZE,
        "axes.labelsize": PLOT_FONT_SIZE,
        "xtick.labelsize": PLOT_FONT_SIZE,
        "ytick.labelsize": PLOT_FONT_SIZE,
        "legend.fontsize": PLOT_FONT_SIZE,
    }
)


# ============================================================================
# 4. Input validation
# ============================================================================

def read_and_validate_table() -> pd.DataFrame:
    """Read and validate the selected OPEN_OCEAN RUN4 plotting rows."""
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

    selected_group_codes = [group_code for group_code, _, _ in GROUPS]
    frame = frame.loc[
        frame["SPATIAL_TYPE"].eq(SPATIAL_TYPE)
        & frame["VERTICAL_PRODUCT"].eq(VERTICAL_PRODUCT)
        & frame["DISTANCE_ZONE"].isin(DISTANCE_ZONES)
        & frame["INTENSITY_GROUP_CODE"].isin(selected_group_codes)
    ].copy()

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[NUMERIC_COLUMNS] = frame[NUMERIC_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )

    duplicate = frame.duplicated(
        ["SPATIAL_TYPE", "VERTICAL_PRODUCT", "INTENSITY_GROUP_CODE", "DISTANCE_ZONE"],
        keep=False,
    )
    if duplicate.any():
        examples = frame.loc[
            duplicate,
            ["SPATIAL_TYPE", "VERTICAL_PRODUCT", "INTENSITY_GROUP_CODE", "DISTANCE_ZONE"],
        ].head(20).to_dict("records")
        raise ValueError(f"Duplicate group-zone plotting rows: {examples}")

    expected_rows = len(GROUPS) * len(DISTANCE_ZONES)
    if len(frame) != expected_rows:
        raise ValueError(
            f"Unexpected filtered row count: found={len(frame)}, "
            f"expected={expected_rows}"
        )

    invalid_numeric = frame[NUMERIC_COLUMNS].isna().any(axis=1)
    if invalid_numeric.any():
        examples = frame.loc[
            invalid_numeric,
            ["SPATIAL_TYPE", "INTENSITY_GROUP_CODE", "DISTANCE_ZONE"]
            + NUMERIC_COLUMNS,
        ].head(20).to_dict("records")
        raise ValueError(f"Missing or invalid plotting values: {examples}")

    invalid_ci = frame["BOOTSTRAP_CI_LOWER"].gt(
        frame["BOOTSTRAP_CI_UPPER"]
    )
    mean_outside_ci = frame["MEAN_CHANGE"].lt(
        frame["BOOTSTRAP_CI_LOWER"]
    ) | frame["MEAN_CHANGE"].gt(frame["BOOTSTRAP_CI_UPPER"])
    if invalid_ci.any():
        raise ValueError(
            f"Bootstrap lower bound exceeds upper bound: {int(invalid_ci.sum())}"
        )
    if mean_outside_ci.any():
        examples = frame.loc[
            mean_outside_ci,
            [
                "SPATIAL_TYPE",
                "INTENSITY_GROUP_CODE",
                "DISTANCE_ZONE",
                "MEAN_CHANGE",
                "BOOTSTRAP_CI_LOWER",
                "BOOTSTRAP_CI_UPPER",
            ],
        ].head(20).to_dict("records")
        raise ValueError(f"Mean values outside their confidence intervals: {examples}")

    invalid_sample_size = frame["COMMON_VALID_WINDOW_COUNT"].lt(0.0) | ~np.isclose(
        frame["COMMON_VALID_WINDOW_COUNT"],
        np.round(frame["COMMON_VALID_WINDOW_COUNT"]),
        rtol=0.0,
        atol=1.0e-9,
    )
    if invalid_sample_size.any():
        raise ValueError(
            f"Invalid COMMON_VALID_WINDOW_COUNT values: "
            f"{int(invalid_sample_size.sum())}"
        )
    frame["COMMON_VALID_WINDOW_COUNT"] = np.round(
        frame["COMMON_VALID_WINDOW_COUNT"]
    ).astype(np.int64)

    print(f"Validated plotting rows: {len(frame)}")
    return frame


# ============================================================================
# 5. Plot one spatial category
# ============================================================================

def format_scaled_y_tick(value, _position):
    """Format 1e-6-scaled ticks with the proper Unicode minus sign."""
    scaled = value / 1.0e-6
    if np.isclose(scaled, 0.0, atol=1.0e-12):
        return "0"
    if scaled < 0.0:
        return f"\N{MINUS SIGN}{abs(scaled):g}"
    return f"{scaled:g}"


def plot_one_spatial(
    frame: pd.DataFrame,
    spatial_type: str,
) -> Path:
    """Plot RW, zero-change, and RI profiles for one spatial category."""
    spatial_frame = frame.loc[frame["SPATIAL_TYPE"].eq(spatial_type)].copy()
    x = np.arange(len(DISTANCE_ZONES), dtype=np.float64)

    fig, axis = plt.subplots(figsize=FIGURE_SIZE)

    for group_code, group_label, color in GROUPS:
        group_frame = (
            spatial_frame.loc[
                spatial_frame["INTENSITY_GROUP_CODE"].eq(group_code)
            ]
            .set_index("DISTANCE_ZONE")
            .reindex(DISTANCE_ZONES)
        )
        plotting_columns = [
            "MEAN_CHANGE",
            "BOOTSTRAP_CI_LOWER",
            "BOOTSTRAP_CI_UPPER",
        ]
        if group_frame[plotting_columns].isna().any(axis=None):
            raise ValueError(
                f"Incomplete plotting data for {spatial_type}/{group_code}"
            )

        mean = group_frame["MEAN_CHANGE"].to_numpy(dtype=np.float64)
        lower = group_frame["BOOTSTRAP_CI_LOWER"].to_numpy(dtype=np.float64)
        upper = group_frame["BOOTSTRAP_CI_UPPER"].to_numpy(dtype=np.float64)

        axis.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.18,
            linewidth=0.0,
        )
        axis.plot(
            x,
            mean,
            color=color,
            linewidth=4.0,
            marker="o",
            markersize=10.0,
            label=group_label,
        )

    axis.axhline(
        0.0,
        color="#333333",
        linestyle="--",
        linewidth=1.4,
    )
    axis.set_xticks(x)
    axis.set_xticklabels(
        DISTANCE_LABELS,
        fontsize=PLOT_FONT_SIZE,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    axis.set_xlabel("")
    axis.set_ylabel(
        r"Change in convergence (s$^{-1}$)",
        fontsize=PLOT_FONT_SIZE,
        labelpad=16,
    )

    # Fix the upper limit at 4 under the 1e-6 multiplier. The lower limit
    # remains data-driven and includes a modest margin below the lowest CI.
    y_upper = 4.0e-6
    data_lower = min(
        0.0,
        float(spatial_frame["BOOTSTRAP_CI_LOWER"].min()),
    )
    y_lower = data_lower - (y_upper - data_lower) * 0.12
    axis.set_ylim(y_lower, y_upper)

    tick_step = 4.0e-6
    first_tick = np.ceil(y_lower / tick_step) * tick_step
    axis.set_yticks(
        np.arange(first_tick, y_upper + 1.0e-12, tick_step)
    )

    axis.set_axisbelow(True)
    axis.yaxis.grid(
        True,
        color="#C4C4C4",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
    )

    axis.yaxis.set_major_formatter(FuncFormatter(format_scaled_y_tick))
    axis.yaxis.get_offset_text().set_visible(False)
    axis.text(
        0.0,
        1.010,
        "1e\N{MINUS SIGN}6",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=PLOT_FONT_SIZE,
        color="#000000",
        clip_on=False,
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

    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(2.0)

    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.25, top=0.96)

    fig.canvas.draw()
    axis.yaxis.get_offset_text().set_visible(False)

    output_path = OUTPUT_DIR / "EX_FIG3C.png"
    fig.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved figure: {output_path}")
    return output_path


# ============================================================================
# 6. Plot one statistics table
# ============================================================================

def plot_statistics_table(frame: pd.DataFrame, spatial_type: str) -> Path:
    """Save the plotted means and confidence intervals as a PNG table."""
    spatial_frame = frame.loc[frame["SPATIAL_TYPE"].eq(spatial_type)].copy()
    group_label_by_code = {
        group_code: group_label for group_code, group_label, _ in GROUPS
    }
    group_color_by_label = {
        group_label: color for _, group_label, color in GROUPS
    }
    distance_label_by_code = dict(zip(DISTANCE_ZONES, DISTANCE_LABELS))
    group_order = {
        group_label: index for index, (_, group_label, _) in enumerate(GROUPS)
    }
    distance_order = {zone: index for index, zone in enumerate(DISTANCE_ZONES)}

    table_frame = spatial_frame[
        [
            "INTENSITY_GROUP_CODE",
            "DISTANCE_ZONE",
            "COMMON_VALID_WINDOW_COUNT",
            "MEAN_CHANGE",
            "BOOTSTRAP_CI_LOWER",
            "BOOTSTRAP_CI_UPPER",
        ]
    ].copy()
    table_frame["Group"] = table_frame["INTENSITY_GROUP_CODE"].map(
        group_label_by_code
    )
    table_frame["Distance zone"] = table_frame["DISTANCE_ZONE"].map(
        distance_label_by_code
    )
    table_frame["_GROUP_ORDER"] = table_frame["Group"].map(group_order)
    table_frame["_DISTANCE_ORDER"] = table_frame["DISTANCE_ZONE"].map(
        distance_order
    )
    table_frame = table_frame.sort_values(
        ["_GROUP_ORDER", "_DISTANCE_ORDER"]
    ).reset_index(drop=True)

    table_frame["N"] = table_frame["COMMON_VALID_WINDOW_COUNT"].map(
        lambda value: f"{int(value):,}"
    )
    table_frame["Mean change"] = table_frame["MEAN_CHANGE"].map(
        lambda value: f"{value:.4e}"
    )
    table_frame["CI lower"] = table_frame["BOOTSTRAP_CI_LOWER"].map(
        lambda value: f"{value:.4e}"
    )
    table_frame["CI upper"] = table_frame["BOOTSTRAP_CI_UPPER"].map(
        lambda value: f"{value:.4e}"
    )

    display_columns = [
        "Group",
        "Distance zone",
        "N",
        "Mean change",
        "CI lower",
        "CI upper",
    ]
    display_frame = table_frame[display_columns]

    fig, axis = plt.subplots(figsize=TABLE_FIGURE_SIZE, dpi=FIGURE_DPI)
    axis.axis("off")
    table = axis.table(
        cellText=display_frame.values,
        colLabels=display_columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.03, 0.03, 0.94, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_FONT_SIZE)
    table.scale(1.0, 1.55)

    for column_index in range(len(display_columns)):
        header_cell = table[(0, column_index)]
        header_cell.set_facecolor("#3E5266")
        header_cell.set_text_props(color="white", fontweight="bold")
        header_cell.set_edgecolor("#2F3E4D")
        header_cell.set_linewidth(1.0)

    for row_index in range(1, len(display_frame) + 1):
        group_label = display_frame.iloc[row_index - 1]["Group"]
        for column_index in range(len(display_columns)):
            cell = table[(row_index, column_index)]
            cell.set_edgecolor("#B7B7B7")
            cell.set_linewidth(0.8)
            if column_index == 0:
                cell.set_facecolor(group_color_by_label[group_label])
                cell.set_text_props(fontweight="bold")
            elif row_index % 2 == 0:
                cell.set_facecolor("#F2F4F5")
            else:
                cell.set_facecolor("#FFFFFF")

    output_path = OUTPUT_DIR / "EX_FIG3C_statistics.png"
    fig.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved statistics table: {output_path}")
    return output_path


# ============================================================================
# 7. Draw one shared legend
# ============================================================================

def plot_shared_legend() -> Path:
    """Save the RW, zero-change, and RI legend as a standalone PNG."""
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=4.0,
            marker="o",
            markersize=10.0,
            label=group_label,
        )
        for _, group_label, color in GROUPS
    ]

    fig, axis = plt.subplots(figsize=LEGEND_FIGURE_SIZE)
    axis.axis("off")
    legend = axis.legend(
        handles=legend_handles,
        loc="center",
        ncol=3,
        fontsize=PLOT_FONT_SIZE,
        frameon=True,
        fancybox=False,
        edgecolor="#000000",
    )
    legend.get_frame().set_linewidth(2.0)

    output_path = OUTPUT_DIR / "EX_FIG3C_legend.png"
    fig.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved shared legend: {output_path}")
    return output_path


# ============================================================================
# 8. Main program
# ============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = read_and_validate_table()
    plot_one_spatial(frame, SPATIAL_TYPE)

    print()
    print("=" * 96)
    print("OPEN_OCEAN FUHE 500-PB RW/0/RI figure completed successfully")
    print("=" * 96)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Spatial category: {SPATIAL_TYPE}")
    print("Display unit: s^-1 with scientific notation")
    print("=" * 96)


if __name__ == "__main__":
    main()
