from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter


# ============================================================
# 1. Global font configuration
# ============================================================

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
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "axes.edgecolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
    }
)


# ============================================================
# 2. Configuration
# ============================================================

THRESHOLD = "30"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT
    / "Data"
    / "Processed"
    / "MSWEP"
    / "Bootstrap"
    / "Area"
    / f"TH{THRESHOLD}"
)

OUTPUT_DIR = PROJECT_ROOT / "Results" / "Main_figures" / "FIG2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. Distance-zone configuration
# ============================================================

DISTANCE_ZONES = ["0_200", "200_500"]
DISTANCE_LABELS = ["0–200 km", "200–500 km"]
DISTANCE_OUTPUT_TAG = "ZONES_0_200_200_500"


# ============================================================
# 4. Intensity-group configuration
# ============================================================

# Plot only the original <=5th group as RW.
GROUPS = [
    ("LE_5TH", "RW", "#56A1BB"),
]


# ============================================================
# 5. Plot configuration
# ============================================================

SPATIAL_TYPE = "OPEN_OCEAN"
METRIC_TYPE = "AREA"

FIGURE_DPI = 600
FIGURE_SIZE = (8.0, 10.0)
PLOT_FONT_SIZE = 30

Y_AXIS_MIN = -6000.0
Y_AXIS_MAX = 4000.0
Y_TICK_INTERVAL = 2000.0

BAR_WIDTH = 0.34
BAR_EDGE_WIDTH = 2.6
ERROR_LINE_WIDTH = 2.4
ERROR_CAP_SIZE = 8.0

GRID_COLOR = "#9E9E9E"
GRID_ALPHA = 0.55
GRID_LINE_WIDTH = 1.0


# ============================================================
# 6. Gradient-bar helper
# ============================================================

def draw_gradient_bar(ax, x_position, value, width, end_color):
    """Draw one white-at-zero to blue-at-value gradient bar."""
    if np.isclose(value, 0.0):
        ax.bar(
            x_position,
            value,
            width=width,
            facecolor="none",
            edgecolor="#000000",
            linewidth=BAR_EDGE_WIDTH,
            zorder=3,
        )
        return

    bar = ax.bar(
        x_position,
        value,
        width=width,
        facecolor="none",
        edgecolor="#000000",
        linewidth=BAR_EDGE_WIDTH,
        zorder=3,
    )[0]

    bottom = min(0.0, value)
    top = max(0.0, value)
    colors = ["#FFFFFF", end_color] if value > 0 else [end_color, "#FFFFFF"]

    cmap = LinearSegmentedColormap.from_list(
        f"gradient_{x_position}",
        colors,
        N=512,
    )
    gradient = np.linspace(0.0, 1.0, 1024).reshape(-1, 1)
    image = ax.imshow(
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
# 7. Input validation
# ============================================================

def read_and_validate_table(spatial_type):
    """Read and validate one threshold-30 AREA bootstrap table."""
    input_path = INPUT_DIR / (
        f"PRE_DATA_IBT_1982_2024_MSWEP_TH{THRESHOLD}_"
        f"{spatial_type}_{METRIC_TYPE}_BOOTSTRAP_TABLE.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Input table does not exist: {input_path}")

    print("=" * 90)
    print(f"Reading {spatial_type} {METRIC_TYPE} bootstrap table")
    print("=" * 90)
    print(f"Input file: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)

    required_columns = [
        "INTENSITY_GROUP_CODE",
        "DISTANCE_ZONE",
        "COMMON_VALID_WINDOW_COUNT",
        "MEAN_CHANGE",
        "BOOTSTRAP_CI_LOWER",
        "BOOTSTRAP_CI_UPPER",
    ]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    selected_group_codes = [group_code for group_code, _, _ in GROUPS]
    df = df.loc[
        df["DISTANCE_ZONE"].isin(DISTANCE_ZONES)
        & df["INTENSITY_GROUP_CODE"].isin(selected_group_codes)
    ].copy()

    numeric_columns = [
        "COMMON_VALID_WINDOW_COUNT",
        "MEAN_CHANGE",
        "BOOTSTRAP_CI_LOWER",
        "BOOTSTRAP_CI_UPPER",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

    duplicate_mask = df.duplicated(
        subset=["INTENSITY_GROUP_CODE", "DISTANCE_ZONE"],
        keep=False,
    )
    if duplicate_mask.any():
        examples = (
            df.loc[
                duplicate_mask,
                ["INTENSITY_GROUP_CODE", "DISTANCE_ZONE"],
            ]
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            "Duplicate group-zone rows were found. "
            f"Examples: {examples}"
        )

    expected_rows = len(GROUPS) * len(DISTANCE_ZONES)
    if len(df) != expected_rows:
        raise ValueError(
            "Unexpected filtered row count: "
            f"found={len(df)}, expected={expected_rows}"
        )

    invalid_numeric_mask = df[numeric_columns].isna().any(axis=1)
    if invalid_numeric_mask.any():
        examples = (
            df.loc[
                invalid_numeric_mask,
                ["INTENSITY_GROUP_CODE", "DISTANCE_ZONE"] + numeric_columns,
            ]
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            "Missing or invalid plotting values were found. "
            f"Examples: {examples}"
        )

    invalid_ci_mask = df["BOOTSTRAP_CI_LOWER"] > df["BOOTSTRAP_CI_UPPER"]
    if invalid_ci_mask.any():
        examples = (
            df.loc[
                invalid_ci_mask,
                [
                    "INTENSITY_GROUP_CODE",
                    "DISTANCE_ZONE",
                    "BOOTSTRAP_CI_LOWER",
                    "BOOTSTRAP_CI_UPPER",
                ],
            ]
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            "Bootstrap lower bounds exceed upper bounds. "
            f"Examples: {examples}"
        )

    invalid_sample_size_mask = (
        (df["COMMON_VALID_WINDOW_COUNT"] < 0)
        | ~np.isclose(
            df["COMMON_VALID_WINDOW_COUNT"],
            np.round(df["COMMON_VALID_WINDOW_COUNT"]),
        )
    )
    if invalid_sample_size_mask.any():
        examples = (
            df.loc[
                invalid_sample_size_mask,
                [
                    "INTENSITY_GROUP_CODE",
                    "DISTANCE_ZONE",
                    "COMMON_VALID_WINDOW_COUNT",
                ],
            ]
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            "Invalid COMMON_VALID_WINDOW_COUNT values were found. "
            f"Examples: {examples}"
        )

    df["COMMON_VALID_WINDOW_COUNT"] = df[
        "COMMON_VALID_WINDOW_COUNT"
    ].astype(np.int64)

    print(f"Validated plotting rows: {len(df)}")
    return df, input_path


# ============================================================
# 8. Plot one spatial category
# ============================================================

def plot_one(spatial_type):
    """Plot AREA change for RW in one spatial category."""
    df, _ = read_and_validate_table(spatial_type)
    x = np.arange(len(DISTANCE_ZONES))

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for group_code, _, color in GROUPS:
        group_df = (
            df.loc[df["INTENSITY_GROUP_CODE"] == group_code]
            .set_index("DISTANCE_ZONE")
            .reindex(DISTANCE_ZONES)
        )

        value_columns = [
            "MEAN_CHANGE",
            "BOOTSTRAP_CI_LOWER",
            "BOOTSTRAP_CI_UPPER",
        ]
        if group_df[value_columns].isna().any(axis=None):
            raise ValueError(f"Incomplete plotting data for {group_code}")

        mean = group_df["MEAN_CHANGE"].to_numpy(dtype=float)
        lower = group_df["BOOTSTRAP_CI_LOWER"].to_numpy(dtype=float)
        upper = group_df["BOOTSTRAP_CI_UPPER"].to_numpy(dtype=float)

        for x_position, value in zip(x, mean):
            draw_gradient_bar(
                ax=ax,
                x_position=x_position,
                value=value,
                width=BAR_WIDTH,
                end_color=color,
            )

        y_error = np.vstack((mean - lower, upper - mean))
        if (y_error < 0).any():
            raise ValueError(
                f"The mean lies outside its bootstrap interval for {group_code}."
            )

        ax.errorbar(
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

    ax.axhline(
        y=0,
        color="#000000",
        linestyle="--",
        linewidth=1.6,
        zorder=6,
    )

    ax.set_xticks(x)
    ax.set_xlim(-0.5, len(DISTANCE_ZONES) - 0.5)
    ax.set_xticklabels(
        DISTANCE_LABELS,
        fontsize=PLOT_FONT_SIZE,
        rotation=0,
        ha="center",
        color="#000000",
    )

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel(
        r"Change in heavy-rainfall area (km$^2$)",
        fontsize=PLOT_FONT_SIZE,
        fontfamily="Arial",
        color="#000000",
        labelpad=10,
    )

    ax.tick_params(
        axis="x",
        labelsize=PLOT_FONT_SIZE,
        pad=8,
        length=10,
        width=2.0,
        colors="#000000",
        direction="out",
    )
    ax.tick_params(
        axis="y",
        labelsize=PLOT_FONT_SIZE,
        length=10,
        width=2.0,
        colors="#000000",
        direction="out",
    )

    # Fixed scientific-scale limits: the displayed axis runs from -6 to 4
    # while the offset text shows 1e3.
    ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)
    ax.set_yticks(
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
    ax.yaxis.set_major_formatter(formatter)

    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_fontsize(PLOT_FONT_SIZE)
    offset_text.set_fontfamily("Arial")
    offset_text.set_color("#000000")
    offset_text.set_y(1.025)
    offset_text.set_verticalalignment("bottom")

    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        visible=True,
        linestyle="--",
        color=GRID_COLOR,
        alpha=GRID_ALPHA,
        linewidth=GRID_LINE_WIDTH,
    )
    ax.grid(axis="x", visible=False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(2.8)

    fig.subplots_adjust(
        left=0.25,
        right=0.98,
        bottom=0.14,
        top=0.90,
    )

    png_path = OUTPUT_DIR / "FIG2A.png"
    fig.savefig(
        png_path,
        dpi=FIGURE_DPI,
        facecolor="white",
    )
    plt.close(fig)

    print(f"Saved PNG: {png_path}")


# ============================================================
# 9. Draw one statistics table
# ============================================================

def plot_statistics_table(df, spatial_type):
    """Save mean changes and bootstrap intervals as a PNG table."""
    group_label_by_code = {
        group_code: group_label
        for group_code, group_label, _ in GROUPS
    }
    distance_label_by_code = dict(zip(DISTANCE_ZONES, DISTANCE_LABELS))

    table_df = df[
        [
            "INTENSITY_GROUP_CODE",
            "DISTANCE_ZONE",
            "COMMON_VALID_WINDOW_COUNT",
            "MEAN_CHANGE",
            "BOOTSTRAP_CI_LOWER",
            "BOOTSTRAP_CI_UPPER",
        ]
    ].copy()

    table_df["Group"] = table_df["INTENSITY_GROUP_CODE"].map(
        group_label_by_code
    )
    table_df["Distance zone"] = table_df["DISTANCE_ZONE"].map(
        distance_label_by_code
    )

    group_order = {
        group_label: index
        for index, (_, group_label, _) in enumerate(GROUPS)
    }
    distance_order = {
        zone: index for index, zone in enumerate(DISTANCE_ZONES)
    }

    table_df["_GROUP_ORDER"] = table_df["Group"].map(group_order)
    table_df["_DISTANCE_ORDER"] = table_df["DISTANCE_ZONE"].map(
        distance_order
    )
    table_df = table_df.sort_values(
        ["_GROUP_ORDER", "_DISTANCE_ORDER"]
    ).reset_index(drop=True)

    table_df = table_df.rename(
        columns={
            "MEAN_CHANGE": "Mean change",
            "BOOTSTRAP_CI_LOWER": "CI lower",
            "BOOTSTRAP_CI_UPPER": "CI upper",
            "COMMON_VALID_WINDOW_COUNT": "N",
        }
    )

    display_columns = [
        "Group",
        "Distance zone",
        "N",
        "Mean change",
        "CI lower",
        "CI upper",
    ]
    table_df = table_df[display_columns]
    table_df["N"] = table_df["N"].map(lambda value: f"{int(value):,}")

    for column in ["Mean change", "CI lower", "CI upper"]:
        table_df[column] = table_df[column].map(
            lambda value: f"{value:.4e}"
        )

    fig, ax = plt.subplots(figsize=(13, 11), dpi=FIGURE_DPI)
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=display_columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.03, 0.03, 0.94, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1.0, 1.55)

    group_color_by_label = {
        group_label: color for _, group_label, color in GROUPS
    }

    for column_index in range(len(display_columns)):
        header_cell = table[(0, column_index)]
        header_cell.set_facecolor("#D9E1E8")
        header_cell.set_text_props(
            color="#000000",
            fontfamily="Arial",
            fontweight="bold",
        )
        header_cell.set_edgecolor("#000000")
        header_cell.set_linewidth(1.0)

    for row_index in range(1, len(table_df) + 1):
        group_label = table_df.iloc[row_index - 1]["Group"]
        for column_index in range(len(display_columns)):
            cell = table[(row_index, column_index)]
            cell.set_edgecolor("#B7B7B7")
            cell.set_linewidth(0.8)
            cell.set_text_props(
                color="#000000",
                fontfamily="Arial",
            )

            if column_index == 0:
                cell.set_facecolor(group_color_by_label[group_label])
                cell.set_text_props(
                    color="#000000",
                    fontfamily="Arial",
                    fontweight="bold",
                )
            elif row_index % 2 == 0:
                cell.set_facecolor("#F2F4F5")
            else:
                cell.set_facecolor("#FFFFFF")

    output_stem = (
        f"PRE_DATA_IBT_1982_2024_MSWEP_TH{THRESHOLD}_"
        f"{spatial_type}_AREA_{DISTANCE_OUTPUT_TAG}_"
        "RW_BOOTSTRAP_TABLE"
    )
    table_png_path = OUTPUT_DIR / f"{output_stem}.png"

    fig.savefig(
        table_png_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print(f"Saved table PNG: {table_png_path}")


# ============================================================
# 10. Draw the legend as a separate figure
# ============================================================

def plot_separate_legend():
    """Save the RW legend as a standalone PNG."""
    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor="#000000",
            linewidth=1.5,
            label=group_label,
        )
        for _, group_label, color in GROUPS
    ]

    fig, ax = plt.subplots(figsize=(6.4, 1.2))
    ax.axis("off")
    legend = ax.legend(
        handles=legend_handles,
        loc="center",
        ncol=1,
        fontsize=PLOT_FONT_SIZE,
        labelcolor="#000000",
        frameon=True,
        fancybox=False,
        edgecolor="#000000",
    )
    legend.get_frame().set_linewidth(2.0)

    legend_path = OUTPUT_DIR / (
        f"PRE_DATA_IBT_1982_2024_MSWEP_TH{THRESHOLD}_"
        f"OPEN_OCEAN_AREA_{DISTANCE_OUTPUT_TAG}_RW_LEGEND.png"
    )
    fig.savefig(
        legend_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print(f"Saved separate legend PNG: {legend_path}")


# ============================================================
# 11. Main program
# ============================================================

def main():
    plot_one(SPATIAL_TYPE)

    print("\n" + "=" * 90)
    print("Threshold-30 OPEN_OCEAN AREA figure completed successfully")
    print("=" * 90)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
