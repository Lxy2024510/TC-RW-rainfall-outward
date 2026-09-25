from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


# ============================================================
# 1. Configuration
# ============================================================

THRESHOLD = "30"

INPUT_DIR = Path(
    PROJECT_ROOT / "Data" / "Processed" / "MSWEP_BOOTSTRAP_RESULTS"
) / f"TH{THRESHOLD}"

OUTPUT_DIR = PROJECT_ROOT / "Results" / "Extended_figures" / "EX_FIG5"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. Distance-zone configuration
# ============================================================

DISTANCE_ZONES = [
    "0_100",
    "100_200",
    "200_300",
    "300_400",
    "400_500",
]

DISTANCE_LABELS = [
    "0\u2013100 km",
    "100\u2013200 km",
    "200\u2013300 km",
    "300\u2013400 km",
    "400\u2013500 km",
]


# ============================================================
# 3. Intensity-group configuration
# ============================================================

# RW uses the original <=5th group, 0 uses the median group,
# and RI uses the original >=95th group.
GROUPS = [
    ("LE_5TH", "RW", "#D5A0B6"),
    ("EQ_50TH", "0", "#D8D8D8"),
    ("GE_95TH", "RI", "#D5A0B6"),
]


# ============================================================
# 4. Plot configuration
# ============================================================

SPATIAL_TYPE = "NEARSHORE"

METRIC_TYPE = "AREA"
FIGURE_DPI = 600
FIGURE_SIZE = (6.4, 12.8)
# Keep the formal panel at the same 1:2 width-to-height ratio as FIG5B/FIG5C.
FIG5A_FIGURE_SIZE = (6.4, 12.8)
PLOT_FONT_SIZE = 32

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
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
    }
)


# ============================================================
# 5. Input validation
# ============================================================

def read_and_validate_table(spatial_type):
    """Read and validate one threshold-30 AREA bootstrap table."""
    input_path = INPUT_DIR / (
        f"PRE_DATA_IBT_1982_2024_MSWEP_TH{THRESHOLD}_"
        f"{spatial_type}_{METRIC_TYPE}_BOOTSTRAP_TABLE.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input table does not exist: {input_path}"
        )

    print("=" * 90)
    print(f"Reading {spatial_type} {METRIC_TYPE} bootstrap table")
    print("=" * 90)
    print(f"Input file: {input_path}")

    df = pd.read_csv(
        input_path,
        low_memory=False,
    )

    required_columns = [
        "INTENSITY_GROUP_CODE",
        "DISTANCE_ZONE",
        "COMMON_VALID_WINDOW_COUNT",
        "MEAN_CHANGE",
        "BOOTSTRAP_CI_LOWER",
        "BOOTSTRAP_CI_UPPER",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}"
        )

    selected_group_codes = [
        group_code
        for group_code, _, _ in GROUPS
    ]

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
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    duplicate_mask = df.duplicated(
        subset=[
            "INTENSITY_GROUP_CODE",
            "DISTANCE_ZONE",
        ],
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
                [
                    "INTENSITY_GROUP_CODE",
                    "DISTANCE_ZONE",
                ]
                + numeric_columns,
            ]
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            "Missing or invalid plotting values were found. "
            f"Examples: {examples}"
        )

    invalid_ci_mask = (
        df["BOOTSTRAP_CI_LOWER"]
        > df["BOOTSTRAP_CI_UPPER"]
    )

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
# 6. Plot one spatial category
# ============================================================

def plot_one(
    spatial_type,
    plot_groups=GROUPS,
    output_group_tag="RW_0_RI",
    save_statistics_table=True,
):
    """Plot AREA change for selected groups in one spatial category."""
    df, _ = read_and_validate_table(spatial_type)
    x = np.arange(len(DISTANCE_ZONES))

    figure_size = (
        FIG5A_FIGURE_SIZE
        if output_group_tag == "RW_ONLY"
        else FIGURE_SIZE
    )
    fig, ax = plt.subplots(figsize=figure_size)

    for group_code, group_label, color in plot_groups:
        group_df = (
            df.loc[
                df["INTENSITY_GROUP_CODE"] == group_code
            ]
            .set_index("DISTANCE_ZONE")
            .reindex(DISTANCE_ZONES)
        )

        if group_df[
            [
                "MEAN_CHANGE",
                "BOOTSTRAP_CI_LOWER",
                "BOOTSTRAP_CI_UPPER",
            ]
        ].isna().any(axis=None):
            raise ValueError(
                f"Incomplete plotting data for {group_code}"
            )

        mean = group_df["MEAN_CHANGE"].to_numpy(dtype=float)
        lower = group_df["BOOTSTRAP_CI_LOWER"].to_numpy(dtype=float)
        upper = group_df["BOOTSTRAP_CI_UPPER"].to_numpy(dtype=float)

        ax.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

        ax.plot(
            x,
            mean,
            color=color,
            linewidth=4.0,
            marker="o",
            markersize=10.0,
            label=group_label,
        )

    ax.axhline(
        y=0,
        color="#333333",
        linestyle="--",
        linewidth=1.4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        DISTANCE_LABELS,
        fontsize=PLOT_FONT_SIZE,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    ax.set_title("")
    ax.set_xlabel("")

    ax.set_ylabel(
        r"Change in heavy-rainfall area (km$^2$)",
        fontsize=PLOT_FONT_SIZE,
        labelpad=16,
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

    plotted_group_codes = [
        group_code
        for group_code, _, _ in plot_groups
    ]
    plotted_df = df.loc[
        df["INTENSITY_GROUP_CODE"].isin(plotted_group_codes)
    ]

    y_min = min(
        0,
        float(plotted_df["BOOTSTRAP_CI_LOWER"].min()),
    )
    y_max = max(
        0,
        float(plotted_df["BOOTSTRAP_CI_UPPER"].max()),
    )
    y_range = y_max - y_min

    if np.isclose(y_range, 0):
        y_range = 1.0

    if output_group_tag == "RW_ONLY":
        # The formal FIG5A contains only RW. Fix its upper limit at 1 × 10^3
        # km² while retaining a data-driven lower limit and modest margin.
        fixed_y_max = 1.0e3
        plot_y_min = y_min - (fixed_y_max - y_min) * 0.12
        ax.set_ylim(plot_y_min, fixed_y_max)

        ax.set_axisbelow(True)
        ax.yaxis.grid(
            True,
            color="#C4C4C4",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
        )
        ax.xaxis.grid(False)
    else:
        ax.set_ylim(
            y_min - y_range * 0.12,
            y_max + y_range * 0.18,
        )

    formatter = ScalarFormatter(
        useMathText=False,
    )
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    formatter.set_useOffset(False)
    ax.yaxis.set_major_formatter(formatter)
    if output_group_tag == "RW_ONLY":
        ax.yaxis.get_offset_text().set_visible(False)
        ax.text(
            0.0,
            1.010,
            "1e3",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=PLOT_FONT_SIZE,
            color="#000000",
            clip_on=False,
        )
    else:
        ax.yaxis.get_offset_text().set_fontsize(PLOT_FONT_SIZE)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(2.0)

    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        bottom=0.25,
        top=0.96 if output_group_tag == "RW_ONLY" else 0.97,
    )

    if output_group_tag == "RW_ONLY":
        png_path = OUTPUT_DIR / "EX_FIG5A.png"
    else:
        png_path = OUTPUT_DIR / "EX_FIG5A_RW_0_RI.png"

    if output_group_tag == "RW_ONLY":
        fig.canvas.draw()
        ax.yaxis.get_offset_text().set_visible(False)

    fig.savefig(
        png_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Saved PNG: {png_path}")

    if save_statistics_table:
        plot_statistics_table(
            df=df,
            spatial_type=spatial_type,
        )


# ============================================================
# 7. Draw one statistics table
# ============================================================

def plot_statistics_table(df, spatial_type):
    """Save mean changes and bootstrap intervals as a PNG table."""
    group_label_by_code = {
        group_code: group_label
        for group_code, group_label, _ in GROUPS
    }
    distance_label_by_code = dict(
        zip(DISTANCE_ZONES, DISTANCE_LABELS)
    )

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

    table_df["Group"] = table_df[
        "INTENSITY_GROUP_CODE"
    ].map(group_label_by_code)
    table_df["Distance zone"] = table_df[
        "DISTANCE_ZONE"
    ].map(distance_label_by_code)

    group_order = {
        group_label: index
        for index, (_, group_label, _) in enumerate(GROUPS)
    }
    distance_order = {
        zone: index
        for index, zone in enumerate(DISTANCE_ZONES)
    }

    table_df["_GROUP_ORDER"] = table_df["Group"].map(group_order)
    table_df["_DISTANCE_ORDER"] = table_df[
        "DISTANCE_ZONE"
    ].map(distance_order)
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

    table_df["N"] = table_df["N"].map(
        lambda value: f"{int(value):,}"
    )

    for column in ["Mean change", "CI lower", "CI upper"]:
        table_df[column] = table_df[column].map(
            lambda value: f"{value:.4e}"
        )

    fig, ax = plt.subplots(
        figsize=(13, 11),
        dpi=FIGURE_DPI,
    )
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
        group_label: color
        for _, group_label, color in GROUPS
    }

    for column_index in range(len(display_columns)):
        header_cell = table[(0, column_index)]
        header_cell.set_facecolor("#3E5266")
        header_cell.set_text_props(
            color="white",
            fontweight="bold",
        )
        header_cell.set_edgecolor("#2F3E4D")
        header_cell.set_linewidth(1.0)

    for row_index in range(1, len(table_df) + 1):
        group_label = table_df.iloc[row_index - 1]["Group"]

        for column_index in range(len(display_columns)):
            cell = table[(row_index, column_index)]
            cell.set_edgecolor("#B7B7B7")
            cell.set_linewidth(0.8)

            if column_index == 0:
                cell.set_facecolor(
                    group_color_by_label[group_label]
                )
                cell.set_text_props(fontweight="bold")
            elif row_index % 2 == 0:
                cell.set_facecolor("#F2F4F5")
            else:
                cell.set_facecolor("#FFFFFF")

    table_png_path = OUTPUT_DIR / "EX_FIG5A_statistics.png"

    fig.savefig(
        table_png_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print(f"Saved table PNG: {table_png_path}")


# ============================================================
# 8. Draw the legend as a separate figure
# ============================================================

def plot_separate_legend():
    """Save the RW, 0, and RI legend as a standalone PNG."""
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

    fig, ax = plt.subplots(figsize=(6.4, 1.2))
    ax.axis("off")
    legend = ax.legend(
        handles=legend_handles,
        loc="center",
        ncol=3,
        fontsize=PLOT_FONT_SIZE,
        frameon=True,
        fancybox=False,
        edgecolor="#000000",
    )
    legend.get_frame().set_linewidth(2.0)

    legend_path = OUTPUT_DIR / "EX_FIG5A_legend.png"
    fig.savefig(
        legend_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print(f"Saved separate legend PNG: {legend_path}")


# ============================================================
# 9. Main program
# ============================================================

def main():
    plot_one(
        spatial_type=SPATIAL_TYPE,
        plot_groups=(GROUPS[0],),
        output_group_tag="RW_ONLY",
        save_statistics_table=False,
    )

    print("\n" + "=" * 90)
    print("Threshold-30 NEARSHORE AREA figure completed successfully")
    print("=" * 90)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
