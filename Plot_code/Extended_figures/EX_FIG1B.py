from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.cursive": ["Arial"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.cal": "Arial",
})


# ============================================================
# 1. File path configuration
# ============================================================

INPUT_CSV = Path(
    str(PROJECT_ROOT / "Data" / "Processed" / "MSWEP_36H_RESULTS") + "/" +
    "PRE_DATA_IBT_1982_2024_MSWEP_TH30_36H_SLIDING_ET_CLEANED.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "Results" / "Extended_figures" / "EX_FIG1"

VIOLIN_OUTPUT_PNG = OUTPUT_DIR / "EX_FIG1B.png"

TABLE_OUTPUT_PNG = OUTPUT_DIR / "EX_FIG1B_statistics.png"


# ============================================================
# 2. Analysis field configuration
# ============================================================

GROUP_SOURCE_COLUMN = "DIFF_USA_WIND"
TARGET_COLUMN = "DIFF_MSWEP_DIST_30"
GROUP_COLUMN = "INTENSITY_GROUP"
POSITIVE_RATIO_COLUMN = "P(DeltaDIST30 > 0)"


# ============================================================
# 3. Fixed 36-hour intensity-group configuration
# ============================================================

GROUP_LABELS = [
    "[0,5th]",
    "(5th,25th]",
    "(25th,50th)",
    "50th",
    "(50th,75th)",
    "[75th,95th)",
    "[95th,100th]",
]

GROUP_RANGES = {
    "[0,5th]": "DIFF_USA_WIND <= -40",
    "(5th,25th]": "-40 < DIFF_USA_WIND <= -10",
    "(25th,50th)": "-10 < DIFF_USA_WIND < 5",
    "50th": "DIFF_USA_WIND = 5",
    "(50th,75th)": "5 < DIFF_USA_WIND < 17",
    "[75th,95th)": "17 <= DIFF_USA_WIND < 45",
    "[95th,100th]": "DIFF_USA_WIND >= 45",
}


# ============================================================
# 4. Figure configuration
# ============================================================

FIGURE_DPI = 300
FIGURE_HEIGHT = 10.25
LEFT_MARGIN_INCHES = 2.55
RIGHT_MARGIN_INCHES = 0.35
TOP_MARGIN_INCHES = 0.25
BOTTOM_MARGIN_INCHES = 2.00
CATEGORY_SLOT_WIDTH_INCHES = (17.0 - 1.55 - RIGHT_MARGIN_INCHES) / 7.0

FONT_SIZE = 34
ANNOTATION_FONT_SIZE = 28
FONT_COLOR = "#000000"
VIOLIN_WIDTH = 0.42
BOX_WIDTH = 0.07
VIOLIN_LINE_WIDTH = 2.6
BOX_LINE_WIDTH = 2.4
FRAME_LINE_WIDTH = 2.8
ZERO_LINE_WIDTH = 1.6
Y_TICK_INTERVAL = 100.0

GROUP_COLORS = [
    "#56A1BB",
    "#6FBFD7",
    "#A3DCEC",
    "#E2E2E2",
    "#FBC8DB",
    "#ECAFC6",
    "#DEAEC3",
]

GROUP_PALETTE = dict(
    zip(
        GROUP_LABELS,
        GROUP_COLORS,
    )
)


# ============================================================
# 5. Read and validate the CLEANED data
# ============================================================

def read_and_validate_data():
    """Read only the two columns required for this analysis."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV does not exist:\n{INPUT_CSV}"
        )

    print("=" * 80)
    print("Reading threshold-30 CLEANED exact 36-hour data")
    print("=" * 80)
    print(f"Input file: {INPUT_CSV}")

    df = pd.read_csv(
        INPUT_CSV,
        usecols=[
            GROUP_SOURCE_COLUMN,
            TARGET_COLUMN,
        ],
        low_memory=False,
    )

    print(f"Rows read: {len(df):,}")

    for column in [GROUP_SOURCE_COLUMN, TARGET_COLUMN]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df[
        [GROUP_SOURCE_COLUMN, TARGET_COLUMN]
    ] = df[
        [GROUP_SOURCE_COLUMN, TARGET_COLUMN]
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    invalid_wind_count = int(
        df[GROUP_SOURCE_COLUMN].isna().sum()
    )
    invalid_target_count = int(
        df[TARGET_COLUMN].isna().sum()
    )
    invalid_any_mask = df[
        [GROUP_SOURCE_COLUMN, TARGET_COLUMN]
    ].isna().any(axis=1)
    invalid_any_count = int(invalid_any_mask.sum())

    if invalid_any_count > 0:
        print("\nData validation warning:")
        print(
            f"Invalid {GROUP_SOURCE_COLUMN}: "
            f"{invalid_wind_count:,}"
        )
        print(
            f"Invalid {TARGET_COLUMN}: "
            f"{invalid_target_count:,}"
        )
        print(
            "Rows excluded because either field is invalid: "
            f"{invalid_any_count:,}"
        )

        df = (
            df.loc[~invalid_any_mask]
            .copy()
            .reset_index(drop=True)
        )

    print(f"Valid rows retained: {len(df):,}")

    if df.empty:
        raise ValueError(
            "No valid rows remain after validation."
        )

    return df


# ============================================================
# 6. Assign the seven fixed 36-hour intensity groups
# ============================================================

def assign_intensity_groups(df):
    """Assign seven mutually exclusive DIFF_USA_WIND groups."""
    wind = df[GROUP_SOURCE_COLUMN]

    conditions = [
        wind <= -40,
        (wind > -40) & (wind <= -10),
        (wind > -10) & (wind < 5),
        wind == 5,
        (wind > 5) & (wind < 17),
        (wind >= 17) & (wind < 45),
        wind >= 45,
    ]

    df[GROUP_COLUMN] = np.select(
        conditions,
        GROUP_LABELS,
        default=None,
    )

    unassigned_count = int(
        df[GROUP_COLUMN].isna().sum()
    )

    if unassigned_count > 0:
        raise RuntimeError(
            "Some valid DIFF_USA_WIND values were not assigned "
            f"to a group: {unassigned_count:,}"
        )

    df[GROUP_COLUMN] = pd.Categorical(
        df[GROUP_COLUMN],
        categories=GROUP_LABELS,
        ordered=True,
    )

    counts = (
        df[GROUP_COLUMN]
        .value_counts(sort=False)
        .reindex(GROUP_LABELS, fill_value=0)
    )

    print("\nIntensity-group counts:")
    for label, count in counts.items():
        print(f"  {label:<15} {int(count):>10,}")

    if int(counts.sum()) != len(df):
        raise RuntimeError(
            "Grouped row count does not equal valid row count."
        )

    return df


# ============================================================
# 7. Calculate full-sample group statistics
# ============================================================

def calculate_group_statistics(df):
    """Calculate statistics from all valid rows in every group."""
    records = []

    for label in GROUP_LABELS:
        values = df.loc[
            df[GROUP_COLUMN] == label,
            TARGET_COLUMN,
        ].astype(float)

        count = len(values)

        if count == 0:
            record = {
                "Intensity group": label,
                "DIFF_USA_WIND range": GROUP_RANGES[label],
                "N": 0,
                "Mean": np.nan,
                "5th": np.nan,
                "25th": np.nan,
                "Median": np.nan,
                "75th": np.nan,
                "95th": np.nan,
                POSITIVE_RATIO_COLUMN: np.nan,
            }
        else:
            quantiles = values.quantile(
                [0.05, 0.25, 0.50, 0.75, 0.95]
            )

            record = {
                "Intensity group": label,
                "DIFF_USA_WIND range": GROUP_RANGES[label],
                "N": count,
                "Mean": values.mean(),
                "5th": quantiles.loc[0.05],
                "25th": quantiles.loc[0.25],
                "Median": quantiles.loc[0.50],
                "75th": quantiles.loc[0.75],
                "95th": quantiles.loc[0.95],
                POSITIVE_RATIO_COLUMN: values.gt(0).mean() * 100.0,
            }

        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# 8. Prepare 5th-95th percentile data for violin rendering
# ============================================================

def build_trimmed_violin_data(df, statistics_df):
    """Trim only density-rendering data, not group statistics."""
    trimmed_chunks = []

    for label in GROUP_LABELS:
        group_df = df.loc[
            df[GROUP_COLUMN] == label
        ].copy()

        if group_df.empty:
            continue

        stats_row = statistics_df.loc[
            statistics_df["Intensity group"] == label
        ].iloc[0]

        q5 = stats_row["5th"]
        q95 = stats_row["95th"]

        trimmed_group = group_df.loc[
            group_df[TARGET_COLUMN].between(
                q5,
                q95,
                inclusive="both",
            )
        ].copy()

        if not trimmed_group.empty:
            trimmed_chunks.append(trimmed_group)

    if not trimmed_chunks:
        raise ValueError(
            "No observations remain for violin rendering."
        )

    trimmed_df = pd.concat(
        trimmed_chunks,
        ignore_index=True,
    )

    trimmed_df[GROUP_COLUMN] = pd.Categorical(
        trimmed_df[GROUP_COLUMN],
        categories=GROUP_LABELS,
        ordered=True,
    )

    return trimmed_df


# ============================================================
# 9. Draw and save the violin figure
# ============================================================

def plot_violin_figure(full_df, trimmed_df, statistics_df):
    """Draw trimmed violins with full-sample quartile boxes."""
    sns.set_theme(
        style="whitegrid",
        rc={
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "grid.linestyle": "--",
            "grid.color": "#9E9E9E",
            "grid.alpha": 0.55,
            "grid.linewidth": 1.0,
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "text.color": FONT_COLOR,
            "axes.labelcolor": FONT_COLOR,
            "xtick.color": FONT_COLOR,
            "ytick.color": FONT_COLOR,
        },
    )

    figure_width = (
        LEFT_MARGIN_INCHES
        + RIGHT_MARGIN_INCHES
        + len(GROUP_LABELS) * CATEGORY_SLOT_WIDTH_INCHES
    )
    fig, ax = plt.subplots(
        figsize=(figure_width, FIGURE_HEIGHT),
        dpi=FIGURE_DPI,
    )
    fig.subplots_adjust(
        left=LEFT_MARGIN_INCHES / figure_width,
        right=1.0 - RIGHT_MARGIN_INCHES / figure_width,
        bottom=BOTTOM_MARGIN_INCHES / FIGURE_HEIGHT,
        top=1.0 - TOP_MARGIN_INCHES / FIGURE_HEIGHT,
    )

    # Render density from values within each group's 5th-95th range.
    sns.violinplot(
        data=trimmed_df,
        x=GROUP_COLUMN,
        y=TARGET_COLUMN,
        hue=GROUP_COLUMN,
        order=GROUP_LABELS,
        hue_order=GROUP_LABELS,
        palette=GROUP_PALETTE,
        legend=False,
        inner=None,
        cut=0,
        width=VIOLIN_WIDTH,
        linewidth=VIOLIN_LINE_WIDTH,
        density_norm="width",
        ax=ax,
    )

    # Render quartile boxes from all valid observations in each group.
    sns.boxplot(
        data=full_df,
        x=GROUP_COLUMN,
        y=TARGET_COLUMN,
        order=GROUP_LABELS,
        width=BOX_WIDTH,
        showfliers=False,
        showcaps=False,
        whis=0,
        boxprops={
            "facecolor": "#FFFFFF",
            "edgecolor": FONT_COLOR,
            "linewidth": BOX_LINE_WIDTH,
            "alpha": 0.90,
            "zorder": 4,
        },
        medianprops={
            "color": FONT_COLOR,
            "linewidth": 2.6,
            "zorder": 5,
        },
        whiskerprops={"linewidth": 0},
        ax=ax,
    )

    ax.axhline(
        y=0,
        color=FONT_COLOR,
        linestyle="--",
        linewidth=ZERO_LINE_WIDTH,
        alpha=0.90,
        zorder=1,
    )

    visible_q5 = statistics_df["5th"].min()
    visible_q95 = statistics_df["95th"].max()

    if not np.isfinite(visible_q5) or not np.isfinite(visible_q95):
        raise ValueError(
            "Unable to determine finite plotting limits."
        )

    data_span = visible_q95 - visible_q5
    if np.isclose(data_span, 0):
        data_span = max(abs(visible_q95), 1.0)

    plot_y_min = visible_q5 - data_span * 0.27
    plot_y_max = 300.0
    annotation_y = visible_q5 - data_span * 0.19

    tick_min = np.ceil(plot_y_min / Y_TICK_INTERVAL) * Y_TICK_INTERVAL
    y_ticks = np.arange(
        tick_min,
        plot_y_max + Y_TICK_INTERVAL * 0.5,
        Y_TICK_INTERVAL,
    )

    ax.set_ylim(plot_y_min, plot_y_max)
    ax.set_yticks(y_ticks)

    for index, row in statistics_df.iterrows():
        count = int(row["N"])
        positive_ratio = row[POSITIVE_RATIO_COLUMN]
        count_math = f"{count:,}".replace(",", "{,}")

        if np.isnan(positive_ratio):
            text = (
                rf"$\mathit{{N}} = \mathrm{{{count_math}}}$"
                "\n"
                r"$\mathit{P} = \mathrm{NA}$"
            )
        else:
            ratio_math = f"{positive_ratio:.1f}"
            text = (
                rf"$\mathit{{N}} = \mathrm{{{count_math}}}$"
                "\n"
                rf"$\mathit{{P}} = \mathrm{{{ratio_math}\%}}$"
            )

        ax.text(
            x=index,
            y=annotation_y,
            s=text,
            ha="center",
            va="center",
            fontsize=ANNOTATION_FONT_SIZE,
            fontweight="normal",
            color=FONT_COLOR,
            linespacing=1.55,
        )

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel(
        "Change in radial distance of\nTC heavy rainfall (km)",
        fontsize=FONT_SIZE,
        fontweight="normal",
        labelpad=16,
        color=FONT_COLOR,
        linespacing=1.30,
    )

    ax.tick_params(
        axis="x", which="major", direction="out", length=8, width=2.0,
        labelsize=FONT_SIZE, pad=14, colors=FONT_COLOR,
        bottom=True, top=False,
    )
    ax.tick_params(
        axis="y", which="major", direction="out", length=8, width=2.0,
        labelsize=FONT_SIZE, colors=FONT_COLOR,
        left=True, right=False,
    )

    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight("normal")

    for tick_label in ax.get_xticklabels():
        tick_label.set_rotation(30)
        tick_label.set_horizontalalignment("center")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(FONT_COLOR)
        spine.set_linewidth(FRAME_LINE_WIDTH)

    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", visible=False)
    for tick_value, gridline in zip(ax.get_yticks(), ax.get_ygridlines()):
        if np.isclose(tick_value, -300.0):
            gridline.set_visible(False)
    ax.set_axisbelow(True)

    fig.savefig(
        VIOLIN_OUTPUT_PNG,
        dpi=FIGURE_DPI,
        facecolor="white",
    )
    plt.close(fig)

    print(f"\nViolin figure written to:\n{VIOLIN_OUTPUT_PNG}")


# ============================================================
# 10. Draw and save the statistics table figure
# ============================================================

def plot_statistics_table(statistics_df):
    """Render the seven-group statistics as a PNG table."""
    display_df = statistics_df.copy()

    display_df["N"] = display_df["N"].map(
        lambda value: f"{int(value):,}"
    )

    for column in [
        "Mean",
        "5th",
        "25th",
        "Median",
        "75th",
        "95th",
    ]:
        display_df[column] = display_df[column].map(
            lambda value: (
                "NA"
                if pd.isna(value)
                else f"{value:.2f}"
            )
        )

    display_df[POSITIVE_RATIO_COLUMN] = display_df[
        POSITIVE_RATIO_COLUMN
    ].map(
        lambda value: (
            "NA"
            if pd.isna(value)
            else f"{value:.1f}%"
        )
    )

    display_df["DIFF_USA_WIND range"] = [
        "w <= -40",
        "-40 < w <= -10",
        "-10 < w < 5",
        "w = 5",
        "5 < w < 17",
        "17 <= w < 45",
        "w >= 45",
    ]

    display_columns = [
        "Intensity group",
        "DIFF_USA_WIND range",
        "N",
        "Mean",
        "5th",
        "25th",
        "Median",
        "75th",
        "95th",
        POSITIVE_RATIO_COLUMN,
    ]
    display_df = display_df[display_columns]

    fig, ax = plt.subplots(
        figsize=(19, 6.2),
        dpi=FIGURE_DPI,
    )
    ax.axis("off")
    ax.set_title(
        "36-Hour Summary Statistics of DIFF_MSWEP_DIST_30 "
        "Across Typhoon Intensity-Change Groups",
        fontsize=20,
        fontweight="bold",
        color="#262626",
        pad=24,
    )

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.01, 0.08, 0.98, 0.80],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12.0)
    table.scale(1.0, 1.7)

    column_widths = [
        0.105,
        0.165,
        0.085,
        0.080,
        0.070,
        0.070,
        0.080,
        0.070,
        0.070,
        0.145,
    ]

    for column_index, width in enumerate(column_widths):
        for row_index in range(len(display_df) + 1):
            table[(row_index, column_index)].set_width(width)

    for column_index in range(len(display_columns)):
        header_cell = table[(0, column_index)]
        header_cell.set_facecolor("#3E5266")
        header_cell.set_text_props(
            color="white",
            fontweight="bold",
        )
        header_cell.set_edgecolor("#2F3E4D")
        header_cell.set_linewidth(1.0)

    for row_index in range(1, len(display_df) + 1):
        group_color = GROUP_COLORS[row_index - 1]

        for column_index in range(len(display_columns)):
            cell = table[(row_index, column_index)]
            cell.set_edgecolor("#B7B7B7")
            cell.set_linewidth(0.8)

            if column_index == 0:
                cell.set_facecolor(group_color)
                cell.set_text_props(
                    color="#202020",
                    fontweight="bold",
                )
            elif row_index % 2 == 0:
                cell.set_facecolor("#F2F4F5")
            else:
                cell.set_facecolor("#FFFFFF")

    fig.text(
        0.5,
        0.035,
        (
            "N, mean, percentiles, and positive-value probability "
            "are calculated from all valid observations in each group."
        ),
        ha="center",
        va="center",
        fontsize=11.5,
        color="#555555",
        style="italic",
    )

    fig.savefig(
        TABLE_OUTPUT_PNG,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print(
        "\nStatistics table figure written to:\n"
        f"{TABLE_OUTPUT_PNG}"
    )


# ============================================================
# 11. Print statistics for verification
# ============================================================

def print_statistics(statistics_df):
    """Print rounded group statistics to the console."""
    console_df = statistics_df.copy()

    for column in [
        "Mean",
        "5th",
        "25th",
        "Median",
        "75th",
        "95th",
        POSITIVE_RATIO_COLUMN,
    ]:
        console_df[column] = console_df[column].round(3)

    print("\n" + "=" * 120)
    print("36-hour group statistics based on full valid data")
    print("=" * 120)
    print(console_df.to_string(index=False))


# ============================================================
# 12. Main program
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = read_and_validate_data()
    df = assign_intensity_groups(df)
    statistics_df = calculate_group_statistics(df)
    trimmed_df = build_trimmed_violin_data(
        df,
        statistics_df,
    )

    plot_violin_figure(
        full_df=df,
        trimmed_df=trimmed_df,
        statistics_df=statistics_df,
    )

    print("\n" + "=" * 80)
    print("36-hour processing completed successfully")
    print("=" * 80)
    print(f"Violin PNG:\n{VIOLIN_OUTPUT_PNG}")


if __name__ == "__main__":
    main()
