"""Plot RW overlap-sensitivity distributions using the established style."""

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
# 1. Input and output configuration
# ============================================================

INPUT_CSV = Path(
    str(PROJECT_ROOT / "Data" / "Processed" / "MSWEP_24H_OVERLAP_SENSITIVITY_RESULTS") + "/" +
    "RW_TH30_24H_OVERLAP_SENSITIVITY_COMBINED.csv"
)

REFERENCE_DIST30_CSV = Path(PROJECT_ROOT / "Data" / "Processed" / "MSWEP_24H_RESULTS") / (
    "PRE_DATA_IBT_1982_2024_MSWEP_TH30_24H_SLIDING_ET_CLEANED.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "Results" / "Extended_figures" / "EX_FIG1"

OUTPUT_PNG = OUTPUT_DIR / "EX_FIG1C.png"

OUTPUT_STATISTICS_CSV = OUTPUT_DIR / "EX_FIG1C.csv"


# ============================================================
# 2. Analysis configuration
# ============================================================

GROUP_SOURCE_COLUMN = "OVERLAP_HOURS"
TARGET_COLUMN = "DIFF_MSWEP_DIST_30"
GROUP_COLUMN = "OVERLAP_GROUP"
POSITIVE_RATE_COLUMN = "P(DeltaDIST30 > 0)"

GROUP_LABELS = [
    "21-h overlap",
    "12-h overlap",
    "0-h overlap",
]

OVERLAP_TO_LABEL = {
    21: "21-h overlap",
    12: "12-h overlap",
    0: "0-h overlap",
}

RW_BLUE = "#56A1BB"
GROUP_PALETTE = {label: RW_BLUE for label in GROUP_LABELS}


# ============================================================
# 3. Figure configuration copied from the intensity-group plot
# ============================================================

DPI = 300
FIGURE_HEIGHT = 9.5
LEFT_MARGIN_INCHES = 2.55
RIGHT_MARGIN_INCHES = 0.35
TOP_MARGIN_INCHES = 0.25
BOTTOM_MARGIN_INCHES = 1.25

# The seven-group reference figure has this physical plotting width.
REFERENCE_GROUP_COUNT = 7
CURRENT_GROUP_COUNT = 3
REFERENCE_CATEGORY_SLOT_WIDTH = (
    17.0 - 1.55 - RIGHT_MARGIN_INCHES
) / REFERENCE_GROUP_COUNT
FIGURE_WIDTH = (
    LEFT_MARGIN_INCHES
    + RIGHT_MARGIN_INCHES
    + REFERENCE_GROUP_COUNT * REFERENCE_CATEGORY_SLOT_WIDTH
)

FONT_SIZE = 34
ANNOTATION_FONT_SIZE = 28
FONT_COLOR = "#000000"

# Scaling by 3/7 keeps the physical widths equal to those in the seven-group
# figure even though Seaborn distributes only three categories across the axis.
REFERENCE_VIOLIN_WIDTH = 0.42
REFERENCE_BOX_WIDTH = 0.07
WIDTH_SCALE = CURRENT_GROUP_COUNT / REFERENCE_GROUP_COUNT
VIOLIN_WIDTH = REFERENCE_VIOLIN_WIDTH * WIDTH_SCALE
BOX_WIDTH = REFERENCE_BOX_WIDTH * WIDTH_SCALE

VIOLIN_LINE_WIDTH = 2.6
BOX_LINE_WIDTH = 2.4
MEDIAN_LINE_WIDTH = 2.6
FRAME_LINE_WIDTH = 2.8
ZERO_LINE_WIDTH = 1.6

Y_TICK_INTERVAL = 100.0
Y_LOWER_SHIFT_KM = 100.0


# ============================================================
# 4. Read and validate the combined sensitivity table
# ============================================================

def read_and_validate_data() -> pd.DataFrame:
    """Read the combined overlap table and validate the plotting fields."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV does not exist:\n{INPUT_CSV}")

    df = pd.read_csv(
        INPUT_CSV,
        usecols=[GROUP_SOURCE_COLUMN, TARGET_COLUMN],
        low_memory=False,
    )

    for column in [GROUP_SOURCE_COLUMN, TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[[GROUP_SOURCE_COLUMN, TARGET_COLUMN]] = df[
        [GROUP_SOURCE_COLUMN, TARGET_COLUMN]
    ].replace([np.inf, -np.inf], np.nan)

    invalid = df[[GROUP_SOURCE_COLUMN, TARGET_COLUMN]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(
            "The combined sensitivity table contains invalid plotting rows: "
            f"{int(invalid.sum()):,}."
        )

    observed_overlaps = set(df[GROUP_SOURCE_COLUMN].astype(int).unique())
    expected_overlaps = set(OVERLAP_TO_LABEL)
    if observed_overlaps != expected_overlaps:
        raise ValueError(
            f"Unexpected overlap groups: observed={sorted(observed_overlaps)}, "
            f"expected={sorted(expected_overlaps)}."
        )

    df[GROUP_COLUMN] = (
        df[GROUP_SOURCE_COLUMN]
        .astype(int)
        .map(OVERLAP_TO_LABEL)
    )
    df[GROUP_COLUMN] = pd.Categorical(
        df[GROUP_COLUMN],
        categories=GROUP_LABELS,
        ordered=True,
    )

    print("=" * 80)
    print("RW overlap-sensitivity violin analysis")
    print("=" * 80)
    print(f"Input: {INPUT_CSV}")
    print(f"Rows: {len(df):,}")
    return df


# ============================================================
# 5. Calculate full-sample statistics
# ============================================================

def calculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics from all values in each overlap group."""
    records = []

    for label in GROUP_LABELS:
        values = df.loc[df[GROUP_COLUMN] == label, TARGET_COLUMN].astype(float)
        if values.empty:
            raise ValueError(f"Group {label} contains no observations.")

        quantiles = values.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        records.append(
            {
                "Overlap group": label,
                "N": len(values),
                "Mean": values.mean(),
                "5th": quantiles.loc[0.05],
                "25th": quantiles.loc[0.25],
                "Median": quantiles.loc[0.50],
                "75th": quantiles.loc[0.75],
                "95th": quantiles.loc[0.95],
                POSITIVE_RATE_COLUMN: values.gt(0).mean() * 100.0,
            }
        )

    return pd.DataFrame(records)


def build_trimmed_violin_data(
    df: pd.DataFrame,
    statistics: pd.DataFrame,
) -> pd.DataFrame:
    """Use each group's 5th-95th percentile values only for violin density."""
    parts = []

    for label in GROUP_LABELS:
        row = statistics.loc[
            statistics["Overlap group"] == label
        ].iloc[0]
        group = df.loc[df[GROUP_COLUMN] == label]
        trimmed = group.loc[
            group[TARGET_COLUMN].between(
                row["5th"],
                row["95th"],
                inclusive="both",
            )
        ].copy()
        parts.append(trimmed)

    result = pd.concat(parts, ignore_index=True)
    result[GROUP_COLUMN] = pd.Categorical(
        result[GROUP_COLUMN],
        categories=GROUP_LABELS,
        ordered=True,
    )
    return result


def calculate_reference_y_layout() -> tuple[float, float, np.ndarray, float]:
    """Reproduce the exact vertical layout of the finalized DIST30 plot."""
    if not REFERENCE_DIST30_CSV.exists():
        raise FileNotFoundError(
            "DIST30 reference CSV does not exist:\n"
            f"{REFERENCE_DIST30_CSV}"
        )

    reference = pd.read_csv(
        REFERENCE_DIST30_CSV,
        usecols=["DIFF_USA_WIND", "DIFF_MSWEP_DIST_30"],
        low_memory=False,
    )
    for column in ["DIFF_USA_WIND", "DIFF_MSWEP_DIST_30"]:
        reference[column] = pd.to_numeric(reference[column], errors="coerce")
    reference = reference.replace([np.inf, -np.inf], np.nan).dropna()

    wind = reference["DIFF_USA_WIND"]
    conditions = [
        wind <= -30,
        (wind > -30) & (wind <= -8),
        (wind > -8) & (wind < 0),
        wind == 0,
        (wind > 0) & (wind < 12),
        (wind >= 12) & (wind < 30),
        wind >= 30,
    ]
    intensity_labels = [
        "[0,5th]", "(5th,25th]", "(25th,50th)", "50th",
        "(50th,75th)", "[75th,95th)", "[95th,100th]",
    ]
    reference["REFERENCE_GROUP"] = np.select(
        conditions, intensity_labels, default=None
    )
    if reference["REFERENCE_GROUP"].isna().any():
        raise RuntimeError("Some DIST30 reference rows were not grouped.")

    q5_values = []
    q95_values = []
    for label in intensity_labels:
        values = reference.loc[
            reference["REFERENCE_GROUP"] == label,
            "DIFF_MSWEP_DIST_30",
        ]
        if values.empty:
            raise ValueError(f"DIST30 reference group {label} is empty.")
        q5_values.append(values.quantile(0.05))
        q95_values.append(values.quantile(0.95))

    q5 = min(q5_values)
    q95 = max(q95_values)
    span = q95 - q5
    if not np.isfinite(span) or np.isclose(span, 0):
        raise ValueError("Cannot calculate the DIST30 reference layout.")

    # Shorten the lower part of the axis by moving both the lower limit and
    # the two-line N/P annotation upward by 100 km.
    y_min = q5 - span * 0.27 + Y_LOWER_SHIFT_KM
    y_max = 300.0
    annotation_y = q5 - span * 0.19 + Y_LOWER_SHIFT_KM
    tick_min = np.ceil(y_min / Y_TICK_INTERVAL) * Y_TICK_INTERVAL
    ticks = np.arange(
        tick_min,
        y_max + Y_TICK_INTERVAL * 0.5,
        Y_TICK_INTERVAL,
    )
    return y_min, y_max, ticks, annotation_y


# ============================================================
# 6. Draw the violin figure
# ============================================================

def plot_violin(
    full_df: pd.DataFrame,
    trimmed_df: pd.DataFrame,
    statistics: pd.DataFrame,
) -> None:
    """Draw three physically matched RW violins without significance tests."""
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

    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        dpi=DPI,
    )
    fig.subplots_adjust(
        left=LEFT_MARGIN_INCHES / FIGURE_WIDTH,
        right=1.0 - RIGHT_MARGIN_INCHES / FIGURE_WIDTH,
        bottom=BOTTOM_MARGIN_INCHES / FIGURE_HEIGHT,
        top=1.0 - TOP_MARGIN_INCHES / FIGURE_HEIGHT,
    )

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
            "linewidth": MEDIAN_LINE_WIDTH,
            "zorder": 5,
        },
        whiskerprops={"linewidth": 0},
        ax=ax,
    )

    ax.axhline(
        0,
        color=FONT_COLOR,
        linestyle="--",
        linewidth=ZERO_LINE_WIDTH,
        alpha=0.90,
        zorder=1,
    )

    y_min, y_max, y_ticks, annotation_y = calculate_reference_y_layout()
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)

    for index, row in statistics.reset_index(drop=True).iterrows():
        count_math = f"{int(row['N']):,}".replace(",", "{,}")
        ratio_math = f"{row[POSITIVE_RATE_COLUMN]:.1f}"
        text = (
            rf"$\mathit{{N}} = \mathrm{{{count_math}}}$"
            "\n"
            rf"$\mathit{{P}} = \mathrm{{{ratio_math}\%}}$"
        )
        ax.text(
            index,
            annotation_y,
            text,
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
        color=FONT_COLOR,
        linespacing=1.30,
        labelpad=16,
    )

    ax.tick_params(
        axis="x",
        which="major",
        direction="out",
        length=8,
        width=2.0,
        labelsize=FONT_SIZE,
        pad=14,
        colors=FONT_COLOR,
        bottom=True,
        top=False,
    )
    ax.tick_params(
        axis="y",
        which="major",
        direction="out",
        length=8,
        width=2.0,
        labelsize=FONT_SIZE,
        colors=FONT_COLOR,
        left=True,
        right=False,
    )

    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight("normal")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(FONT_COLOR)
        spine.set_linewidth(FRAME_LINE_WIDTH)

    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", visible=False)
    # Hide the horizontal gridline nearest the N/P annotation so that the
    # dashed line does not run through the text after the 100-km shift.
    annotation_tick = y_ticks[np.argmin(np.abs(y_ticks - annotation_y))]
    for tick_value, gridline in zip(ax.get_yticks(), ax.get_ygridlines()):
        if np.isclose(tick_value, annotation_tick):
            gridline.set_visible(False)
    ax.set_axisbelow(True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"\nViolin PNG: {OUTPUT_PNG}")


# ============================================================
# 7. Main program
# ============================================================

def main() -> None:
    """Run the overlap-sensitivity violin analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = read_and_validate_data()
    statistics = calculate_statistics(df)
    trimmed_df = build_trimmed_violin_data(df, statistics)

    plot_violin(
        full_df=df,
        trimmed_df=trimmed_df,
        statistics=statistics,
    )

    print("Processing completed successfully.")


if __name__ == "__main__":
    main()
