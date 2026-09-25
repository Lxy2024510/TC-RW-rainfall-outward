"""Plot RW radial-distance changes across four rainfall-rate thresholds."""

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

INPUT_DIR = Path(PROJECT_ROOT / "Data" / "Processed" / "MSWEP_24H_RESULTS")
OUTPUT_DIR = PROJECT_ROOT / "Results" / "Extended_figures" / "EX_FIG2"

OUTPUT_PNG = OUTPUT_DIR / "EX_FIG2B.png"
OUTPUT_STATISTICS_CSV = OUTPUT_DIR / "EX_FIG2B_statistics.csv"
OUTPUT_SAMPLES_CSV = OUTPUT_DIR / "EX_FIG2B_samples.csv"

THRESHOLDS = [10, 20, 30, 40]
GROUP_LABELS = [str(value) for value in THRESHOLDS]

INPUT_FILES = {
    threshold: INPUT_DIR
    / (
        "PRE_DATA_IBT_1982_2024_"
        f"MSWEP_TH{threshold}_24H_SLIDING_ET_CLEANED.csv"
    )
    for threshold in THRESHOLDS
}

# The finalized R30 intensity figure supplies the shared vertical layout.
REFERENCE_R30_CSV = INPUT_FILES[30]


# ============================================================
# 2. Analysis fields and definitions
# ============================================================

WIND_COLUMN = "DIFF_USA_WIND"
GROUP_COLUMN = "RAINFALL_RATE_THRESHOLD"
TARGET_COLUMN = "DIFF_MEAN_RADIAL_DISTANCE"
SOURCE_TARGET_COLUMN = {
    threshold: f"DIFF_MSWEP_DIST_{threshold}" for threshold in THRESHOLDS
}

RW_LIMIT = -30.0
POSITIVE_RATE_COLUMN = "P(DeltaR > 0)"

RW_BLUE = "#56A1BB"
GROUP_PALETTE = {label: RW_BLUE for label in GROUP_LABELS}


# ============================================================
# 3. Figure configuration matching the established RUN7 style
# ============================================================

DPI = 300
FIGURE_WIDTH = 17.0
FIGURE_HEIGHT = 9.5

# The larger left margin prevents the longer y-axis title from being clipped.
LEFT_MARGIN_INCHES = 2.55
RIGHT_MARGIN_INCHES = 0.35
TOP_MARGIN_INCHES = 0.25
BOTTOM_MARGIN_INCHES = 1.55

REFERENCE_GROUP_COUNT = 7
CURRENT_GROUP_COUNT = len(GROUP_LABELS)

FONT_SIZE = 34
ANNOTATION_FONT_SIZE = 28
FONT_COLOR = "#000000"

# Scaling the relative widths by 4/7 preserves the physical violin and box
# widths used in the seven-group intensity figure.
WIDTH_SCALE = CURRENT_GROUP_COUNT / REFERENCE_GROUP_COUNT
VIOLIN_WIDTH = 0.42 * WIDTH_SCALE
BOX_WIDTH = 0.07 * WIDTH_SCALE

VIOLIN_LINE_WIDTH = 2.6
BOX_LINE_WIDTH = 2.4
MEDIAN_LINE_WIDTH = 2.6
FRAME_LINE_WIDTH = 2.8
ZERO_LINE_WIDTH = 1.6

Y_TICK_INTERVAL = 100.0
Y_AXIS_MAX = 300.0
Y_LOWER_SHIFT_KM = 100.0


# ============================================================
# 4. Read and validate the four independent RW samples
# ============================================================

def finite_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric and replace non-finite values with NaN."""
    return pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def read_one_threshold(threshold: int) -> pd.DataFrame:
    """Read one cleaned table and retain valid RW observations only."""
    path = INPUT_FILES[threshold]
    target = SOURCE_TARGET_COLUMN[threshold]

    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist:\n{path}")

    header = pd.read_csv(path, nrows=0)
    missing = sorted({WIND_COLUMN, target} - set(header.columns))
    if missing:
        raise KeyError(
            f"{path.name} is missing required columns: {', '.join(missing)}"
        )

    data = pd.read_csv(
        path,
        usecols=[WIND_COLUMN, target],
        low_memory=False,
    )
    data[WIND_COLUMN] = finite_numeric(data[WIND_COLUMN])
    data[target] = finite_numeric(data[target])

    source_rows = len(data)
    valid = data[[WIND_COLUMN, target]].notna().all(axis=1)
    rw = data.loc[valid & data[WIND_COLUMN].le(RW_LIMIT)].copy()
    if rw.empty:
        raise ValueError(f"No valid RW observations remain for threshold {threshold}.")

    result = pd.DataFrame(
        {
            GROUP_COLUMN: str(threshold),
            TARGET_COLUMN: rw[target].to_numpy(dtype=float),
            WIND_COLUMN: rw[WIND_COLUMN].to_numpy(dtype=float),
        }
    )

    print(f"Threshold {threshold} mm 3h^-1")
    print(f"  Input: {path}")
    print(f"  Source rows: {source_rows:,}")
    print(f"  Valid RW rows: {len(result):,}")
    return result


def build_all_samples() -> pd.DataFrame:
    """Combine independently filtered RW samples from all four thresholds."""
    parts = [read_one_threshold(threshold) for threshold in THRESHOLDS]
    samples = pd.concat(parts, ignore_index=True)
    samples[GROUP_COLUMN] = pd.Categorical(
        samples[GROUP_COLUMN], categories=GROUP_LABELS, ordered=True
    )
    return samples


# ============================================================
# 5. Full-sample statistics and trimmed density samples
# ============================================================

def calculate_statistics(samples: pd.DataFrame) -> pd.DataFrame:
    """Calculate full-sample statistics for each rainfall threshold."""
    records = []
    for threshold, label in zip(THRESHOLDS, GROUP_LABELS):
        values = samples.loc[
            samples[GROUP_COLUMN] == label, TARGET_COLUMN
        ].astype(float)
        if values.empty:
            raise ValueError(f"Threshold group {label} is empty.")

        quantiles = values.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        records.append(
            {
                "RAINFALL_RATE_THRESHOLD_MM_3H": threshold,
                "N": len(values),
                "MEAN_CHANGE_KM": values.mean(),
                "Q05_KM": quantiles.loc[0.05],
                "Q25_KM": quantiles.loc[0.25],
                "MEDIAN_CHANGE_KM": quantiles.loc[0.50],
                "Q75_KM": quantiles.loc[0.75],
                "Q95_KM": quantiles.loc[0.95],
                POSITIVE_RATE_COLUMN: values.gt(0).mean() * 100.0,
            }
        )
    return pd.DataFrame(records)


def build_trimmed_density_samples(
    samples: pd.DataFrame,
    statistics: pd.DataFrame,
) -> pd.DataFrame:
    """Use each group's 5th-95th percentiles only for violin density."""
    parts = []
    for label in GROUP_LABELS:
        threshold = int(label)
        row = statistics.loc[
            statistics["RAINFALL_RATE_THRESHOLD_MM_3H"] == threshold
        ].iloc[0]
        group = samples.loc[samples[GROUP_COLUMN] == label]
        parts.append(
            group.loc[
                group[TARGET_COLUMN].between(
                    row["Q05_KM"], row["Q95_KM"], inclusive="both"
                )
            ].copy()
        )

    trimmed = pd.concat(parts, ignore_index=True)
    trimmed[GROUP_COLUMN] = pd.Categorical(
        trimmed[GROUP_COLUMN], categories=GROUP_LABELS, ordered=True
    )
    return trimmed


# ============================================================
# 6. Reproduce the vertical layout of the finalized R30 figure
# ============================================================

def calculate_reference_y_layout() -> tuple[float, np.ndarray, float]:
    """Calculate the established y limit, ticks, and N/P annotation level."""
    required = [WIND_COLUMN, "DIFF_MSWEP_DIST_30"]
    if not REFERENCE_R30_CSV.exists():
        raise FileNotFoundError(
            f"R30 reference CSV does not exist:\n{REFERENCE_R30_CSV}"
        )

    reference = pd.read_csv(
        REFERENCE_R30_CSV,
        usecols=required,
        low_memory=False,
    )
    for column in required:
        reference[column] = finite_numeric(reference[column])
    reference = reference.dropna(subset=required)

    wind = reference[WIND_COLUMN]
    labels = [
        "[0,5th]",
        "(5th,25th]",
        "(25th,50th)",
        "50th",
        "(50th,75th)",
        "[75th,95th)",
        "[95th,100th]",
    ]
    conditions = [
        wind <= -30,
        (wind > -30) & (wind <= -8),
        (wind > -8) & (wind < 0),
        wind == 0,
        (wind > 0) & (wind < 12),
        (wind >= 12) & (wind < 30),
        wind >= 30,
    ]
    reference["REFERENCE_GROUP"] = np.select(
        conditions, labels, default=None
    )
    if reference["REFERENCE_GROUP"].isna().any():
        raise RuntimeError("Some R30 reference rows were not grouped.")

    q05_values = []
    q95_values = []
    for label in labels:
        values = reference.loc[
            reference["REFERENCE_GROUP"] == label,
            "DIFF_MSWEP_DIST_30",
        ]
        if values.empty:
            raise ValueError(f"R30 reference group {label} is empty.")
        q05_values.append(values.quantile(0.05))
        q95_values.append(values.quantile(0.95))

    q05 = min(q05_values)
    q95 = max(q95_values)
    span = q95 - q05
    if not np.isfinite(span) or np.isclose(span, 0.0):
        raise ValueError("Cannot calculate a finite R30 reference layout.")

    # Move the lower limit and the N/P annotation upward together by 100 km.
    y_min = q05 - span * 0.27 + Y_LOWER_SHIFT_KM
    annotation_y = q05 - span * 0.19 + Y_LOWER_SHIFT_KM
    tick_min = np.ceil(y_min / Y_TICK_INTERVAL) * Y_TICK_INTERVAL
    y_ticks = np.arange(
        tick_min,
        Y_AXIS_MAX + Y_TICK_INTERVAL * 0.5,
        Y_TICK_INTERVAL,
    )
    return y_min, y_ticks, annotation_y


# ============================================================
# 7. Draw the four-violin threshold-sensitivity figure
# ============================================================

def plot_violin(
    samples: pd.DataFrame,
    trimmed: pd.DataFrame,
    statistics: pd.DataFrame,
) -> None:
    """Draw four RW violins in the established scientific-figure style."""
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

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=DPI)
    fig.subplots_adjust(
        left=LEFT_MARGIN_INCHES / FIGURE_WIDTH,
        right=1.0 - RIGHT_MARGIN_INCHES / FIGURE_WIDTH,
        bottom=BOTTOM_MARGIN_INCHES / FIGURE_HEIGHT,
        top=1.0 - TOP_MARGIN_INCHES / FIGURE_HEIGHT,
    )

    sns.violinplot(
        data=trimmed,
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
        data=samples,
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

    y_min, y_ticks, annotation_y = calculate_reference_y_layout()
    ax.set_ylim(y_min, Y_AXIS_MAX)
    ax.set_yticks(y_ticks)

    for index, row in statistics.reset_index(drop=True).iterrows():
        n_math = f"{int(row['N']):,}".replace(",", "{,}")
        p_math = f"{row[POSITIVE_RATE_COLUMN]:.1f}"
        ax.text(
            index,
            annotation_y,
            rf"$\mathit{{N}} = \mathrm{{{n_math}}}$"
            + "\n"
            + rf"$\mathit{{P}} = \mathrm{{{p_math}\%}}$",
            ha="center",
            va="center",
            fontsize=ANNOTATION_FONT_SIZE,
            fontweight="normal",
            color=FONT_COLOR,
            linespacing=1.55,
        )

    ax.set_title("")
    ax.set_xlabel(
        r"Rainfall rate threshold (mm 3 h$^{-1}$)",
        fontsize=FONT_SIZE,
        fontweight="normal",
        color=FONT_COLOR,
        labelpad=18,
    )
    ax.set_ylabel(
        "Change in radial distance of\nTC heavy rainfall (km)",
        fontsize=FONT_SIZE,
        fontweight="normal",
        color=FONT_COLOR,
        labelpad=18,
        linespacing=1.30,
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
    # Remove the gridline nearest the N/P text so it does not cross the labels.
    annotation_tick = y_ticks[np.argmin(np.abs(y_ticks - annotation_y))]
    for tick_value, gridline in zip(ax.get_yticks(), ax.get_ygridlines()):
        if np.isclose(tick_value, annotation_tick):
            gridline.set_visible(False)
    ax.set_axisbelow(True)

    fig.savefig(OUTPUT_PNG, dpi=DPI, facecolor="white")
    plt.close(fig)


# ============================================================
# 8. Main program
# ============================================================

def main() -> None:
    """Run the RW rainfall-threshold sensitivity analysis and plotting."""
    print("=" * 88)
    print("RW rainfall-rate threshold sensitivity analysis")
    print("=" * 88)
    print(f"RW definition: {WIND_COLUMN} <= {RW_LIMIT}")
    print("Each threshold is filtered independently.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = build_all_samples()
    statistics = calculate_statistics(samples)
    trimmed = build_trimmed_density_samples(samples, statistics)

    plot_violin(samples, trimmed, statistics)

    print(f"\nFigure: {OUTPUT_PNG}")
    print("Processing completed successfully.")


if __name__ == "__main__":
    main()
