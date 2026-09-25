"""Compare 24-hour RW DIST30 changes across precipitation products."""

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

MSWEP_24H_CLEANED = Path(PROJECT_ROOT / "Data" / "Processed" / "MSWEP_24H_RESULTS") / (
    "PRE_DATA_IBT_1982_2024_MSWEP_TH30_24H_SLIDING_ET_CLEANED.csv"
)

CMORPH_SOURCE = Path(PROJECT_ROOT / "Data" / "Processed" / "CMORPH_RESULTS") / (
    "PRE_DATA_IBT_1998_2024_CMORPH_THRESHOLD_30.csv"
)

IMERG_SOURCE = Path(PROJECT_ROOT / "Data" / "Processed" / "IMERG_RESULTS") / (
    "PRE_DATA_IBT_1998_2024_METRICS_IMERG_THRESHOLD_30.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "Results" / "Extended_figures" / "EX_FIG2"

OUTPUT_PNG = OUTPUT_DIR / "EX_FIG2A.png"
OUTPUT_STATISTICS_CSV = OUTPUT_DIR / "EX_FIG2A_statistics.csv"
OUTPUT_SAMPLES_CSV = OUTPUT_DIR / "EX_FIG2A_samples.csv"


# ============================================================
# 2. Analysis configuration
# ============================================================

WINDOW_HOURS = 24
RW_LIMIT = -30.0
PERIOD_START = pd.Timestamp("2001-01-01 00:00:00")
PERIOD_END_EXCLUSIVE = pd.Timestamp("2025-01-01 00:00:00")
EARLY_PERIOD_START = pd.Timestamp("1982-01-01 00:00:00")
EARLY_PERIOD_END_EXCLUSIVE = pd.Timestamp("2001-01-01 00:00:00")

GROUP_COLUMN = "PRODUCT_GROUP"
TARGET_COLUMN = "DIFF_DIST30"
POSITIVE_RATE_COLUMN = "P(DeltaDIST30 > 0)"

GROUP_LABELS = [
    "MSWEP\n1982-2024",
    "MSWEP\n1982-2000",
    "MSWEP\n2001-2024",
    "IMERG\n2001-2024",
    "CMORPH\n2001-2024",
]

RW_BLUE = "#56A1BB"
GROUP_PALETTE = {label: RW_BLUE for label in GROUP_LABELS}


# ============================================================
# 3. Figure configuration copied from the finalized RUN7 plot
# ============================================================

DPI = 300
FIGURE_HEIGHT = 9.5
LEFT_MARGIN_INCHES = 2.55
RIGHT_MARGIN_INCHES = 0.35
TOP_MARGIN_INCHES = 0.25
BOTTOM_MARGIN_INCHES = 1.25

REFERENCE_GROUP_COUNT = 7
CURRENT_GROUP_COUNT = 5
REFERENCE_SLOT_WIDTH = (
    17.0 - 1.55 - RIGHT_MARGIN_INCHES
) / REFERENCE_GROUP_COUNT
FIGURE_WIDTH = (
    LEFT_MARGIN_INCHES
    + RIGHT_MARGIN_INCHES
    + REFERENCE_GROUP_COUNT * REFERENCE_SLOT_WIDTH
)

FONT_SIZE = 34
ANNOTATION_FONT_SIZE = 28
FONT_COLOR = "#000000"

# Scale relative Seaborn widths so physical violin widths match RUN7.
WIDTH_SCALE = CURRENT_GROUP_COUNT / REFERENCE_GROUP_COUNT
VIOLIN_WIDTH = 0.42 * WIDTH_SCALE
BOX_WIDTH = 0.07 * WIDTH_SCALE

VIOLIN_LINE_WIDTH = 2.6
BOX_LINE_WIDTH = 2.4
MEDIAN_LINE_WIDTH = 2.6
FRAME_LINE_WIDTH = 2.8
ZERO_LINE_WIDTH = 1.6
Y_TICK_INTERVAL = 100.0
Y_LOWER_SHIFT_KM = 100.0


# ============================================================
# 4. General validation helpers
# ============================================================

def require_file(path: Path) -> None:
    """Raise a clear error when an input file is unavailable."""
    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist:\n{path}")


def numeric_series(series: pd.Series) -> pd.Series:
    """Convert a series to finite numeric values with invalid values as NaN."""
    return pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


# ============================================================
# 5. Read the two MSWEP groups from the existing 24-hour table
# ============================================================

def read_mswep_groups() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return full-period, 1982-2000, and 2001-2024 MSWEP RW samples."""
    require_file(MSWEP_24H_CLEANED)
    required = [
        "SID",
        "ISO_TIME_S",
        "ISO_TIME_E",
        "DIFF_USA_WIND",
        "DIFF_MSWEP_DIST_30",
    ]
    df = pd.read_csv(
        MSWEP_24H_CLEANED,
        usecols=required,
        dtype={"SID": "string"},
        low_memory=False,
    )
    df["ISO_TIME_S"] = pd.to_datetime(df["ISO_TIME_S"], errors="coerce")
    df["ISO_TIME_E"] = pd.to_datetime(df["ISO_TIME_E"], errors="coerce")
    df["DIFF_USA_WIND"] = numeric_series(df["DIFF_USA_WIND"])
    df["DIFF_MSWEP_DIST_30"] = numeric_series(df["DIFF_MSWEP_DIST_30"])

    valid = df[
        ["SID", "ISO_TIME_S", "ISO_TIME_E", "DIFF_USA_WIND", "DIFF_MSWEP_DIST_30"]
    ].notna().all(axis=1)
    df = df.loc[valid & df["DIFF_USA_WIND"].le(RW_LIMIT)].copy()

    interval_hours = (
        (df["ISO_TIME_E"] - df["ISO_TIME_S"]).dt.total_seconds() / 3600.0
    )
    if not np.isclose(interval_hours, WINDOW_HOURS).all():
        raise ValueError("The MSWEP table contains non-24-hour RW rows.")

    full = df[["SID", "ISO_TIME_S", "ISO_TIME_E"]].copy()
    full[TARGET_COLUMN] = df["DIFF_MSWEP_DIST_30"].to_numpy()
    full[GROUP_COLUMN] = GROUP_LABELS[0]

    early_mask = (
        df["ISO_TIME_S"].ge(EARLY_PERIOD_START)
        & df["ISO_TIME_E"].lt(EARLY_PERIOD_END_EXCLUSIVE)
    )
    early = df.loc[early_mask, ["SID", "ISO_TIME_S", "ISO_TIME_E"]].copy()
    early[TARGET_COLUMN] = df.loc[
        early_mask, "DIFF_MSWEP_DIST_30"
    ].to_numpy()
    early[GROUP_COLUMN] = GROUP_LABELS[1]

    period_mask = (
        df["ISO_TIME_S"].ge(PERIOD_START)
        & df["ISO_TIME_E"].lt(PERIOD_END_EXCLUSIVE)
    )
    period = df.loc[period_mask, ["SID", "ISO_TIME_S", "ISO_TIME_E"]].copy()
    period[TARGET_COLUMN] = df.loc[
        period_mask, "DIFF_MSWEP_DIST_30"
    ].to_numpy()
    period[GROUP_COLUMN] = GROUP_LABELS[2]
    return full, early, period


# ============================================================
# 6. Build independent 24-hour samples for IMERG and CMORPH
# ============================================================

def build_product_group(
    source_path: Path,
    distance_column: str,
    group_label: str,
) -> pd.DataFrame:
    """Build exact 24-hour RW samples for one precipitation product."""
    require_file(source_path)
    required = ["SID", "ISO_TIME", "USA_WIND", distance_column]
    header = pd.read_csv(source_path, nrows=0)
    missing = sorted(set(required) - set(header.columns))
    if missing:
        raise KeyError(
            f"{source_path.name} is missing columns: {', '.join(missing)}"
        )

    source = pd.read_csv(
        source_path,
        usecols=required,
        dtype={"SID": "string"},
        low_memory=False,
    )
    source["ISO_TIME"] = pd.to_datetime(source["ISO_TIME"], errors="coerce")
    source["USA_WIND"] = numeric_series(source["USA_WIND"])
    source[distance_column] = numeric_series(source[distance_column])

    if source[["SID", "ISO_TIME"]].isna().any(axis=None):
        raise ValueError(f"{source_path.name} contains invalid SID or ISO_TIME.")
    if source.duplicated(["SID", "ISO_TIME"]).any():
        raise ValueError(f"{source_path.name} contains duplicate SID-time keys.")

    source = source.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
    start = source.rename(
        columns={
            "ISO_TIME": "ISO_TIME_S",
            "USA_WIND": "USA_WIND_S",
            distance_column: "DIST30_S",
        }
    )
    start["TARGET_TIME_E"] = start["ISO_TIME_S"] + pd.Timedelta(
        hours=WINDOW_HOURS
    )

    end = source[["SID", "ISO_TIME", "USA_WIND", distance_column]].rename(
        columns={
            "ISO_TIME": "ISO_TIME_E",
            "USA_WIND": "USA_WIND_E",
            distance_column: "DIST30_E",
        }
    )
    pairs = start.merge(
        end,
        left_on=["SID", "TARGET_TIME_E"],
        right_on=["SID", "ISO_TIME_E"],
        how="inner",
        validate="one_to_one",
        sort=False,
    ).drop(columns="TARGET_TIME_E")

    # The 71 non-synoptic source timestamps are retained when they have an
    # endpoint exactly 24 hours later; intermediate observations are irrelevant.
    period_mask = (
        pairs["ISO_TIME_S"].ge(PERIOD_START)
        & pairs["ISO_TIME_E"].lt(PERIOD_END_EXCLUSIVE)
    )
    wind_valid = pairs[["USA_WIND_S", "USA_WIND_E"]].notna().all(axis=1)
    distance_valid = pairs[["DIST30_S", "DIST30_E"]].notna().all(axis=1)
    pairs["DIFF_USA_WIND"] = pairs["USA_WIND_E"] - pairs["USA_WIND_S"]
    rw_mask = wind_valid & pairs["DIFF_USA_WIND"].le(RW_LIMIT)

    selected = pairs.loc[
        period_mask & rw_mask & distance_valid,
        ["SID", "ISO_TIME_S", "ISO_TIME_E", "DIST30_S", "DIST30_E"],
    ].copy()
    selected[TARGET_COLUMN] = selected["DIST30_E"] - selected["DIST30_S"]
    selected[GROUP_COLUMN] = group_label
    return selected[["SID", "ISO_TIME_S", "ISO_TIME_E", TARGET_COLUMN, GROUP_COLUMN]]


# ============================================================
# 7. Assemble independent product samples
# ============================================================

def build_all_samples() -> pd.DataFrame:
    """Build the four independently sampled comparison groups."""
    mswep_full, mswep_early, mswep_period = read_mswep_groups()
    imerg = build_product_group(
        IMERG_SOURCE,
        "IMERG_DIST_30",
        GROUP_LABELS[3],
    )
    cmorph = build_product_group(
        CMORPH_SOURCE,
        "CMORPH_DIST_30",
        GROUP_LABELS[4],
    )

    samples = pd.concat(
        [mswep_full, mswep_early, mswep_period, imerg, cmorph],
        ignore_index=True,
    )
    samples[GROUP_COLUMN] = pd.Categorical(
        samples[GROUP_COLUMN], categories=GROUP_LABELS, ordered=True
    )
    if samples[TARGET_COLUMN].isna().any():
        raise RuntimeError("The final comparison samples contain missing DIST30 changes.")

    print("\nIndependent group counts:")
    counts = samples[GROUP_COLUMN].value_counts(sort=False)
    for label in GROUP_LABELS:
        print(f"  {label.replace(chr(10), ' '):<22} {int(counts[label]):>8,}")
    return samples


# ============================================================
# 8. Statistics, violin trimming, and RUN7 vertical reference
# ============================================================

def calculate_statistics(samples: pd.DataFrame) -> pd.DataFrame:
    """Calculate full-sample distribution statistics for each group."""
    records = []
    for label in GROUP_LABELS:
        values = samples.loc[samples[GROUP_COLUMN] == label, TARGET_COLUMN]
        if values.empty:
            raise ValueError(f"Group {label} contains no observations.")
        q = values.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        records.append(
            {
                "Product group": label.replace("\n", " "),
                "N": len(values),
                "Mean": values.mean(),
                "5th": q.loc[0.05],
                "25th": q.loc[0.25],
                "Median": q.loc[0.50],
                "75th": q.loc[0.75],
                "95th": q.loc[0.95],
                POSITIVE_RATE_COLUMN: values.gt(0).mean() * 100.0,
            }
        )
    return pd.DataFrame(records)


def build_trimmed_data(
    samples: pd.DataFrame,
    statistics: pd.DataFrame,
) -> pd.DataFrame:
    """Use each group's 5th-95th values only for density rendering."""
    parts = []
    for label in GROUP_LABELS:
        row = statistics.loc[
            statistics["Product group"] == label.replace("\n", " ")
        ].iloc[0]
        group = samples.loc[samples[GROUP_COLUMN] == label]
        parts.append(
            group.loc[
                group[TARGET_COLUMN].between(
                    row["5th"], row["95th"], inclusive="both"
                )
            ].copy()
        )
    trimmed = pd.concat(parts, ignore_index=True)
    trimmed[GROUP_COLUMN] = pd.Categorical(
        trimmed[GROUP_COLUMN], categories=GROUP_LABELS, ordered=True
    )
    return trimmed


def calculate_reference_y_layout() -> tuple[float, float, np.ndarray, float]:
    """Reproduce RUN7 y limits, ticks, and N/P annotation height."""
    reference = pd.read_csv(
        MSWEP_24H_CLEANED,
        usecols=["DIFF_USA_WIND", "DIFF_MSWEP_DIST_30"],
        low_memory=False,
    )
    for column in reference.columns:
        reference[column] = numeric_series(reference[column])
    reference = reference.dropna()

    wind = reference["DIFF_USA_WIND"]
    labels = [
        "[0,5th]", "(5th,25th]", "(25th,50th)", "50th",
        "(50th,75th)", "[75th,95th)", "[95th,100th]",
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
    reference["REFERENCE_GROUP"] = np.select(conditions, labels, default=None)

    q5_values = []
    q95_values = []
    for label in labels:
        values = reference.loc[
            reference["REFERENCE_GROUP"] == label,
            "DIFF_MSWEP_DIST_30",
        ]
        q5_values.append(values.quantile(0.05))
        q95_values.append(values.quantile(0.95))

    q5 = min(q5_values)
    q95 = max(q95_values)
    span = q95 - q5
    # Move the lower limit and the N/P annotation upward together by 100 km.
    y_min = q5 - span * 0.27 + Y_LOWER_SHIFT_KM
    y_max = 300.0
    annotation_y = q5 - span * 0.19 + Y_LOWER_SHIFT_KM
    tick_min = np.ceil(y_min / Y_TICK_INTERVAL) * Y_TICK_INTERVAL
    ticks = np.arange(
        tick_min, y_max + Y_TICK_INTERVAL * 0.5, Y_TICK_INTERVAL
    )
    return y_min, y_max, ticks, annotation_y


# ============================================================
# 9. Draw the four-group violin figure
# ============================================================

def plot_violin(
    samples: pd.DataFrame,
    trimmed: pd.DataFrame,
    statistics: pd.DataFrame,
) -> None:
    """Draw four physically matched blue violins without significance tests."""
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

    y_min, y_max, ticks, annotation_y = calculate_reference_y_layout()
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(ticks)

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
    ax.set_xlabel("")
    ax.set_ylabel(
        "Change in radial distance of\nTC heavy rainfall (km)",
        fontsize=FONT_SIZE,
        fontweight="normal",
        color=FONT_COLOR,
        labelpad=16,
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
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(FONT_COLOR)
        spine.set_linewidth(FRAME_LINE_WIDTH)

    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", visible=False)
    # Remove the gridline nearest the N/P text so it does not cross the labels.
    annotation_tick = ticks[np.argmin(np.abs(ticks - annotation_y))]
    for tick_value, gridline in zip(ax.get_yticks(), ax.get_ygridlines()):
        if np.isclose(tick_value, annotation_tick):
            gridline.set_visible(False)
    ax.set_axisbelow(True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"\nViolin PNG: {OUTPUT_PNG}")


# ============================================================
# 10. Main program
# ============================================================

def main() -> None:
    """Run the complete independent-product sensitivity analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = build_all_samples()
    statistics = calculate_statistics(samples)
    trimmed = build_trimmed_data(samples, statistics)

    plot_violin(samples, trimmed, statistics)

    print("Processing completed successfully.")


if __name__ == "__main__":
    main()
