from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ============================================================
# Global font configuration
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
    }
)


# ============================================================
# 1. Paths and fields
# ============================================================

MSWEP_THRESHOLD = 30
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MSWEP_PROCESSED_ROOT = PROJECT_ROOT / "Data" / "Processed" / "MSWEP"
INPUT_CSV = MSWEP_PROCESSED_ROOT / "Windows" / "24H" / (
    "PRE_DATA_IBT_1982_2024_"
    f"MSWEP_TH{MSWEP_THRESHOLD}_24H_SLIDING_ET_CLEANED.csv"
)
REFERENCE_FIG1B_CSV = MSWEP_PROCESSED_ROOT / "Windows" / "36H" / (
    "PRE_DATA_IBT_1982_2024_"
    "MSWEP_TH30_36H_SLIDING_ET_CLEANED.csv"
)
MAIN_OUTPUT_DIR = PROJECT_ROOT / "Results" / "Main_figures" / "FIG1"
INTENSITY_PNG = MAIN_OUTPUT_DIR / "FIG1A.png"
SPATIAL_PNG = MAIN_OUTPUT_DIR / "FIG1B.png"

WIND = "DIFF_USA_WIND"
TARGET = f"DIFF_MSWEP_DIST_{MSWEP_THRESHOLD}"
LAT = "USA_LAT_S"
BASIN = "BASIN_S"
INTENSITY_GROUP = "INTENSITY_GROUP"
SPATIAL_GROUP = "RW_SPATIAL_GROUP"
POSITIVE_RATE = f"P(DeltaDIST{MSWEP_THRESHOLD} > 0)"


# ============================================================
# 2. Groups and colors
# ============================================================

INTENSITY_LABELS = [
    "[0,5th]", "(5th,25th]", "(25th,50th)", "50th",
    "(50th,75th)", "[75th,95th)", "[95th,100th]",
]
INTENSITY_RANGES = [
    "DIFF_USA_WIND <= -30", "-30 < DIFF_USA_WIND <= -8",
    "-8 < DIFF_USA_WIND < 0", "DIFF_USA_WIND = 0",
    "0 < DIFF_USA_WIND < 12", "12 <= DIFF_USA_WIND < 30",
    "DIFF_USA_WIND >= 30",
]
# Use the original IBTrACS BASIN_S codes for both filtering and display.
BASIN_DISPLAY_TO_SOURCE = {
    "WP": "WP",
    "EP": "EP",
    "NA": "NA",
    "SI": "SI",
    "SP": "SP",
    "NI": "NI",
}
SPATIAL_LABELS = ["NH", "SH", *BASIN_DISPLAY_TO_SOURCE.keys()]
EXPECTED_BASINS = set(BASIN_DISPLAY_TO_SOURCE.values())

RW_BLUE = "#56A1BB"
FONT_COLOR = "#000000"
INTENSITY_COLORS = [
    RW_BLUE, "#6FBFD7", "#A3DCEC", "#E2E2E2",
    "#FBC8DB", "#ECAFC6", "#DEAEC3",
]
INTENSITY_PALETTE = dict(zip(INTENSITY_LABELS, INTENSITY_COLORS))
SPATIAL_PALETTE = {label: RW_BLUE for label in SPATIAL_LABELS}


# ============================================================
# 3. Figure settings
# ============================================================

DPI = 300
# Preserve the original axes height while enlarging only the outer canvas
# margins so axis titles and tick labels are not clipped.
AXES_HEIGHT_IN = 10.25 - 2.00 - 0.25
LEFT_IN = 2.95
RIGHT_IN = 0.35
TOP_IN = 0.25
BOTTOM_IN = 2.40
HEIGHT = AXES_HEIGHT_IN + BOTTOM_IN + TOP_IN
# Preserve the original physical width allocated to each category. Increasing
# the left margin therefore enlarges the canvas instead of compressing violins.
SLOT_IN = (17.0 - 1.55 - RIGHT_IN) / 7.0
FONT_SIZE = 34
ANNOTATION_SIZE = 28
VIOLIN_WIDTH = 0.42
BOX_WIDTH = 0.07
VIOLIN_LINE_WIDTH = 2.6
BOX_LINE_WIDTH = 2.4
FRAME_LINE_WIDTH = 2.8
Y_TICK_INTERVAL = 100.0


# ============================================================
# 4. Data preparation
# ============================================================

def read_data():
    """Read the fields required by both analyses."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV does not exist:\n{INPUT_CSV}")

    print("=" * 80)
    print(f"Reading CLEANED threshold-{MSWEP_THRESHOLD} data")
    print("=" * 80)
    print(f"Input: {INPUT_CSV}")

    # This option preserves the literal North Atlantic code "NA".
    df = pd.read_csv(
        INPUT_CSV,
        usecols=[WIND, TARGET, LAT, BASIN],
        dtype={BASIN: "string"},
        keep_default_na=False,
        low_memory=False,
    )
    for column in [WIND, TARGET, LAT]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[[WIND, TARGET, LAT]] = df[[WIND, TARGET, LAT]].replace(
        [np.inf, -np.inf], np.nan
    )
    df[BASIN] = df[BASIN].astype("string").str.strip().str.upper()

    invalid = df[[WIND, TARGET]].isna().any(axis=1)
    if invalid.any():
        print(f"Invalid base-analysis rows removed: {int(invalid.sum()):,}")
        df = df.loc[~invalid].copy()
    if df.empty:
        raise ValueError("No valid observations remain.")
    print(f"Valid rows: {len(df):,}")
    return df.reset_index(drop=True)


def make_intensity_data(df):
    """Assign seven mutually exclusive intensity-change groups."""
    result = df.copy()
    wind = result[WIND]
    conditions = [
        wind <= -30,
        (wind > -30) & (wind <= -8),
        (wind > -8) & (wind < 0),
        wind == 0,
        (wind > 0) & (wind < 12),
        (wind >= 12) & (wind < 30),
        wind >= 30,
    ]
    result[INTENSITY_GROUP] = np.select(
        conditions, INTENSITY_LABELS, default=None
    )
    if result[INTENSITY_GROUP].isna().any():
        raise RuntimeError("Some wind-change values were not assigned.")
    result[INTENSITY_GROUP] = pd.Categorical(
        result[INTENSITY_GROUP], categories=INTENSITY_LABELS, ordered=True
    )
    return result


def make_rw_spatial_data(df):
    """Create two hemisphere groups and six basin groups for RW only."""
    rw = df.loc[df[WIND] <= -30].copy()
    if rw[LAT].isna().any():
        raise ValueError("RW rows contain missing USA_LAT_S values.")
    if rw[LAT].eq(0).any():
        raise ValueError("RW rows contain USA_LAT_S == 0.")

    # Diagnostics established that blanks in this source are the original NA code.
    # Only empty strings are restored; unrelated missing values are not filled.
    rw[BASIN] = rw[BASIN].replace("", "NA")
    unexpected = sorted(set(rw[BASIN].dropna()) - EXPECTED_BASINS)
    if unexpected:
        raise ValueError(f"Unexpected basin codes in RW rows: {unexpected}")

    chunks = []
    for label, mask in [("NH", rw[LAT] > 0), ("SH", rw[LAT] < 0)]:
        part = rw.loc[mask].copy()
        part[SPATIAL_GROUP] = label
        chunks.append(part)
    for display_label, source_code in BASIN_DISPLAY_TO_SOURCE.items():
        part = rw.loc[rw[BASIN] == source_code].copy()
        part[SPATIAL_GROUP] = display_label
        chunks.append(part)

    hemisphere_n = sum(len(part) for part in chunks[:2])
    basin_n = sum(len(part) for part in chunks[2:])
    if hemisphere_n != len(rw) or basin_n != len(rw):
        raise RuntimeError(
            f"Spatial coverage failed: RW={len(rw):,}, "
            f"hemisphere={hemisphere_n:,}, basin={basin_n:,}."
        )

    result = pd.concat(chunks, ignore_index=True)
    result[SPATIAL_GROUP] = pd.Categorical(
        result[SPATIAL_GROUP], categories=SPATIAL_LABELS, ordered=True
    )
    print(f"\nRW rows: {len(rw):,}")
    print("RW spatial-group counts:")
    counts = result[SPATIAL_GROUP].value_counts(sort=False)
    for label in SPATIAL_LABELS:
        print(f"  {label:<3} {int(counts.get(label, 0)):>10,}")
    return result


# ============================================================
# 5. Statistics and shared axis
# ============================================================

def group_statistics(df, group_column, labels):
    """Calculate statistics from all valid observations."""
    records = []
    for label in labels:
        values = df.loc[df[group_column] == label, TARGET].astype(float)
        if values.empty:
            raise ValueError(f"Group {label} contains no observations.")
        q = values.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        records.append({
            "Group": label,
            "N": len(values),
            "Mean": values.mean(),
            "5th": q.loc[0.05],
            "25th": q.loc[0.25],
            "Median": q.loc[0.50],
            "75th": q.loc[0.75],
            "95th": q.loc[0.95],
            POSITIVE_RATE: values.gt(0).mean() * 100.0,
        })
    return pd.DataFrame(records)


def trimmed_violin_data(df, statistics, group_column, labels):
    """Trim density rendering to each group's 5th-95th percentile range."""
    parts = []
    for label in labels:
        row = statistics.loc[statistics["Group"] == label].iloc[0]
        group = df.loc[df[group_column] == label]
        parts.append(group.loc[group[TARGET].between(row["5th"], row["95th"])] .copy())
    result = pd.concat(parts, ignore_index=True)
    result[group_column] = pd.Categorical(
        result[group_column], categories=labels, ordered=True
    )
    return result


def reference_y_axis():
    """Reproduce EX_FIG1B's original y-axis and annotation height."""
    if not REFERENCE_FIG1B_CSV.exists():
        raise FileNotFoundError(
            f"FIG1B reference CSV does not exist:\n{REFERENCE_FIG1B_CSV}"
        )

    reference = pd.read_csv(
        REFERENCE_FIG1B_CSV,
        usecols=[WIND, TARGET],
        low_memory=False,
    )
    reference[WIND] = pd.to_numeric(reference[WIND], errors="coerce")
    reference[TARGET] = pd.to_numeric(reference[TARGET], errors="coerce")
    reference[[WIND, TARGET]] = reference[[WIND, TARGET]].replace(
        [np.inf, -np.inf], np.nan
    )
    reference = reference.dropna(subset=[WIND, TARGET]).copy()
    if reference.empty:
        raise ValueError("No valid FIG1B reference rows remain.")

    # Original EX_FIG1B fixed 36-hour intensity boundaries.
    wind = reference[WIND]
    conditions = [
        wind <= -40,
        (wind > -40) & (wind <= -10),
        (wind > -10) & (wind < 5),
        wind == 5,
        (wind > 5) & (wind < 17),
        (wind >= 17) & (wind < 45),
        wind >= 45,
    ]
    reference[INTENSITY_GROUP] = np.select(
        conditions,
        INTENSITY_LABELS,
        default=None,
    )

    q5_values = []
    q95_values = []
    for label in INTENSITY_LABELS:
        values = reference.loc[
            reference[INTENSITY_GROUP] == label,
            TARGET,
        ].astype(float)
        if values.empty:
            raise ValueError(f"FIG1B reference group is empty: {label}")
        q5_values.append(values.quantile(0.05))
        q95_values.append(values.quantile(0.95))

    q5 = float(np.min(q5_values))
    q95 = float(np.max(q95_values))
    span = q95 - q5
    if not np.isfinite(span) or np.isclose(span, 0):
        raise ValueError("Cannot calculate a finite shared y-axis.")

    # Use FIG1B's original label position and lower-limit calculation.
    annotation_y = q5 - span * 0.19
    raw_min = q5 - span * 0.27
    raw_max = 300.0
    tick_min = np.ceil(raw_min / Y_TICK_INTERVAL) * Y_TICK_INTERVAL
    tick_max = np.floor(raw_max / Y_TICK_INTERVAL) * Y_TICK_INTERVAL
    ticks = np.arange(
        tick_min, tick_max + Y_TICK_INTERVAL * 0.5, Y_TICK_INTERVAL
    )
    print("\nEX_FIG1B reference layout applied to main FIG1A and FIG1B:")
    print(f"  lower limit: {raw_min:.6f}")
    print(f"  upper limit: {raw_max:.6f}")
    print(f"  ticks: {ticks.tolist()}")
    print(f"  annotation y: {annotation_y:.6f}")

    return raw_min, raw_max, ticks, annotation_y


# ============================================================
# 6. Plotting
# ============================================================

def plot_violin(
    full_df, trimmed_df, statistics, group_column, labels,
    palette, output_path, y_axis,
    annotation_size=ANNOTATION_SIZE,
    annotation_linespacing=1.0,
    x_label_rotation=30,
    show_y_grid=True,
):
    """Draw a violin plot with a fixed physical category-slot width."""
    sns.set_theme(style="whitegrid" if show_y_grid else "white", rc={
        "font.family": "Arial", "font.sans-serif": ["Arial"],
        "font.cursive": ["Arial"],
        "grid.linestyle": "--", "grid.color": "#9E9E9E",
        "grid.alpha": 0.55, "grid.linewidth": 1.0,
        "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
        "text.color": FONT_COLOR, "axes.labelcolor": FONT_COLOR,
        "xtick.color": FONT_COLOR, "ytick.color": FONT_COLOR,
    })

    width = LEFT_IN + RIGHT_IN + len(labels) * SLOT_IN
    fig, ax = plt.subplots(figsize=(width, HEIGHT), dpi=DPI)
    fig.subplots_adjust(
        left=LEFT_IN / width,
        right=1 - RIGHT_IN / width,
        bottom=BOTTOM_IN / HEIGHT,
        top=1 - TOP_IN / HEIGHT,
    )

    sns.violinplot(
        data=trimmed_df, x=group_column, y=TARGET, hue=group_column,
        order=labels, hue_order=labels, palette=palette, legend=False,
        inner=None, cut=0, width=VIOLIN_WIDTH,
        linewidth=VIOLIN_LINE_WIDTH, density_norm="width", ax=ax,
    )
    sns.boxplot(
        data=full_df, x=group_column, y=TARGET, order=labels,
        width=BOX_WIDTH, showfliers=False, showcaps=False, whis=0,
        boxprops={
            "facecolor": "#FFFFFF", "edgecolor": "#000000",
            "linewidth": BOX_LINE_WIDTH, "alpha": 0.90, "zorder": 4,
        },
        medianprops={"color": "#000000", "linewidth": 2.6, "zorder": 5},
        whiskerprops={"linewidth": 0}, ax=ax,
    )
    ax.axhline(0, color="#000000", linestyle="--", linewidth=1.6, alpha=0.90)

    y_min, y_max, ticks, annotation_y = y_axis
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(ticks)

    for index, row in statistics.reset_index(drop=True).iterrows():
        n_text = f"{int(row['N']):,}".replace(",", "{,}")
        p_text = f"{row[POSITIVE_RATE]:.1f}"
        ax.text(
            index, annotation_y,
            rf"$\mathit{{N}} = \mathrm{{{n_text}}}$" + "\n" +
            rf"$\mathit{{P}} = \mathrm{{{p_text}\%}}$",
            ha="center", va="center", fontsize=annotation_size,
            fontweight="normal", color=FONT_COLOR,
            linespacing=annotation_linespacing,
        )

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel(
        "Change in radial distance of\nTC heavy rainfall (km)",
        fontsize=FONT_SIZE, fontweight="normal", labelpad=16,
        color=FONT_COLOR, linespacing=1.30,
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
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("normal")
    for label in ax.get_xticklabels():
        label.set_rotation(x_label_rotation)
        label.set_horizontalalignment("center")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(FRAME_LINE_WIDTH)
    if show_y_grid:
        ax.grid(axis="y", visible=True)
        ax.grid(axis="x", visible=False)
        # Keep the -300 tick label but remove its gridline so it cannot cross
        # the lower N/P annotations.
        for tick_value, gridline in zip(ax.get_yticks(), ax.get_ygridlines()):
            if np.isclose(tick_value, -300.0):
                gridline.set_visible(False)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)

    # Avoid bbox_inches="tight" so both figures retain identical physical slots.
    # Preserve the complete original canvas and margins. Do not use
    # bbox_inches="tight", because it changes the exported image bounds.
    fig.savefig(
        output_path,
        dpi=DPI,
        facecolor="white",
        bbox_inches=None,
    )
    plt.close(fig)
    print(f"\nPNG figure: {output_path}")


# ============================================================
# 7. Main program
# ============================================================

def main():
    MAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = read_data()
    intensity = make_intensity_data(source)
    spatial = make_rw_spatial_data(source)

    intensity_stats = group_statistics(
        intensity, INTENSITY_GROUP, INTENSITY_LABELS
    )
    spatial_stats = group_statistics(spatial, SPATIAL_GROUP, SPATIAL_LABELS)
    intensity_trimmed = trimmed_violin_data(
        intensity, intensity_stats, INTENSITY_GROUP, INTENSITY_LABELS
    )
    spatial_trimmed = trimmed_violin_data(
        spatial, spatial_stats, SPATIAL_GROUP, SPATIAL_LABELS
    )
    # Both main panels use the original EX_FIG1B vertical layout, including
    # limits, ticks, and the N/P annotation height.
    y_axis = reference_y_axis()

    plot_violin(
        intensity, intensity_trimmed, intensity_stats,
        INTENSITY_GROUP, INTENSITY_LABELS, INTENSITY_PALETTE,
        INTENSITY_PNG, y_axis,
        annotation_size=ANNOTATION_SIZE,
        annotation_linespacing=1.55,
        x_label_rotation=30,
        show_y_grid=True,
    )
    plot_violin(
        spatial, spatial_trimmed, spatial_stats,
        SPATIAL_GROUP, SPATIAL_LABELS, SPATIAL_PALETTE,
        SPATIAL_PNG, y_axis,
        annotation_size=ANNOTATION_SIZE,
        annotation_linespacing=1.70,
        x_label_rotation=0,
        show_y_grid=True,
    )
    print("\n" + "=" * 80)
    print("Processing completed successfully")
    print("=" * 80)
    print(f"Intensity violin PNG: {INTENSITY_PNG}")
    print(f"RW spatial violin PNG: {SPATIAL_PNG}")


if __name__ == "__main__":
    main()
