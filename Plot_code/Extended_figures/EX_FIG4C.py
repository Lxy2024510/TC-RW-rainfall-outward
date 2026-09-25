import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter


# ============================================================
# 1. Configuration and paths
# ============================================================

INPUT_CSV = Path(
    str(PROJECT_ROOT / "Data" / "Processed" / "MSWEP_24H_SPEED_RESULTS") + "/" +
    "PRE_DATA_IBT_1982_2024_MSWEP_TH30_"
    "24H_SLIDING_ET_SPEED_CLEANED.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "Results" / "Extended_figures" / "EX_FIG4"

OUTPUT_WITH_SIGNIFICANCE = OUTPUT_DIR / "EX_FIG4C.png"
OUTPUT_WITHOUT_SIGNIFICANCE = OUTPUT_DIR / "EX_FIG4C_WITHOUT_SIGNIFICANCE.png"
PLOT_DATA_CSV = OUTPUT_DIR / "EX_FIG4C_DATA.csv"
SIGNIFICANCE_CSV = OUTPUT_DIR / "EX_FIG4C_SIGNIFICANCE_TEST.csv"

RW_WIND_THRESHOLD = -30.0
VALUE_COLUMN = "DIFF_STORM_SPEED"
WIND_DIFF_COLUMN = "DIFF_USA_WIND"
OUTER_AREA_DIFF_COLUMN = "DIFF_MSWEP_AREA_30_200_500"
DISTANCE_DIFF_COLUMN = "DIFF_MSWEP_DIST_30"
ET_COUNT_COLUMN = "COUNT_ET_24H"

BOOTSTRAP_ITERATIONS = 5000
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_RANDOM_SEED = 20260814
BOOTSTRAP_BATCH_SIZE = 250

PERMUTATION_ITERATIONS = 50000
PERMUTATION_RANDOM_SEED = 20260823
SIGNIFICANCE_LEVEL = 0.05
MARGINAL_SIGNIFICANCE_LEVEL = 0.10

GROUP_ORDER = [
    "ALL_RW_NO_ET",
    "AREA_DECREASE_DISTANCE_DECREASE_NO_ET",
]
GROUP_LABELS = {
    "ALL_RW_NO_ET": "All RW",
    "AREA_DECREASE_DISTANCE_DECREASE_NO_ET": (
        r"$\Delta R30 < 0$"
        "\n"
        r"$\Delta\mathit{Outer\ area} < 0$"
    ),
}
GROUP_GRADIENT_COLORS = {
    "ALL_RW_NO_ET": ["#56A1BB", "#6FBFD7", "#A3DCEC"],
    "AREA_DECREASE_DISTANCE_DECREASE_NO_ET": [
        "#DEAEC3", "#ECAFC6", "#FBC8DB"
    ],
}

BAR_WIDTH = 0.46
FIGSIZE = (7.2, 10.2)
FIGURE_DPI = 600
FONT_SIZE = 28
SIGNIFICANCE_STAR_FONT_SIZE = 36


# ============================================================
# 2. Input validation and sample selection
# ============================================================

def read_and_validate_input():
    """Read the cleaned speed table and validate required variables."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input sample table does not exist: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    pair_columns = ["SID", "ROW_ID_S", "ROW_ID_E"]
    numeric_columns = [
        VALUE_COLUMN,
        WIND_DIFF_COLUMN,
        OUTER_AREA_DIFF_COLUMN,
        DISTANCE_DIFF_COLUMN,
        ET_COUNT_COLUMN,
    ]
    required_columns = pair_columns + numeric_columns
    missing = set(required_columns).difference(df.columns)

    if missing:
        raise KeyError(
            "Input table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    duplicate_mask = df.duplicated(pair_columns, keep=False)
    if duplicate_mask.any():
        examples = df.loc[duplicate_mask, pair_columns].head(20).to_dict("records")
        raise ValueError(f"Duplicate 24-hour pair keys were found: {examples}")

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_mask = ~np.isfinite(
        df[numeric_columns].to_numpy(dtype=float)
    ).all(axis=1)
    if invalid_mask.any():
        examples = (
            df.loc[invalid_mask, pair_columns + numeric_columns]
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            "Input table contains non-finite required values. "
            f"Count={int(invalid_mask.sum())}; examples={examples}"
        )

    rw_df = df.loc[df[WIND_DIFF_COLUMN] <= RW_WIND_THRESHOLD].copy()
    rw_df.reset_index(drop=True, inplace=True)
    if rw_df.empty:
        raise RuntimeError(
            f"No RW samples satisfy {WIND_DIFF_COLUMN} <= {RW_WIND_THRESHOLD}"
        )
    return df, rw_df


def construct_groups(rw_df):
    """Construct the full no-ET RW sample and its nested subset."""
    non_et = rw_df[ET_COUNT_COLUMN].eq(0)
    masks = {
        "ALL_RW_NO_ET": non_et,
        "AREA_DECREASE_DISTANCE_DECREASE_NO_ET": (
            non_et
            & rw_df[OUTER_AREA_DIFF_COLUMN].lt(0)
            & rw_df[DISTANCE_DIFF_COLUMN].lt(0)
        ),
    }
    groups = {}
    sample_tables = {}

    for group_name in GROUP_ORDER:
        group_df = rw_df.loc[masks[group_name]].copy()
        if group_df.empty:
            raise ValueError(f"Sample group is empty: {group_name}")
        groups[group_name] = group_df[VALUE_COLUMN].to_numpy(dtype=float)
        sample_tables[group_name] = group_df

    key_columns = ["SID", "ROW_ID_S", "ROW_ID_E"]
    full_keys = set(map(tuple, sample_tables[GROUP_ORDER[0]][key_columns].astype(str).values))
    subset_keys = set(map(tuple, sample_tables[GROUP_ORDER[1]][key_columns].astype(str).values))
    if not subset_keys.issubset(full_keys):
        raise RuntimeError("The nested group is not a subset of all no-ET RW samples")
    return groups


# ============================================================
# 3. Statistical calculations
# ============================================================

def bootstrap_mean_ci(values, seed_sequence):
    """Calculate a sample-level bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed_sequence)
    sample_count = values.size
    bootstrap_means = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)

    for start in range(0, BOOTSTRAP_ITERATIONS, BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, BOOTSTRAP_ITERATIONS)
        indices = rng.integers(
            0, sample_count, size=(stop - start, sample_count), endpoint=False
        )
        bootstrap_means[start:stop] = values[indices].mean(axis=1)

    alpha = 1.0 - CONFIDENCE_LEVEL
    return tuple(
        map(float, np.quantile(bootstrap_means, [alpha / 2, 1 - alpha / 2]))
    )


def calculate_statistics(groups):
    """Calculate sample sizes, means, bootstrap CIs, and medians."""
    seeds = np.random.SeedSequence(BOOTSTRAP_RANDOM_SEED).spawn(len(GROUP_ORDER))
    rows = []
    for group_name, seed in zip(GROUP_ORDER, seeds):
        values = groups[group_name]
        ci_low, ci_high = bootstrap_mean_ci(values, seed)
        rows.append({
            "SAMPLE_GROUP": group_name,
            "SAMPLE_COUNT": int(values.size),
            "DIFF_STORM_SPEED_MEAN": float(values.mean()),
            "DIFF_STORM_SPEED_MEAN_BOOTSTRAP_CI_LOW": ci_low,
            "DIFF_STORM_SPEED_MEAN_BOOTSTRAP_CI_HIGH": ci_high,
            "DIFF_STORM_SPEED_MEDIAN": float(np.median(values)),
            "BOOTSTRAP_ITERATIONS": BOOTSTRAP_ITERATIONS,
            "BOOTSTRAP_CONFIDENCE_LEVEL": CONFIDENCE_LEVEL,
            "BOOTSTRAP_RANDOM_SEED": BOOTSTRAP_RANDOM_SEED,
            "BOOTSTRAP_METHOD": (
                "sample-level resampling with replacement; not clustered by SID"
            ),
        })
    return pd.DataFrame(rows)


def nested_subset_permutation_test(groups):
    """Test whether the target subset mean is lower than the full mean."""
    all_values = groups[GROUP_ORDER[0]]
    subset_values = groups[GROUP_ORDER[1]]
    full_count = all_values.size
    subset_count = subset_values.size
    if subset_count >= full_count:
        raise ValueError("The target subset must be smaller than the full sample")

    full_mean = float(all_values.mean())
    subset_mean = float(subset_values.mean())
    observed_difference = subset_mean - full_mean
    rng = np.random.default_rng(PERMUTATION_RANDOM_SEED)
    random_differences = np.empty(PERMUTATION_ITERATIONS, dtype=float)

    for iteration in range(PERMUTATION_ITERATIONS):
        indices = rng.choice(full_count, size=subset_count, replace=False)
        random_differences[iteration] = all_values[indices].mean() - full_mean

    extreme_count = int(
        np.count_nonzero(random_differences <= observed_difference)
    )
    p_value = (extreme_count + 1) / (PERMUTATION_ITERATIONS + 1)
    null_low, null_high = map(
        float, np.quantile(random_differences, [0.05, 0.95])
    )
    if p_value < SIGNIFICANCE_LEVEL:
        stars, category = "**", "95% significant"
    elif p_value < MARGINAL_SIGNIFICANCE_LEVEL:
        stars, category = "*", "90% significant"
    else:
        stars, category = "", "Not significant"

    return pd.DataFrame([{
        "TEST_NAME": (
            "One-sided lower-tail Monte Carlo permutation test for a nested subset mean difference"
        ),
        "ALTERNATIVE_HYPOTHESIS": "Subset mean is lower than full mean",
        "VALUE_COLUMN": VALUE_COLUMN,
        "FULL_SAMPLE_COUNT": int(full_count),
        "SUBSET_COUNT": int(subset_count),
        "FULL_SAMPLE_MEAN": full_mean,
        "SUBSET_MEAN": subset_mean,
        "OBSERVED_MEAN_DIFFERENCE_SUBSET_MINUS_FULL": observed_difference,
        "NULL_DISTRIBUTION_5_PERCENTILE": null_low,
        "NULL_DISTRIBUTION_95_PERCENTILE": null_high,
        "PERMUTATION_ITERATIONS": PERMUTATION_ITERATIONS,
        "EXTREME_PERMUTATIONS": extreme_count,
        "ONE_SIDED_P_VALUE": float(p_value),
        "SIGNIFICANCE_STARS": stars,
        "SIGNIFICANCE_CATEGORY": category,
        "ALPHA_FOR_95_PERCENT_SIGNIFICANCE": SIGNIFICANCE_LEVEL,
        "ALPHA_FOR_90_PERCENT_SIGNIFICANCE": MARGINAL_SIGNIFICANCE_LEVEL,
        "PERMUTATION_RANDOM_SEED": PERMUTATION_RANDOM_SEED,
    }])


# ============================================================
# 4. Output helpers
# ============================================================

def save_figure_atomic(fig, output_path):
    """Write one PNG figure atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    try:
        fig.savefig(
            temporary_path, format="png", dpi=FIGURE_DPI,
            bbox_inches="tight", facecolor="white"
        )
        if not temporary_path.exists() or temporary_path.stat().st_size == 0:
            raise RuntimeError(f"Temporary figure is missing or empty: {temporary_path}")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def write_csv_atomic(df, output_path):
    """Write one CSV atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    try:
        df.to_csv(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


# ============================================================
# 5. Plot styling and continuous gradient rendering
# ============================================================

def apply_plot_style():
    """Apply the manuscript-oriented plotting style."""
    plt.rcParams.update({
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.cursive": ["Arial"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.cal": "Arial",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "axes.linewidth": 2.0,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
    })


def format_speed_y_tick(value, position):
    """Show zero without a decimal and retain one decimal elsewhere."""
    if np.isclose(value, 0.0):
        return "0"
    return f"{value:.1f}"


def add_vertical_gradient_to_bar(ax, bar, colors, zorder=1.8):
    """Fill one bar with a smooth PowerPoint-style vertical gradient."""
    x_left = bar.get_x()
    x_right = x_left + bar.get_width()
    height = bar.get_height()
    y_bottom, y_top = min(0.0, height), max(0.0, height)
    if np.isclose(y_bottom, y_top):
        return

    deep_color, middle_color, light_color = colors
    stops_from_zero = [
        (0.00, "#FFFFFF"),
        (0.56, light_color),
        (0.79, middle_color),
        (1.00, deep_color),
    ]
    color_stops = (
        [(1.0 - position, color) for position, color in reversed(stops_from_zero)]
        if height < 0.0 else stops_from_zero
    )
    color_map = LinearSegmentedColormap.from_list(
        "powerpoint_speed_bar_gradient", color_stops, N=2048
    )
    gradient = np.linspace(0.0, 1.0, 2048).reshape(2048, 1)
    image = ax.imshow(
        gradient,
        extent=[x_left, x_right, y_bottom, y_top],
        origin="lower",
        aspect="auto",
        cmap=color_map,
        interpolation="bicubic",
        zorder=zorder,
        clip_on=True,
    )
    image.set_clip_path(bar)


def calculate_shared_axis_limits(stats):
    """Calculate shared limits and a bracket position near the bar heads."""
    means = stats["DIFF_STORM_SPEED_MEAN"].to_numpy(dtype=float)
    ci_low = stats["DIFF_STORM_SPEED_MEAN_BOOTSTRAP_CI_LOW"].to_numpy(dtype=float)
    ci_high = stats["DIFF_STORM_SPEED_MEAN_BOOTSTRAP_CI_HIGH"].to_numpy(dtype=float)
    data_min = min(0.0, float(ci_low.min()))
    data_max = max(0.0, float(ci_high.max()))
    data_span = max(data_max - data_min, 1.0)
    all_negative = bool(np.all(means < 0.0))
    all_positive = bool(np.all(means > 0.0))

    if all_negative:
        bracket_near = float(ci_low.min()) - 0.06 * data_span
        bracket_far = bracket_near - 0.04 * data_span
        star_y = bracket_far - 0.025 * data_span
        y_min = star_y - 0.12 * data_span
        y_max = data_max + 0.12 * data_span
    else:
        bracket_near = float(ci_high.max()) + 0.06 * data_span
        bracket_far = bracket_near + 0.04 * data_span
        star_y = bracket_far + 0.025 * data_span
        # Place zero on the lower frame when all plotted means are positive.
        y_min = 0.0 if all_positive else data_min - 0.16 * data_span
        y_max = star_y + 0.12 * data_span

    return {
        "all_negative": all_negative,
        "bracket_near": bracket_near,
        "bracket_far": bracket_far,
        "star_y": star_y,
        "y_min": y_min,
        "y_max": y_max,
    }


def create_main_figure(stats, significance_test, axis_limits, show_significance):
    """Create the two-group speed chart with optional significance marks."""
    apply_plot_style()
    means = stats["DIFF_STORM_SPEED_MEAN"].to_numpy(dtype=float)
    ci_low = stats["DIFF_STORM_SPEED_MEAN_BOOTSTRAP_CI_LOW"].to_numpy(dtype=float)
    ci_high = stats["DIFF_STORM_SPEED_MEAN_BOOTSTRAP_CI_HIGH"].to_numpy(dtype=float)
    counts = stats["SAMPLE_COUNT"].to_numpy(dtype=int)
    errors = np.vstack([means - ci_low, ci_high - means])
    x = np.arange(len(GROUP_ORDER), dtype=float)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(
        x, means, width=BAR_WIDTH, color="none", edgecolor="black",
        linewidth=2.0, yerr=errors, capsize=8,
        error_kw={
            "elinewidth": 2.3, "capthick": 2.3,
            "ecolor": "black", "zorder": 3.5,
        },
        zorder=2.5,
    )
    for bar, group_name in zip(bars, GROUP_ORDER):
        add_vertical_gradient_to_bar(ax, bar, GROUP_GRADIENT_COLORS[group_name])

    ax.set_xlim(-0.55, len(GROUP_ORDER) - 0.45)
    ax.set_ylim(axis_limits["y_min"], 2.7)
    ax.yaxis.set_major_formatter(FuncFormatter(format_speed_y_tick))
    ax.set_xticks(x, [
        f"{GROUP_LABELS[group]}\n$N$ = {counts[index]:,}"
        for index, group in enumerate(GROUP_ORDER)
    ])
    if show_significance:
        ax.plot(
            [x[0], x[0], x[1], x[1]],
            [
                axis_limits["bracket_near"], axis_limits["bracket_far"],
                axis_limits["bracket_far"], axis_limits["bracket_near"],
            ],
            color="black", linewidth=2.0, clip_on=False, zorder=4.0,
        )
        stars = str(significance_test.loc[0, "SIGNIFICANCE_STARS"])
        if stars:
            ax.text(
                x.mean(), axis_limits["star_y"], stars,
                ha="center",
                va="top" if axis_limits["all_negative"] else "bottom",
                fontsize=SIGNIFICANCE_STAR_FONT_SIZE,
                fontweight="bold", zorder=4.0,
            )

    # Keep the original y-axis title unchanged.
    ax.set_ylabel("Change in translation speed (kt)", labelpad=14)
    ax.tick_params(
        axis="x",
        direction="out",
        length=6,
        pad=12,
        labelrotation=10,
    )
    ax.tick_params(axis="y", direction="out", length=6)
    ax.grid(False)

    # Use the same horizontal-guide styling as the other EX_FIG4 panels,
    # while leaving the upper 2.5 tick without a guide line.
    for y_tick in ax.get_yticks():
        if np.isclose(y_tick, 2.5):
            continue
        if np.isclose(y_tick, 0.0):
            ax.axhline(
                y_tick,
                color="#333333",
                linestyle="--",
                linewidth=1.9,
                zorder=0.7,
            )
        else:
            ax.axhline(
                y_tick,
                color="#C4C4C4",
                linestyle="--",
                linewidth=1.7,
                alpha=0.85,
                zorder=0.6,
            )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
    fig.subplots_adjust(left=0.25, right=0.96, bottom=0.16, top=0.97)
    return fig


# ============================================================
# 6. Main program
# ============================================================

def main():
    print("=" * 90)
    print("RW DIFF_STORM_SPEED nested-subset gradient comparison")
    print("=" * 90)
    print(f"Input: {INPUT_CSV}")
    print(f"RW definition: {WIND_DIFF_COLUMN} <= {RW_WIND_THRESHOLD}")
    print(f"ET exclusion: {ET_COUNT_COLUMN} == 0")
    print(
        f"Nested subset: {OUTER_AREA_DIFF_COLUMN} < 0 and "
        f"{DISTANCE_DIFF_COLUMN} < 0"
    )
    print(f"Value: {VALUE_COLUMN}")
    print(f"Bar width: {BAR_WIDTH}")

    source_df, rw_df = read_and_validate_input()
    groups = construct_groups(rw_df)
    stats = calculate_statistics(groups)
    significance_test = nested_subset_permutation_test(groups)
    axis_limits = calculate_shared_axis_limits(stats)

    figure = create_main_figure(
        stats, significance_test, axis_limits, show_significance=True
    )
    try:
        save_figure_atomic(figure, OUTPUT_WITH_SIGNIFICANCE)
    finally:
        plt.close(figure)

    print("\n" + "=" * 90)
    print("Analysis completed successfully")
    print("=" * 90)
    print(f"Input cleaned pairs: {len(source_df):,}")
    print(f"RW pairs: {len(rw_df):,}")
    print(f"\nFigure with significance: {OUTPUT_WITH_SIGNIFICANCE}")


if __name__ == "__main__":
    main()
