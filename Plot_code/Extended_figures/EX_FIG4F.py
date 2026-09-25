import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch


# ============================================================
# 1. Paths and analysis settings
# ============================================================

INPUT_CSV = Path(
    str(PROJECT_ROOT / "Data" / "Processed" / "MSWEP_RH600_24H_RESULTS") + "/" +
    "RW_DIFF_RH600_BOOTSTRAP_COMPARISON/"
    "ALL_RW_SAMPLES_WITH_CHANGE_GROUPS.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "Results" / "Extended_figures" / "EX_FIG4"

OUTPUT_STEM = "EX_FIG4F"
OUTPUT_PNG = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
PLOT_DATA_CSV = OUTPUT_DIR / f"{OUTPUT_STEM}_DATA.csv"
SIGNIFICANCE_CSV = OUTPUT_DIR / f"{OUTPUT_STEM}_SIGNIFICANCE_TEST.csv"

VALUE_COLUMN = "DIFF_RH600_AWMEAN_200_800"
BOOTSTRAP_ITERATIONS = 5000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 20260814
BOOTSTRAP_BATCH_SIZE = 250

PERMUTATION_ITERATIONS = 50000
PERMUTATION_RANDOM_SEED = 20260823
SIGNIFICANCE_LEVEL = 0.05
MARGINAL_SIGNIFICANCE_LEVEL = 0.10


# ============================================================
# 2. Group and color settings
# ============================================================

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

# Colors are ordered from the deepest shade to the lightest supplied shade.
# The plotted gradient adds pure white at the zero baseline.
GROUP_GRADIENT_COLORS = {
    "ALL_RW_NO_ET": [
        "#56A1BB",
        "#6FBFD7",
        "#A3DCEC",
    ],
    "AREA_DECREASE_DISTANCE_DECREASE_NO_ET": [
        "#DEAEC3",
        "#ECAFC6",
        "#FBC8DB",
    ],
}

# Solid representative colors are used only in the separate legend.
GROUP_LEGEND_COLORS = {
    "ALL_RW_NO_ET": "#56A1BB",
    "AREA_DECREASE_DISTANCE_DECREASE_NO_ET": "#DEAEC3",
}

BAR_WIDTH = 0.46
FIGSIZE = (7.2, 10.2)
LEGEND_FIGSIZE = (8.0, 1.35)
FIGURE_DPI = 600
FONT_SIZE = 28


# ============================================================
# 3. Input and statistical calculations
# ============================================================

def read_and_validate_input():
    """Read and validate the existing RW sample table."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input sample table does not exist: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    required_columns = [
        VALUE_COLUMN,
        "COUNT_ET_24H",
        "DIFF_MSWEP_AREA_30_200_500",
        "DIFF_MSWEP_DIST_30",
    ]

    missing = set(required_columns).difference(df.columns)
    if missing:
        raise KeyError(
            "Input table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if not np.isfinite(
        df[required_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Input table contains non-finite required values")

    return df


def construct_groups(df):
    """Construct the full no-ET RW group and its nested subset."""
    non_et = df["COUNT_ET_24H"].eq(0)
    area_decrease = df["DIFF_MSWEP_AREA_30_200_500"].lt(0)
    distance_decrease = df["DIFF_MSWEP_DIST_30"].lt(0)

    masks = {
        "ALL_RW_NO_ET": non_et,
        "AREA_DECREASE_DISTANCE_DECREASE_NO_ET": (
            non_et & area_decrease & distance_decrease
        ),
    }

    groups = {}

    for group_name in GROUP_ORDER:
        values = df.loc[masks[group_name], VALUE_COLUMN].to_numpy(dtype=float)

        if values.size == 0:
            raise ValueError(f"Sample group is empty: {group_name}")

        groups[group_name] = values

    return groups


def bootstrap_mean_ci(values, seed_sequence):
    """Calculate a sample-level bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed_sequence)
    sample_count = values.size
    bootstrap_means = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)

    for start in range(0, BOOTSTRAP_ITERATIONS, BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, BOOTSTRAP_ITERATIONS)
        indices = rng.integers(
            0,
            sample_count,
            size=(stop - start, sample_count),
            endpoint=False,
        )
        bootstrap_means[start:stop] = values[indices].mean(axis=1)

    alpha = 1.0 - CONFIDENCE_LEVEL
    ci_low, ci_high = np.quantile(
        bootstrap_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )

    return float(ci_low), float(ci_high)


def calculate_statistics(groups):
    """Calculate group means, confidence intervals, and medians."""
    child_seeds = np.random.SeedSequence(RANDOM_SEED).spawn(
        len(GROUP_ORDER)
    )
    rows = []

    for group_name, child_seed in zip(GROUP_ORDER, child_seeds):
        values = groups[group_name]
        ci_low, ci_high = bootstrap_mean_ci(values, child_seed)

        rows.append(
            {
                "SAMPLE_GROUP": group_name,
                "SAMPLE_COUNT": int(values.size),
                "DIFF_RH600_MEAN": float(np.mean(values)),
                "DIFF_RH600_MEAN_BOOTSTRAP_CI_LOW": ci_low,
                "DIFF_RH600_MEAN_BOOTSTRAP_CI_HIGH": ci_high,
                "DIFF_RH600_MEDIAN": float(np.median(values)),
                "BOOTSTRAP_ITERATIONS": BOOTSTRAP_ITERATIONS,
                "BOOTSTRAP_CONFIDENCE_LEVEL": CONFIDENCE_LEVEL,
                "BOOTSTRAP_RANDOM_SEED": RANDOM_SEED,
            }
        )

    return pd.DataFrame(rows)


def nested_subset_permutation_test(groups):
    """Test whether the target subset mean is smaller than the full mean."""
    all_values = groups["ALL_RW_NO_ET"]
    subset_values = groups[
        "AREA_DECREASE_DISTANCE_DECREASE_NO_ET"
    ]
    full_count = all_values.size
    subset_count = subset_values.size

    if subset_count >= full_count:
        raise ValueError(
            "The target subset must be smaller than the full sample"
        )

    full_mean = float(all_values.mean())
    subset_mean = float(subset_values.mean())
    observed_difference = subset_mean - full_mean

    rng = np.random.default_rng(PERMUTATION_RANDOM_SEED)
    random_differences = np.empty(PERMUTATION_ITERATIONS, dtype=float)

    for iteration in range(PERMUTATION_ITERATIONS):
        random_indices = rng.choice(
            full_count,
            size=subset_count,
            replace=False,
        )
        random_differences[iteration] = (
            all_values[random_indices].mean() - full_mean
        )

    extreme_count = int(
        np.count_nonzero(random_differences <= observed_difference)
    )
    p_value = (extreme_count + 1) / (PERMUTATION_ITERATIONS + 1)

    if p_value < SIGNIFICANCE_LEVEL:
        significance_stars = "**"
        significance_category = "95% significant"
    elif p_value < MARGINAL_SIGNIFICANCE_LEVEL:
        significance_stars = "*"
        significance_category = "90% significant"
    else:
        significance_stars = ""
        significance_category = "Not significant"

    null_ci_low, null_ci_high = np.quantile(
        random_differences,
        [0.05, 0.95],
    )

    return pd.DataFrame(
        [
            {
                "TEST_NAME": (
                    "One-sided lower-tail Monte Carlo permutation test for a nested "
                    "subset mean difference"
                ),
                "ALTERNATIVE_HYPOTHESIS": "Subset mean is smaller than full mean",
                "FULL_SAMPLE_COUNT": int(full_count),
                "SUBSET_COUNT": int(subset_count),
                "FULL_SAMPLE_MEAN": full_mean,
                "SUBSET_MEAN": subset_mean,
                "OBSERVED_MEAN_DIFFERENCE_SUBSET_MINUS_FULL": (
                    observed_difference
                ),
                "NULL_DISTRIBUTION_5_PERCENTILE": float(null_ci_low),
                "NULL_DISTRIBUTION_95_PERCENTILE": float(null_ci_high),
                "PERMUTATION_ITERATIONS": PERMUTATION_ITERATIONS,
                "EXTREME_PERMUTATIONS": extreme_count,
                "ONE_SIDED_P_VALUE": float(p_value),
                "SIGNIFICANCE_STARS": significance_stars,
                "SIGNIFICANCE_CATEGORY": significance_category,
                "ALPHA_FOR_95_PERCENT_SIGNIFICANCE": SIGNIFICANCE_LEVEL,
                "ALPHA_FOR_90_PERCENT_SIGNIFICANCE": (
                    MARGINAL_SIGNIFICANCE_LEVEL
                ),
                "PERMUTATION_RANDOM_SEED": PERMUTATION_RANDOM_SEED,
            }
        ]
    )


# ============================================================
# 4. Atomic output helpers
# ============================================================

def save_figure_atomic(fig, output_path):
    """Write a figure atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        output_path.name + f".tmp.{os.getpid()}"
    )

    try:
        fig.savefig(
            temporary_path,
            format=output_path.suffix.lstrip(".").lower(),
            dpi=FIGURE_DPI,
            bbox_inches="tight",
            facecolor="white",
        )

        if not temporary_path.exists() or temporary_path.stat().st_size == 0:
            raise RuntimeError(
                f"Temporary figure is missing or empty: {temporary_path}"
            )

        os.replace(temporary_path, output_path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def write_csv_atomic(df, output_path):
    """Write a CSV atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        output_path.name + f".tmp.{os.getpid()}"
    )

    try:
        df.to_csv(temporary_path, index=False)
        os.replace(temporary_path, output_path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


# ============================================================
# 5. Gradient bar rendering
# ============================================================

def apply_plot_style():
    """Apply the existing manuscript-oriented plot style."""
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
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "axes.linewidth": 2.0,
            "xtick.major.width": 2.0,
            "ytick.major.width": 2.0,
        }
    )


def add_vertical_gradient_to_bar(ax, bar, colors, zorder=1.8):
    """Fill one bar with a PowerPoint-style multistop color gradient."""
    x_left = bar.get_x()
    x_right = x_left + bar.get_width()
    bar_height = bar.get_height()
    y_bottom = min(0.0, bar_height)
    y_top = max(0.0, bar_height)

    if np.isclose(y_bottom, y_top):
        return

    gradient = np.linspace(0.0, 1.0, 1024).reshape(1024, 1)
    deep_color, middle_color, light_color = colors

    # Positions are measured from the zero baseline toward the bar head.
    # The transition starts immediately at zero. There is no repeated white
    # stop or flat white section, which avoids a visible boundary in the bar.
    stops_from_zero = [
        (0.00, "#FFFFFF"),
        (0.56, light_color),
        (0.79, middle_color),
        (1.00, deep_color),
    ]

    if bar_height < 0.0:
        # imshow runs upward from the negative head to the zero baseline.
        color_stops = [
            (1.0 - position, color)
            for position, color in reversed(stops_from_zero)
        ]
    else:
        # imshow runs upward from zero to the positive bar head.
        color_stops = stops_from_zero

    color_map = LinearSegmentedColormap.from_list(
        "powerpoint_bar_gradient",
        color_stops,
        N=1024,
    )

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


def create_main_figure(stats, significance_test):
    """Create the narrower two-bar figure with gradient fills."""
    apply_plot_style()

    means = stats["DIFF_RH600_MEAN"].to_numpy(dtype=float)
    ci_low = stats[
        "DIFF_RH600_MEAN_BOOTSTRAP_CI_LOW"
    ].to_numpy(dtype=float)
    ci_high = stats[
        "DIFF_RH600_MEAN_BOOTSTRAP_CI_HIGH"
    ].to_numpy(dtype=float)
    counts = stats["SAMPLE_COUNT"].to_numpy(dtype=int)
    errors = np.vstack([means - ci_low, ci_high - means])
    x = np.arange(len(GROUP_ORDER), dtype=float)

    data_min = min(0.0, float(ci_low.min()))
    data_max = max(0.0, float(ci_high.max()))
    data_span = max(data_max - data_min, 1.0)

    all_negative = bool(np.all(means < 0.0))

    if all_negative:
        # Place the bracket beyond the negative bar heads and their CIs.
        bracket_near = float(ci_low.min()) - 0.06 * data_span
        bracket_far = bracket_near - 0.04 * data_span
        star_y = bracket_far - 0.025 * data_span
        y_min = star_y - 0.12 * data_span
        # Place zero directly on the upper frame for an all-negative chart.
        y_max = 0.0
    else:
        # Retain the conventional upper placement for non-negative cases.
        bracket_near = max(0.0, float(ci_high.max())) + 0.08 * data_span
        bracket_far = bracket_near + 0.04 * data_span
        star_y = bracket_far + 0.025 * data_span
        y_min = data_min - 0.16 * data_span
        y_max = star_y + 0.12 * data_span

    fig, ax = plt.subplots(figsize=FIGSIZE)

    bars = ax.bar(
        x,
        means,
        width=BAR_WIDTH,
        color="none",
        edgecolor="black",
        linewidth=2.0,
        yerr=errors,
        capsize=8,
        error_kw={
            "elinewidth": 2.3,
            "capthick": 2.3,
            "ecolor": "black",
            "zorder": 3.5,
        },
        zorder=2.5,
    )

    for bar, group_name in zip(bars, GROUP_ORDER):
        add_vertical_gradient_to_bar(
            ax,
            bar,
            GROUP_GRADIENT_COLORS[group_name],
        )

    # Restore the intended axes because imshow participates in autoscaling.
    ax.set_xlim(-0.55, len(GROUP_ORDER) - 0.45)
    ax.set_ylim(y_min, y_max)

    if all_negative:
        # Keep integer RH ticks at a fixed interval of one percentage point.
        ax.set_yticks(np.arange(np.ceil(y_min), 0.0 + 1.0, 1.0))

    ax.set_xticks(
        x,
        [
            f"{GROUP_LABELS[group]}\n$N$ = {counts[index]:,}"
            for index, group in enumerate(GROUP_ORDER)
        ],
    )
    ax.plot(
        [x[0], x[0], x[1], x[1]],
        [bracket_near, bracket_far, bracket_far, bracket_near],
        color="black",
        linewidth=2.0,
        clip_on=False,
        zorder=4.0,
    )

    significance_stars = str(
        significance_test.loc[0, "SIGNIFICANCE_STARS"]
    )

    if significance_stars:
        ax.text(
            x.mean(),
            star_y,
            significance_stars,
            ha="center",
            va="top" if all_negative else "bottom",
            fontsize=FONT_SIZE,
            zorder=4.0,
        )

    ax.set_ylabel("Change in RH (%)", labelpad=14)
    ax.tick_params(
        axis="x",
        direction="out",
        length=6,
        pad=12,
        labelrotation=10,
    )
    ax.tick_params(axis="y", direction="out", length=6)
    ax.grid(False)

    # Draw the horizontal guides explicitly so the lower tick at -4 has no
    # guide line. Zero retains the darker reference-line styling used in the
    # other manuscript panels.
    for y_tick in ax.get_yticks():
        if np.isclose(y_tick, -4.0):
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

    fig.subplots_adjust(
        left=0.25,
        right=0.96,
        bottom=0.16,
        top=0.97,
    )

    return fig


def create_legend_figure():
    """Create the existing separate legend with representative colors."""
    apply_plot_style()

    handles = [
        Patch(
            facecolor=GROUP_LEGEND_COLORS[group],
            edgecolor="black",
            linewidth=1.7,
            label=GROUP_LABELS[group].replace("\n", " & "),
        )
        for group in GROUP_ORDER
    ]

    fig, ax = plt.subplots(figsize=LEGEND_FIGSIZE)
    ax.axis("off")
    legend = ax.legend(
        handles=handles,
        loc="center",
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=1.4,
        borderpad=0.6,
    )
    legend.get_frame().set_linewidth(2.0)

    return fig


# ============================================================
# 6. Main program
# ============================================================

def main():
    print("=" * 90)
    print("Creating gradient RW ΔRH600 comparison figure")
    print("=" * 90)
    print(f"Bar width: {BAR_WIDTH}")
    print(
        "All RW PowerPoint-style gradient: white -> "
        + " -> ".join(
            reversed(GROUP_GRADIENT_COLORS["ALL_RW_NO_ET"])
        )
    )
    print(
        "Nested-subset PowerPoint-style gradient: white -> "
        + " -> ".join(
            reversed(
                GROUP_GRADIENT_COLORS[
                    "AREA_DECREASE_DISTANCE_DECREASE_NO_ET"
                ]
            )
        )
    )

    source = read_and_validate_input()
    groups = construct_groups(source)
    stats = calculate_statistics(groups)
    significance_test = nested_subset_permutation_test(groups)

    main_figure = create_main_figure(stats, significance_test)

    try:
        save_figure_atomic(main_figure, OUTPUT_PNG)
    finally:
        plt.close(main_figure)

    print(f"\nMain figure: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
