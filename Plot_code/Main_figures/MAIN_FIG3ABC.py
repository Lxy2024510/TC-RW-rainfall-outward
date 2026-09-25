#!/usr/bin/env python3
"""Plot TC-level new population exposure for RW and RI events."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter


plt.rcParams.update({
    "font.family": "Arial",
    "text.color": "#000000",
    "axes.labelcolor": "#000000",
    "axes.edgecolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
})


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "Data" / "Processed" / "Exposure"
INPUT_FILES = {
    "RW": DATA_ROOT / "RW" / "RUN6_FINAL_NEW_POPULATION_EXPOSURE.csv",
    "RI": DATA_ROOT / "RI" / "RUN6_FINAL_NEW_POPULATION_EXPOSURE.csv",
}
OUTPUT_DIR = PROJECT_ROOT / "Results" / "Main_figures" / "FIG3"
PANELS = {
    "A": {"basin": None, "description": "All basins", "region_label": "Global"},
    "B": {"basin": "WP", "description": "Western North Pacific", "region_label": "WP"},
    "C": {"basin": "NA", "description": "North Atlantic", "region_label": "NA"},
}

LABELS = ["RW", "RI"]
BAR_COLORS = ["#56A1BB", "#D5A0B6"]
PEOPLE_PER_MILLION = 1_000_000.0
BOOTSTRAP_REPETITIONS = 5000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

FIGURE_DPI = 600
FIGURE_SIZE = (8.0, 10.0)
PLOT_FONT_SIZE = 30
BAR_WIDTH = 0.34
BAR_EDGE_WIDTH = 2.6
ERROR_LINE_WIDTH = 2.4
ERROR_CAP_SIZE = 8.0
GRID_COLOR = "#9E9E9E"
GRID_ALPHA = 0.55
GRID_LINE_WIDTH = 1.0


def clean_decimal_tick(value, position=None):
    """Show zero as 0 and remove unnecessary trailing zeros."""
    if np.isclose(value, 0.0):
        return "0"
    return f"{value:g}"


def load_tc_level_data(group):
    """Sum all window-level new exposure values within each SID."""
    path = INPUT_FILES[group]
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    # Keep literal basin code "NA" instead of treating it as a missing value.
    data = pd.read_csv(path, keep_default_na=False)
    required = {"SID", "BASIN_S", "NEW_EXPOSURE_POP_VALUE"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise KeyError(f"{path} is missing columns: {', '.join(missing)}")
    if data["SID"].isna().any():
        raise ValueError(f"{path} contains missing SID values")

    exposure = pd.to_numeric(data["NEW_EXPOSURE_POP_VALUE"], errors="coerce")
    if exposure.isna().any():
        raise ValueError(f"{path} contains invalid exposure values")
    if exposure.lt(0).any():
        raise ValueError(f"{path} contains negative exposure values")

    basin = data["BASIN_S"].astype(str).str.strip()
    # Compatibility with files previously read and saved using pandas' default
    # NA parser, which may have converted the North Atlantic code to a blank.
    basin = basin.replace({"": "NA", "nan": "NA", "NaN": "NA"})
    working = pd.DataFrame({
        "SID": data["SID"].astype(str),
        "BASIN": basin,
        "WINDOW_NEW_EXPOSURE": exposure.astype(float),
    })
    tc_level = (
        working.groupby(["SID", "BASIN"], as_index=False, sort=True)
        .agg(
            WINDOW_COUNT=("WINDOW_NEW_EXPOSURE", "size"),
            NEW_EXPOSURE_POP_VALUE=("WINDOW_NEW_EXPOSURE", "sum"),
        )
    )
    tc_level.insert(0, "GROUP", group)
    tc_level["NEW_EXPOSURE_MILLION"] = (
        tc_level["NEW_EXPOSURE_POP_VALUE"] / PEOPLE_PER_MILLION
    )
    return tc_level


def bootstrap_mean(values, repetitions, rng):
    """Calculate a percentile 95% confidence interval for the TC mean."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Bootstrap input must be a non-empty one-dimensional array")

    means = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        means[index] = rng.choice(values, size=values.size, replace=True).mean()

    alpha = 1.0 - CONFIDENCE_LEVEL
    lower, upper = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(values.mean()), float(lower), float(upper)


def calculate_summary(tc_data):
    """Calculate RW and RI means using SID as the statistical unit."""
    seed_sequence = np.random.SeedSequence(RANDOM_SEED)
    child_seeds = seed_sequence.spawn(len(LABELS))
    rows = []

    for group, child_seed in zip(LABELS, child_seeds):
        subset = tc_data.loc[tc_data["GROUP"].eq(group)]
        mean, lower, upper = bootstrap_mean(
            subset["NEW_EXPOSURE_MILLION"].to_numpy(),
            BOOTSTRAP_REPETITIONS,
            np.random.default_rng(child_seed),
        )
        rows.append({
            "GROUP": group,
            "TC_COUNT": len(subset),
            "WINDOW_COUNT": int(subset["WINDOW_COUNT"].sum()),
            "TOTAL_NEW_EXPOSURE": float(subset["NEW_EXPOSURE_POP_VALUE"].sum()),
            "MEAN_MILLION": mean,
            "CI_LOWER_MILLION": lower,
            "CI_UPPER_MILLION": upper,
            "BOOTSTRAP_REPETITIONS": BOOTSTRAP_REPETITIONS,
            "CONFIDENCE_LEVEL": CONFIDENCE_LEVEL,
        })
    return pd.DataFrame(rows)


def draw_gradient_bar(ax, x_position, value, width, end_color):
    """Draw one white-to-color gradient bar."""
    bar = ax.bar(
        x_position,
        value,
        width=width,
        facecolor="none",
        edgecolor="#000000",
        linewidth=BAR_EDGE_WIDTH,
        zorder=3,
    )[0]
    color_map = LinearSegmentedColormap.from_list(
        f"gradient_{x_position}", ["#FFFFFF", end_color], N=512
    )
    gradient = np.linspace(0.0, 1.0, 1024).reshape(-1, 1)
    image = ax.imshow(
        gradient,
        extent=[x_position - width / 2.0, x_position + width / 2.0, 0.0, value],
        origin="lower",
        aspect="auto",
        cmap=color_map,
        interpolation="bicubic",
        zorder=2,
    )
    image.set_clip_path(bar)


def plot_comparison(summary, output_path, region_label):
    """Plot TC means with SID-bootstrap 95% confidence intervals."""
    means = summary["MEAN_MILLION"].to_numpy(dtype=float)
    lower = summary["CI_LOWER_MILLION"].to_numpy(dtype=float)
    upper = summary["CI_UPPER_MILLION"].to_numpy(dtype=float)
    x_positions = np.arange(len(LABELS))

    if np.any(lower > means) or np.any(means > upper):
        raise ValueError("Each mean must lie inside its confidence interval")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for x_position, mean, color in zip(x_positions, means, BAR_COLORS):
        draw_gradient_bar(ax, x_position, mean, BAR_WIDTH, color)

    errors = np.vstack((means - lower, upper - means))
    ax.errorbar(
        x_positions,
        means,
        yerr=errors,
        fmt="none",
        ecolor="#000000",
        elinewidth=ERROR_LINE_WIDTH,
        capsize=ERROR_CAP_SIZE,
        capthick=ERROR_LINE_WIDTH,
        zorder=5,
    )
    ax.axhline(0.0, color="#000000", linestyle="--", linewidth=1.6, zorder=6)

    tc_counts = summary["TC_COUNT"].to_numpy(dtype=int)
    tick_labels = [
        f"{group}\n" + rf"$\mathit{{N}}_{{\mathrm{{TC}}}} = {tc_count}$"
        for group, tc_count in zip(LABELS, tc_counts)
    ]
    ax.set_xticks(x_positions)
    ax.set_xlim(-0.5, len(LABELS) - 0.5)
    ax.set_xticklabels(
        tick_labels,
        fontsize=PLOT_FONT_SIZE,
        rotation=0,
        ha="center",
        multialignment="center",
        color="#000000",
    )
    ax.set_xlabel("")
    ax.set_ylabel(
        "New population exposure (million)",
        fontsize=PLOT_FONT_SIZE,
        fontfamily="Arial",
        color="#000000",
        labelpad=10,
    )
    ax.text(
        0.5,
        0.97,
        region_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PLOT_FONT_SIZE,
        fontfamily="Arial",
        color="#000000",
        zorder=10,
    )
    ax.tick_params(
        axis="x", labelsize=PLOT_FONT_SIZE, pad=8, length=10,
        width=2.0, colors="#000000", direction="out",
    )
    ax.tick_params(
        axis="y", labelsize=PLOT_FONT_SIZE, length=10,
        width=2.8, colors="#000000", direction="out",
    )

    # Use the same scale for FIG3A-C so the three panels remain comparable.
    # The upper limit accommodates the WP RW bootstrap upper bound (~3.56).
    ax.set_ylim(0.0, 4.0)
    ax.set_yticks(np.arange(0.0, 4.01, 1.0))
    ax.yaxis.set_major_formatter(FuncFormatter(clean_decimal_tick))
    ax.set_axisbelow(True)
    ax.grid(
        axis="y", visible=True, linestyle="--", color=GRID_COLOR,
        alpha=GRID_ALPHA, linewidth=GRID_LINE_WIDTH,
    )
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(2.8)

    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.14, top=0.90)
    fig.savefig(output_path, dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_tc_data = pd.concat(
        [load_tc_level_data(group) for group in LABELS],
        ignore_index=True,
    )
    for panel, settings in PANELS.items():
        basin = settings["basin"]
        if basin is None:
            # A TC can cross a basin boundary. For the all-basin panel, merge
            # every basin segment back to one sample per SID and event group.
            tc_data = (
                all_tc_data.groupby(["GROUP", "SID"], as_index=False, sort=True)
                .agg(
                    WINDOW_COUNT=("WINDOW_COUNT", "sum"),
                    NEW_EXPOSURE_POP_VALUE=("NEW_EXPOSURE_POP_VALUE", "sum"),
                )
            )
            tc_data.insert(2, "BASIN", "ALL")
            tc_data["NEW_EXPOSURE_MILLION"] = (
                tc_data["NEW_EXPOSURE_POP_VALUE"] / PEOPLE_PER_MILLION
            )
        else:
            tc_data = all_tc_data.loc[all_tc_data["BASIN"].eq(basin)].copy()

        missing_groups = sorted(set(LABELS).difference(tc_data["GROUP"].unique()))
        if missing_groups:
            raise ValueError(
                f"FIG3{panel} ({settings['description']}) has no TC samples for: "
                f"{', '.join(missing_groups)}"
            )

        summary = calculate_summary(tc_data)
        summary.insert(1, "BASIN", basin if basin is not None else "ALL")
        output_path = OUTPUT_DIR / f"FIG3{panel}.png"
        plot_comparison(summary, output_path, settings["region_label"])

        print("=" * 80)
        print(f"MAIN FIG3{panel}: {settings['description']}")
        print("=" * 80)
        print(f"Saved PNG: {output_path}")


if __name__ == "__main__":
    main()
