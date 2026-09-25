from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter


# ============================================================
# 1. Configuration
# ============================================================

START_YEAR = 1982
END_YEAR = 2024
THRESHOLD = "30"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "Data"
    / "Processed"
    / "ERA5"
    / "Vertical_velocity"
    / "Bootstrap"
    / "W500_REVERSED_1982_2024_MSWEP30_BOOTSTRAP_LONG.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "Results" / "Main_figures" / "FIG2"

SPATIAL_TYPE = "OPEN_OCEAN"
DISTANCE_ZONES = ["0_200", "200_500"]
DISTANCE_LABELS = ["0–200 km", "200–500 km"]

GROUP_CODE = "LE_5TH"
GROUP_COLOR = "#56A1BB"
EXPECTED_WIND_RANGE = "<= -30 kt"

FIGURE_DPI = 600
FIGURE_SIZE = (8.0, 10.0)
PLOT_FONT_SIZE = 30

# The displayed axis uses a 1e-1 multiplier. Therefore, -0.20 and 0.05
# appear as -2 and 0.5, respectively.
Y_AXIS_MIN = -0.20
Y_AXIS_MAX = 0.05
Y_TICKS = np.array([-0.20, -0.10, 0.00])

BAR_WIDTH = 0.34
BAR_EDGE_WIDTH = 2.6
ERROR_LINE_WIDTH = 2.4
ERROR_CAP_SIZE = 8.0

GRID_COLOR = "#9E9E9E"
GRID_ALPHA = 0.55
GRID_LINE_WIDTH = 1.0

REQUIRED_COLUMNS = [
    "START_YEAR",
    "END_YEAR",
    "PRECIPITATION_DATASET",
    "RAINFALL_THRESHOLD",
    "PAIR_SELECTION",
    "SPATIAL_TYPE",
    "METRIC_TYPE",
    "INTENSITY_BOUNDARY_TYPE",
    "INTENSITY_GROUP_CODE",
    "WIND_CHANGE_RANGE",
    "DISTANCE_ZONE",
    "GROUP_WINDOW_COUNT",
    "COMMON_VALID_WINDOW_COUNT",
    "INVALID_WINDOW_COUNT",
    "MEAN_CHANGE",
    "BOOTSTRAP_CI_LOWER",
    "BOOTSTRAP_CI_UPPER",
    "BOOTSTRAP_METHOD",
]

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": PLOT_FONT_SIZE,
        "axes.labelsize": PLOT_FONT_SIZE,
        "xtick.labelsize": PLOT_FONT_SIZE,
        "ytick.labelsize": PLOT_FONT_SIZE,
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "axes.edgecolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
    }
)


# ============================================================
# 2. Input validation
# ============================================================

def require_single_text_value(df, column, expected):
    """Require one exact text value in a metadata field."""
    actual = set(df[column].dropna().astype(str))
    if actual != {str(expected)}:
        raise ValueError(
            f"Unexpected {column}: actual={sorted(actual)}, "
            f"expected={expected}"
        )


def read_and_validate():
    """Read the bootstrap table and select the two RW plotting rows."""
    print("=" * 90)
    print("Reading and validating MSWEP reversed-W500 bootstrap results")
    print("=" * 90)
    print(f"Input file: {INPUT_CSV}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    missing = sorted(set(REQUIRED_COLUMNS).difference(df.columns))
    if missing:
        raise KeyError("Input CSV is missing: " + ", ".join(missing))

    numeric_columns = [
        "START_YEAR",
        "END_YEAR",
        "RAINFALL_THRESHOLD",
        "GROUP_WINDOW_COUNT",
        "COMMON_VALID_WINDOW_COUNT",
        "INVALID_WINDOW_COUNT",
        "MEAN_CHANGE",
        "BOOTSTRAP_CI_LOWER",
        "BOOTSTRAP_CI_UPPER",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    invalid_numeric = df[numeric_columns].isna().any(axis=1)
    if invalid_numeric.any():
        raise ValueError(
            "Missing or invalid numeric values: "
            f"count={int(invalid_numeric.sum())}"
        )

    if not (
        df["START_YEAR"].eq(START_YEAR)
        & df["END_YEAR"].eq(END_YEAR)
    ).all():
        raise ValueError(
            f"Bootstrap table does not consistently describe "
            f"{START_YEAR}-{END_YEAR}"
        )
    if not df["RAINFALL_THRESHOLD"].eq(float(THRESHOLD)).all():
        raise ValueError(
            f"Bootstrap table does not consistently use threshold {THRESHOLD}"
        )

    require_single_text_value(df, "PRECIPITATION_DATASET", "MSWEP")
    require_single_text_value(
        df, "PAIR_SELECTION", "BOTH_ENDPOINTS_HAVE_RAIN30"
    )
    require_single_text_value(df, "METRIC_TYPE", "W500_REVERSED_CHANGE")
    require_single_text_value(
        df,
        "INTENSITY_BOUNDARY_TYPE",
        "FIXED_REFERENCE_THRESHOLDS",
    )
    require_single_text_value(
        df, "BOOTSTRAP_METHOD", "WINDOW_LEVEL_PERCENTILE"
    )

    actual_ranges = set(
        df.loc[
            df["INTENSITY_GROUP_CODE"].eq(GROUP_CODE),
            "WIND_CHANGE_RANGE",
        ].astype(str)
    )
    if actual_ranges != {EXPECTED_WIND_RANGE}:
        raise ValueError(
            f"Unexpected wind range for {GROUP_CODE}: "
            f"{sorted(actual_ranges)}"
        )

    plot_df = df.loc[
        df["SPATIAL_TYPE"].eq(SPATIAL_TYPE)
        & df["INTENSITY_GROUP_CODE"].eq(GROUP_CODE)
        & df["DISTANCE_ZONE"].isin(DISTANCE_ZONES)
    ].copy()

    if len(plot_df) != len(DISTANCE_ZONES):
        raise ValueError(
            f"Incomplete plotting rows: found={len(plot_df)}, "
            f"expected={len(DISTANCE_ZONES)}"
        )

    duplicate = plot_df.duplicated(
        ["SPATIAL_TYPE", "INTENSITY_GROUP_CODE", "DISTANCE_ZONE"],
        keep=False,
    )
    if duplicate.any():
        raise ValueError("Duplicate RW distance-zone rows were found")

    plot_df = (
        plot_df.set_index("DISTANCE_ZONE")
        .reindex(DISTANCE_ZONES)
        .reset_index()
    )

    value_columns = [
        "MEAN_CHANGE",
        "BOOTSTRAP_CI_LOWER",
        "BOOTSTRAP_CI_UPPER",
    ]
    if plot_df[value_columns].isna().any(axis=None):
        raise ValueError("Incomplete plotting values after distance-zone ordering")

    lower = plot_df["BOOTSTRAP_CI_LOWER"].to_numpy(dtype=float)
    mean = plot_df["MEAN_CHANGE"].to_numpy(dtype=float)
    upper = plot_df["BOOTSTRAP_CI_UPPER"].to_numpy(dtype=float)
    if np.any(lower > mean) or np.any(mean > upper):
        raise ValueError(
            "A mean lies outside its bootstrap confidence interval"
        )

    if lower.min() < Y_AXIS_MIN or upper.max() > Y_AXIS_MAX:
        raise ValueError(
            "The requested fixed Y-axis would clip a confidence interval: "
            f"data range={lower.min():.6g} to {upper.max():.6g}, "
            f"axis range={Y_AXIS_MIN:.6g} to {Y_AXIS_MAX:.6g}"
        )

    print(f"Input rows: {len(df):,}")
    print(f"Selected plotting rows: {len(plot_df)}")
    print("[PASS] Plotting-data validation completed")
    return plot_df


# ============================================================
# 3. Gradient bar
# ============================================================

def draw_gradient_bar(ax, x_position, value, width, end_color):
    """Draw one bar fading smoothly from white at zero to blue at its end."""
    bar = ax.bar(
        x_position,
        value,
        width=width,
        facecolor="none",
        edgecolor="#000000",
        linewidth=BAR_EDGE_WIDTH,
        zorder=3,
    )[0]

    if np.isclose(value, 0.0):
        return

    bottom = min(0.0, value)
    top = max(0.0, value)
    colors = (
        ["#FFFFFF", end_color]
        if value > 0
        else [end_color, "#FFFFFF"]
    )
    cmap = LinearSegmentedColormap.from_list(
        f"gradient_{x_position}", colors, N=512
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
# 4. Plot
# ============================================================

def plot_figure(plot_df):
    """Create FIG2B using the same layout as the updated FIG2A."""
    x = np.arange(len(DISTANCE_ZONES))
    mean = plot_df["MEAN_CHANGE"].to_numpy(dtype=float)
    lower = plot_df["BOOTSTRAP_CI_LOWER"].to_numpy(dtype=float)
    upper = plot_df["BOOTSTRAP_CI_UPPER"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for x_position, value in zip(x, mean):
        draw_gradient_bar(
            ax=ax,
            x_position=x_position,
            value=value,
            width=BAR_WIDTH,
            end_color=GROUP_COLOR,
        )

    y_error = np.vstack((mean - lower, upper - mean))
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
        rotation=0,
        ha="center",
        fontsize=PLOT_FONT_SIZE,
        color="#000000",
    )

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel(
        r"Change in vertical velocity (Pa s$^{-1}$)",
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

    ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)
    ax.set_yticks(Y_TICKS)

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

    output_path = OUTPUT_DIR / "FIG2B.png"
    fig.savefig(
        output_path,
        dpi=FIGURE_DPI,
        facecolor="white",
    )
    plt.close(fig)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Figure output is missing or empty: {output_path}")

    print(f"Saved PNG: {output_path}")
    return output_path


# ============================================================
# 5. Main program
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = read_and_validate()

    print("\n" + "=" * 90)
    print("Creating MSWEP RW reversed-W500 FIG2B")
    print("=" * 90)

    plot_figure(plot_df)

    print("\n" + "=" * 90)
    print("FIG2B completed successfully")
    print("=" * 90)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
