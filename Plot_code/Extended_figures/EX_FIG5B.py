from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


START_YEAR = 1982
END_YEAR = 2024
THRESHOLD = "30"
FIGURE_DPI = 600
FIGURE_SIZE = (6.4, 12.8)
# Keep the formal panel at the same 1:2 width-to-height ratio as FIG5C.
FIG5B_FIGURE_SIZE = (6.4, 12.8)
PLOT_FONT_SIZE = 32
TABLE_FONT_SIZE = 12

INPUT_CSV = Path(
    str(PROJECT_ROOT / "Data" / "Processed" / "MSWEP_W500_BOOTSTRAP_RESULTS") + "/" +
    "W500_REVERSED_1982_2024_MSWEP30_BOOTSTRAP_LONG.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "Results" / "Extended_figures" / "EX_FIG5"

SPATIAL_TYPE = "NEARSHORE"
DISTANCE_ZONES = ["0_100", "100_200", "200_300", "300_400", "400_500"]
DISTANCE_LABELS = [
    "0\u2013100 km",
    "100\u2013200 km",
    "200\u2013300 km",
    "300\u2013400 km",
    "400\u2013500 km",
]

GROUPS = [
    {
        "code": "LE_5TH",
        "label": "RW",
        "expected_range": "<= -30 kt",
        "color": "#D5A0B6",
    },
    {
        "code": "EQ_50TH",
        "label": "0",
        "expected_range": "0 kt",
        "color": "#D4D4D4",
    },
    {
        "code": "GE_95TH",
        "label": "RI",
        "expected_range": ">= 30 kt",
        "color": "#D5A0B6",
    },
]

REQUIRED_COLUMNS = [
    "START_YEAR", "END_YEAR", "PRECIPITATION_DATASET", "RAINFALL_THRESHOLD",
    "PAIR_SELECTION", "SPATIAL_TYPE", "METRIC_TYPE",
    "INTENSITY_BOUNDARY_TYPE", "INTENSITY_GROUP_CODE", "WIND_CHANGE_RANGE",
    "DISTANCE_ZONE", "GROUP_WINDOW_COUNT", "COMMON_VALID_WINDOW_COUNT",
    "INVALID_WINDOW_COUNT", "MEAN_CHANGE", "BOOTSTRAP_CI_LOWER",
    "BOOTSTRAP_CI_UPPER", "BOOTSTRAP_METHOD",
]

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
        "legend.fontsize": PLOT_FONT_SIZE,
    }
)


def require_single_text_value(df, column, expected):
    actual = set(df[column].dropna().astype(str))
    if actual != {str(expected)}:
        raise ValueError(
            f"Unexpected {column}: actual={sorted(actual)}, expected={expected}"
        )


def read_and_validate():
    print("=" * 90)
    print("Reading and validating MSWEP reversed-W500 Bootstrap results")
    print("=" * 90)
    print(f"Input file: {INPUT_CSV}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    missing = set(REQUIRED_COLUMNS).difference(df.columns)
    if missing:
        raise KeyError("Input CSV is missing: " + ", ".join(sorted(missing)))

    if len(df) != 112:
        raise ValueError(f"Unexpected input row count: {len(df)}; expected 112")

    numeric_columns = [
        "START_YEAR", "END_YEAR", "RAINFALL_THRESHOLD", "GROUP_WINDOW_COUNT",
        "COMMON_VALID_WINDOW_COUNT", "INVALID_WINDOW_COUNT", "MEAN_CHANGE",
        "BOOTSTRAP_CI_LOWER", "BOOTSTRAP_CI_UPPER",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

    invalid_numeric = df[numeric_columns].isna().any(axis=1)
    if invalid_numeric.any():
        examples = df.loc[
            invalid_numeric,
            ["SPATIAL_TYPE", "INTENSITY_GROUP_CODE", "DISTANCE_ZONE"]
            + numeric_columns,
        ].head(20).to_dict("records")
        raise ValueError(
            "Missing or invalid numeric values: "
            f"count={int(invalid_numeric.sum())}; examples={examples}"
        )

    if not (df["START_YEAR"].eq(START_YEAR) & df["END_YEAR"].eq(END_YEAR)).all():
        raise ValueError(
            f"Bootstrap table does not consistently describe {START_YEAR}-{END_YEAR}"
        )
    if not df["RAINFALL_THRESHOLD"].eq(float(THRESHOLD)).all():
        raise ValueError(f"Bootstrap table does not consistently use threshold {THRESHOLD}")

    require_single_text_value(df, "PRECIPITATION_DATASET", "MSWEP")
    require_single_text_value(df, "PAIR_SELECTION", "BOTH_ENDPOINTS_HAVE_RAIN30")
    require_single_text_value(df, "METRIC_TYPE", "W500_REVERSED_CHANGE")
    require_single_text_value(
        df, "INTENSITY_BOUNDARY_TYPE", "FIXED_REFERENCE_THRESHOLDS"
    )
    require_single_text_value(df, "BOOTSTRAP_METHOD", "WINDOW_LEVEL_PERCENTILE")

    if SPATIAL_TYPE not in set(df["SPATIAL_TYPE"].astype(str)):
        raise ValueError(
            f"Input table does not contain spatial type {SPATIAL_TYPE}"
        )

    for group in GROUPS:
        actual_ranges = set(
            df.loc[
                df["INTENSITY_GROUP_CODE"] == group["code"], "WIND_CHANGE_RANGE"
            ].astype(str)
        )
        if actual_ranges != {group["expected_range"]}:
            raise ValueError(
                f"Unexpected wind range for {group['code']}: {sorted(actual_ranges)}"
            )

    selected_codes = {group["code"] for group in GROUPS}
    plot_df = df.loc[
        df["SPATIAL_TYPE"].eq(SPATIAL_TYPE)
        & df["INTENSITY_GROUP_CODE"].isin(selected_codes)
        & df["DISTANCE_ZONE"].isin(DISTANCE_ZONES)
    ].copy()

    expected_rows = len(GROUPS) * len(DISTANCE_ZONES)
    if len(plot_df) != expected_rows:
        raise ValueError(
            f"Incomplete plotting rows: found={len(plot_df)}, expected={expected_rows}"
        )

    duplicate = plot_df.duplicated(
        ["SPATIAL_TYPE", "INTENSITY_GROUP_CODE", "DISTANCE_ZONE"], keep=False
    )
    if duplicate.any():
        examples = plot_df.loc[
            duplicate,
            ["SPATIAL_TYPE", "INTENSITY_GROUP_CODE", "DISTANCE_ZONE"],
        ].head(20).to_dict("records")
        raise ValueError(f"Duplicate plotting combinations: {examples}")

    lower = plot_df["BOOTSTRAP_CI_LOWER"].to_numpy(dtype=float)
    mean = plot_df["MEAN_CHANGE"].to_numpy(dtype=float)
    upper = plot_df["BOOTSTRAP_CI_UPPER"].to_numpy(dtype=float)
    if np.any(lower > mean) or np.any(mean > upper):
        raise ValueError("A mean lies outside its Bootstrap confidence interval")

    for column in [
        "GROUP_WINDOW_COUNT", "COMMON_VALID_WINDOW_COUNT", "INVALID_WINDOW_COUNT"
    ]:
        values = plot_df[column].to_numpy(dtype=float)
        if np.any(values < 0) or not np.isclose(values, np.round(values)).all():
            raise ValueError(f"Invalid count values in {column}")
        plot_df[column] = np.round(values).astype(np.int64)

    if not np.array_equal(
        plot_df["GROUP_WINDOW_COUNT"].to_numpy(dtype=np.int64),
        (plot_df["COMMON_VALID_WINDOW_COUNT"] + plot_df["INVALID_WINDOW_COUNT"])
        .to_numpy(dtype=np.int64),
    ):
        raise ValueError("Valid and invalid counts do not sum to group count")
    if not plot_df["INVALID_WINDOW_COUNT"].eq(0).all():
        raise ValueError("Selected groups contain invalid W500 windows")

    for group in GROUPS:
        group_df = plot_df.loc[
            plot_df["INTENSITY_GROUP_CODE"] == group["code"]
        ]
        if group_df["COMMON_VALID_WINDOW_COUNT"].nunique() != 1:
            raise ValueError(
                f"Inconsistent sample count for {SPATIAL_TYPE} {group['code']}"
            )

    print(f"Input rows: {len(df)}")
    print(f"Selected plotting rows: {len(plot_df)}")
    print("[PASS] Plotting-data validation completed")
    return plot_df


def ordered_group_data(spatial_df, group_code):
    group_df = (
        spatial_df.loc[spatial_df["INTENSITY_GROUP_CODE"] == group_code]
        .set_index("DISTANCE_ZONE")
        .reindex(DISTANCE_ZONES)
    )
    required = [
        "MEAN_CHANGE", "BOOTSTRAP_CI_LOWER", "BOOTSTRAP_CI_UPPER",
        "COMMON_VALID_WINDOW_COUNT",
    ]
    if group_df[required].isna().any(axis=None):
        raise ValueError(f"Incomplete plotting data for {group_code}")
    return group_df


def format_scaled_y_tick(value, _position):
    """Format 1e-1-scaled ticks with the proper Unicode minus sign."""
    scaled = value / 1.0e-1
    if np.isclose(scaled, 0.0, atol=1.0e-12):
        return "0"
    if scaled < 0.0:
        return f"\N{MINUS SIGN}{abs(scaled):g}"
    return f"{scaled:g}"


def plot_one(spatial_df, spatial_type):
    x = np.arange(len(DISTANCE_ZONES))
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for group in GROUPS:
        group_df = ordered_group_data(spatial_df, group["code"])
        mean = group_df["MEAN_CHANGE"].to_numpy(dtype=float)
        lower = group_df["BOOTSTRAP_CI_LOWER"].to_numpy(dtype=float)
        upper = group_df["BOOTSTRAP_CI_UPPER"].to_numpy(dtype=float)

        ax.fill_between(
            x, lower, upper, color=group["color"], alpha=0.18, linewidth=0
        )
        ax.plot(
            x,
            mean,
            color=group["color"],
            linewidth=4.0,
            marker="o",
            markersize=10.0,
            label=group["label"],
        )

    ax.axhline(y=0, color="#333333", linestyle="--", linewidth=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(
        DISTANCE_LABELS,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel(
        r"Change in vertical velocity (Pa s$^{-1}$)", labelpad=16
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

    y_min = min(0.0, float(spatial_df["BOOTSTRAP_CI_LOWER"].min()))
    y_max = max(0.0, float(spatial_df["BOOTSTRAP_CI_UPPER"].max()))
    y_range = y_max - y_min
    if np.isclose(y_range, 0.0):
        y_range = 1.0
    ax.set_ylim(y_min - y_range * 0.12, y_max + y_range * 0.18)

    ax.yaxis.set_major_formatter(FuncFormatter(format_scaled_y_tick))
    ax.yaxis.get_offset_text().set_visible(False)
    ax.text(
        0.0,
        1.010,
        "1e\N{MINUS SIGN}1",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PLOT_FONT_SIZE,
        color="#000000",
        clip_on=False,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(2.0)

    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.25, top=0.97)

    path = OUTPUT_DIR / "EX_FIG5B_RW_0_RI.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Figure output is missing or empty: {path}")
    print(f"Saved PNG: {path}")
    return path


def plot_statistics_table(spatial_df, spatial_type):
    group_label_by_code = {group["code"]: group["label"] for group in GROUPS}
    group_color_by_label = {group["label"]: group["color"] for group in GROUPS}
    distance_label_by_code = dict(zip(DISTANCE_ZONES, DISTANCE_LABELS))

    table_df = spatial_df[
        [
            "INTENSITY_GROUP_CODE", "DISTANCE_ZONE", "COMMON_VALID_WINDOW_COUNT",
            "MEAN_CHANGE", "BOOTSTRAP_CI_LOWER", "BOOTSTRAP_CI_UPPER",
        ]
    ].copy()
    table_df["Group"] = table_df["INTENSITY_GROUP_CODE"].map(group_label_by_code)
    table_df["Distance zone"] = table_df["DISTANCE_ZONE"].map(
        distance_label_by_code
    )

    group_order = {group["label"]: i for i, group in enumerate(GROUPS)}
    distance_order = {zone: i for i, zone in enumerate(DISTANCE_ZONES)}
    table_df["_GROUP_ORDER"] = table_df["Group"].map(group_order)
    table_df["_DISTANCE_ORDER"] = table_df["DISTANCE_ZONE"].map(distance_order)
    table_df = table_df.sort_values(
        ["_GROUP_ORDER", "_DISTANCE_ORDER"]
    ).reset_index(drop=True)

    table_df = table_df.rename(
        columns={
            "COMMON_VALID_WINDOW_COUNT": "N",
            "MEAN_CHANGE": "Mean change",
            "BOOTSTRAP_CI_LOWER": "CI lower",
            "BOOTSTRAP_CI_UPPER": "CI upper",
        }
    )
    display_columns = [
        "Group", "Distance zone", "N", "Mean change", "CI lower", "CI upper"
    ]
    table_df = table_df[display_columns]
    table_df["N"] = table_df["N"].map(lambda value: f"{int(value):,}")
    for column in ["Mean change", "CI lower", "CI upper"]:
        table_df[column] = table_df[column].map(lambda value: f"{value:.4e}")

    fig, ax = plt.subplots(figsize=(13, 8), dpi=FIGURE_DPI)
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
    table.set_fontsize(TABLE_FONT_SIZE)
    table.scale(1.0, 1.55)

    for column_index in range(len(display_columns)):
        cell = table[(0, column_index)]
        cell.set_facecolor("#3E5266")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#2F3E4D")
        cell.set_linewidth(1.0)

    for row_index in range(1, len(table_df) + 1):
        group_label = table_df.iloc[row_index - 1]["Group"]
        for column_index in range(len(display_columns)):
            cell = table[(row_index, column_index)]
            cell.set_edgecolor("#B7B7B7")
            cell.set_linewidth(0.8)
            if column_index == 0:
                cell.set_facecolor(group_color_by_label[group_label])
                cell.set_text_props(fontweight="bold")
            elif row_index % 2 == 0:
                cell.set_facecolor("#F2F4F5")
            else:
                cell.set_facecolor("#FFFFFF")

    path = OUTPUT_DIR / "EX_FIG5B_statistics.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Table output is missing or empty: {path}")
    print(f"Saved table PNG: {path}")
    return path


def plot_nearshore_rw_auto_y(spatial_df):
    """Draw the formal nearshore RW FIG5B with a fixed upper limit."""
    group = next(item for item in GROUPS if item["code"] == "LE_5TH")
    group_df = ordered_group_data(spatial_df, group["code"])
    x = np.arange(len(DISTANCE_ZONES))
    mean = group_df["MEAN_CHANGE"].to_numpy(dtype=float)
    lower = group_df["BOOTSTRAP_CI_LOWER"].to_numpy(dtype=float)
    upper = group_df["BOOTSTRAP_CI_UPPER"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=FIG5B_FIGURE_SIZE)
    ax.fill_between(
        x, lower, upper, color=group["color"], alpha=0.18, linewidth=0
    )
    ax.plot(
        x,
        mean,
        color=group["color"],
        linewidth=4.0,
        marker="o",
        markersize=10.0,
        label=group["label"],
    )

    ax.axhline(y=0, color="#333333", linestyle="--", linewidth=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(
        DISTANCE_LABELS,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel(
        r"Change in vertical velocity (Pa s$^{-1}$)", labelpad=16
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

    # Use -4.9 so the axis retains nearly the same range without displaying
    # a -5 major tick label.
    plot_y_min = -4.9e-1
    fixed_y_max = 1.0e-1
    ax.set_ylim(plot_y_min, fixed_y_max)

    tick_step = 1.0e-1
    first_tick = np.ceil(plot_y_min / tick_step) * tick_step
    ax.set_yticks(
        np.arange(first_tick, fixed_y_max + 1.0e-12, tick_step)
    )

    ax.set_axisbelow(True)
    ax.yaxis.grid(
        True,
        color="#C4C4C4",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
    )
    ax.xaxis.grid(False)

    ax.yaxis.set_major_formatter(FuncFormatter(format_scaled_y_tick))
    ax.yaxis.get_offset_text().set_visible(False)
    ax.text(
        0.0,
        1.010,
        "1e\N{MINUS SIGN}1",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PLOT_FONT_SIZE,
        color="#000000",
        clip_on=False,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(2.0)

    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.25, top=0.96)

    fig.canvas.draw()
    ax.yaxis.get_offset_text().set_visible(False)

    path = OUTPUT_DIR / "EX_FIG5B.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Nearshore RW output is missing or empty: {path}")
    print(f"Saved nearshore RW auto-y PNG: {path}")
    return path


def plot_separate_legend():
    """Save the RW, 0, and RI legend as a standalone PNG."""
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=group["color"],
            linewidth=4.0,
            marker="o",
            markersize=10.0,
            label=group["label"],
        )
        for group in GROUPS
    ]

    fig, ax = plt.subplots(figsize=(6.4, 1.5))
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

    path = OUTPUT_DIR / "EX_FIG5B_legend.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Legend output is missing or empty: {path}")
    print(f"Saved separate legend PNG: {path}")
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = read_and_validate()

    print("\n" + "=" * 90)
    print("Creating the formal MSWEP nearshore RW reversed-W500 figure")
    print("=" * 90)

    plot_nearshore_rw_auto_y(plot_df)

    print("\n" + "=" * 90)
    print("NEARSHORE MSWEP reversed-W500 figure completed successfully")
    print("=" * 90)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
