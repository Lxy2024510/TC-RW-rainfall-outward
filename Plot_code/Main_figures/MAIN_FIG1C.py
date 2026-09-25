"""Build 24-hour rapid-weakening segments and a 5-degree global map.

The script processes only the MSWEP threshold-30 table. Each candidate
segment must contain exact observations at T0, T0+12 h, and T0+24 h for the
same storm. Rapid weakening (RW) is defined as a USA_WIND decrease of at
least 30 kt over 24 hours.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "Data"
    / "Processed"
    / "MSWEP"
    / "Thresholds"
    / "PRE_DATA_IBT_1982_2024_MSWEP_THRESHOLD_30.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Results" / "Main_figures" / "FIG1"

TIME_TO_MIDDLE_HOURS = 12
TIME_TO_END_HOURS = 24
RW_WIND_CHANGE_LIMIT = -30.0
GRID_SIZE_DEGREES = 5.0
MIN_SEGMENTS_PER_GRID = 10

# Spatial smoothing is applied to counts, not directly to probabilities.
# One grid-cell standard deviation corresponds to approximately 5 degrees.
GAUSSIAN_SIGMA_GRID_CELLS = 1.0
MIN_SMOOTHED_TOTAL_WEIGHT = 0.75

DEEP_RED = "#DEAEC3"
LIGHT_RED = "#FBC8DB"
DEEP_BLUE = "#56A1BB"
LAND_GRAY = "#D9D9D9"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract threshold-30 RW segments and draw a 5-degree global "
            "map of positive MSWEP distance-change proportions."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the threshold-30 source CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output CSV and map files.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution in dots per inch (default: 600).",
    )
    return parser.parse_args()


def validate_and_prepare_source(input_path: Path) -> pd.DataFrame:
    """Read the source table and validate fields needed by this analysis."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    df = pd.read_csv(
        input_path,
        dtype={"SID": "string", "NAME": "string"},
        low_memory=False,
    )

    required_columns = [
        "ROW_ID",
        "SID",
        "NAME",
        "ISO_TIME",
        "USA_LAT",
        "USA_LON",
        "USA_WIND",
        "MSWEP_DIST_30",
    ]
    missing = sorted(set(required_columns).difference(df.columns))
    if missing:
        raise KeyError("Missing required columns: " + ", ".join(missing))

    df["ROW_ID"] = pd.to_numeric(df["ROW_ID"], errors="coerce")
    if df["ROW_ID"].isna().any():
        raise ValueError("ROW_ID contains missing or non-numeric values.")
    df["ROW_ID"] = df["ROW_ID"].astype(np.int64)
    if df["ROW_ID"].duplicated().any():
        raise ValueError("ROW_ID contains duplicate values.")

    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    if df["ISO_TIME"].isna().any():
        raise ValueError("ISO_TIME contains missing or invalid timestamps.")

    exact_3h = (
        df["ISO_TIME"].dt.hour.isin(range(0, 24, 3))
        & df["ISO_TIME"].dt.minute.eq(0)
        & df["ISO_TIME"].dt.second.eq(0)
        & df["ISO_TIME"].dt.microsecond.eq(0)
    )
    if not exact_3h.all():
        raise ValueError(
            f"ISO_TIME contains {int((~exact_3h).sum())} non-exact "
            "3-hourly timestamps."
        )

    if df.duplicated(["SID", "ISO_TIME"]).any():
        raise ValueError("Duplicate (SID, ISO_TIME) keys were found.")

    numeric_columns = [
        "USA_LAT",
        "USA_LON",
        "USA_WIND",
        "MSWEP_DIST_30",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

    return df.sort_values(["SID", "ISO_TIME", "ROW_ID"]).reset_index(drop=True)


def build_three_time_segments(source_df: pd.DataFrame) -> pd.DataFrame:
    """Build exact T0, T12, and T24 records for the same storm."""
    value_columns = [
        "ROW_ID",
        "ISO_TIME",
        "USA_LAT",
        "USA_LON",
        "USA_WIND",
        "MSWEP_DIST_30",
    ]

    t0 = source_df[["SID", "NAME"] + value_columns].copy()
    t0["TARGET_TIME_T12"] = t0["ISO_TIME"] + pd.Timedelta(
        hours=TIME_TO_MIDDLE_HOURS
    )
    t0["TARGET_TIME_T24"] = t0["ISO_TIME"] + pd.Timedelta(
        hours=TIME_TO_END_HOURS
    )
    t0 = t0.rename(columns={column: f"{column}_T0" for column in value_columns})

    t12 = source_df[["SID"] + value_columns].copy().rename(
        columns={column: f"{column}_T12" for column in value_columns}
    )
    segments = t0.merge(
        t12,
        left_on=["SID", "TARGET_TIME_T12"],
        right_on=["SID", "ISO_TIME_T12"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    t24 = source_df[["SID"] + value_columns].copy().rename(
        columns={column: f"{column}_T24" for column in value_columns}
    )
    segments = segments.merge(
        t24,
        left_on=["SID", "TARGET_TIME_T24"],
        right_on=["SID", "ISO_TIME_T24"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    segments = segments.drop(columns=["TARGET_TIME_T12", "TARGET_TIME_T24"])

    segments["DIFF_USA_WIND"] = (
        segments["USA_WIND_T24"] - segments["USA_WIND_T0"]
    )
    segments["DIFF_MSWEP_DIST_30"] = (
        segments["MSWEP_DIST_30_T24"] - segments["MSWEP_DIST_30_T0"]
    )

    return segments.sort_values(["ROW_ID_T0", "ROW_ID_T24"]).reset_index(drop=True)


def select_rw_segments(segments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select RW segments and identify rows eligible for spatial analysis."""
    wind_valid = segments[["USA_WIND_T0", "USA_WIND_T24"]].notna().all(axis=1)
    rw_mask = wind_valid & segments["DIFF_USA_WIND"].le(RW_WIND_CHANGE_LIMIT)
    rw_all = segments.loc[rw_mask].copy().reset_index(drop=True)

    rw_all["DIST_VALID_FLAG"] = (
        rw_all[["MSWEP_DIST_30_T0", "MSWEP_DIST_30_T24"]]
        .notna()
        .all(axis=1)
    )
    rw_all["LOCATION_VALID_FLAG"] = (
        rw_all[["USA_LAT_T12", "USA_LON_T12"]].notna().all(axis=1)
        & rw_all["USA_LAT_T12"].between(-90.0, 90.0, inclusive="both")
        & rw_all["USA_LON_T12"].between(-180.0, 360.0, inclusive="both")
    )
    rw_all["USED_IN_MAP_FLAG"] = (
        rw_all["DIST_VALID_FLAG"] & rw_all["LOCATION_VALID_FLAG"]
    )
    rw_all["DIST_CHANGE_SIGN"] = pd.Series(pd.NA, index=rw_all.index, dtype="string")
    valid_distance = rw_all["DIST_VALID_FLAG"]
    rw_all.loc[
        valid_distance & rw_all["DIFF_MSWEP_DIST_30"].gt(0),
        "DIST_CHANGE_SIGN",
    ] = "POSITIVE"
    rw_all.loc[
        valid_distance & rw_all["DIFF_MSWEP_DIST_30"].lt(0),
        "DIST_CHANGE_SIGN",
    ] = "NEGATIVE"
    rw_all.loc[
        valid_distance & rw_all["DIFF_MSWEP_DIST_30"].eq(0),
        "DIST_CHANGE_SIGN",
    ] = "ZERO"

    rw_cleaned = rw_all.loc[rw_all["USED_IN_MAP_FLAG"]].copy().reset_index(drop=True)
    return rw_all, rw_cleaned


def normalize_longitude(longitude: pd.Series) -> pd.Series:
    """Convert longitudes to the [0, 360) convention."""
    return longitude % 360.0


def add_grid_coordinates(rw_cleaned: pd.DataFrame) -> pd.DataFrame:
    """Assign each RW segment to a 5-degree cell using its T12 location."""
    result = rw_cleaned.copy()
    result["USA_LON_T12_NORMALIZED"] = normalize_longitude(result["USA_LON_T12"])

    lon_min = (
        np.floor(result["USA_LON_T12_NORMALIZED"] / GRID_SIZE_DEGREES)
        * GRID_SIZE_DEGREES
    )
    lat_for_bin = result["USA_LAT_T12"].clip(
        lower=-90.0,
        upper=np.nextafter(90.0, -np.inf),
    )
    lat_min = np.floor(lat_for_bin / GRID_SIZE_DEGREES) * GRID_SIZE_DEGREES

    result["GRID_LON_MIN"] = lon_min
    result["GRID_LON_MAX"] = lon_min + GRID_SIZE_DEGREES
    result["GRID_LAT_MIN"] = lat_min
    result["GRID_LAT_MAX"] = lat_min + GRID_SIZE_DEGREES
    result["GRID_CENTER_LON"] = lon_min + GRID_SIZE_DEGREES / 2.0
    result["GRID_CENTER_LAT"] = lat_min + GRID_SIZE_DEGREES / 2.0
    return result


def classify_probability(positive_count: int, total_count: int) -> tuple[str, str]:
    """Return one of the three requested probability classes."""
    probability = positive_count / total_count
    if probability < 0.50:
        return "0.00 <= P < 0.50", DEEP_BLUE
    if probability < 0.70:
        return "0.50 <= P < 0.70", LIGHT_RED
    return "0.70 <= P <= 1.00", DEEP_RED


def build_grid_statistics(rw_cleaned: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid RW segments into 5-degree grid cells."""
    grouped = rw_cleaned.groupby(
        ["GRID_LON_MIN", "GRID_LAT_MIN"],
        as_index=False,
        observed=True,
    ).agg(
        RW_COUNT=("SID", "size"),
        POSITIVE_COUNT=("DIFF_MSWEP_DIST_30", lambda values: int(values.gt(0).sum())),
        NEGATIVE_COUNT=("DIFF_MSWEP_DIST_30", lambda values: int(values.lt(0).sum())),
        ZERO_COUNT=("DIFF_MSWEP_DIST_30", lambda values: int(values.eq(0).sum())),
    )

    grouped["GRID_LON_MAX"] = grouped["GRID_LON_MIN"] + GRID_SIZE_DEGREES
    grouped["GRID_LAT_MAX"] = grouped["GRID_LAT_MIN"] + GRID_SIZE_DEGREES
    grouped["GRID_CENTER_LON"] = grouped["GRID_LON_MIN"] + GRID_SIZE_DEGREES / 2.0
    grouped["GRID_CENTER_LAT"] = grouped["GRID_LAT_MIN"] + GRID_SIZE_DEGREES / 2.0
    grouped["P"] = grouped["POSITIVE_COUNT"] / grouped["RW_COUNT"]
    grouped["MEETS_MIN_SAMPLE"] = grouped["RW_COUNT"].ge(MIN_SEGMENTS_PER_GRID)
    grouped["P_CLASS"] = pd.Series(pd.NA, index=grouped.index, dtype="string")
    grouped["COLOR_HEX"] = pd.Series(pd.NA, index=grouped.index, dtype="string")

    for index, row in grouped.loc[grouped["MEETS_MIN_SAMPLE"]].iterrows():
        label, color = classify_probability(
            int(row["POSITIVE_COUNT"]), int(row["RW_COUNT"])
        )
        grouped.at[index, "P_CLASS"] = label
        grouped.at[index, "COLOR_HEX"] = color

    ordered_columns = [
        "GRID_LON_MIN",
        "GRID_LON_MAX",
        "GRID_LAT_MIN",
        "GRID_LAT_MAX",
        "GRID_CENTER_LON",
        "GRID_CENTER_LAT",
        "RW_COUNT",
        "POSITIVE_COUNT",
        "NEGATIVE_COUNT",
        "ZERO_COUNT",
        "P",
        "MEETS_MIN_SAMPLE",
        "P_CLASS",
        "COLOR_HEX",
    ]
    return grouped[ordered_columns].sort_values(
        ["GRID_LAT_MIN", "GRID_LON_MIN"]
    ).reset_index(drop=True)


def build_smoothed_probability_field(
    grid_stats: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Smooth positive and total counts separately on the 5-degree grid."""
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as error:
        raise ImportError(
            "Spatial smoothing requires scipy. Install it with "
            "'conda install scipy' or an equivalent command."
        ) from error

    longitude_edges = np.arange(
        0.0,
        360.0 + GRID_SIZE_DEGREES,
        GRID_SIZE_DEGREES,
    )
    latitude_edges = np.arange(
        -90.0,
        90.0 + GRID_SIZE_DEGREES,
        GRID_SIZE_DEGREES,
    )
    grid_shape = (
        latitude_edges.size - 1,
        longitude_edges.size - 1,
    )
    positive_counts = np.zeros(grid_shape, dtype=float)
    total_counts = np.zeros(grid_shape, dtype=float)

    for row in grid_stats.itertuples(index=False):
        longitude_index = int(
            np.floor(row.GRID_LON_MIN / GRID_SIZE_DEGREES)
        ) % grid_shape[1]
        latitude_index = int(
            np.floor((row.GRID_LAT_MIN + 90.0) / GRID_SIZE_DEGREES)
        )
        if not 0 <= latitude_index < grid_shape[0]:
            raise ValueError(
                f"Grid latitude is outside the raster: {row.GRID_LAT_MIN}"
            )
        positive_counts[latitude_index, longitude_index] = row.POSITIVE_COUNT
        total_counts[latitude_index, longitude_index] = row.RW_COUNT

    # Latitude is bounded, whereas longitude is periodic across 0/360 degrees.
    smoothing_mode = ("constant", "wrap")
    smoothed_positive = gaussian_filter(
        positive_counts,
        sigma=GAUSSIAN_SIGMA_GRID_CELLS,
        mode=smoothing_mode,
        cval=0.0,
    )
    smoothed_total = gaussian_filter(
        total_counts,
        sigma=GAUSSIAN_SIGMA_GRID_CELLS,
        mode=smoothing_mode,
        cval=0.0,
    )

    smoothed_probability = np.full(grid_shape, np.nan, dtype=float)
    displayed = smoothed_total >= MIN_SMOOTHED_TOTAL_WEIGHT
    smoothed_probability[displayed] = (
        smoothed_positive[displayed] / smoothed_total[displayed]
    )
    smoothed_probability = np.clip(smoothed_probability, 0.0, 1.0)

    longitude_centers = (
        longitude_edges[:-1] + GRID_SIZE_DEGREES / 2.0
    )
    latitude_centers = latitude_edges[:-1] + GRID_SIZE_DEGREES / 2.0
    records = []
    for latitude_index, latitude in enumerate(latitude_centers):
        for longitude_index, longitude in enumerate(longitude_centers):
            records.append(
                {
                    "GRID_CENTER_LON": longitude,
                    "GRID_CENTER_LAT": latitude,
                    "SMOOTHED_POSITIVE_WEIGHT": smoothed_positive[
                        latitude_index, longitude_index
                    ],
                    "SMOOTHED_TOTAL_WEIGHT": smoothed_total[
                        latitude_index, longitude_index
                    ],
                    "SMOOTHED_P": smoothed_probability[
                        latitude_index, longitude_index
                    ],
                    "DISPLAYED": bool(displayed[
                        latitude_index, longitude_index
                    ]),
                    "GAUSSIAN_SIGMA_GRID_CELLS": (
                        GAUSSIAN_SIGMA_GRID_CELLS
                    ),
                    "MIN_SMOOTHED_TOTAL_WEIGHT": (
                        MIN_SMOOTHED_TOTAL_WEIGHT
                    ),
                }
            )

    return (
        longitude_centers,
        latitude_centers,
        smoothed_probability,
        pd.DataFrame(records),
    )


def draw_global_map(
    grid_stats: pd.DataFrame,
    png_path: Path,
    dpi: int,
) -> None:
    """Draw the original 5-degree grid cells without spatial smoothing."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch, Rectangle
    except ImportError as error:
        raise ImportError(
            "Map drawing requires matplotlib and cartopy. Install them with "
            "'conda install -c conda-forge matplotlib cartopy' or an equivalent "
            "environment command."
        ) from error

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.44,
            "axes.linewidth": 0.5,
        }
    )

    # Increase only the outer canvas height to accommodate the legend below
    # the map while preserving the original map-frame width and height.
    original_height = 2.9
    figure_height = 3.45
    original_axes_bottom_in = 0.15 * original_height
    original_axes_height_in = 0.79 * original_height
    added_bottom_space_in = figure_height - original_height

    fig = plt.figure(figsize=(7.2, figure_height), facecolor="white")
    data_crs = ccrs.PlateCarree()
    projection = ccrs.PlateCarree(central_longitude=180)
    ax = fig.add_axes(
        [
            0.065,
            (original_axes_bottom_in + added_bottom_space_in) / figure_height,
            0.91,
            original_axes_height_in / figure_height,
        ],
        projection=projection,
    )
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-50.0, 50.0)
    ax.set_facecolor("#FFFFFF")
    # Keep land below the grid cells so valid colored cells remain visible.
    ax.add_feature(cfeature.LAND, facecolor=LAND_GRAY, edgecolor="none", zorder=0)
    ax.add_feature(
        cfeature.COASTLINE.with_scale("110m"),
        edgecolor="#6E716F",
        linewidth=0.45,
        zorder=4,
    )
    ax.spines["geo"].set_edgecolor("#000000")
    ax.spines["geo"].set_linewidth(0.50)

    projected_longitudes = [-120, -60, 0, 60, 120]
    longitude_labels = [
        "60°E",
        "120°E",
        "180°",
        "120°W",
        "60°W",
    ]
    latitude_ticks = [-30, 0, 30]
    latitude_labels = ["30°S", "0°", "30°N"]
    ax.set_xticks(projected_longitudes)
    ax.set_xticklabels(longitude_labels)
    ax.set_yticks(latitude_ticks)
    ax.set_yticklabels(latitude_labels)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=2.2,
        width=0.45,
        pad=2.5,
        labelsize=6.59,
        colors="#000000",
        top=False,
        right=False,
    )

    visible_cells = grid_stats.loc[grid_stats["MEETS_MIN_SAMPLE"]]
    for row in visible_cells.itertuples(index=False):
        ax.add_patch(
            Rectangle(
                (row.GRID_LON_MIN, row.GRID_LAT_MIN),
                GRID_SIZE_DEGREES,
                GRID_SIZE_DEGREES,
                facecolor=row.COLOR_HEX,
                edgecolor="none",
                linewidth=0,
                transform=data_crs,
                zorder=2,
            )
        )

    legend_handles = [
        Patch(
            facecolor=DEEP_BLUE,
            edgecolor="#444444",
            linewidth=0.45,
            label=r"$P(\Delta R30 > 0) < 0.50$",
        ),
        Patch(
            facecolor=LIGHT_RED,
            edgecolor="#444444",
            linewidth=0.45,
            label=r"$0.50 \leq P(\Delta R30 > 0) < 0.70$",
        ),
        Patch(
            facecolor=DEEP_RED,
            edgecolor="#444444",
            linewidth=0.45,
            label=r"$P(\Delta R30 > 0) \geq 0.70$",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.225),
        ncol=3,
        frameon=False,
        fontsize=6.59,
        handlelength=2.2,
        handleheight=0.55,
        handletextpad=0.75,
        columnspacing=2.6,
        borderaxespad=0.0,
    )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    """Run the complete threshold-30 RW analysis."""
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "FIG1C.png"

    print(f"Reading: {args.input}")
    source_df = validate_and_prepare_source(args.input)
    print(f"Source rows: {len(source_df):,}")

    segments = build_three_time_segments(source_df)
    print(f"Complete T0-T12-T24 segments: {len(segments):,}")

    rw_all, rw_cleaned = select_rw_segments(segments)
    rw_cleaned = add_grid_coordinates(rw_cleaned)
    print(f"All RW segments: {len(rw_all):,}")
    print(f"RW segments eligible for mapping: {len(rw_cleaned):,}")

    grid_stats = build_grid_statistics(rw_cleaned)
    visible_grid_count = int(grid_stats["MEETS_MIN_SAMPLE"].sum())
    print(f"Grid cells with at least {MIN_SEGMENTS_PER_GRID} segments: {visible_grid_count:,}")

    draw_global_map(
        grid_stats,
        png_path,
        args.dpi,
    )

    print("Processing completed successfully.")
    print(f"PNG map: {png_path}")


if __name__ == "__main__":
    main()
