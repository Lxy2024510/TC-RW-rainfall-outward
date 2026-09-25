#!/usr/bin/env python3
"""RUN1: Filter RW/RI events and match MSWEP rainfall around TC centers."""

from __future__ import annotations

import argparse
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


PROJECT_ROOT = Path(
    os.environ.get("TC_RW_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
SOURCE_CSV = (
    PROJECT_ROOT
    / "Data/Processed/MSWEP/Windows/24H"
    / "PRE_DATA_IBT_1982_2024_MSWEP_TH30_24H_SLIDING_ET_CLEANED.csv"
)
MSWEP_ROOT = PROJECT_ROOT / "Data/Raw/MSWEP/MSWEP_3h"
OUTPUT_ROOT = PROJECT_ROOT / "Data/Intermediate/Exposure/MSWEP_DIST30"
QC_ROOT = PROJECT_ROOT / "Results/Quality_control/Exposure/RUN1"

START_YEAR = 2000
END_YEAR = 2024
MAX_TC_DISTANCE_KM = 500.0
MIN_PRECIPITATION = 30.0
EARTH_RADIUS_KM = 6371.009
DEFAULT_WORKERS = 16

EXPECTED_COUNTS = {"RW": 2445, "RI": 1600}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter RW/RI records and extract MSWEP grid cells with precipitation "
            ">= 30 mm/3h within 500 km of S and E tropical-cyclone centers."
        )
    )
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--mswep-root", type=Path, default=MSWEP_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("RW", "RI"),
        default=["RW", "RI"],
        help="Event groups to process (default: RW RI).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of worker processes (default: 16).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute rows that already have a successful completion marker.",
    )
    return parser.parse_args()


def normalize_longitude(longitude: np.ndarray | float) -> np.ndarray | float:
    """Normalize longitude to the half-open interval [-180, 180)."""
    return (longitude + 180.0) % 360.0 - 180.0


def circular_longitude_difference(
    longitude: np.ndarray | float, reference: float
) -> np.ndarray | float:
    """Return the shortest signed longitude difference in degrees."""
    return normalize_longitude(longitude - reference)


def haversine_distance_matrix(
    center_lat: float,
    center_lon: float,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
) -> np.ndarray:
    """Calculate great-circle distances from one center to a coordinate mesh."""
    lat1 = np.radians(center_lat)
    lat2 = np.radians(grid_lat)
    dlat = lat2 - lat1
    dlon = np.radians(circular_longitude_difference(grid_lon, center_lon))

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def mswep_file_for_time(mswep_root: Path, timestamp: pd.Timestamp) -> Path:
    """Build the expected MSWEP path for a three-hour timestamp."""
    year = timestamp.strftime("%Y")
    day_of_year = timestamp.strftime("%j")
    hour = timestamp.strftime("%H")
    return mswep_root / year / f"{year}{day_of_year}.{hour}.nc"


def output_columns(prefix: str) -> list[str]:
    return [
        "SID",
        "BASIN_S",
        f"USA_LAT_{prefix}",
        f"USA_LON_{prefix}",
        "TIME",
        "RAIN_LAT",
        "RAIN_LON",
        "RAIN_VALUE",
        "RAIN_DIST",
        "RAIN_DIST_LAT",
        "RAIN_DIST_LON",
    ]


PROJECTION_COLUMNS = [
    "SID",
    "TIME",
    "USA_LAT_E",
    "USA_LON_E",
    "RAIN_LAT",
    "RAIN_LON",
    "RAIN_VALUE",
    "RAIN_DIST",
    "RAIN_DIST_LAT",
    "RAIN_DIST_LON",
]


def empty_result(prefix: str) -> pd.DataFrame:
    return pd.DataFrame(columns=output_columns(prefix))


def select_local_indices(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    center_lat: float,
    center_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a conservative bounding box before loading rainfall values."""
    latitude_margin = MAX_TC_DISTANCE_KM / 110.0 + 0.2
    maximum_abs_latitude = min(abs(center_lat) + latitude_margin, 89.9)
    cosine_limit = max(np.cos(np.radians(maximum_abs_latitude)), 0.01)
    longitude_margin = min(MAX_TC_DISTANCE_KM / (111.0 * cosine_limit) + 0.2, 180.0)

    latitude_indices = np.flatnonzero(
        np.abs(latitudes.astype(float) - center_lat) <= latitude_margin
    )
    longitude_indices = np.flatnonzero(
        np.abs(circular_longitude_difference(longitudes.astype(float), center_lon))
        <= longitude_margin
    )
    return latitude_indices, longitude_indices


def extract_rainfall_points(
    mswep_root: Path,
    tc_lat: float,
    tc_lon: float,
    iso_time: Any,
    tc_sid: Any,
    tc_basin: Any,
    prefix: str,
) -> tuple[pd.DataFrame, str, str, str]:
    """Extract qualifying MSWEP cells for one TC center and one timestamp."""
    try:
        timestamp = pd.to_datetime(iso_time, errors="raise")
        nc_path = mswep_file_for_time(mswep_root, timestamp)

        if not nc_path.is_file():
            return empty_result(prefix), "missing_nc", str(nc_path), "File not found"

        with xr.open_dataset(nc_path) as dataset:
            required = {"precipitation", "lat", "lon", "time"}
            missing = sorted(required.difference(dataset.variables))
            if missing:
                message = f"Missing NetCDF variables: {', '.join(missing)}"
                return empty_result(prefix), "invalid_nc", str(nc_path), message

            if dataset.sizes.get("time", 0) != 1:
                message = f"Expected one time value, found {dataset.sizes.get('time', 0)}"
                return empty_result(prefix), "invalid_nc", str(nc_path), message

            file_time = pd.to_datetime(dataset["time"].values[0])
            if file_time != timestamp:
                message = f"Expected time {timestamp}, found {file_time}"
                return empty_result(prefix), "time_mismatch", str(nc_path), message

            latitudes = dataset["lat"].values
            longitudes = dataset["lon"].values
            lat_idx, lon_idx = select_local_indices(
                latitudes, longitudes, float(tc_lat), float(tc_lon)
            )

            if lat_idx.size == 0 or lon_idx.size == 0:
                return empty_result(prefix), "no_rain", str(nc_path), "Empty local grid"

            rain = (
                dataset["precipitation"]
                .isel(time=0, lat=lat_idx, lon=lon_idx)
                .transpose("lat", "lon")
                .values
            )
            local_latitudes = latitudes[lat_idx].astype(float)
            local_longitudes = longitudes[lon_idx].astype(float)
            lon_mesh, lat_mesh = np.meshgrid(local_longitudes, local_latitudes)
            distances = haversine_distance_matrix(
                float(tc_lat), float(tc_lon), lat_mesh, lon_mesh
            )

            mask = (
                np.isfinite(rain)
                & (rain >= MIN_PRECIPITATION)
                & (distances <= MAX_TC_DISTANCE_KM)
            )
            valid_lat_idx, valid_lon_idx = np.where(mask)

            if valid_lat_idx.size == 0:
                return empty_result(prefix), "no_rain", str(nc_path), "No qualifying cells"

            result_lats = local_latitudes[valid_lat_idx]
            result_lons = local_longitudes[valid_lon_idx]
            result_rains = rain[valid_lat_idx, valid_lon_idx].astype(float)
            result_distances = distances[valid_lat_idx, valid_lon_idx]

            lat_diff_km = (result_lats - float(tc_lat)) * 111.132
            lon_delta = circular_longitude_difference(result_lons, float(tc_lon))
            average_latitude = np.radians((result_lats + float(tc_lat)) / 2.0)
            lon_diff_km = lon_delta * 111.320 * np.cos(average_latitude)

            result = pd.DataFrame(
                {
                    "SID": tc_sid,
                    "BASIN_S": tc_basin,
                    f"USA_LAT_{prefix}": float(tc_lat),
                    f"USA_LON_{prefix}": float(tc_lon),
                    "TIME": file_time,
                    "RAIN_LAT": result_lats,
                    "RAIN_LON": result_lons,
                    "RAIN_VALUE": result_rains,
                    "RAIN_DIST": result_distances,
                    "RAIN_DIST_LAT": lat_diff_km,
                    "RAIN_DIST_LON": lon_diff_km,
                }
            )
            result = result.sort_values("RAIN_VALUE", ascending=False).reset_index(drop=True)
            return result, "success", str(nc_path), ""

    except Exception as exc:
        nc_path_text = str(locals().get("nc_path", ""))
        message = f"{type(exc).__name__}: {exc}"
        return empty_result(prefix), "error", nc_path_text, message


def project_s_pattern_to_e(
    result_s: pd.DataFrame,
    tc_lat_e: float,
    tc_lon_e: float,
    iso_time_e: Any,
) -> pd.DataFrame:
    """Translate S rainfall offsets to the E center and E timestamp."""
    if result_s.empty:
        return pd.DataFrame(columns=PROJECTION_COLUMNS)

    projection = result_s[
        ["SID", "TIME", "RAIN_DIST_LAT", "RAIN_DIST_LON", "RAIN_VALUE"]
    ].copy()
    # S_ON_E is an end-point counterfactual.  Its population matching must
    # therefore use the LandScan year at E, including for cross-year windows.
    projection["TIME"] = pd.to_datetime(iso_time_e, errors="raise")
    projection["USA_LAT_E"] = float(tc_lat_e)
    projection["USA_LON_E"] = float(tc_lon_e)

    projection["RAIN_LAT"] = (
        projection["USA_LAT_E"] + projection["RAIN_DIST_LAT"] / 111.132
    )
    average_latitude = np.radians(
        (projection["RAIN_LAT"] + projection["USA_LAT_E"]) / 2.0
    )
    denominator = 111.320 * np.cos(average_latitude)
    projection["RAIN_LON"] = normalize_longitude(
        projection["USA_LON_E"] + projection["RAIN_DIST_LON"] / denominator
    )

    projection["RAIN_DIST"] = haversine_distance_matrix(
        float(tc_lat_e),
        float(tc_lon_e),
        projection["RAIN_LAT"].to_numpy(dtype=float),
        projection["RAIN_LON"].to_numpy(dtype=float),
    )
    return projection[PROJECTION_COLUMNS]


def write_csv_atomic(dataframe: pd.DataFrame, path: Path) -> None:
    """Write a CSV through a temporary file to avoid partial final outputs."""
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    dataframe.to_csv(temporary_path, index=False, encoding="utf-8")
    os.replace(temporary_path, path)


def write_text_atomic(text: str, path: Path) -> None:
    """Write a small text file atomically."""
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, path)


def process_one_row(task: dict[str, Any]) -> dict[str, Any]:
    """Process S, E, and S-on-E products for one filtered source row."""
    source_index = int(task["source_index"])
    row = task["row"]
    group = task["group"]
    output_dir = Path(task["output_dir"])
    mswep_root = Path(task["mswep_root"])

    path_s = output_dir / f"row_{source_index}_S.csv"
    path_e = output_dir / f"row_{source_index}_E.csv"
    path_projection = output_dir / f"row_{source_index}_S_ON_E.csv"
    done_marker = output_dir / f"row_{source_index}.done"

    log_record: dict[str, Any] = {
        "GROUP": group,
        "SOURCE_INDEX": source_index,
        "SID": row.get("SID"),
        "STATUS": "failed",
        "S_STATUS": "not_started",
        "E_STATUS": "not_started",
        "S_POINT_COUNT": 0,
        "E_POINT_COUNT": 0,
        "S_NC_FILE": "",
        "E_NC_FILE": "",
        "MESSAGE": "",
    }

    try:
        result_s, status_s, file_s, message_s = extract_rainfall_points(
            mswep_root,
            row["USA_LAT_S"],
            row["USA_LON_S"],
            row["ISO_TIME_S"],
            row["SID"],
            row["BASIN_S"],
            "S",
        )
        result_e, status_e, file_e, message_e = extract_rainfall_points(
            mswep_root,
            row["USA_LAT_E"],
            row["USA_LON_E"],
            row["ISO_TIME_E"],
            row["SID"],
            row["BASIN_S"],
            "E",
        )
        projection = project_s_pattern_to_e(
            result_s,
            row["USA_LAT_E"],
            row["USA_LON_E"],
            row["ISO_TIME_E"],
        )

        write_csv_atomic(result_s, path_s)
        write_csv_atomic(result_e, path_e)
        write_csv_atomic(projection, path_projection)

        acceptable = {"success", "no_rain"}
        task_succeeded = status_s in acceptable and status_e in acceptable
        if task_succeeded:
            write_text_atomic(
                f"Completed by RUN1_TC_RAIN_MATCHING.py at {datetime.now().isoformat()}\n",
                done_marker,
            )

        messages = [message for message in (message_s, message_e) if message]
        log_record.update(
            {
                "STATUS": "completed" if task_succeeded else "failed",
                "S_STATUS": status_s,
                "E_STATUS": status_e,
                "S_POINT_COUNT": len(result_s),
                "E_POINT_COUNT": len(result_e),
                "S_NC_FILE": file_s,
                "E_NC_FILE": file_e,
                "MESSAGE": " | ".join(messages),
            }
        )
        return log_record

    except Exception as exc:
        log_record["MESSAGE"] = (
            f"{type(exc).__name__}: {exc} | {traceback.format_exc(limit=3)}"
        )
        return log_record


def load_and_filter_events(source_csv: Path) -> dict[str, pd.DataFrame]:
    """Load the source data and apply the confirmed shared, RW, and RI filters."""
    if not source_csv.is_file():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    data = pd.read_csv(source_csv)
    required_columns = {
        "SID",
        "BASIN_S",
        "USA_LAT_S",
        "USA_LON_S",
        "ISO_TIME_S",
        "USA_LAT_E",
        "USA_LON_E",
        "ISO_TIME_E",
        "CAL_DIST_REAL_S",
        "CAL_DIST_REAL_E",
        "DIFF_USA_WIND",
    }
    missing = sorted(required_columns.difference(data.columns))
    if missing:
        raise KeyError(f"Source CSV is missing columns: {', '.join(missing)}")

    data.insert(0, "SOURCE_INDEX", data.index.astype(int))
    data["ISO_TIME_S"] = pd.to_datetime(data["ISO_TIME_S"], errors="coerce")
    data["ISO_TIME_E"] = pd.to_datetime(data["ISO_TIME_E"], errors="coerce")

    numeric_columns = [
        "USA_LAT_S",
        "USA_LON_S",
        "USA_LAT_E",
        "USA_LON_E",
        "CAL_DIST_REAL_S",
        "CAL_DIST_REAL_E",
        "DIFF_USA_WIND",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    base_mask = (
        data["ISO_TIME_S"].dt.year.between(START_YEAR, END_YEAR, inclusive="both")
        & data["ISO_TIME_E"].dt.year.between(START_YEAR, END_YEAR, inclusive="both")
        & data["CAL_DIST_REAL_S"].le(MAX_TC_DISTANCE_KM)
        & data["CAL_DIST_REAL_E"].le(MAX_TC_DISTANCE_KM)
    )

    return {
        "RW": data.loc[base_mask & data["DIFF_USA_WIND"].le(-30)].copy(),
        "RI": data.loc[base_mask & data["DIFF_USA_WIND"].ge(30)].copy(),
    }


def save_filtered_table(data: pd.DataFrame, path: Path) -> None:
    """Save the filtered event table with ISO timestamps in standard text format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(data, path)


def build_tasks(
    group: str,
    filtered: pd.DataFrame,
    output_dir: Path,
    mswep_root: Path,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Create worker tasks while honoring successful completion markers."""
    tasks: list[dict[str, Any]] = []
    skipped = 0

    for row in filtered.to_dict(orient="records"):
        source_index = int(row["SOURCE_INDEX"])
        done_marker = output_dir / f"row_{source_index}.done"
        if done_marker.is_file() and not overwrite:
            skipped += 1
            continue

        tasks.append(
            {
                "source_index": source_index,
                "row": row,
                "group": group,
                "output_dir": str(output_dir),
                "mswep_root": str(mswep_root),
            }
        )
    return tasks, skipped


def run_group(
    group: str,
    filtered: pd.DataFrame,
    output_root: Path,
    mswep_root: Path,
    workers: int,
    overwrite: bool,
) -> None:
    """Run all pending tasks for one event group and save a run log."""
    group_dir = output_root / group
    result_dir = group_dir / "RAIN_POINT_MATCH"
    log_dir = QC_ROOT / group
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    filtered_path = group_dir / f"FILTERED_{group}.csv"
    save_filtered_table(filtered, filtered_path)

    actual_count = len(filtered)
    expected_count = EXPECTED_COUNTS[group]
    print(f"[{group}] Filtered rows: {actual_count} (expected: {expected_count})")
    if actual_count != expected_count:
        print(f"[{group}] WARNING: Filtered count differs from the confirmed count.")

    tasks, skipped = build_tasks(
        group, filtered, result_dir, mswep_root, overwrite
    )
    print(f"[{group}] Completed rows skipped: {skipped}")
    print(f"[{group}] Rows submitted: {len(tasks)}")

    if not tasks:
        print(f"[{group}] No pending work.")
        return

    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(process_one_row, task): task["source_index"]
            for task in tasks
        }
        progress = tqdm(
            as_completed(future_to_index),
            total=len(future_to_index),
            desc=f"{group} TC-rain matching",
        )
        for future in progress:
            source_index = future_to_index[future]
            try:
                records.append(future.result())
            except Exception as exc:
                records.append(
                    {
                        "GROUP": group,
                        "SOURCE_INDEX": source_index,
                        "SID": "",
                        "STATUS": "failed",
                        "S_STATUS": "worker_crash",
                        "E_STATUS": "worker_crash",
                        "S_POINT_COUNT": 0,
                        "E_POINT_COUNT": 0,
                        "S_NC_FILE": "",
                        "E_NC_FILE": "",
                        "MESSAGE": f"{type(exc).__name__}: {exc}",
                    }
                )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"RUN1_{group}_task_log_{run_stamp}.csv"
    log_data = pd.DataFrame(records).sort_values("SOURCE_INDEX")
    write_csv_atomic(log_data, log_path)

    completed = int((log_data["STATUS"] == "completed").sum())
    failed = int((log_data["STATUS"] == "failed").sum())
    print(f"[{group}] Completed in this run: {completed}")
    print(f"[{group}] Failed in this run: {failed}")
    print(f"[{group}] Log: {log_path}")


def main() -> None:
    args = parse_arguments()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not args.mswep_root.is_dir():
        raise FileNotFoundError(f"MSWEP root not found: {args.mswep_root}")

    print(f"Source CSV: {args.source_csv}")
    print(f"MSWEP root: {args.mswep_root}")
    print(f"Output root: {args.output_root}")
    print(f"Workers: {args.workers}")

    filtered_groups = load_and_filter_events(args.source_csv)
    for group in args.groups:
        run_group(
            group=group,
            filtered=filtered_groups[group],
            output_root=args.output_root,
            mswep_root=args.mswep_root,
            workers=args.workers,
            overwrite=args.overwrite,
        )

    print("RUN1 finished.")


if __name__ == "__main__":
    main()
