#!/usr/bin/env python3
"""RUN2: Match RUN1 rainfall points with LandScan population grid cells."""

from __future__ import annotations

import argparse
import atexit
import os
import traceback
from collections import OrderedDict
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
INPUT_ROOT = PROJECT_ROOT / "Data/Intermediate/Exposure/MSWEP_DIST30"
LANDSCAN_ROOT = PROJECT_ROOT / "Data/Raw/LandScan"
QC_ROOT = PROJECT_ROOT / "Results/Quality_control/Exposure/RUN2"
DEFAULT_WORKERS = 16
BUFFER_DEGREES = 0.05
GROUPS = ("RW", "RI")
EXPECTED_INPUT_COUNTS = {"RW": 7335, "RI": 4800}
POPULATION_COLUMNS = ["POP_LAT", "POP_LON", "POP_VALUE"]

# Each worker keeps a small LRU cache of open annual LandScan datasets.
_DATASET_CACHE: OrderedDict[str, xr.Dataset] = OrderedDict()
_MAX_CACHED_DATASETS = 3


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand each RUN1 rainfall point to all non-NaN LandScan grid cells "
            "within a +/-0.05 degree square."
        )
    )
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--landscan-root", type=Path, default=LANDSCAN_ROOT)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=GROUPS,
        default=list(GROUPS),
        help="Event groups to process (default: RW RI).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of worker processes (default: 16).",
    )
    parser.add_argument(
        "--buffer-degrees",
        type=float,
        default=BUFFER_DEGREES,
        help="Half-width of the population search square (default: 0.05).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute files that already have a successful RUN2 marker.",
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


def close_cached_datasets() -> None:
    """Close all LandScan datasets cached by the current worker."""
    while _DATASET_CACHE:
        _, dataset = _DATASET_CACHE.popitem(last=False)
        try:
            dataset.close()
        except Exception:
            pass


atexit.register(close_cached_datasets)


def get_landscan_dataset(path: Path) -> xr.Dataset:
    """Return an open LandScan dataset from the worker-local LRU cache."""
    cache_key = str(path)
    if cache_key in _DATASET_CACHE:
        dataset = _DATASET_CACHE.pop(cache_key)
        _DATASET_CACHE[cache_key] = dataset
        return dataset

    dataset = xr.open_dataset(path)
    required = {"population", "lat", "lon"}
    missing = sorted(required.difference(dataset.variables))
    if missing:
        dataset.close()
        raise KeyError(f"Missing LandScan variables: {', '.join(missing)}")

    population = dataset["population"].squeeze(drop=True)
    if set(population.dims) != {"lat", "lon"}:
        dataset.close()
        raise ValueError(
            f"Expected population dimensions lat/lon, found {population.dims}"
        )

    _DATASET_CACHE[cache_key] = dataset
    while len(_DATASET_CACHE) > _MAX_CACHED_DATASETS:
        _, old_dataset = _DATASET_CACHE.popitem(last=False)
        old_dataset.close()
    return dataset


def write_csv_atomic(dataframe: pd.DataFrame, path: Path) -> None:
    """Write a CSV through a temporary file to prevent partial final outputs."""
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    dataframe.to_csv(temporary_path, index=False, encoding="utf-8")
    os.replace(temporary_path, path)


def write_text_atomic(text: str, path: Path) -> None:
    """Write a small text file atomically."""
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, path)


def population_file_for_year(landscan_root: Path, year: int) -> Path:
    """Build the expected annual LandScan file path."""
    return landscan_root / f"landscan-global-{year}.nc"


def extract_population_cells(
    dataset: xr.Dataset,
    rain_lat: float,
    rain_lon: float,
    buffer_degrees: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract all non-NaN population cells in a square around one rain point."""
    latitudes = dataset["lat"].values.astype(float)
    longitudes = dataset["lon"].values.astype(float)

    lat_indices = np.flatnonzero(
        np.abs(latitudes - rain_lat) <= buffer_degrees + 1.0e-10
    )
    lon_indices = np.flatnonzero(
        np.abs(circular_longitude_difference(longitudes, rain_lon))
        <= buffer_degrees + 1.0e-10
    )

    if lat_indices.size == 0 or lon_indices.size == 0:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )

    population = (
        dataset["population"]
        .squeeze(drop=True)
        .transpose("lat", "lon")
        .isel(lat=lat_indices, lon=lon_indices)
        .values
    )
    local_latitudes = latitudes[lat_indices]
    local_longitudes = longitudes[lon_indices]
    lon_mesh, lat_mesh = np.meshgrid(local_longitudes, local_latitudes)
    valid = ~pd.isna(population)

    return (
        lat_mesh[valid].astype(float),
        lon_mesh[valid].astype(float),
        np.asarray(population[valid]),
    )


def expand_rainfall_row(
    row: pd.Series,
    dataset: xr.Dataset,
    buffer_degrees: float,
) -> tuple[pd.DataFrame, bool]:
    """Cross one rainfall row with matching population cells."""
    rain_lat = float(row["RAIN_LAT"])
    rain_lon = float(row["RAIN_LON"])
    pop_lat, pop_lon, pop_value = extract_population_cells(
        dataset, rain_lat, rain_lon, buffer_degrees
    )

    fallback_used = pop_value.size == 0
    if fallback_used:
        pop_lat = np.array([rain_lat], dtype=float)
        pop_lon = np.array([rain_lon], dtype=float)
        pop_value = np.array([0])

    repeated = pd.DataFrame(
        np.repeat(row.to_numpy()[None, :], pop_value.size, axis=0),
        columns=row.index,
    )
    repeated["POP_LAT"] = pop_lat
    repeated["POP_LON"] = pop_lon
    repeated["POP_VALUE"] = pop_value
    return repeated[list(row.index) + POPULATION_COLUMNS], fallback_used


def process_one_csv(task: dict[str, Any]) -> dict[str, Any]:
    """Process one RUN1 CSV and write its population-expanded counterpart."""
    input_path = Path(task["input_path"])
    output_path = Path(task["output_path"])
    done_marker = Path(task["done_marker"])
    landscan_root = Path(task["landscan_root"])
    group = task["group"]
    buffer_degrees = float(task["buffer_degrees"])

    record: dict[str, Any] = {
        "GROUP": group,
        "INPUT_FILE": input_path.name,
        "STATUS": "failed",
        "YEAR": "",
        "RAIN_ROW_COUNT": 0,
        "OUTPUT_ROW_COUNT": 0,
        "OCEAN_FALLBACK_COUNT": 0,
        "LANDSCAN_FILE": "",
        "MESSAGE": "",
    }

    try:
        rain_data = pd.read_csv(input_path)
        record["RAIN_ROW_COUNT"] = len(rain_data)

        missing_columns = sorted({"TIME", "RAIN_LAT", "RAIN_LON"} - set(rain_data.columns))
        if missing_columns:
            raise KeyError(f"Missing input columns: {', '.join(missing_columns)}")

        # Preserve the legacy behavior for empty rainfall files: keep their headers.
        if rain_data.empty:
            write_csv_atomic(rain_data, output_path)
            write_text_atomic(
                f"Completed by RUN2_LANDSCAN_POPULATION_MATCHING.py at "
                f"{datetime.now().isoformat()}\n",
                done_marker,
            )
            record["STATUS"] = "completed_empty"
            return record

        first_time = pd.to_datetime(rain_data.iloc[0]["TIME"], errors="raise")
        year = int(first_time.year)
        landscan_path = population_file_for_year(landscan_root, year)
        record["YEAR"] = year
        record["LANDSCAN_FILE"] = str(landscan_path)

        if not landscan_path.is_file():
            raise FileNotFoundError(f"LandScan file not found: {landscan_path}")

        # All rows in one RUN1 file are expected to share the same timestamp year.
        parsed_years = pd.to_datetime(rain_data["TIME"], errors="coerce").dt.year
        if parsed_years.isna().any() or not parsed_years.eq(year).all():
            raise ValueError("The input CSV contains invalid or mixed TIME years")

        rain_data["RAIN_LAT"] = pd.to_numeric(rain_data["RAIN_LAT"], errors="coerce")
        rain_data["RAIN_LON"] = pd.to_numeric(rain_data["RAIN_LON"], errors="coerce")
        if rain_data[["RAIN_LAT", "RAIN_LON"]].isna().any().any():
            raise ValueError("The input CSV contains invalid rainfall coordinates")

        dataset = get_landscan_dataset(landscan_path)
        expanded_rows: list[pd.DataFrame] = []
        fallback_count = 0

        for _, row in rain_data.iterrows():
            expanded, fallback_used = expand_rainfall_row(
                row, dataset, buffer_degrees
            )
            expanded_rows.append(expanded)
            fallback_count += int(fallback_used)

        final_data = pd.concat(expanded_rows, ignore_index=True)
        write_csv_atomic(final_data, output_path)
        write_text_atomic(
            f"Completed by RUN2_LANDSCAN_POPULATION_MATCHING.py at "
            f"{datetime.now().isoformat()}\n",
            done_marker,
        )

        record.update(
            {
                "STATUS": "completed",
                "OUTPUT_ROW_COUNT": len(final_data),
                "OCEAN_FALLBACK_COUNT": fallback_count,
            }
        )
        return record

    except Exception as exc:
        record["MESSAGE"] = (
            f"{type(exc).__name__}: {exc} | {traceback.format_exc(limit=3)}"
        )
        return record


def discover_input_files(input_dir: Path) -> list[Path]:
    """Return only the three RUN1 rainfall CSV product types."""
    files = []
    for path in input_dir.glob("row_*.csv"):
        name = path.name
        if name.endswith(("_S.csv", "_E.csv", "_S_ON_E.csv")):
            files.append(path)
    return sorted(files, key=lambda path: path.name)


def build_tasks(
    group: str,
    input_files: list[Path],
    output_dir: Path,
    landscan_root: Path,
    buffer_degrees: float,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Create pending file tasks and skip files with RUN2 completion markers."""
    tasks: list[dict[str, Any]] = []
    skipped = 0

    for input_path in input_files:
        output_path = output_dir / input_path.name
        done_marker = output_dir / f"{input_path.name}.done"
        if done_marker.is_file() and not overwrite:
            skipped += 1
            continue
        tasks.append(
            {
                "group": group,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "done_marker": str(done_marker),
                "landscan_root": str(landscan_root),
                "buffer_degrees": buffer_degrees,
            }
        )
    return tasks, skipped


def run_group(
    group: str,
    input_root: Path,
    landscan_root: Path,
    workers: int,
    buffer_degrees: float,
    overwrite: bool,
) -> None:
    """Process all RUN1 CSV files for one RW or RI group."""
    group_dir = input_root / group
    input_dir = group_dir / "RAIN_POINT_MATCH"
    output_dir = group_dir / "POP_POINT_MATCH"
    log_dir = QC_ROOT / group

    if not input_dir.is_dir():
        raise FileNotFoundError(f"RUN1 input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    input_files = discover_input_files(input_dir)
    actual_count = len(input_files)
    expected_count = EXPECTED_INPUT_COUNTS[group]
    print(f"[{group}] RUN1 input CSV files: {actual_count} (expected: {expected_count})")
    if actual_count != expected_count:
        print(f"[{group}] WARNING: Input file count differs from the confirmed count.")

    tasks, skipped = build_tasks(
        group,
        input_files,
        output_dir,
        landscan_root,
        buffer_degrees,
        overwrite,
    )
    print(f"[{group}] Completed files skipped: {skipped}")
    print(f"[{group}] Files submitted: {len(tasks)}")

    if not tasks:
        print(f"[{group}] No pending work.")
        return

    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_file = {
            executor.submit(process_one_csv, task): Path(task["input_path"]).name
            for task in tasks
        }
        progress = tqdm(
            as_completed(future_to_file),
            total=len(future_to_file),
            desc=f"{group} LandScan matching",
        )
        for future in progress:
            input_name = future_to_file[future]
            try:
                records.append(future.result())
            except Exception as exc:
                records.append(
                    {
                        "GROUP": group,
                        "INPUT_FILE": input_name,
                        "STATUS": "failed",
                        "YEAR": "",
                        "RAIN_ROW_COUNT": 0,
                        "OUTPUT_ROW_COUNT": 0,
                        "OCEAN_FALLBACK_COUNT": 0,
                        "LANDSCAN_FILE": "",
                        "MESSAGE": f"{type(exc).__name__}: {exc}",
                    }
                )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"RUN2_{group}_task_log_{run_stamp}.csv"
    log_data = pd.DataFrame(records).sort_values("INPUT_FILE")
    write_csv_atomic(log_data, log_path)

    completed = int(log_data["STATUS"].isin(["completed", "completed_empty"]).sum())
    empty = int((log_data["STATUS"] == "completed_empty").sum())
    failed = int((log_data["STATUS"] == "failed").sum())
    print(f"[{group}] Completed in this run: {completed}")
    print(f"[{group}] Empty rainfall files: {empty}")
    print(f"[{group}] Failed in this run: {failed}")
    print(f"[{group}] Log: {log_path}")


def main() -> None:
    args = parse_arguments()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.buffer_degrees <= 0:
        raise ValueError("--buffer-degrees must be greater than 0")
    if not args.landscan_root.is_dir():
        raise FileNotFoundError(f"LandScan root not found: {args.landscan_root}")

    print(f"Input root: {args.input_root}")
    print(f"LandScan root: {args.landscan_root}")
    print(f"Workers: {args.workers}")
    print(f"Population buffer: +/-{args.buffer_degrees} degrees")

    for group in args.groups:
        run_group(
            group=group,
            input_root=args.input_root,
            landscan_root=args.landscan_root,
            workers=args.workers,
            buffer_degrees=args.buffer_degrees,
            overwrite=args.overwrite,
        )

    print("RUN2 finished.")


if __name__ == "__main__":
    main()
