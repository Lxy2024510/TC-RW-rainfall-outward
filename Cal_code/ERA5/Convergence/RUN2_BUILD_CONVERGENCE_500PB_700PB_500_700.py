#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build three ERA5 pressure-weighted convergence products.

For every tropical-cyclone time step, this program:

1. Reads the collocated surface-pressure grid and 16 pressure-level
   convergence grids produced by RUN1.
2. Defines PB independently at every grid cell as the largest available
   pressure level that does not exceed surface pressure.
3. Computes pressure-weighted mean convergence for 500-PB, 700-PB, and
   500-700 hPa using trapezoidal integration. The residual layer from PB to
   the true surface pressure is intentionally excluded.
4. Saves one combined grid product containing all pressure levels and all
   three derived convergence products.
5. Computes area-weighted statistics for seven radial zones.
6. Builds annual and all-year radial summaries.
7. Merges the all-year summary one-to-one with the cleaned MSWEP base table.

Positive values mean convergence; negative values mean divergence.
"""

from __future__ import annotations

import gc
import gzip
import os
import re
import sys
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================================
# 1. User configuration
# ============================================================================

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(
    os.environ.get("TC_RW_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))
).expanduser().resolve()
CONVERGENCE_INTERMEDIATE_ROOT = (
    PROJECT_ROOT / "Data" / "Intermediate" / "ERA5" / "Convergence"
)
INPUT_ROOT = CONVERGENCE_INTERMEDIATE_ROOT / "Grids_500KM"
SP0_ROOT = INPUT_ROOT / "SP0_500KM"
OUTPUT_ROOT = CONVERGENCE_INTERMEDIATE_ROOT / "FUHE_CONVERGENCE_LAYERS"
SUMMARY_DIR = (
    PROJECT_ROOT / ".work" / "ERA5" / "Convergence" / "RUN2" / "summary"
)
LOG_DIR = (
    PROJECT_ROOT / "Results" / "Quality_control" / "ERA5"
    / "Convergence" / "RUN2"
)

BASE_CSV = (
    PROJECT_ROOT / "Data" / "Processed" / "MSWEP" / "Thresholds"
    / "PRE_DATA_IBT_1982_2024_MSWEP_THRESHOLD_30.csv"
)
FINAL_OUTPUT_DIR = (
    PROJECT_ROOT / "Data" / "Processed" / "ERA5"
    / "Convergence" / "Endpoints"
)
FINAL_MERGED_CSV = FINAL_OUTPUT_DIR / (
    "PRE_DATA_IBT_1982_2024_MSWEP_THRESHOLD_30_VALID_DIST30_"
    "FUHE_CONVERGENCE_LAYERS.csv"
)

START_YEAR = 1982
END_YEAR = 2024

# Start conservatively because every task reads 17 gzip files and writes one.
# Increase this only if storage throughput and memory remain healthy.
NUM_WORKERS = 8
POOL_CHUNKSIZE = 1
MAX_TASKS_PER_CHILD = 200

GZIP_COMPRESS_LEVEL = 1
SKIP_COMPLETED = True
STRICT_COMPLETION_CHECK = False
COORDINATE_ATOL = 1.0e-6
LOW_VALID_AREA_FRACTION = 0.90

PRESSURE_LEVELS = np.array(
    [
        500, 550, 600, 650, 700, 750, 775, 800,
        825, 850, 875, 900, 925, 950, 975, 1000,
    ],
    dtype=np.float64,
)

RADIAL_ZONES = (
    ("R000_100", 0.0, 100.0, False),
    ("R100_200", 100.0, 200.0, False),
    ("R200_300", 200.0, 300.0, False),
    ("R300_400", 300.0, 400.0, False),
    ("R400_500", 400.0, 500.0, True),
    ("R000_200", 0.0, 200.0, False),
    ("R200_500", 200.0, 500.0, True),
)

GRID_KEYS = ["LAT", "LON"]
SP0_COLUMNS = ["LAT", "LON", "SP0", "DIST_KM", "AREA_KM2"]
FILENAME_PATTERN = re.compile(
    r"^row_(?P<row_id>\d{9})_(?P<rain_time_id>\d{7}\.\d{2})\.csv\.gz$"
)

PRODUCTS = (
    ("500_PB", 500.0, "PB"),
    ("700_PB", 700.0, "PB"),
    ("500_700", 500.0, 700.0),
)

COMMON_DERIVED_GRID_COLUMNS = [
    "PS_HPA",
    "PB_HPA",
    "PS_MINUS_PB_HPA",
]

COMMON_ZONE_METRICS = [
    "TOTAL_GRID_COUNT",
    "TOTAL_AREA_KM2",
    "PS_HPA_AW",
    "PB_HPA_AW",
    "PS_MINUS_PB_AW",
]

PRODUCT_ZONE_METRICS = [
    "VALID_GRID_COUNT",
    "VALID_AREA_KM2",
    "VALID_AREA_FRACTION",
    "DIV_AW",
    "CONV_AW",
    "CONVERGENCE_AREA_KM2",
    "CONVERGENCE_AREA_FRACTION",
    "VALID_DP_AW",
]


# ============================================================================
# 2. Schema and path helpers
# ============================================================================

def conv_column(level: float) -> str:
    return f"CONV{int(level)}"


def conv_columns() -> list[str]:
    return [conv_column(level) for level in PRESSURE_LEVELS]


def product_grid_columns() -> list[str]:
    columns: list[str] = []
    for product, _, _ in PRODUCTS:
        columns.extend(
            [
                f"VALID_DP_{product}",
                f"LEVEL_COUNT_{product}",
                f"VERTICAL_VALID_FLAG_{product}",
                f"DIV_WEIGHTED_{product}",
                f"CONV_WEIGHTED_{product}",
            ]
        )
    return columns


def full_grid_columns() -> list[str]:
    return [
        "ROW_ID",
        "RAIN_TIME_ID",
        "LAT",
        "LON",
        *conv_columns(),
        "SP0",
        *COMMON_DERIVED_GRID_COLUMNS,
        *product_grid_columns(),
        "DIST_KM",
        "AREA_KM2",
    ]


def radial_summary_columns() -> list[str]:
    columns = ["ROW_ID", "RAIN_TIME_ID"]
    for zone_name, _, _, _ in RADIAL_ZONES:
        columns.extend(f"{metric}_{zone_name}" for metric in COMMON_ZONE_METRICS)
        for product, _, _ in PRODUCTS:
            columns.extend(
                f"{metric}_{product}_{zone_name}"
                for metric in PRODUCT_ZONE_METRICS
            )
    return columns


def conv_path(year: int, filename: str, level: float) -> Path:
    return INPUT_ROOT / f"CONV{int(level)}" / str(year) / filename


def output_path(year: int, filename: str) -> Path:
    return OUTPUT_ROOT / str(year) / filename


def annual_summary_path(year: int) -> Path:
    return SUMMARY_DIR / f"radial_summary_{year}.csv.gz"


def parse_filename(filename: str) -> tuple[int, str]:
    match = FILENAME_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(f"Unexpected input filename: {filename}")
    return int(match.group("row_id")), match.group("rain_time_id")


def require_columns(frame: pd.DataFrame, columns: list[str], path: Path | str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")


# ============================================================================
# 3. Safe file writing and completion checks
# ============================================================================

def write_csv_gzip_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with gzip.open(
            temporary,
            mode="wt",
            encoding="utf-8",
            newline="",
            compresslevel=GZIP_COMPRESS_LEVEL,
        ) as handle:
            frame.to_csv(handle, index=False, float_format="%.10g")
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError(f"Temporary output is missing or empty: {temporary}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False, date_format="%Y-%m-%d %H:%M:%S")
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError(f"Temporary output is missing or empty: {temporary}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def grid_output_is_complete(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    if not STRICT_COMPLETION_CHECK:
        return True
    try:
        header = pd.read_csv(path, compression="gzip", nrows=0)
        return set(full_grid_columns()).issubset(header.columns)
    except Exception:
        return False


# ============================================================================
# 4. Input reading and coordinate alignment
# ============================================================================

def numeric_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            path,
            compression="gzip",
            usecols=columns,
            low_memory=False,
        )
    except ValueError as error:
        raise ValueError(f"Cannot read required columns from {path}: {error}") from error

    require_columns(frame, columns, path)
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[GRID_KEYS].isna().any().any():
        raise ValueError(f"Invalid LAT/LON values in {path}")
    if frame.duplicated(GRID_KEYS).any():
        raise ValueError(f"Duplicate LAT/LON grid cells in {path}")
    return frame


def coordinates_match(reference: pd.DataFrame, candidate: pd.DataFrame) -> bool:
    if len(reference) != len(candidate):
        return False
    return bool(
        np.allclose(
            reference["LAT"].to_numpy(dtype=np.float64),
            candidate["LAT"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=COORDINATE_ATOL,
            equal_nan=False,
        )
        and np.allclose(
            reference["LON"].to_numpy(dtype=np.float64),
            candidate["LON"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=COORDINATE_ATOL,
            equal_nan=False,
        )
    )


def align_convergence(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    value_column: str,
    path: Path,
) -> tuple[np.ndarray, bool]:
    if coordinates_match(reference, candidate):
        return candidate[value_column].to_numpy(dtype=np.float64), False

    # A strict one-to-one fallback handles a different row order safely.
    aligned = reference[GRID_KEYS].merge(
        candidate,
        on=GRID_KEYS,
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if len(aligned) != len(reference) or not aligned["_merge"].eq("both").all():
        raise ValueError(f"LAT/LON grid set differs from SP0 in {path}")
    return aligned[value_column].to_numpy(dtype=np.float64), True


def build_full_input_grid(
    sp0_path: Path,
    year: int,
    filename: str,
    row_id: int,
    rain_time_id: str,
) -> tuple[pd.DataFrame, int]:
    frame = numeric_frame(sp0_path, SP0_COLUMNS)
    fallback_alignment_count = 0

    for level in PRESSURE_LEVELS:
        path = conv_path(year, filename, level)
        if not path.is_file():
            raise FileNotFoundError(f"Missing convergence file: {path}")
        column = conv_column(level)
        candidate = numeric_frame(path, ["LAT", "LON", column])
        values, used_fallback = align_convergence(frame, candidate, column, path)
        frame[column] = values
        fallback_alignment_count += int(used_fallback)
        del candidate

    frame.insert(0, "RAIN_TIME_ID", rain_time_id)
    frame.insert(0, "ROW_ID", row_id)
    return frame, fallback_alignment_count


# ============================================================================
# 5. Vertical pressure-weighted divergence
# ============================================================================

def calculate_vertical_fields(frame: pd.DataFrame) -> pd.DataFrame:
    ps_hpa = frame["SP0"].to_numpy(dtype=np.float64) / 100.0
    row_count = len(frame)

    bottom_indices = np.full(row_count, -1, dtype=np.int16)
    finite_ps = np.isfinite(ps_hpa) & (ps_hpa > 0.0)
    if finite_ps.any():
        selected = np.searchsorted(
            PRESSURE_LEVELS, ps_hpa[finite_ps], side="right"
        ) - 1
        selected = np.minimum(selected, len(PRESSURE_LEVELS) - 1)
        bottom_indices[finite_ps] = selected.astype(np.int16)

    pb_hpa = np.full(row_count, np.nan, dtype=np.float64)
    has_boundary = bottom_indices >= 0
    pb_hpa[has_boundary] = PRESSURE_LEVELS[bottom_indices[has_boundary]]

    residual_dp = np.full(row_count, np.nan, dtype=np.float64)
    residual_dp[has_boundary] = ps_hpa[has_boundary] - pb_hpa[has_boundary]

    convergence_values = frame[conv_columns()].to_numpy(dtype=np.float64)

    frame["PS_HPA"] = ps_hpa
    frame["PB_HPA"] = pb_hpa
    frame["PS_MINUS_PB_HPA"] = residual_dp

    for product, top_pressure, bottom_definition in PRODUCTS:
        top_index = int(np.flatnonzero(PRESSURE_LEVELS == top_pressure)[0])
        weighted_convergence = np.full(row_count, np.nan, dtype=np.float64)
        valid_dp = np.full(row_count, np.nan, dtype=np.float64)
        level_count = np.zeros(row_count, dtype=np.int16)

        if bottom_definition == "PB":
            # A positive pressure depth is required, so PB must lie below the
            # selected top pressure. The PB-to-surface residual is excluded.
            for bottom_index in range(top_index + 1, len(PRESSURE_LEVELS)):
                positions = np.flatnonzero(bottom_indices == bottom_index)
                if positions.size == 0:
                    continue
                levels = PRESSURE_LEVELS[top_index : bottom_index + 1]
                profiles = convergence_values[
                    positions, top_index : bottom_index + 1
                ]
                complete = np.all(np.isfinite(profiles), axis=1)
                if not complete.any():
                    continue
                valid_positions = positions[complete]
                valid_profiles = profiles[complete]
                intervals = np.diff(levels)
                interval_means = (
                    valid_profiles[:, :-1] + valid_profiles[:, 1:]
                ) / 2.0
                pressure_depth = float(levels[-1] - levels[0])
                weighted_convergence[valid_positions] = np.sum(
                    interval_means * intervals[np.newaxis, :], axis=1
                ) / pressure_depth
                valid_dp[valid_positions] = pressure_depth
                level_count[valid_positions] = len(levels)
        else:
            bottom_pressure = float(bottom_definition)
            bottom_index = int(
                np.flatnonzero(PRESSURE_LEVELS == bottom_pressure)[0]
            )
            # Fixed 500-700 hPa is valid only when the surface is at or below
            # 700 hPa (PS >= 700 hPa) and every required level is finite.
            positions = np.flatnonzero(bottom_indices >= bottom_index)
            if positions.size:
                levels = PRESSURE_LEVELS[top_index : bottom_index + 1]
                profiles = convergence_values[
                    positions, top_index : bottom_index + 1
                ]
                complete = np.all(np.isfinite(profiles), axis=1)
                valid_positions = positions[complete]
                valid_profiles = profiles[complete]
                if valid_positions.size:
                    intervals = np.diff(levels)
                    interval_means = (
                        valid_profiles[:, :-1] + valid_profiles[:, 1:]
                    ) / 2.0
                    pressure_depth = float(bottom_pressure - top_pressure)
                    weighted_convergence[valid_positions] = np.sum(
                        interval_means * intervals[np.newaxis, :], axis=1
                    ) / pressure_depth
                    valid_dp[valid_positions] = pressure_depth
                    level_count[valid_positions] = len(levels)

        vertical_valid = np.isfinite(weighted_convergence)
        frame[f"VALID_DP_{product}"] = valid_dp
        frame[f"LEVEL_COUNT_{product}"] = level_count
        frame[f"VERTICAL_VALID_FLAG_{product}"] = vertical_valid
        frame[f"CONV_WEIGHTED_{product}"] = weighted_convergence
        frame[f"DIV_WEIGHTED_{product}"] = -weighted_convergence
    return frame


# ============================================================================
# 6. Horizontal area-weighted radial summaries
# ============================================================================

def weighted_mean(values: np.ndarray, areas: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(areas) & (areas > 0.0)
    if not valid.any():
        return np.nan
    denominator = float(np.sum(areas[valid]))
    if denominator <= 0.0:
        return np.nan
    return float(np.sum(values[valid] * areas[valid]) / denominator)


def zone_mask(distances: np.ndarray, lower: float, upper: float, close_upper: bool) -> np.ndarray:
    if close_upper:
        return (distances >= lower) & (distances <= upper)
    return (distances >= lower) & (distances < upper)


def summarize_radial_zones(
    frame: pd.DataFrame,
    row_id: int,
    rain_time_id: str,
) -> dict[str, object]:
    result: dict[str, object] = {
        "ROW_ID": row_id,
        "RAIN_TIME_ID": rain_time_id,
    }

    distances = frame["DIST_KM"].to_numpy(dtype=np.float64)
    areas = frame["AREA_KM2"].to_numpy(dtype=np.float64)
    ps_hpa = frame["PS_HPA"].to_numpy(dtype=np.float64)
    pb_hpa = frame["PB_HPA"].to_numpy(dtype=np.float64)
    residual_dp = frame["PS_MINUS_PB_HPA"].to_numpy(dtype=np.float64)

    valid_area_base = np.isfinite(areas) & (areas > 0.0)

    for zone_name, lower, upper, close_upper in RADIAL_ZONES:
        in_zone = zone_mask(distances, lower, upper, close_upper)
        total_area_mask = in_zone & valid_area_base

        total_grid_count = int(np.count_nonzero(in_zone))
        total_area = float(np.sum(areas[total_area_mask])) if total_area_mask.any() else 0.0
        result[f"TOTAL_GRID_COUNT_{zone_name}"] = total_grid_count
        result[f"TOTAL_AREA_KM2_{zone_name}"] = total_area
        result[f"PS_HPA_AW_{zone_name}"] = weighted_mean(
            ps_hpa[total_area_mask], areas[total_area_mask]
        )
        result[f"PB_HPA_AW_{zone_name}"] = weighted_mean(
            pb_hpa[total_area_mask], areas[total_area_mask]
        )
        result[f"PS_MINUS_PB_AW_{zone_name}"] = weighted_mean(
            residual_dp[total_area_mask], areas[total_area_mask]
        )

        for product, _, _ in PRODUCTS:
            divergence = frame[f"DIV_WEIGHTED_{product}"].to_numpy(
                dtype=np.float64
            )
            convergence = frame[f"CONV_WEIGHTED_{product}"].to_numpy(
                dtype=np.float64
            )
            valid_dp = frame[f"VALID_DP_{product}"].to_numpy(dtype=np.float64)
            vertical_valid = (
                frame[f"VERTICAL_VALID_FLAG_{product}"].to_numpy(dtype=bool)
                & np.isfinite(divergence)
                & np.isfinite(convergence)
                & valid_area_base
            )
            valid_mask = in_zone & vertical_valid
            valid_grid_count = int(np.count_nonzero(valid_mask))
            valid_area = (
                float(np.sum(areas[valid_mask])) if valid_mask.any() else 0.0
            )
            valid_area_fraction = (
                valid_area / total_area if total_area > 0.0 else np.nan
            )
            convergent_mask = valid_mask & (convergence > 0.0)
            convergence_area = (
                float(np.sum(areas[convergent_mask]))
                if convergent_mask.any()
                else 0.0
            )
            convergence_area_fraction = (
                convergence_area / valid_area if valid_area > 0.0 else np.nan
            )
            suffix = f"{product}_{zone_name}"
            result[f"VALID_GRID_COUNT_{suffix}"] = valid_grid_count
            result[f"VALID_AREA_KM2_{suffix}"] = valid_area
            result[f"VALID_AREA_FRACTION_{suffix}"] = valid_area_fraction
            result[f"DIV_AW_{suffix}"] = weighted_mean(
                divergence[valid_mask], areas[valid_mask]
            )
            result[f"CONV_AW_{suffix}"] = weighted_mean(
                convergence[valid_mask], areas[valid_mask]
            )
            result[f"CONVERGENCE_AREA_KM2_{suffix}"] = convergence_area
            result[f"CONVERGENCE_AREA_FRACTION_{suffix}"] = (
                convergence_area_fraction
            )
            result[f"VALID_DP_AW_{suffix}"] = weighted_mean(
                valid_dp[valid_mask], areas[valid_mask]
            )

    return result


# ============================================================================
# 7. One-file worker
# ============================================================================

def process_one_file(task: tuple[str, int, str]) -> dict[str, object]:
    sp0_path_text, year, filename = task
    sp0_path = Path(sp0_path_text)
    target = output_path(year, filename)

    try:
        row_id, rain_time_id = parse_filename(filename)
        used_existing = SKIP_COMPLETED and grid_output_is_complete(target)

        if used_existing:
            frame = pd.read_csv(target, compression="gzip", low_memory=False)
            require_columns(frame, full_grid_columns(), target)
            fallback_alignment_count = 0
            status = "skipped"
        else:
            frame, fallback_alignment_count = build_full_input_grid(
                sp0_path, year, filename, row_id, rain_time_id
            )
            frame = calculate_vertical_fields(frame)
            frame = frame[full_grid_columns()].copy()
            write_csv_gzip_atomic(frame, target)
            status = "success"

        radial_summary = summarize_radial_zones(frame, row_id, rain_time_id)
        valid_counts = {
            product: int(
                frame[f"VERTICAL_VALID_FLAG_{product}"].astype(bool).sum()
            )
            for product, _, _ in PRODUCTS
        }
        ps_at_or_below_500_count = int(
            (pd.to_numeric(frame["PS_HPA"], errors="coerce") <= 500.0).sum()
        )
        low_coverage_zones = []
        for product, _, _ in PRODUCTS:
            for zone_name, _, _, _ in RADIAL_ZONES:
                value = radial_summary[
                    f"VALID_AREA_FRACTION_{product}_{zone_name}"
                ]
                if np.isfinite(value) and value < LOW_VALID_AREA_FRACTION:
                    low_coverage_zones.append(f"{product}:{zone_name}")

        response = {
            "status": status,
            "year": year,
            "filename": filename,
            "row_id": row_id,
            "rain_time_id": rain_time_id,
            "grid_count": len(frame),
            "valid_vertical_count_500_pb": valid_counts["500_PB"],
            "invalid_vertical_count_500_pb": len(frame) - valid_counts["500_PB"],
            "valid_vertical_count_700_pb": valid_counts["700_PB"],
            "invalid_vertical_count_700_pb": len(frame) - valid_counts["700_PB"],
            "valid_vertical_count_500_700": valid_counts["500_700"],
            "invalid_vertical_count_500_700": len(frame) - valid_counts["500_700"],
            "ps_at_or_below_500_count": ps_at_or_below_500_count,
            "fallback_alignment_count": fallback_alignment_count,
            "low_coverage_zones": "|".join(low_coverage_zones),
            "message": "",
            "traceback": "",
            "summary": radial_summary,
        }
        del frame
        gc.collect()
        return response

    except Exception as error:
        return {
            "status": "failed",
            "year": year,
            "filename": filename,
            "row_id": np.nan,
            "rain_time_id": "",
            "grid_count": 0,
            "valid_vertical_count_500_pb": 0,
            "invalid_vertical_count_500_pb": 0,
            "valid_vertical_count_700_pb": 0,
            "invalid_vertical_count_700_pb": 0,
            "valid_vertical_count_500_700": 0,
            "invalid_vertical_count_500_700": 0,
            "ps_at_or_below_500_count": 0,
            "fallback_alignment_count": 0,
            "low_coverage_zones": "",
            "message": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "summary": None,
        }


# ============================================================================
# 8. Validation, task scanning, and annual processing
# ============================================================================

def validate_configuration() -> None:
    if not BASE_CSV.is_file():
        raise FileNotFoundError(f"MSWEP threshold-30 CSV does not exist: {BASE_CSV}")
    if not SP0_ROOT.is_dir():
        raise FileNotFoundError(f"SP0 directory does not exist: {SP0_ROOT}")
    for level in PRESSURE_LEVELS:
        directory = INPUT_ROOT / f"CONV{int(level)}"
        if not directory.is_dir():
            raise FileNotFoundError(f"Convergence directory does not exist: {directory}")
    if not np.all(np.diff(PRESSURE_LEVELS) > 0.0):
        raise ValueError("PRESSURE_LEVELS must be strictly increasing")


def tasks_for_year(year: int) -> list[tuple[str, int, str]]:
    directory = SP0_ROOT / str(year)
    if not directory.is_dir():
        raise FileNotFoundError(f"SP0 year directory does not exist: {directory}")

    tasks = []
    for path in sorted(directory.glob("row_*.csv.gz")):
        parse_filename(path.name)
        tasks.append((str(path), year, path.name))
    if not tasks:
        raise RuntimeError(f"No SP0 files found for {year}: {directory}")
    return tasks


def log_record(result: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in result.items() if key != "summary"}


def process_year(year: int, pool) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = tasks_for_year(year)
    summary_rows: list[dict[str, object]] = []
    log_rows: list[dict[str, object]] = []

    iterator = pool.imap_unordered(process_one_file, tasks, chunksize=POOL_CHUNKSIZE)
    for result in tqdm(
        iterator,
        total=len(tasks),
        desc=f"FUHE {year}",
        unit="file",
        dynamic_ncols=True,
    ):
        log_rows.append(log_record(result))
        if result["summary"] is not None:
            summary_rows.append(result["summary"])

    log_frame = pd.DataFrame(log_rows).sort_values("filename").reset_index(drop=True)
    failed = log_frame["status"].eq("failed")
    if failed.any():
        year_failure_path = LOG_DIR / f"processing_failures_{year}.csv"
        write_csv_atomic(log_frame.loc[failed].copy(), year_failure_path)
        raise RuntimeError(
            f"{year} has {int(failed.sum())} failed tasks. "
            f"Review {year_failure_path}; later years were not processed."
        )

    summary_frame = pd.DataFrame(summary_rows, columns=radial_summary_columns())
    summary_frame = summary_frame.sort_values("ROW_ID").reset_index(drop=True)
    if len(summary_frame) != len(tasks):
        raise RuntimeError(
            f"{year} summary row count {len(summary_frame)} differs from task count {len(tasks)}"
        )
    if summary_frame["ROW_ID"].duplicated().any():
        raise ValueError(f"Duplicate ROW_ID values in the {year} radial summary")

    write_csv_gzip_atomic(summary_frame, annual_summary_path(year))
    write_csv_atomic(log_frame, LOG_DIR / f"processing_row_log_{year}.csv")
    return summary_frame, log_frame


# ============================================================================
# 9. All-year summary and strict merge with the cleaned base CSV
# ============================================================================

def build_all_year_summary(annual_frames: list[pd.DataFrame]) -> pd.DataFrame:
    summary = pd.concat(annual_frames, ignore_index=True)
    summary = summary.sort_values("ROW_ID").reset_index(drop=True)
    if summary["ROW_ID"].duplicated().any():
        examples = summary.loc[
            summary["ROW_ID"].duplicated(keep=False), "ROW_ID"
        ].head(20).tolist()
        raise ValueError(f"Duplicate ROW_ID values in all-year summary: {examples}")
    path = SUMMARY_DIR / "FUHE_CONVERGENCE_LAYERS_radial_summary_1982_2024.csv.gz"
    write_csv_gzip_atomic(summary, path)
    return summary


def merge_with_base(summary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    base = pd.read_csv(
        BASE_CSV,
        dtype={"SID": "string", "RAIN_TIME_ID": "string"},
        low_memory=False,
    )
    source_base_row_count = len(base)
    require_columns(
        base, ["ROW_ID", "RAIN_TIME_ID", "ISO_TIME", "MSWEP_DIST_30"], BASE_CSV
    )

    base["MSWEP_DIST_30"] = pd.to_numeric(
        base["MSWEP_DIST_30"], errors="coerce"
    )
    base["ISO_TIME"] = pd.to_datetime(base["ISO_TIME"], errors="coerce")
    valid_base = (
        np.isfinite(base["MSWEP_DIST_30"].to_numpy(dtype=np.float64))
        & base["ISO_TIME"].dt.year.between(
            START_YEAR, END_YEAR, inclusive="both"
        )
    )
    base = base.loc[valid_base].copy().reset_index(drop=True)

    base["ROW_ID"] = pd.to_numeric(base["ROW_ID"], errors="raise").astype(np.int64)
    summary["ROW_ID"] = pd.to_numeric(
        summary["ROW_ID"], errors="raise"
    ).astype(np.int64)
    base["RAIN_TIME_ID"] = base["RAIN_TIME_ID"].astype("string")
    summary["RAIN_TIME_ID"] = summary["RAIN_TIME_ID"].astype("string")

    if base["ROW_ID"].duplicated().any():
        raise ValueError("The cleaned base CSV contains duplicate ROW_ID values")
    if summary["ROW_ID"].duplicated().any():
        raise ValueError("The radial summary contains duplicate ROW_ID values")

    base_ids = set(base["ROW_ID"].tolist())
    summary_ids = set(summary["ROW_ID"].tolist())
    missing_summary_ids = sorted(base_ids - summary_ids)
    extra_summary_ids = sorted(summary_ids - base_ids)

    time_check = base[["ROW_ID", "RAIN_TIME_ID"]].merge(
        summary[["ROW_ID", "RAIN_TIME_ID"]],
        on="ROW_ID",
        how="inner",
        validate="one_to_one",
        suffixes=("_BASE", "_SUMMARY"),
    )
    time_mismatch = time_check[
        time_check["RAIN_TIME_ID_BASE"] != time_check["RAIN_TIME_ID_SUMMARY"]
    ]
    if not time_mismatch.empty:
        mismatch_path = LOG_DIR / "rain_time_id_mismatches.csv"
        write_csv_atomic(time_mismatch, mismatch_path)
        raise ValueError(
            f"RAIN_TIME_ID mismatch for {len(time_mismatch)} ROW_ID values; "
            f"review {mismatch_path}"
        )

    summary_metrics = summary.drop(columns=["RAIN_TIME_ID"])
    merged = base.merge(
        summary_metrics,
        on="ROW_ID",
        how="left",
        sort=False,
        validate="one_to_one",
        indicator="FUHE_MERGE_STATUS",
    )

    audit = {
        "SOURCE_BASE_ROW_COUNT": source_base_row_count,
        "BASE_ROW_COUNT": len(base),
        "SUMMARY_ROW_COUNT": len(summary),
        "FINAL_ROW_COUNT": len(merged),
        "MATCHED_ROW_COUNT": int(merged["FUHE_MERGE_STATUS"].eq("both").sum()),
        "MISSING_SUMMARY_ROW_COUNT": len(missing_summary_ids),
        "EXTRA_SUMMARY_ROW_COUNT": len(extra_summary_ids),
        "RAIN_TIME_ID_MISMATCH_COUNT": len(time_mismatch),
        "FINAL_COLUMN_COUNT": len(merged.columns),
    }

    if missing_summary_ids or extra_summary_ids:
        mismatch_rows = pd.DataFrame(
            {
                "TYPE": (
                    ["BASE_WITHOUT_SUMMARY"] * len(missing_summary_ids)
                    + ["SUMMARY_WITHOUT_BASE"] * len(extra_summary_ids)
                ),
                "ROW_ID": missing_summary_ids + extra_summary_ids,
            }
        )
        write_csv_atomic(mismatch_rows, LOG_DIR / "row_id_merge_mismatches.csv")
        raise ValueError(
            "ROW_ID coverage is not identical between the cleaned base CSV and radial summary"
        )

    merged = merged.drop(columns=["FUHE_MERGE_STATUS"])
    audit["FINAL_COLUMN_COUNT"] = len(merged.columns)
    write_csv_atomic(merged, FINAL_MERGED_CSV)
    write_csv_atomic(pd.DataFrame([audit]), LOG_DIR / "final_merge_audit.csv")
    return merged, audit


# ============================================================================
# 10. Main program
# ============================================================================

def main() -> None:
    print("=" * 88)
    print("ERA5 multi-layer convergence and area-weighted radial summary")
    print("=" * 88)
    print(f"Input root: {INPUT_ROOT}")
    print(f"SP0 root: {SP0_ROOT}")
    print(f"Grid output root: {OUTPUT_ROOT}")
    print(f"MSWEP threshold-30 CSV: {BASE_CSV}")
    print("Final merge retains only rows with finite MSWEP_DIST_30")
    print(f"Final merged CSV: {FINAL_MERGED_CSV}")
    print(f"Years: {START_YEAR}-{END_YEAR}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Skip complete grid outputs: {SKIP_COMPLETED}")
    print("Vertical products: 500-PB, 700-PB, and 500-700 hPa")
    print("500-700 is retained only where PS >= 700 hPa")
    print("700-PB requires PB > 700 hPa")
    print("Residual layer from PB to surface pressure: excluded")
    print("Horizontal radial statistics: AREA_KM2 weighted")
    print("=" * 88)

    validate_configuration()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    annual_frames: list[pd.DataFrame] = []
    annual_logs: list[pd.DataFrame] = []

    context = get_context("fork")
    with context.Pool(
        processes=NUM_WORKERS,
        maxtasksperchild=MAX_TASKS_PER_CHILD,
    ) as pool:
        for year in range(START_YEAR, END_YEAR + 1):
            summary_frame, log_frame = process_year(year, pool)
            annual_frames.append(summary_frame)
            annual_logs.append(log_frame)

    all_year_summary = build_all_year_summary(annual_frames)
    all_logs = pd.concat(annual_logs, ignore_index=True)
    write_csv_atomic(all_logs, LOG_DIR / "processing_row_log_1982_2024.csv")

    low_coverage = all_logs[all_logs["low_coverage_zones"].astype(str).ne("")].copy()
    write_csv_atomic(low_coverage, LOG_DIR / "low_valid_area_records.csv")

    processing_summary = {
        "TASK_COUNT": len(all_logs),
        "SUCCESS_COUNT": int(all_logs["status"].eq("success").sum()),
        "SKIPPED_COUNT": int(all_logs["status"].eq("skipped").sum()),
        "FAILED_COUNT": int(all_logs["status"].eq("failed").sum()),
        "GRID_COUNT": int(all_logs["grid_count"].sum()),
        "VALID_VERTICAL_GRID_COUNT_500_PB": int(
            all_logs["valid_vertical_count_500_pb"].sum()
        ),
        "INVALID_VERTICAL_GRID_COUNT_500_PB": int(
            all_logs["invalid_vertical_count_500_pb"].sum()
        ),
        "VALID_VERTICAL_GRID_COUNT_700_PB": int(
            all_logs["valid_vertical_count_700_pb"].sum()
        ),
        "INVALID_VERTICAL_GRID_COUNT_700_PB": int(
            all_logs["invalid_vertical_count_700_pb"].sum()
        ),
        "VALID_VERTICAL_GRID_COUNT_500_700": int(
            all_logs["valid_vertical_count_500_700"].sum()
        ),
        "INVALID_VERTICAL_GRID_COUNT_500_700": int(
            all_logs["invalid_vertical_count_500_700"].sum()
        ),
        "PS_AT_OR_BELOW_500_GRID_COUNT": int(
            all_logs["ps_at_or_below_500_count"].sum()
        ),
        "FALLBACK_ALIGNMENT_COUNT": int(all_logs["fallback_alignment_count"].sum()),
        "LOW_VALID_AREA_RECORD_COUNT": len(low_coverage),
        "LOW_VALID_AREA_THRESHOLD": LOW_VALID_AREA_FRACTION,
    }
    write_csv_atomic(
        pd.DataFrame([processing_summary]), LOG_DIR / "processing_summary.csv"
    )

    merged, merge_audit = merge_with_base(all_year_summary)

    print()
    print("=" * 88)
    print("Processing completed successfully")
    print("=" * 88)
    for key, value in processing_summary.items():
        print(f"{key}: {value}")
    for key, value in merge_audit.items():
        print(f"{key}: {value}")
    print(f"Grid-level products: {OUTPUT_ROOT}")
    print(
        "All-year radial summary: "
        f"{SUMMARY_DIR / 'FUHE_CONVERGENCE_LAYERS_radial_summary_1982_2024.csv.gz'}"
    )
    print(f"Final merged table: {FINAL_MERGED_CSV}")
    print(f"Final merged shape: {merged.shape}")
    print("=" * 88)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        raise
