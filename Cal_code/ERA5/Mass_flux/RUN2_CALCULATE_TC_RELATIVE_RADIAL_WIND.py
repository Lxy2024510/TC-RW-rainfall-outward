#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Calculate TC-motion-relative inward radial wind from ERA5 U/V grids.

This stage reads the 550-km U/V grid files produced by RUN1 and writes one
matching radial-wind grid file for every available TC record and pressure
level. Positive RADIAL_VEL denotes flow toward the TC centre; negative values
denote flow away from the TC centre.
"""

import gzip
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from tqdm import tqdm


warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. Paths and parameters
# ============================================================

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(
    os.environ.get("TC_RW_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))
).expanduser().resolve()
SOURCE_CSV = (
    PROJECT_ROOT / "Data" / "Processed" / "IBTrACS"
    / "PRE_DATA_IBT_1982_2024_BASE.csv"
)

# RUN1 input products and RUN2 output products share this root directory.
ERA5_ROOT = (
    PROJECT_ROOT / "Data" / "Intermediate" / "ERA5"
    / "Mass_flux" / "UV_550KM"
)
OVERALL_LOG_DIR = ERA5_ROOT / "_logs" / "RUN2_RADIAL_WIND"

PRESSURE_LEVELS_HPA = [
    500, 550, 600, 650, 700, 750, 775, 800,
    825, 850, 875, 900, 925, 950, 975, 1000,
]

REFERENCE_LEVEL_HPA = 500
START_YEAR = 1982
END_YEAR = 2024

MAX_WORKERS = 24
POOL_CHUNKSIZE = 1
GZIP_COMPRESS_LEVEL = 1

SKIP_EXISTING = True
STRICT_EXISTING_CHECK = False

TEST_MODE = False
TEST_MAX_ROWS = 500


SUMMARY_COLUMNS = [
    "LEVEL_HPA",
    "YEAR_MONTH",
    "RECORD_COUNT",
    "SUCCESS_COUNT",
    "SKIPPED_EXISTING_COUNT",
    "MISSING_INPUT_COUNT",
    "FAILED_COUNT",
    "GRID_COUNT",
    "VALID_RADIAL_GRID_COUNT",
    "NAN_RADIAL_GRID_COUNT",
]

ERROR_COLUMNS = [
    "LEVEL_HPA",
    "ROW_ID",
    "SID",
    "ISO_TIME",
    "RAIN_TIME_ID",
    "INPUT_PATH",
    "OUTPUT_PATH",
    "ERROR",
]


# These globals are configured once in every worker process.
CURRENT_LEVEL_HPA = None
CURRENT_INPUT_DIR = None
CURRENT_OUTPUT_DIR = None


# ============================================================
# 2. General helpers
# ============================================================

def configure_worker(level_hpa):
    """Configure field-specific paths inside one worker process."""
    global CURRENT_LEVEL_HPA
    global CURRENT_INPUT_DIR
    global CURRENT_OUTPUT_DIR

    CURRENT_LEVEL_HPA = int(level_hpa)
    CURRENT_INPUT_DIR = ERA5_ROOT / f"UV{CURRENT_LEVEL_HPA}"
    CURRENT_OUTPUT_DIR = ERA5_ROOT / f"INFLOW_{CURRENT_LEVEL_HPA}"


def normalize_rain_time_id(value):
    """Return RAIN_TIME_ID in the fixed YYYYDDD.HH representation."""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        raise ValueError("RAIN_TIME_ID is empty")

    if "." in text:
        left, right = text.split(".", maxsplit=1)
        right = right[:2].ljust(2, "0")
        return f"{left}.{right}"

    # A numeric reader may remove the decimal point from an integer-hour ID.
    if len(text) == 9 and text.isdigit():
        return f"{text[:7]}.{text[7:]}"

    raise ValueError(f"Invalid RAIN_TIME_ID: {value!r}")


def build_grid_path(base_dir, row_id, rain_time_id):
    """Build the RUN1/RUN2 row-level gzip CSV path."""
    return (
        base_dir
        / rain_time_id[:4]
        / f"row_{int(row_id):09d}_{rain_time_id}.csv.gz"
    )


def input_path(level_hpa, row_id, rain_time_id):
    return build_grid_path(
        ERA5_ROOT / f"UV{int(level_hpa)}",
        row_id,
        rain_time_id,
    )


def output_path(level_hpa, row_id, rain_time_id):
    return build_grid_path(
        ERA5_ROOT / f"INFLOW_{int(level_hpa)}",
        row_id,
        rain_time_id,
    )


def output_is_complete(path):
    """Return whether an existing output can be reused."""
    if not path.exists() or path.stat().st_size <= 0:
        return False
    if not STRICT_EXISTING_CHECK:
        return True

    try:
        with gzip.open(path, mode="rt", encoding="utf-8") as handle:
            columns = handle.readline().strip().split(",")
        return "RADIAL_VEL" in columns
    except (OSError, EOFError, gzip.BadGzipFile, UnicodeDecodeError):
        return False


def write_csv_gzip_atomic(frame, path):
    """Write one gzip CSV atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")

    try:
        with gzip.open(
            temporary,
            mode="wt",
            encoding="utf-8",
            newline="",
            compresslevel=GZIP_COMPRESS_LEVEL,
        ) as handle:
            frame.to_csv(handle, index=False, float_format="%.10g")

        if not temporary.exists() or temporary.stat().st_size <= 0:
            raise RuntimeError(f"Temporary output is missing or empty: {temporary}")

        os.replace(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def write_csv_atomic(frame, path):
    """Write a normal CSV atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


# ============================================================
# 3. Input-table preparation
# ============================================================

def read_source_table():
    """Read and validate the new BASE table."""
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"BASE table does not exist: {SOURCE_CSV}")

    required = [
        "SID",
        "ISO_TIME",
        "RAIN_TIME_ID",
        "USA_LAT",
        "USA_LON",
        "u_speed_ms",
        "v_speed_ms",
    ]

    source_columns = pd.read_csv(SOURCE_CSV, nrows=0).columns.tolist()
    missing = [column for column in required if column not in source_columns]
    if missing:
        raise KeyError(f"BASE table is missing required columns: {missing}")

    # The current BASE table has no stored ROW_ID. RUN1 assigned ROW_ID from
    # its original zero-based physical row order before applying any filter.
    # Reconstructing it here preserves the exact mapping to every RUN1 file.
    usecols = required.copy()
    if "ROW_ID" in source_columns:
        usecols.insert(0, "ROW_ID")

    frame = pd.read_csv(
        SOURCE_CSV,
        usecols=usecols,
        dtype={"SID": "string", "RAIN_TIME_ID": "string"},
        low_memory=False,
    )

    if "ROW_ID" not in frame.columns:
        frame.insert(0, "ROW_ID", np.arange(len(frame), dtype=np.int64))

    frame["ISO_TIME"] = pd.to_datetime(frame["ISO_TIME"], errors="coerce")
    for column in [
        "ROW_ID", "USA_LAT", "USA_LON", "u_speed_ms", "v_speed_ms"
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid_row_id = (
        frame["ROW_ID"].isna()
        | (frame["ROW_ID"] < 0)
        | (frame["ROW_ID"] % 1 != 0)
    )
    if invalid_row_id.any():
        raise ValueError(f"Invalid ROW_ID values: {int(invalid_row_id.sum())}")

    frame["ROW_ID"] = frame["ROW_ID"].astype(np.int64)
    if frame["ROW_ID"].duplicated().any():
        raise ValueError("Duplicate ROW_ID values exist in the BASE table")

    invalid_time = frame["ISO_TIME"].isna()
    if invalid_time.any():
        raise ValueError(f"Invalid ISO_TIME values: {int(invalid_time.sum())}")

    frame = frame.loc[
        frame["ISO_TIME"].dt.year.between(START_YEAR, END_YEAR)
    ].copy()

    frame["RAIN_TIME_ID"] = frame["RAIN_TIME_ID"].map(
        normalize_rain_time_id
    )
    frame["YEAR_MONTH"] = frame["ISO_TIME"].dt.strftime("%Y%m")

    finite_columns = ["USA_LAT", "USA_LON", "u_speed_ms", "v_speed_ms"]
    finite_mask = np.isfinite(
        frame[finite_columns].to_numpy(dtype=np.float64)
    ).all(axis=1)

    invalid_motion = frame.loc[~finite_mask].copy()
    frame = frame.loc[finite_mask].copy()

    # RUN1 intentionally extracted only records with valid MSWEP_DIST_30.
    # The UV500 file inventory is therefore used as the authoritative mask.
    reference_dir = ERA5_ROOT / f"UV{REFERENCE_LEVEL_HPA}"
    available_mask = np.fromiter(
        (
            build_grid_path(reference_dir, row.ROW_ID, row.RAIN_TIME_ID).is_file()
            for row in frame.itertuples(index=False)
        ),
        dtype=bool,
        count=len(frame),
    )

    unavailable = frame.loc[~available_mask].copy()
    frame = frame.loc[available_mask].copy()

    if TEST_MODE:
        frame = frame.head(TEST_MAX_ROWS).copy()

    return frame, invalid_motion, unavailable


def records_by_month(frame):
    """Convert a table into compact monthly worker payloads."""
    columns = [
        "ROW_ID",
        "SID",
        "ISO_TIME",
        "RAIN_TIME_ID",
        "USA_LAT",
        "USA_LON",
        "u_speed_ms",
        "v_speed_ms",
    ]

    tasks = []
    for year_month, group in frame.groupby("YEAR_MONTH", sort=True):
        records = []
        for row in group[columns].itertuples(index=False, name=None):
            records.append(
                (
                    int(row[0]),
                    str(row[1]),
                    pd.Timestamp(row[2]).isoformat(),
                    str(row[3]),
                    float(row[4]),
                    float(row[5]),
                    float(row[6]),
                    float(row[7]),
                )
            )
        tasks.append((str(year_month), records))
    return tasks


# ============================================================
# 4. Radial-wind calculation
# ============================================================

def calculate_inward_radial_wind(
    frame,
    level_hpa,
    center_lat,
    center_lon,
    tc_u_speed,
    tc_v_speed,
):
    """Add TC-motion-relative inward radial velocity to one grid."""
    u_column = f"U{int(level_hpa)}"
    v_column = f"V{int(level_hpa)}"
    required = ["LAT", "LON", u_column, v_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing input columns: {missing}")

    result = frame.copy()
    for column in required:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    grid_lat = result["LAT"].to_numpy(dtype=np.float64)
    grid_lon = result["LON"].to_numpy(dtype=np.float64)
    u_wind = result[u_column].to_numpy(dtype=np.float64)
    v_wind = result[v_column].to_numpy(dtype=np.float64)

    if not np.isfinite(grid_lat).all() or not np.isfinite(grid_lon).all():
        raise ValueError("Grid LAT or LON contains invalid values")

    # Express the centre longitude in the same convention as the grid.
    if np.nanmin(grid_lon) >= 0.0:
        center_lon_normalized = float(center_lon) % 360.0
    else:
        center_lon_normalized = (
            (float(center_lon) + 180.0) % 360.0
        ) - 180.0

    lat_rad = np.radians(grid_lat)
    lon_rad = np.radians(grid_lon)
    center_lat_rad = np.radians(float(center_lat))
    center_lon_rad = np.radians(center_lon_normalized)

    delta_lon = (
        lon_rad - center_lon_rad + np.pi
    ) % (2.0 * np.pi) - np.pi

    # Bearing from the TC centre to each grid cell, clockwise from north.
    bearing = np.arctan2(
        np.sin(delta_lon) * np.cos(lat_rad),
        np.cos(center_lat_rad) * np.sin(lat_rad)
        - np.sin(center_lat_rad)
        * np.cos(lat_rad)
        * np.cos(delta_lon),
    )

    relative_u = u_wind - float(tc_u_speed)
    relative_v = v_wind - float(tc_v_speed)

    # Outward projection is negated so positive values indicate inflow.
    result["RADIAL_VEL"] = -(
        relative_u * np.sin(bearing)
        + relative_v * np.cos(bearing)
    )

    return result


# ============================================================
# 5. Monthly worker
# ============================================================

def process_month(task):
    """Process all eligible TC records in one month."""
    year_month, records = task

    summary = {
        "LEVEL_HPA": CURRENT_LEVEL_HPA,
        "YEAR_MONTH": year_month,
        "RECORD_COUNT": len(records),
        "SUCCESS_COUNT": 0,
        "SKIPPED_EXISTING_COUNT": 0,
        "MISSING_INPUT_COUNT": 0,
        "FAILED_COUNT": 0,
        "GRID_COUNT": 0,
        "VALID_RADIAL_GRID_COUNT": 0,
        "NAN_RADIAL_GRID_COUNT": 0,
    }
    errors = []

    for record in records:
        (
            row_id,
            sid,
            iso_time,
            rain_time_id,
            center_lat,
            center_lon,
            tc_u_speed,
            tc_v_speed,
        ) = record

        source = build_grid_path(CURRENT_INPUT_DIR, row_id, rain_time_id)
        target = build_grid_path(CURRENT_OUTPUT_DIR, row_id, rain_time_id)

        if SKIP_EXISTING and output_is_complete(target):
            summary["SKIPPED_EXISTING_COUNT"] += 1
            continue

        if not source.is_file():
            summary["MISSING_INPUT_COUNT"] += 1
            errors.append(
                {
                    "LEVEL_HPA": CURRENT_LEVEL_HPA,
                    "ROW_ID": row_id,
                    "SID": sid,
                    "ISO_TIME": iso_time,
                    "RAIN_TIME_ID": rain_time_id,
                    "INPUT_PATH": str(source),
                    "OUTPUT_PATH": str(target),
                    "ERROR": "Input U/V grid file does not exist",
                }
            )
            continue

        try:
            source_frame = pd.read_csv(source, low_memory=False)
            result = calculate_inward_radial_wind(
                source_frame,
                CURRENT_LEVEL_HPA,
                center_lat,
                center_lon,
                tc_u_speed,
                tc_v_speed,
            )

            radial = result["RADIAL_VEL"].to_numpy(dtype=np.float64)
            valid_count = int(np.isfinite(radial).sum())
            nan_count = int(len(radial) - valid_count)

            write_csv_gzip_atomic(result, target)

            summary["SUCCESS_COUNT"] += 1
            summary["GRID_COUNT"] += len(result)
            summary["VALID_RADIAL_GRID_COUNT"] += valid_count
            summary["NAN_RADIAL_GRID_COUNT"] += nan_count

        except Exception as error:
            summary["FAILED_COUNT"] += 1
            errors.append(
                {
                    "LEVEL_HPA": CURRENT_LEVEL_HPA,
                    "ROW_ID": row_id,
                    "SID": sid,
                    "ISO_TIME": iso_time,
                    "RAIN_TIME_ID": rain_time_id,
                    "INPUT_PATH": str(source),
                    "OUTPUT_PATH": str(target),
                    "ERROR": f"{type(error).__name__}: {error}",
                }
            )

    return summary, errors


# ============================================================
# 6. Process one pressure level
# ============================================================

def process_level(level_hpa, monthly_tasks):
    """Run all monthly tasks for one pressure level."""
    input_dir = ERA5_ROOT / f"UV{level_hpa}"
    if not input_dir.is_dir():
        raise FileNotFoundError(f"RUN1 input directory is missing: {input_dir}")

    output_dir = ERA5_ROOT / f"INFLOW_{level_hpa}"
    log_dir = output_dir / "_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    errors = []
    worker_count = min(MAX_WORKERS, len(monthly_tasks))

    with Pool(
        processes=worker_count,
        initializer=configure_worker,
        initargs=(level_hpa,),
        maxtasksperchild=24,
    ) as pool:
        iterator = pool.imap_unordered(
            process_month,
            monthly_tasks,
            chunksize=POOL_CHUNKSIZE,
        )
        for summary, task_errors in tqdm(
            iterator,
            total=len(monthly_tasks),
            desc=f"UV{level_hpa} radial wind",
            unit="month",
        ):
            summaries.append(summary)
            errors.extend(task_errors)

    summary_frame = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
    summary_frame = summary_frame.sort_values("YEAR_MONTH").reset_index(drop=True)
    error_frame = pd.DataFrame(errors, columns=ERROR_COLUMNS)

    write_csv_atomic(summary_frame, log_dir / "processing_month_log.csv")
    write_csv_atomic(error_frame, log_dir / "processing_error_log.csv")

    totals = {
        column: int(summary_frame[column].sum())
        for column in SUMMARY_COLUMNS[2:]
    }
    return totals, log_dir


# ============================================================
# 7. Main program
# ============================================================

def main():
    """Calculate radial wind at all 16 pressure levels."""
    print("=" * 88)
    print("Reading and validating the new IBTrACS BASE table")
    print("=" * 88)

    frame, invalid_motion, unavailable = read_source_table()
    monthly_tasks = records_by_month(frame)

    print(f"BASE input: {SOURCE_CSV}")
    print(f"BASE records in {START_YEAR}-{END_YEAR}: "
          f"{len(frame) + len(invalid_motion) + len(unavailable):,}")
    print(f"Records with unavailable TC translation velocity: {len(invalid_motion):,}")
    print(f"Records without a RUN1 UV500 file: {len(unavailable):,}")
    print(f"Eligible RUN2 records: {len(frame):,}")
    print(f"Monthly tasks per pressure level: {len(monthly_tasks):,}")
    print(f"Worker processes: {min(MAX_WORKERS, len(monthly_tasks))}")

    OVERALL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(
        invalid_motion,
        OVERALL_LOG_DIR / "records_with_invalid_translation_velocity.csv",
    )
    write_csv_atomic(
        unavailable,
        OVERALL_LOG_DIR / "records_without_run1_uv500.csv",
    )

    print("\n" + "=" * 88)
    print("Starting TC-motion-relative radial-wind calculation")
    print("=" * 88)
    print("Positive RADIAL_VEL: inward flow toward the TC centre")
    print("Negative RADIAL_VEL: outward flow away from the TC centre")
    print(f"RUN1/RUN2 root: {ERA5_ROOT}")
    print("Pressure levels (hPa): " + ", ".join(map(str, PRESSURE_LEVELS_HPA)))

    overall_rows = []

    for index, level_hpa in enumerate(PRESSURE_LEVELS_HPA, start=1):
        print("\n" + "-" * 88)
        print(
            f"Pressure level {index}/{len(PRESSURE_LEVELS_HPA)}: "
            f"{level_hpa} hPa"
        )

        totals, log_dir = process_level(level_hpa, monthly_tasks)
        overall_row = {"LEVEL_HPA": level_hpa, **totals, "LOG_DIR": str(log_dir)}
        overall_rows.append(overall_row)

        print(f"Records: {totals['RECORD_COUNT']:,}")
        print(f"New outputs: {totals['SUCCESS_COUNT']:,}")
        print(f"Existing outputs reused: {totals['SKIPPED_EXISTING_COUNT']:,}")
        print(f"Missing U/V inputs: {totals['MISSING_INPUT_COUNT']:,}")
        print(f"Failed records: {totals['FAILED_COUNT']:,}")
        print(f"Output directory: {ERA5_ROOT / f'INFLOW_{level_hpa}'}")

        if totals["MISSING_INPUT_COUNT"] or totals["FAILED_COUNT"]:
            raise RuntimeError(
                f"Level {level_hpa} completed with missing or failed records; "
                f"review {log_dir}"
            )

    overall_frame = pd.DataFrame(overall_rows)
    overall_log = OVERALL_LOG_DIR / "pressure_level_processing_summary.csv"
    write_csv_atomic(overall_frame, overall_log)

    print("\n" + "=" * 88)
    print("RUN2 completed successfully")
    print("=" * 88)
    print(f"Pressure levels completed: {len(overall_rows)}")
    print(f"Eligible TC records per level: {len(frame):,}")
    print(f"Overall summary: {overall_log}")


if __name__ == "__main__":
    main()
