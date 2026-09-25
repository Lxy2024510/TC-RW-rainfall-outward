import os
import re
import gzip
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm


warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. Project paths
# ============================================================

# The script is expected at TC-RW-V1/Cal_code/IMERG/.
# TC_RW_PROJECT_ROOT can override automatic project-root detection.
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(
    os.environ.get("TC_RW_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))
).expanduser().resolve()

BASE_CSV = (
    PROJECT_ROOT / "Data" / "Processed" / "IBTrACS"
    / "PRE_DATA_IBT_1982_2024_BASE.csv"
)
IMERG_GRID_DIR = (
    PROJECT_ROOT / "Data" / "Intermediate" / "IMERG" / "Grids_500KM"
)

# v2 identifies the NaN-aware cache schema in this script.
YEAR_CACHE_DIR = (
    PROJECT_ROOT / ".work" / "IMERG" / "RUN2" / "year_cache_v2"
)
LOG_DIR = (
    PROJECT_ROOT / "Results" / "Quality_control" / "IMERG" / "RUN2"
)
FINAL_OUTPUT_DIR = (
    PROJECT_ROOT / "Data" / "Processed" / "IMERG" / "Thresholds"
)


# ============================================================
# 2. Parameters
# ============================================================

START_YEAR = 1998
END_YEAR = 2024

MAX_WORKERS = min(24, cpu_count())
POOL_CHUNKSIZE = 1

THRESHOLDS = [
    1.5,
    10,
    20,
    30,
    40,
    50,
]

REUSE_YEAR_CACHE = True
FAIL_ON_FILE_ERROR = True

EXPECTED_RADIUS_KM = 500.0
DISTANCE_TOLERANCE_KM = 0.01


# ============================================================
# 3. Input file definitions
# ============================================================

GRID_FILE_PATTERN = re.compile(
    r"^row_(\d+)_(\d{7}\.\d{2})\.csv\.gz$"
)

GRID_COLUMNS = [
    "PRECIP_3H",
    "DIST_KM",
    "AREA_KM2",
    "RAIN_VOLUME_M3",
]

GRID_DTYPES = {
    "PRECIP_3H": np.float32,
    "DIST_KM": np.float32,
    "AREA_KM2": np.float32,
    "RAIN_VOLUME_M3": np.float64,
}


# ============================================================
# 4. Zones and output columns
# ============================================================

BASE_ZONE_NAMES = [
    "0_100",
    "100_200",
    "200_300",
    "300_400",
    "400_500",
]

ALL_ZONE_NAMES = [
    "0_100",
    "100_200",
    "200_300",
    "300_400",
    "400_500",
    "0_200",
    "200_500",
    "0_500",
]

# These columns describe source-data coverage and are included in every
# threshold-specific final table.
COVERAGE_COLUMNS = [
    "RAIN_POINT",
    "VALID_RAIN_POINT",
    "NAN_RAIN_POINT",
    "TOTAL_AREA_KM2",
    "VALID_AREA_KM2",
    "NAN_AREA_KM2",
    "VALID_AREA_FRACTION",
    "HAS_NAN_PRECIP",
]


def threshold_label(threshold):
    threshold = float(threshold)
    return str(int(threshold)) if threshold.is_integer() else str(threshold)


def build_threshold_metric_columns(threshold):
    label = threshold_label(threshold)
    columns = [f"RAIN_POINT_{label}"]

    for zone_name in ALL_ZONE_NAMES:
        columns.append(f"IMERG_AREA_{label}_{zone_name}")

    for zone_name in ALL_ZONE_NAMES:
        columns.append(f"IMERG_VOLUME_{label}_{zone_name}")

    columns.append(f"IMERG_DIST_{label}")
    return columns


def build_all_summary_columns():
    columns = ["ROW_ID"] + COVERAGE_COLUMNS.copy()
    for threshold in THRESHOLDS:
        columns.extend(build_threshold_metric_columns(threshold))
    return columns


ALL_SUMMARY_COLUMNS = build_all_summary_columns()


# ============================================================
# 5. General helpers
# ============================================================

def parse_grid_file_name(file_path):
    match = GRID_FILE_PATTERN.fullmatch(file_path.name)
    if match is None:
        raise ValueError(f"Unexpected grid filename: {file_path.name}")
    return int(match.group(1)), match.group(2)


def write_gzip_csv_atomic(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    try:
        with gzip.open(
            temp_path,
            mode="wt",
            encoding="utf-8",
            newline="",
            compresslevel=1,
        ) as gzip_file:
            df.to_csv(gzip_file, index=False)

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise RuntimeError(f"Temporary output missing or empty: {temp_path}")
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def write_csv_atomic(df, output_path, date_format=None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    try:
        df.to_csv(temp_path, index=False, date_format=date_format)
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise RuntimeError(f"Temporary output missing or empty: {temp_path}")
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


# ============================================================
# 6. BASE table: preserve original ROW_ID, then filter years
# ============================================================

def read_base_table():
    print("=" * 80)
    print("Reading and filtering the BASE table")
    print("=" * 80)

    if not BASE_CSV.exists():
        raise FileNotFoundError(f"BASE CSV does not exist: {BASE_CSV}")

    base_df = pd.read_csv(
        BASE_CSV,
        dtype={"SID": "string", "RAIN_TIME_ID": "string"},
        low_memory=False,
    )

    required = {"SID", "ISO_TIME", "RAIN_TIME_ID"}
    missing = required.difference(base_df.columns)
    if missing:
        raise KeyError("BASE CSV is missing: " + ", ".join(sorted(missing)))

    base_df = base_df.reset_index(drop=True)
    base_df.insert(0, "ROW_ID", np.arange(len(base_df), dtype=np.int64))
    original_count = len(base_df)

    base_df["ISO_TIME"] = pd.to_datetime(base_df["ISO_TIME"], errors="coerce")
    if base_df["ISO_TIME"].isna().any():
        raise ValueError(
            "BASE CSV contains invalid ISO_TIME values: "
            f"{int(base_df['ISO_TIME'].isna().sum())}"
        )

    valid_id = base_df["RAIN_TIME_ID"].str.fullmatch(r"\d{7}\.\d{2}", na=False)
    if not valid_id.all():
        examples = base_df.loc[
            ~valid_id, ["ROW_ID", "ISO_TIME", "RAIN_TIME_ID"]
        ].head(20).to_dict("records")
        raise ValueError(
            f"Invalid RAIN_TIME_ID format: count={int((~valid_id).sum())}; "
            f"examples={examples}"
        )

    base_df["YEAR"] = base_df["ISO_TIME"].dt.year.astype(np.int16)
    eligible = base_df["YEAR"].between(START_YEAR, END_YEAR, inclusive="both")
    base_df = base_df.loc[eligible].copy()

    if base_df.empty:
        raise RuntimeError(f"No BASE records in {START_YEAR}-{END_YEAR}")
    if base_df["ROW_ID"].duplicated().any():
        raise RuntimeError("Duplicate ROW_ID values after BASE filtering")

    print(f"Original BASE rows: {original_count}")
    print(f"Eligible {START_YEAR}-{END_YEAR} rows: {len(base_df)}")
    print(
        f"Eligible ROW_ID range: {base_df['ROW_ID'].min()} to "
        f"{base_df['ROW_ID'].max()}"
    )
    print(
        f"Eligible time range: {base_df['ISO_TIME'].min()} to "
        f"{base_df['ISO_TIME'].max()}"
    )
    return base_df


# ============================================================
# 7. Annual tasks
# ============================================================

def build_year_tasks(base_df):
    tasks = []
    for year in range(START_YEAR, END_YEAR + 1):
        year_rows = base_df.loc[base_df["YEAR"] == year]
        if year_rows.empty:
            raise RuntimeError(f"No eligible BASE records for calendar year {year}")

        year_dir = IMERG_GRID_DIR / str(year)
        if not year_dir.is_dir():
            raise FileNotFoundError(f"IMERG year directory is missing: {year_dir}")

        expected_records = [
            (int(row.ROW_ID), str(row.RAIN_TIME_ID))
            for row in year_rows[["ROW_ID", "RAIN_TIME_ID"]].itertuples(index=False)
        ]
        tasks.append((year, str(year_dir), expected_records))

    print(f"Year tasks: {len(tasks)}")
    print(f"Worker processes: {MAX_WORKERS}")
    return tasks


# ============================================================
# 8. Distance zones
# ============================================================

def create_distance_zone_masks(distance):
    return {
        "0_100": (distance >= 0.0) & (distance <= 100.0),
        "100_200": (distance > 100.0) & (distance <= 200.0),
        "200_300": (distance > 200.0) & (distance <= 300.0),
        "300_400": (distance > 300.0) & (distance <= 400.0),
        "400_500": (
            (distance > 400.0)
            & (distance <= EXPECTED_RADIUS_KM + DISTANCE_TOLERANCE_KM)
        ),
    }


# ============================================================
# 9. Summarise one TC grid file
# ============================================================

def summarise_grid_file(file_path, expected_year, expected_rain_time_id):
    row_id, rain_time_id = parse_grid_file_name(file_path)

    if int(rain_time_id[:4]) != expected_year:
        raise ValueError(
            f"Filename year differs from directory year: {file_path.name}"
        )
    if rain_time_id != expected_rain_time_id:
        raise ValueError(
            "Grid filename RAIN_TIME_ID differs from BASE: "
            f"file={rain_time_id}, base={expected_rain_time_id}"
        )

    grid_df = pd.read_csv(
        file_path,
        usecols=GRID_COLUMNS,
        dtype=GRID_DTYPES,
        low_memory=False,
    )
    if grid_df.empty:
        raise ValueError(f"Grid file is empty: {file_path}")

    precipitation = grid_df["PRECIP_3H"].to_numpy(dtype=np.float64, copy=False)
    distance = grid_df["DIST_KM"].to_numpy(dtype=np.float64, copy=False)
    area = grid_df["AREA_KM2"].to_numpy(dtype=np.float64, copy=False)
    volume = grid_df["RAIN_VOLUME_M3"].to_numpy(dtype=np.float64, copy=False)

    # Geometry must never be missing.
    if not np.isfinite(distance).all():
        raise ValueError(f"Non-finite distances in {file_path}")
    if not np.isfinite(area).all():
        raise ValueError(f"Non-finite areas in {file_path}")
    if np.any(distance < 0.0):
        raise ValueError(f"Negative distances in {file_path}")
    if np.any(area <= 0.0):
        raise ValueError(f"Non-positive areas in {file_path}")

    # Precipitation and volume are allowed to be NaN only as a matching pair.
    precip_nan = np.isnan(precipitation)
    volume_nan = np.isnan(volume)
    if not np.array_equal(precip_nan, volume_nan):
        mismatch_count = int(np.count_nonzero(precip_nan != volume_nan))
        raise ValueError(
            "PRECIP_3H and RAIN_VOLUME_M3 NaN masks differ in "
            f"{file_path}; count={mismatch_count}"
        )

    # Infinity remains invalid even though NaN is permitted.
    if np.isinf(precipitation).any() or np.isinf(volume).any():
        raise ValueError(f"Infinite precipitation or volume in {file_path}")

    valid_precip = ~precip_nan
    if np.any(precipitation[valid_precip] < 0.0):
        raise ValueError(f"Negative precipitation in {file_path}")
    if np.any(volume[valid_precip] < 0.0):
        raise ValueError(f"Negative rain volume in {file_path}")

    max_distance = float(np.max(distance))
    if max_distance > EXPECTED_RADIUS_KM + DISTANCE_TOLERANCE_KM:
        raise ValueError(
            f"Distance exceeds 500-km radius in {file_path}: {max_distance}"
        )

    zone_masks = create_distance_zone_masks(distance)
    covered_by_zones = np.logical_or.reduce(
        [zone_masks[name] for name in BASE_ZONE_NAMES]
    )
    if not covered_by_zones.all():
        raise ValueError(
            f"Some grid distances do not belong to any zone in {file_path}; "
            f"count={int((~covered_by_zones).sum())}"
        )

    total_area = float(area.sum(dtype=np.float64))
    valid_area = float(area[valid_precip].sum(dtype=np.float64))
    nan_area = float(area[precip_nan].sum(dtype=np.float64))
    valid_fraction = valid_area / total_area

    record = {
        "ROW_ID": row_id,
        "RAIN_POINT": int(len(grid_df)),
        "VALID_RAIN_POINT": int(valid_precip.sum()),
        "NAN_RAIN_POINT": int(precip_nan.sum()),
        "TOTAL_AREA_KM2": total_area,
        "VALID_AREA_KM2": valid_area,
        "NAN_AREA_KM2": nan_area,
        "VALID_AREA_FRACTION": valid_fraction,
        "HAS_NAN_PRECIP": bool(precip_nan.any()),
    }

    for threshold in THRESHOLDS:
        label = threshold_label(threshold)

        # Critical NaN rule:
        # A grid contributes only when precipitation is finite AND >= threshold.
        # Therefore NaN contributes to neither IMERG_DIST_30's numerator nor
        # denominator (and the same rule applies to every other threshold).
        threshold_mask = valid_precip & (precipitation >= float(threshold))
        record[f"RAIN_POINT_{label}"] = int(threshold_mask.sum())

        area_values = {}
        volume_values = {}

        for zone_name in BASE_ZONE_NAMES:
            selected = threshold_mask & zone_masks[zone_name]
            area_values[zone_name] = float(area[selected].sum(dtype=np.float64))
            volume_values[zone_name] = float(
                volume[selected].sum(dtype=np.float64)
            )

        area_values["0_200"] = area_values["0_100"] + area_values["100_200"]
        area_values["200_500"] = (
            area_values["200_300"]
            + area_values["300_400"]
            + area_values["400_500"]
        )
        area_values["0_500"] = area_values["0_200"] + area_values["200_500"]

        volume_values["0_200"] = (
            volume_values["0_100"] + volume_values["100_200"]
        )
        volume_values["200_500"] = (
            volume_values["200_300"]
            + volume_values["300_400"]
            + volume_values["400_500"]
        )
        volume_values["0_500"] = (
            volume_values["0_200"] + volume_values["200_500"]
        )

        for zone_name in ALL_ZONE_NAMES:
            record[f"IMERG_AREA_{label}_{zone_name}"] = area_values[zone_name]
        for zone_name in ALL_ZONE_NAMES:
            record[f"IMERG_VOLUME_{label}_{zone_name}"] = volume_values[zone_name]

        selected_volume = volume[threshold_mask]
        total_selected_volume = float(selected_volume.sum(dtype=np.float64))

        if total_selected_volume > 0.0:
            weighted_distance = float(
                np.sum(
                    selected_volume * distance[threshold_mask],
                    dtype=np.float64,
                )
                / total_selected_volume
            )
        else:
            # No positive volume at/above the threshold: distance is undefined.
            weighted_distance = np.nan

        record[f"IMERG_DIST_{label}"] = weighted_distance

    return record


# ============================================================
# 10. Annual cache validation
# ============================================================

def validate_year_cache(cache_path, expected_records):
    if not cache_path.exists() or cache_path.stat().st_size == 0:
        return False

    try:
        cache_df = pd.read_csv(cache_path, low_memory=False)
        if list(cache_df.columns) != ALL_SUMMARY_COLUMNS:
            return False
        if cache_df["ROW_ID"].duplicated().any():
            return False

        actual_ids = set(cache_df["ROW_ID"].astype(np.int64).tolist())
        expected_ids = {int(row_id) for row_id, _ in expected_records}
        if actual_ids != expected_ids:
            return False

        # Coverage fields must be complete and internally plausible.
        coverage_numeric = [
            "RAIN_POINT",
            "VALID_RAIN_POINT",
            "NAN_RAIN_POINT",
            "TOTAL_AREA_KM2",
            "VALID_AREA_KM2",
            "NAN_AREA_KM2",
            "VALID_AREA_FRACTION",
        ]
        if cache_df[coverage_numeric].isna().any().any():
            return False
        if not np.array_equal(
            cache_df["RAIN_POINT"].to_numpy(dtype=np.int64),
            (
                cache_df["VALID_RAIN_POINT"] + cache_df["NAN_RAIN_POINT"]
            ).to_numpy(dtype=np.int64),
        ):
            return False
        if not cache_df["VALID_AREA_FRACTION"].between(0.0, 1.0).all():
            return False
        return True
    except Exception:
        return False


# ============================================================
# 11. Process one year
# ============================================================

def process_year_task(task):
    year, year_dir_text, expected_records = task
    year_dir = Path(year_dir_text)
    cache_path = YEAR_CACHE_DIR / f"summary_{year}.csv.gz"
    error_path = LOG_DIR / f"errors_{year}.csv"

    result = {
        "YEAR": year,
        "EXPECTED_FILES": len(expected_records),
        "FOUND_FILES": 0,
        "PROCESSED_FILES": 0,
        "CACHE_REUSED": 0,
        "ERROR_COUNT": 0,
        "NAN_RECORD_COUNT": 0,
        "NAN_GRID_COUNT": 0,
        "STATUS": "UNKNOWN",
        "CACHE_PATH": str(cache_path),
        "ERROR_PATH": str(error_path),
    }

    if REUSE_YEAR_CACHE and validate_year_cache(cache_path, expected_records):
        cached = pd.read_csv(
            cache_path,
            usecols=["NAN_RAIN_POINT", "HAS_NAN_PRECIP"],
            low_memory=False,
        )
        result["FOUND_FILES"] = len(expected_records)
        result["PROCESSED_FILES"] = len(expected_records)
        result["CACHE_REUSED"] = 1
        # Derive this from the numeric count rather than CSV boolean parsing.
        result["NAN_RECORD_COUNT"] = int((cached["NAN_RAIN_POINT"] > 0).sum())
        result["NAN_GRID_COUNT"] = int(cached["NAN_RAIN_POINT"].sum())
        result["STATUS"] = "CACHE_REUSED"
        return result

    expected_id_to_time = {row_id: time_id for row_id, time_id in expected_records}
    expected_ids = set(expected_id_to_time)
    grid_files = sorted(year_dir.glob("row_*.csv.gz"))
    result["FOUND_FILES"] = len(grid_files)

    file_map = {}
    errors = []

    for file_path in grid_files:
        try:
            row_id, rain_time_id = parse_grid_file_name(file_path)
            if row_id in file_map:
                errors.append({
                    "YEAR": year,
                    "ROW_ID": row_id,
                    "FILE_PATH": str(file_path),
                    "ERROR_TYPE": "DUPLICATE_ROW_ID",
                    "ERROR_MESSAGE": "More than one grid file uses this ROW_ID",
                })
            else:
                file_map[row_id] = (file_path, rain_time_id)
        except Exception as error:
            errors.append({
                "YEAR": year,
                "ROW_ID": np.nan,
                "FILE_PATH": str(file_path),
                "ERROR_TYPE": type(error).__name__,
                "ERROR_MESSAGE": str(error),
            })

    found_ids = set(file_map)
    for row_id in sorted(expected_ids - found_ids):
        errors.append({
            "YEAR": year,
            "ROW_ID": row_id,
            "FILE_PATH": "",
            "ERROR_TYPE": "MISSING_GRID_FILE",
            "ERROR_MESSAGE": "No grid file was found for this BASE ROW_ID",
        })
    for row_id in sorted(found_ids - expected_ids):
        errors.append({
            "YEAR": year,
            "ROW_ID": row_id,
            "FILE_PATH": str(file_map[row_id][0]),
            "ERROR_TYPE": "UNEXPECTED_ROW_ID",
            "ERROR_MESSAGE": "Grid ROW_ID is not expected for this year",
        })

    records = []
    for row_id in sorted(expected_ids & found_ids):
        file_path, parsed_time_id = file_map[row_id]
        expected_time_id = expected_id_to_time[row_id]

        if parsed_time_id != expected_time_id:
            errors.append({
                "YEAR": year,
                "ROW_ID": row_id,
                "FILE_PATH": str(file_path),
                "ERROR_TYPE": "RAIN_TIME_ID_MISMATCH",
                "ERROR_MESSAGE": (
                    f"filename={parsed_time_id}; BASE={expected_time_id}"
                ),
            })
            continue

        try:
            records.append(
                summarise_grid_file(file_path, year, expected_time_id)
            )
        except Exception as error:
            errors.append({
                "YEAR": year,
                "ROW_ID": row_id,
                "FILE_PATH": str(file_path),
                "ERROR_TYPE": type(error).__name__,
                "ERROR_MESSAGE": str(error),
            })

    result["PROCESSED_FILES"] = len(records)
    result["ERROR_COUNT"] = len(errors)

    if errors:
        write_csv_atomic(pd.DataFrame(errors), error_path)
        result["STATUS"] = "FAILED"
        return result

    summary_df = pd.DataFrame(records)
    if summary_df.empty:
        result["ERROR_COUNT"] = 1
        result["STATUS"] = "FAILED"
        write_csv_atomic(
            pd.DataFrame([{
                "YEAR": year,
                "ROW_ID": np.nan,
                "FILE_PATH": str(year_dir),
                "ERROR_TYPE": "EMPTY_YEAR_RESULT",
                "ERROR_MESSAGE": "No annual summary records were generated",
            }]),
            error_path,
        )
        return result

    summary_df = summary_df[ALL_SUMMARY_COLUMNS]
    summary_df = summary_df.sort_values("ROW_ID").reset_index(drop=True)

    actual_ids = set(summary_df["ROW_ID"].astype(np.int64).tolist())
    if summary_df["ROW_ID"].duplicated().any() or actual_ids != expected_ids:
        result["ERROR_COUNT"] = 1
        result["STATUS"] = "FAILED"
        write_csv_atomic(
            pd.DataFrame([{
                "YEAR": year,
                "ROW_ID": np.nan,
                "FILE_PATH": str(year_dir),
                "ERROR_TYPE": "ANNUAL_ROW_ID_VALIDATION_FAILED",
                "ERROR_MESSAGE": "Annual summary ROW_ID coverage is invalid",
            }]),
            error_path,
        )
        return result

    result["NAN_RECORD_COUNT"] = int(summary_df["HAS_NAN_PRECIP"].sum())
    result["NAN_GRID_COUNT"] = int(summary_df["NAN_RAIN_POINT"].sum())
    write_gzip_csv_atomic(summary_df, cache_path)
    result["STATUS"] = "SUCCESS"
    return result


# ============================================================
# 12. Combine annual caches
# ============================================================

def combine_year_caches(year_tasks, expected_row_ids):
    frames = []

    for year, _, expected_records in tqdm(
        year_tasks,
        desc="Reading annual caches",
        unit="year",
    ):
        cache_path = YEAR_CACHE_DIR / f"summary_{year}.csv.gz"
        if not validate_year_cache(cache_path, expected_records):
            raise RuntimeError(f"Annual cache validation failed: {cache_path}")
        frames.append(pd.read_csv(cache_path, low_memory=False))

    summary_df = pd.concat(frames, ignore_index=True)
    summary_df["ROW_ID"] = pd.to_numeric(
        summary_df["ROW_ID"], errors="raise"
    ).astype(np.int64)
    summary_df = summary_df.sort_values("ROW_ID").reset_index(drop=True)

    expected_ids = np.sort(np.asarray(expected_row_ids, dtype=np.int64))
    actual_ids = summary_df["ROW_ID"].to_numpy(dtype=np.int64)

    if len(summary_df) != len(expected_ids):
        raise RuntimeError(
            f"Combined row count mismatch: summary={len(summary_df)}, "
            f"expected={len(expected_ids)}"
        )
    if summary_df["ROW_ID"].duplicated().any():
        raise RuntimeError("Duplicate ROW_ID values in combined summary")
    if not np.array_equal(actual_ids, expected_ids):
        raise RuntimeError("Combined ROW_ID set differs from eligible BASE ROW_ID set")

    return summary_df


# ============================================================
# 13. Final threshold-specific BASE tables
# ============================================================

def create_final_threshold_tables(base_df, summary_df):
    output_paths = []
    base_output = base_df.drop(columns=["YEAR"], errors="ignore").copy()

    for threshold in THRESHOLDS:
        label = threshold_label(threshold)
        metric_columns = (
            ["ROW_ID"]
            + COVERAGE_COLUMNS
            + build_threshold_metric_columns(threshold)
        )
        threshold_summary = summary_df[metric_columns].copy()

        final_df = base_output.merge(
            threshold_summary,
            on="ROW_ID",
            how="left",
            validate="one_to_one",
            sort=False,
        )

        if len(final_df) != len(base_output):
            raise RuntimeError(f"Final row count changed for threshold {label}")

        # IMERG_DIST may legitimately be NaN if no grid reaches the threshold.
        # Coverage fields and count/area/volume metrics must never be missing.
        required_nonmissing = COVERAGE_COLUMNS + [f"RAIN_POINT_{label}"]
        required_nonmissing += [
            f"IMERG_AREA_{label}_{zone}" for zone in ALL_ZONE_NAMES
        ]
        required_nonmissing += [
            f"IMERG_VOLUME_{label}_{zone}" for zone in ALL_ZONE_NAMES
        ]

        if final_df[required_nonmissing].isna().any().any():
            bad_ids = final_df.loc[
                final_df[required_nonmissing].isna().any(axis=1), "ROW_ID"
            ].head(20).tolist()
            raise RuntimeError(
                f"Missing required metrics for threshold {label}: {bad_ids}"
            )

        output_path = FINAL_OUTPUT_DIR / (
            f"PRE_DATA_IBT_{START_YEAR}_{END_YEAR}_"
            f"IMERG_THRESHOLD_{label}.csv"
        )
        print(f"Writing threshold {label}: {output_path}")
        write_csv_atomic(
            final_df,
            output_path,
            date_format="%Y-%m-%d %H:%M:%S",
        )
        output_paths.append(output_path)

    return output_paths


# ============================================================
# 14. Logs
# ============================================================

def save_overall_log(processing_results):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame(processing_results).sort_values("YEAR").reset_index(drop=True)
    log_path = LOG_DIR / "year_processing_log.csv"
    write_csv_atomic(log_df, log_path)
    return log_df, log_path


def write_nan_coverage_log(base_df, summary_df):
    nan_summary = summary_df.loc[
        summary_df["NAN_RAIN_POINT"] > 0,
        ["ROW_ID"] + COVERAGE_COLUMNS,
    ].copy()

    base_columns = [
        column
        for column in [
            "ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID", "USA_LAT", "USA_LON"
        ]
        if column in base_df.columns
    ]
    nan_log = base_df[base_columns].merge(
        nan_summary,
        on="ROW_ID",
        how="inner",
        validate="one_to_one",
    )
    path = LOG_DIR / "nan_coverage_records.csv"
    write_csv_atomic(nan_log, path, date_format="%Y-%m-%d %H:%M:%S")
    return nan_log, path


# ============================================================
# 15. Main
# ============================================================

def main():
    YEAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_df = read_base_table()
    year_tasks = build_year_tasks(base_df)

    print("\n" + "=" * 80)
    print("Starting IMERG threshold summary calculation")
    print("=" * 80)
    print(
        "Thresholds (mm/3h): "
        + ", ".join(threshold_label(value) for value in THRESHOLDS)
    )
    print(f"Worker processes: {MAX_WORKERS}")
    print(f"Grid input directory: {IMERG_GRID_DIR}")
    print(f"Annual cache directory: {YEAR_CACHE_DIR}")
    print("NaN precipitation does not participate in threshold or distance metrics")

    processing_results = []
    with Pool(processes=MAX_WORKERS, maxtasksperchild=2) as pool:
        iterator = pool.imap_unordered(
            process_year_task,
            year_tasks,
            chunksize=POOL_CHUNKSIZE,
        )
        for result in tqdm(
            iterator,
            total=len(year_tasks),
            desc="Processing years",
            unit="year",
        ):
            processing_results.append(result)

    processing_log, processing_log_path = save_overall_log(processing_results)
    failed_years = processing_log.loc[
        processing_log["STATUS"] == "FAILED", "YEAR"
    ].astype(int).tolist()
    total_errors = int(processing_log["ERROR_COUNT"].sum())

    print("\n" + "=" * 80)
    print("Year processing summary")
    print("=" * 80)
    print(f"Years: {len(processing_log)}")
    print(f"Expected files: {int(processing_log['EXPECTED_FILES'].sum())}")
    print(f"Found files: {int(processing_log['FOUND_FILES'].sum())}")
    print(f"Processed files: {int(processing_log['PROCESSED_FILES'].sum())}")
    print(f"Reused annual caches: {int(processing_log['CACHE_REUSED'].sum())}")
    print(f"Records containing NaN: {int(processing_log['NAN_RECORD_COUNT'].sum())}")
    print(f"NaN grid cells: {int(processing_log['NAN_GRID_COUNT'].sum())}")
    print(f"Errors: {total_errors}")
    print(f"Processing log: {processing_log_path}")

    if failed_years:
        print("Failed years: " + ", ".join(str(year) for year in failed_years))
    if total_errors > 0 and FAIL_ON_FILE_ERROR:
        raise RuntimeError(
            f"Grid errors detected; review {LOG_DIR}. Final tables were not created."
        )

    print("\n" + "=" * 80)
    print("Combining annual summary caches")
    print("=" * 80)
    summary_df = combine_year_caches(
        year_tasks,
        expected_row_ids=base_df["ROW_ID"].to_numpy(dtype=np.int64),
    )
    print(f"Combined rows: {len(summary_df)}")
    print(f"Combined columns: {len(summary_df.columns)}")

    nan_log, nan_log_path = write_nan_coverage_log(base_df, summary_df)
    print(f"NaN-affected records: {len(nan_log)}")
    print(f"NaN coverage log: {nan_log_path}")

    print("\n" + "=" * 80)
    print("Creating threshold-specific BASE tables")
    print("=" * 80)
    output_paths = create_final_threshold_tables(base_df, summary_df)

    print("\n" + "=" * 80)
    print("Processing completed successfully")
    print("=" * 80)
    print(f"Final output files: {len(output_paths)}")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
