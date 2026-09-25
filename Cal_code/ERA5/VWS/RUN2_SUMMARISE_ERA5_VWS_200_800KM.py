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
# 1. Paths
# ============================================================

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(
    os.environ.get("TC_RW_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))
).expanduser().resolve()
BASE_CSV = (
    PROJECT_ROOT / "Data" / "Processed" / "IBTrACS"
    / "PRE_DATA_IBT_1982_2024_BASE.csv"
)
VWS_GRID_DIR = (
    PROJECT_ROOT / "Data" / "Intermediate" / "ERA5"
    / "VWS" / "Grids_200_800KM"
)
SUMMARY_ROOT = PROJECT_ROOT / ".work" / "ERA5" / "VWS" / "RUN2"
YEAR_CACHE_DIR = SUMMARY_ROOT / "year_cache_v1"
LOG_DIR = (
    PROJECT_ROOT / "Results" / "Quality_control" / "ERA5" / "VWS" / "RUN2"
)
FINAL_OUTPUT = (
    PROJECT_ROOT / "Data" / "Processed" / "ERA5" / "VWS" / "Endpoints"
    / "PRE_DATA_IBT_1982_2024_VWS_200_800KM.csv"
)


# ============================================================
# 2. Processing parameters
# ============================================================

START_YEAR = 1982
END_YEAR = 2024

MAX_WORKERS = min(24, cpu_count())
POOL_CHUNKSIZE = 1

# Reuse a cache only when its columns and ROW_ID coverage are valid.
# If the statistical definition changes, use a new cache directory version.
REUSE_YEAR_CACHE = True

# Stop before final merging if any expected file is missing or invalid.
FAIL_ON_FILE_ERROR = True

EXPECTED_INNER_RADIUS_KM = 200.0
EXPECTED_OUTER_RADIUS_KM = 800.0
DISTANCE_TOLERANCE_KM = 0.01


# ============================================================
# 3. Input definitions
# ============================================================

GRID_FILE_PATTERN = re.compile(r"^row_(\d+)_(\d{7}\.\d{2})\.csv\.gz$")

GRID_COLUMNS = ["VWS", "DIST_KM", "AREA_KM2"]

GRID_DTYPES = {
    "VWS": np.float32,
    "DIST_KM": np.float32,
    "AREA_KM2": np.float32,
}

SUMMARY_COLUMNS = [
    "ROW_ID",
    "VWS_AWMEAN_200_800",
    "VWS_VALID_POINT_200_800",
    "VWS_VALID_AREA_200_800",
]


# ============================================================
# 4. General helpers
# ============================================================

def parse_grid_file_name(file_path):
    match = GRID_FILE_PATTERN.fullmatch(file_path.name)
    if match is None:
        raise ValueError(f"Unexpected grid file name: {file_path.name}")
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
            raise RuntimeError(f"Temporary output is missing or empty: {temp_path}")
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
            raise RuntimeError(f"Temporary output is missing or empty: {temp_path}")
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


# ============================================================
# 5. Read and validate BASE table
# ============================================================

def read_base_table():
    print("=" * 80)
    print("Reading the BASE table")
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
        raise KeyError("BASE CSV is missing columns: " + ", ".join(sorted(missing)))

    base_df = base_df.reset_index(drop=True)
    base_df.insert(0, "ROW_ID", np.arange(len(base_df), dtype=np.int64))
    base_df["ISO_TIME"] = pd.to_datetime(base_df["ISO_TIME"], errors="coerce")

    if base_df["ISO_TIME"].isna().any():
        raise ValueError(
            "BASE CSV contains invalid ISO_TIME values. "
            f"Count: {int(base_df['ISO_TIME'].isna().sum())}"
        )

    valid_rain_id = base_df["RAIN_TIME_ID"].str.fullmatch(r"\d{7}\.\d{2}", na=False)
    if not valid_rain_id.all():
        examples = base_df.loc[
            ~valid_rain_id, ["ROW_ID", "ISO_TIME", "RAIN_TIME_ID"]
        ].head(20).to_dict("records")
        raise ValueError(
            "BASE CSV contains invalid RAIN_TIME_ID values. "
            f"Count: {int((~valid_rain_id).sum())}; examples={examples}"
        )

    expected_rain_id = base_df["ISO_TIME"].dt.strftime("%Y%j.%H")
    mismatch = expected_rain_id != base_df["RAIN_TIME_ID"]
    if mismatch.any():
        examples = (
            base_df.loc[mismatch, ["ROW_ID", "ISO_TIME", "RAIN_TIME_ID"]]
            .assign(EXPECTED_RAIN_TIME_ID=expected_rain_id[mismatch].values)
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            "ISO_TIME and RAIN_TIME_ID mismatch. "
            f"Count: {int(mismatch.sum())}; examples={examples}"
        )

    base_df["YEAR"] = base_df["ISO_TIME"].dt.year.astype(np.int16)
    outside_years = ~base_df["YEAR"].between(START_YEAR, END_YEAR, inclusive="both")
    if outside_years.any():
        examples = base_df.loc[
            outside_years, ["ROW_ID", "ISO_TIME", "RAIN_TIME_ID"]
        ].head(20).to_dict("records")
        raise ValueError(
            f"BASE rows outside {START_YEAR}-{END_YEAR}. "
            f"Count: {int(outside_years.sum())}; examples={examples}"
        )

    duplicate = base_df.duplicated(subset=["SID", "ISO_TIME"], keep=False)
    if duplicate.any():
        examples = base_df.loc[
            duplicate, ["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID"]
        ].head(20).to_dict("records")
        raise ValueError(
            "Duplicate SID + ISO_TIME records. "
            f"Count: {int(duplicate.sum())}; examples={examples}"
        )

    print(f"BASE rows: {len(base_df)}")
    print(f"ROW_ID range: {base_df['ROW_ID'].min()} to {base_df['ROW_ID'].max()}")
    print(f"Year range: {base_df['YEAR'].min()} to {base_df['YEAR'].max()}")
    return base_df


# ============================================================
# 6. Construct annual tasks
# ============================================================

def build_year_tasks(base_df):
    year_tasks = []
    for year in sorted(base_df["YEAR"].astype(int).unique().tolist()):
        year_dir = VWS_GRID_DIR / str(year)
        if not year_dir.is_dir():
            raise FileNotFoundError(f"VWS year directory is missing: {year_dir}")

        expected_rows = base_df.loc[
            base_df["YEAR"] == year,
            ["ROW_ID", "RAIN_TIME_ID"],
        ]
        expected_records = [
            (int(row.ROW_ID), str(row.RAIN_TIME_ID))
            for row in expected_rows.itertuples(index=False)
        ]
        year_tasks.append((int(year), str(year_dir), expected_records))

    print(f"Year tasks: {len(year_tasks)}")
    print(f"Worker processes: {MAX_WORKERS}")
    return year_tasks


# ============================================================
# 7. VWS area-weighted summary for the 200-800 km annulus
# ============================================================


def summarise_grid_file(file_path, expected_year, expected_rain_time_id):
    row_id, rain_time_id = parse_grid_file_name(file_path)
    file_year = int(rain_time_id[:4])

    if file_year != expected_year:
        raise ValueError(
            "File year does not match its directory: "
            f"file={file_path.name}, directory_year={expected_year}"
        )
    if rain_time_id != expected_rain_time_id:
        raise ValueError(
            "File RAIN_TIME_ID does not match BASE: "
            f"file={rain_time_id}, expected={expected_rain_time_id}"
        )

    grid_df = pd.read_csv(
        file_path,
        usecols=GRID_COLUMNS,
        dtype=GRID_DTYPES,
        low_memory=False,
    )
    if grid_df.empty:
        raise ValueError(f"Grid file is empty: {file_path}")

    vws = grid_df["VWS"].to_numpy(dtype=np.float64, copy=False)
    distance = grid_df["DIST_KM"].to_numpy(dtype=np.float64, copy=False)
    area = grid_df["AREA_KM2"].to_numpy(dtype=np.float64, copy=False)

    if not np.isfinite(distance).all():
        raise ValueError(f"Non-finite distances were found in {file_path}")
    if not np.isfinite(area).all():
        raise ValueError(f"Non-finite grid areas were found in {file_path}")
    if np.any(area <= 0.0):
        raise ValueError(f"Non-positive grid areas were found in {file_path}")
    if np.isinf(vws).any():
        raise ValueError(f"Infinite VWS values were found in {file_path}")
    finite_vws = np.isfinite(vws)
    if np.any(vws[finite_vws] < 0.0):
        min_vws = float(np.min(vws[finite_vws]))
        raise ValueError(
            f"Negative VWS magnitude was found in {file_path}: {min_vws}"
        )

    min_distance = float(np.min(distance))
    max_distance = float(np.max(distance))
    if min_distance < EXPECTED_INNER_RADIUS_KM - DISTANCE_TOLERANCE_KM:
        raise ValueError(
            "A distance is below the expected 200-km inner radius in "
            f"{file_path}: {min_distance}"
        )
    if max_distance > EXPECTED_OUTER_RADIUS_KM + DISTANCE_TOLERANCE_KM:
        raise ValueError(
            "A distance exceeds the expected 800-km outer radius in "
            f"{file_path}: {max_distance}"
        )

    valid_vws = finite_vws
    valid_point = int(valid_vws.sum())
    if valid_point == 0:
        raise ValueError(f"No finite VWS values were found in {file_path}")

    valid_area = float(area[valid_vws].sum(dtype=np.float64))
    if not np.isfinite(valid_area) or valid_area <= 0.0:
        raise ValueError(f"Invalid total VWS area in {file_path}: {valid_area}")

    weighted_sum = np.sum(
        vws[valid_vws] * area[valid_vws],
        dtype=np.float64,
    )
    area_weighted_mean = float(weighted_sum / valid_area)
    if not np.isfinite(area_weighted_mean):
        raise ValueError(
            f"Non-finite VWS area-weighted mean in {file_path}"
        )

    record = {
        "ROW_ID": row_id,
        "VWS_AWMEAN_200_800": area_weighted_mean,
        "VWS_VALID_POINT_200_800": valid_point,
        "VWS_VALID_AREA_200_800": valid_area,
    }
    return record


# ============================================================
# 8. Annual cache validation and processing
# ============================================================

def validate_year_cache(cache_path, expected_records):
    if not cache_path.exists() or cache_path.stat().st_size == 0:
        return False
    try:
        cache_df = pd.read_csv(cache_path, low_memory=False)
        if list(cache_df.columns) != SUMMARY_COLUMNS:
            return False
        if cache_df["ROW_ID"].duplicated().any():
            return False
        actual_ids = set(cache_df["ROW_ID"].astype(np.int64).tolist())
        expected_ids = {row_id for row_id, _ in expected_records}
        return actual_ids == expected_ids
    except Exception:
        return False


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
        "STATUS": "UNKNOWN",
        "CACHE_PATH": str(cache_path),
        "ERROR_PATH": str(error_path),
    }

    if REUSE_YEAR_CACHE and validate_year_cache(cache_path, expected_records):
        result["FOUND_FILES"] = len(expected_records)
        result["PROCESSED_FILES"] = len(expected_records)
        result["CACHE_REUSED"] = 1
        result["STATUS"] = "CACHE_REUSED"
        return result

    grid_files = sorted(year_dir.glob("row_*.csv.gz"))
    result["FOUND_FILES"] = len(grid_files)

    expected_map = {row_id: rain_time_id for row_id, rain_time_id in expected_records}
    file_map = {}
    errors = []

    for file_path in grid_files:
        try:
            row_id, _ = parse_grid_file_name(file_path)
            if row_id in file_map:
                errors.append(
                    {
                        "YEAR": year,
                        "ROW_ID": row_id,
                        "FILE_PATH": str(file_path),
                        "ERROR_TYPE": "DUPLICATE_ROW_ID",
                        "ERROR_MESSAGE": "More than one grid file uses this ROW_ID.",
                    }
                )
            else:
                file_map[row_id] = file_path
        except Exception as error:
            errors.append(
                {
                    "YEAR": year,
                    "ROW_ID": np.nan,
                    "FILE_PATH": str(file_path),
                    "ERROR_TYPE": type(error).__name__,
                    "ERROR_MESSAGE": str(error),
                }
            )

    expected_ids = set(expected_map)
    found_ids = set(file_map)

    for row_id in sorted(expected_ids - found_ids):
        errors.append(
            {
                "YEAR": year,
                "ROW_ID": row_id,
                "FILE_PATH": "",
                "ERROR_TYPE": "MISSING_GRID_FILE",
                "ERROR_MESSAGE": "No VWS grid file was found for this BASE ROW_ID.",
            }
        )

    for row_id in sorted(found_ids - expected_ids):
        errors.append(
            {
                "YEAR": year,
                "ROW_ID": row_id,
                "FILE_PATH": str(file_map[row_id]),
                "ERROR_TYPE": "UNEXPECTED_ROW_ID",
                "ERROR_MESSAGE": "Grid file ROW_ID is not expected for this year.",
            }
        )

    records = []
    for row_id in sorted(expected_ids & found_ids):
        file_path = file_map[row_id]
        try:
            record = summarise_grid_file(
                file_path,
                expected_year=year,
                expected_rain_time_id=expected_map[row_id],
            )
            if record["ROW_ID"] != row_id:
                raise ValueError(
                    f"Parsed ROW_ID changed unexpectedly: {record['ROW_ID']} != {row_id}"
                )
            records.append(record)
        except Exception as error:
            errors.append(
                {
                    "YEAR": year,
                    "ROW_ID": row_id,
                    "FILE_PATH": str(file_path),
                    "ERROR_TYPE": type(error).__name__,
                    "ERROR_MESSAGE": str(error),
                }
            )

    result["PROCESSED_FILES"] = len(records)
    result["ERROR_COUNT"] = len(errors)

    if errors:
        write_csv_atomic(pd.DataFrame(errors), error_path)
        result["STATUS"] = "FAILED"
        return result

    summary_df = pd.DataFrame(records)
    if summary_df.empty:
        error_df = pd.DataFrame(
            [{
                "YEAR": year,
                "ROW_ID": np.nan,
                "FILE_PATH": str(year_dir),
                "ERROR_TYPE": "EMPTY_YEAR_RESULT",
                "ERROR_MESSAGE": "No annual VWS summary records were generated.",
            }]
        )
        write_csv_atomic(error_df, error_path)
        result["ERROR_COUNT"] = 1
        result["STATUS"] = "FAILED"
        return result

    summary_df = summary_df[SUMMARY_COLUMNS].sort_values("ROW_ID").reset_index(drop=True)
    actual_ids = set(summary_df["ROW_ID"].astype(np.int64).tolist())
    if summary_df["ROW_ID"].duplicated().any() or actual_ids != expected_ids:
        result["ERROR_COUNT"] = 1
        result["STATUS"] = "FAILED"
        return result

    write_gzip_csv_atomic(summary_df, cache_path)
    result["STATUS"] = "SUCCESS"
    return result


# ============================================================
# 9. Combine annual caches and create final table
# ============================================================

def combine_year_caches(year_tasks, expected_total_rows):
    annual_frames = []
    for year, _, expected_records in tqdm(
        year_tasks,
        desc="Reading annual caches",
        unit="year",
    ):
        cache_path = YEAR_CACHE_DIR / f"summary_{year}.csv.gz"
        if not validate_year_cache(cache_path, expected_records):
            raise RuntimeError(f"Annual cache validation failed: {cache_path}")
        annual_frames.append(pd.read_csv(cache_path, low_memory=False))

    summary_df = pd.concat(annual_frames, axis=0, ignore_index=True)
    summary_df["ROW_ID"] = pd.to_numeric(
        summary_df["ROW_ID"], errors="raise"
    ).astype(np.int64)
    summary_df = summary_df.sort_values("ROW_ID").reset_index(drop=True)

    if len(summary_df) != expected_total_rows:
        raise RuntimeError(
            "Combined summary row count does not match BASE: "
            f"summary={len(summary_df)}, base={expected_total_rows}"
        )
    if summary_df["ROW_ID"].duplicated().any():
        raise RuntimeError("Duplicate ROW_ID values exist in the combined summary")

    expected_ids = np.arange(expected_total_rows, dtype=np.int64)
    actual_ids = summary_df["ROW_ID"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_ids, expected_ids):
        raise RuntimeError("Combined summary does not fully cover BASE ROW_ID values")
    return summary_df


def create_final_table(base_df, summary_df):
    base_output_df = base_df.drop(columns=["YEAR"], errors="ignore").copy()
    final_df = base_output_df.merge(
        summary_df,
        on="ROW_ID",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if len(final_df) != len(base_output_df):
        raise RuntimeError("Final merged row count changed")

    completely_missing = final_df[SUMMARY_COLUMNS[1:]].isna().all(axis=1)
    if completely_missing.any():
        missing_ids = final_df.loc[completely_missing, "ROW_ID"].head(20).tolist()
        raise RuntimeError(f"Some BASE rows have no VWS metrics: {missing_ids}")

    print(f"Writing final table: {FINAL_OUTPUT}")
    write_csv_atomic(
        final_df,
        FINAL_OUTPUT,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    return FINAL_OUTPUT


def save_overall_log(processing_results):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame(processing_results).sort_values("YEAR").reset_index(drop=True)
    log_path = LOG_DIR / "year_processing_log.csv"
    write_csv_atomic(log_df, log_path)
    return log_df, log_path


# ============================================================
# 10. Main
# ============================================================

def main():
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    YEAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    base_df = read_base_table()
    year_tasks = build_year_tasks(base_df)

    print("\n" + "=" * 80)
    print("Starting VWS 200-800-km area-weighted summaries")
    print("=" * 80)
    print("Distance range: 200-800 km (no sub-zones)")
    print(f"Worker processes: {MAX_WORKERS}")
    print(f"Grid input directory: {VWS_GRID_DIR}")
    print(f"Annual cache directory: {YEAR_CACHE_DIR}")
    print("VWS values are not clipped or thresholded")

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

    processing_log_df, processing_log_path = save_overall_log(processing_results)
    failed_years = processing_log_df.loc[
        processing_log_df["STATUS"] == "FAILED", "YEAR"
    ].astype(int).tolist()
    total_errors = int(processing_log_df["ERROR_COUNT"].sum())

    print("\n" + "=" * 80)
    print("Year processing summary")
    print("=" * 80)
    print(f"Years processed: {len(processing_log_df)}")
    print(f"Expected grid files: {int(processing_log_df['EXPECTED_FILES'].sum())}")
    print(f"Found grid files: {int(processing_log_df['FOUND_FILES'].sum())}")
    print(f"Successfully summarised files: {int(processing_log_df['PROCESSED_FILES'].sum())}")
    print(f"Reused annual caches: {int(processing_log_df['CACHE_REUSED'].sum())}")
    print(f"Errors: {total_errors}")
    print(f"Year processing log: {processing_log_path}")

    if failed_years:
        print("Failed years: " + ", ".join(str(year) for year in failed_years))
    if total_errors > 0 and FAIL_ON_FILE_ERROR:
        raise RuntimeError(
            "VWS grid-file errors were detected. Final table was not created. "
            f"Review logs in {LOG_DIR}."
        )

    print("\n" + "=" * 80)
    print("Combining annual summary caches")
    print("=" * 80)
    summary_df = combine_year_caches(year_tasks, expected_total_rows=len(base_df))
    print(f"Combined summary rows: {len(summary_df)}")
    print(f"Combined summary columns: {len(summary_df.columns)}")

    print("\n" + "=" * 80)
    print("Creating final BASE-wide VWS table")
    print("=" * 80)
    output_path = create_final_table(base_df, summary_df)

    print("\n" + "=" * 80)
    print("Processing completed successfully")
    print("=" * 80)
    print(f"Final output: {output_path}")


if __name__ == "__main__":
    main()

