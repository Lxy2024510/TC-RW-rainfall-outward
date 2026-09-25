import os
import gzip
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. Paths and parameters
# ============================================================

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(
    os.environ.get("TC_RW_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))
).expanduser().resolve()
BASE_CSV = (
    PROJECT_ROOT / "Data" / "Processed" / "IBTrACS"
    / "PRE_DATA_IBT_1982_2024_BASE.csv"
)
ERA5_BASE_DIR = (
    PROJECT_ROOT / "Data" / "Raw" / "ERA5" / "PRESS_LEV_HOU"
    / "w700_1980_2025_nc"
)
OUTPUT_BASE_DIR = (
    PROJECT_ROOT / "Data" / "Intermediate" / "ERA5"
    / "W700" / "Grids_500KM"
)
LOG_DIR = (
    PROJECT_ROOT / "Results" / "Quality_control" / "ERA5" / "W700" / "RUN1"
)

START_YEAR = 1982
END_YEAR = 2024

EARTH_RADIUS_KM = 6371.0
RADIUS_KM = 500.0
EXPECTED_PLEV_PA = 70000.0

# Each worker can open a roughly 500 MB monthly file. Adjust for available RAM/I/O.
MAX_WORKERS = 24
POOL_CHUNKSIZE = 1
GZIP_COMPRESS_LEVEL = 1
SKIP_EXISTING = True

# First run: process only the first 500 eligible TC records.
# Set to False after checking the test outputs and logs.
TEST_MODE = False
TEST_MAX_ROWS = 500

# Only flag records in the audit log; do not remove any grid cells.
HIGH_NAN_FRACTION_THRESHOLD = 0.10

OUTPUT_COLUMNS = ["LAT", "LON", "W700", "DIST_KM", "AREA_KM2"]

ROW_LOG_COLUMNS = [
    "ROW_ID",
    "SID",
    "ISO_TIME",
    "RAIN_TIME_ID",
    "CENTER_LAT",
    "CENTER_LON",
    "OUTPUT_PATH",
    "STATUS",
    "GRID_COUNT",
    "VALID_W700_GRIDS",
    "NAN_W700_GRIDS",
    "NAN_FRACTION",
    "HIGH_NAN_FLAG",
    "ERROR",
]


# ============================================================
# 2. Helpers
# ============================================================

def normalize_longitude_180(lon):
    return ((float(lon) + 180.0) % 360.0) - 180.0


def normalize_longitude_360(lon):
    return float(lon) % 360.0


def angular_lon_difference(lon_array, center_lon):
    return ((lon_array - center_lon + 180.0) % 360.0) - 180.0


def rain_time_id_to_timestamp(rain_time_id):
    date_value = pd.to_datetime(rain_time_id[:7], format="%Y%j", errors="raise")
    return date_value + pd.Timedelta(hours=int(rain_time_id[8:10]))


def calculate_grid_area_km2(lat_centers, d_lat_deg, d_lon_deg):
    half_dlat = d_lat_deg / 2.0
    lat_north = np.minimum(90.0, lat_centers + half_dlat)
    lat_south = np.maximum(-90.0, lat_centers - half_dlat)
    return (
        EARTH_RADIUS_KM**2
        * np.radians(d_lon_deg)
        * np.abs(
            np.sin(np.radians(lat_north))
            - np.sin(np.radians(lat_south))
        )
    )


def calculate_distance_grid_km(lat_values, lon_values, center_lat, center_lon):
    grid_lat_rad = np.radians(lat_values[:, None])
    grid_lon_rad = np.radians(lon_values[None, :])
    center_lat_rad = np.radians(center_lat)
    center_lon_rad = np.radians(center_lon)

    dlat = grid_lat_rad - center_lat_rad
    dlon = (grid_lon_rad - center_lon_rad + np.pi) % (2.0 * np.pi) - np.pi
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(center_lat_rad)
        * np.cos(grid_lat_rad)
        * np.sin(dlon / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_output_path(row_id, rain_time_id):
    output_dir = OUTPUT_BASE_DIR / rain_time_id[:4]
    return output_dir / f"row_{row_id:09d}_{rain_time_id}.csv.gz"


def output_is_complete(output_path):
    if not output_path.exists():
        return False
    try:
        return output_path.stat().st_size > 0
    except OSError:
        return False


def write_csv_gzip_atomic(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    try:
        with gzip.open(
            temp_path,
            mode="wt",
            encoding="utf-8",
            newline="",
            compresslevel=GZIP_COMPRESS_LEVEL,
        ) as gzip_file:
            df.to_csv(
                gzip_file,
                index=False,
                columns=OUTPUT_COLUMNS,
                float_format="%.10g",
            )
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise RuntimeError(f"Temporary output is missing or empty: {temp_path}")
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def empty_row_log(record, output_path, status):
    row_id, sid, iso_time, rain_time_id, center_lat, center_lon = record
    return {
        "ROW_ID": row_id,
        "SID": sid,
        "ISO_TIME": iso_time,
        "RAIN_TIME_ID": rain_time_id,
        "CENTER_LAT": center_lat,
        "CENTER_LON": center_lon,
        "OUTPUT_PATH": str(output_path),
        "STATUS": status,
        "GRID_COUNT": 0,
        "VALID_W700_GRIDS": 0,
        "NAN_W700_GRIDS": 0,
        "NAN_FRACTION": np.nan,
        "HIGH_NAN_FLAG": False,
        "ERROR": "",
    }


# ============================================================
# 3. Read and validate BASE CSV
# ============================================================

def read_and_validate_base_csv():
    print("=" * 80)
    print("Reading and validating the BASE CSV")
    print("=" * 80)

    if not BASE_CSV.exists():
        raise FileNotFoundError(f"BASE CSV does not exist: {BASE_CSV}")

    df = pd.read_csv(
        BASE_CSV,
        dtype={"SID": "string", "RAIN_TIME_ID": "string"},
        low_memory=False,
    )

    required = {"SID", "ISO_TIME", "RAIN_TIME_ID", "USA_LAT", "USA_LON"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError("BASE CSV is missing: " + ", ".join(sorted(missing)))

    df = df.reset_index(drop=True)
    df.insert(0, "ROW_ID", np.arange(len(df), dtype=np.int64))
    original_count = len(df)

    df["USA_LAT"] = pd.to_numeric(df["USA_LAT"], errors="coerce")
    df["USA_LON"] = pd.to_numeric(df["USA_LON"], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")

    invalid = (
        df["ISO_TIME"].isna()
        | df["RAIN_TIME_ID"].isna()
        | df["USA_LAT"].isna()
        | df["USA_LON"].isna()
    )
    if invalid.any():
        columns = ["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID", "USA_LAT", "USA_LON"]
        examples = df.loc[invalid, columns].head(20).to_dict("records")
        raise ValueError(
            f"Invalid required BASE fields: count={int(invalid.sum())}; examples={examples}"
        )

    invalid_lat = ~df["USA_LAT"].between(-90.0, 90.0)
    if invalid_lat.any():
        examples = df.loc[invalid_lat, ["ROW_ID", "USA_LAT"]].head(20).to_dict("records")
        raise ValueError(
            f"Out-of-range latitudes: count={int(invalid_lat.sum())}; examples={examples}"
        )

    df["USA_LON"] = ((df["USA_LON"] + 180.0) % 360.0) - 180.0

    valid_format = df["RAIN_TIME_ID"].str.fullmatch(r"\d{7}\.\d{2}", na=False)
    if not valid_format.all():
        examples = df.loc[~valid_format, ["ROW_ID", "RAIN_TIME_ID"]].head(20).to_dict("records")
        raise ValueError(
            f"Invalid RAIN_TIME_ID format: count={int((~valid_format).sum())}; examples={examples}"
        )

    invalid_ids = []
    for rain_time_id in df["RAIN_TIME_ID"].drop_duplicates():
        try:
            parsed = rain_time_id_to_timestamp(rain_time_id)
            if parsed.strftime("%Y%j.%H") != rain_time_id:
                invalid_ids.append(rain_time_id)
        except Exception:
            invalid_ids.append(rain_time_id)
    if invalid_ids:
        raise ValueError(f"Invalid calendar values in RAIN_TIME_ID: {invalid_ids[:20]}")

    expected_id = df["ISO_TIME"].dt.strftime("%Y%j.%H")
    mismatch = expected_id != df["RAIN_TIME_ID"]
    if mismatch.any():
        examples = (
            df.loc[mismatch, ["ROW_ID", "ISO_TIME", "RAIN_TIME_ID"]]
            .assign(EXPECTED_RAIN_TIME_ID=expected_id[mismatch].values)
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            f"ISO_TIME and RAIN_TIME_ID mismatch: count={int(mismatch.sum())}; examples={examples}"
        )

    invalid_hour = ~df["ISO_TIME"].dt.hour.isin(range(0, 24, 3))
    non_exact_hour = (
        (df["ISO_TIME"].dt.minute != 0)
        | (df["ISO_TIME"].dt.second != 0)
        | (df["ISO_TIME"].dt.microsecond != 0)
    )
    if invalid_hour.any() or non_exact_hour.any():
        bad = invalid_hour | non_exact_hour
        examples = df.loc[bad, ["ROW_ID", "ISO_TIME", "RAIN_TIME_ID"]].head(20).to_dict("records")
        raise ValueError(
            f"BASE records outside exact ERA5 3-hour slots: count={int(bad.sum())}; "
            f"examples={examples}"
        )

    duplicate = df.duplicated(subset=["SID", "ISO_TIME"], keep=False)
    if duplicate.any():
        examples = df.loc[
            duplicate, ["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID"]
        ].head(20).to_dict("records")
        raise ValueError(
            f"Duplicate SID + ISO_TIME records: count={int(duplicate.sum())}; examples={examples}"
        )

    eligible = df["ISO_TIME"].dt.year.between(START_YEAR, END_YEAR, inclusive="both")
    df = df.loc[eligible].copy()
    eligible_count = len(df)
    if eligible_count == 0:
        raise RuntimeError(f"No BASE records in {START_YEAR}-{END_YEAR}")

    print(f"Original BASE records: {original_count}")
    print(f"Eligible {START_YEAR}-{END_YEAR} records: {eligible_count}")
    print(f"Eligible time range: {df['ISO_TIME'].min()} to {df['ISO_TIME'].max()}")

    if TEST_MODE:
        df = df.iloc[: min(TEST_MAX_ROWS, len(df))].copy()
        print(f"TEST_MODE: processing the first {len(df)} eligible records")
    else:
        print("TEST_MODE disabled: processing all eligible records")

    return df


# ============================================================
# 4. Monthly tasks and spatial extraction
# ============================================================

def generate_month_tasks(df):
    month_key = df["ISO_TIME"].dt.strftime("%Y%m")
    for year_month, group in df.groupby(month_key, sort=True):
        records = []
        for row in group.itertuples(index=False):
            records.append(
                (
                    int(row.ROW_ID),
                    str(row.SID),
                    row.ISO_TIME.isoformat(sep=" "),
                    str(row.RAIN_TIME_ID),
                    float(row.USA_LAT),
                    float(row.USA_LON),
                )
            )
        yield str(year_month), records


def extract_tc_grid(
    w700,
    all_lat,
    all_lon,
    d_lat_deg,
    d_lon_deg,
    center_lat,
    center_lon,
):
    center_lon_360 = normalize_longitude_360(center_lon)
    radius_lat_deg = np.degrees(RADIUS_KM / EARTH_RADIUS_KM)
    lat_margin_deg = radius_lat_deg + d_lat_deg / 2.0 + 0.05
    lat_indices = np.flatnonzero(np.abs(all_lat - center_lat) <= lat_margin_deg)
    if lat_indices.size == 0:
        raise RuntimeError("No candidate latitude cells")

    candidate_lat = all_lat[lat_indices]
    max_abs_lat = min(89.999, float(np.max(np.abs(candidate_lat))))
    cos_lat = np.cos(np.radians(max_abs_lat))
    if cos_lat <= 1e-6:
        lon_mask = np.ones(all_lon.shape, dtype=bool)
    else:
        lon_margin_deg = radius_lat_deg / cos_lat + d_lon_deg / 2.0 + 0.05
        if lon_margin_deg >= 180.0:
            lon_mask = np.ones(all_lon.shape, dtype=bool)
        else:
            lon_mask = (
                np.abs(angular_lon_difference(all_lon, center_lon_360))
                <= lon_margin_deg
            )

    lon_indices = np.flatnonzero(lon_mask)
    if lon_indices.size == 0:
        raise RuntimeError("No candidate longitude cells")

    candidate_lon = all_lon[lon_indices]
    distance_grid = calculate_distance_grid_km(
        candidate_lat, candidate_lon, center_lat, center_lon_360
    )
    radius_mask = distance_grid <= RADIUS_KM
    if not radius_mask.any():
        raise RuntimeError(f"No grid-cell centres within {RADIUS_KM} km")

    w700_subset = np.asarray(
        w700.isel(lat=lat_indices, lon=lon_indices).load().values,
        dtype=np.float32,
    )
    expected_shape = (len(candidate_lat), len(candidate_lon))
    if w700_subset.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected W700 shape {w700_subset.shape}; expected {expected_shape}"
        )

    lat_grid = np.broadcast_to(candidate_lat[:, None], distance_grid.shape)
    lon_grid = np.broadcast_to(candidate_lon[None, :], distance_grid.shape)
    area_by_lat = calculate_grid_area_km2(candidate_lat, d_lat_deg, d_lon_deg)
    area_grid = np.broadcast_to(area_by_lat[:, None], distance_grid.shape)

    # Use wrapped angular differences for selection and distance calculations,
    # then write longitude in the conventional -180 to 180 range.
    output_lon = ((lon_grid[radius_mask] + 180.0) % 360.0) - 180.0

    result = pd.DataFrame(
        {
            "LAT": lat_grid[radius_mask].astype(np.float32),
            "LON": output_lon.astype(np.float32),
            "W700": w700_subset[radius_mask].astype(np.float32),
            "DIST_KM": distance_grid[radius_mask].astype(np.float32),
            "AREA_KM2": area_grid[radius_mask].astype(np.float32),
        }
    )
    return result.sort_values("DIST_KM", kind="mergesort").reset_index(drop=True)


# ============================================================
# 5. Process one monthly file
# ============================================================

def process_single_month(task):
    year_month, records = task
    nc_path = ERA5_BASE_DIR / f"w700_{year_month}.nc"

    task_result = {
        "YEAR_MONTH": year_month,
        "NC_PATH": str(nc_path),
        "TOTAL_RECORDS": len(records),
        "SUCCESS": 0,
        "SKIPPED_EXISTING": 0,
        "MISSING_FILE": 0,
        "FAILED": 0,
        "OUTPUT_GRIDS": 0,
        "VALID_W700_GRIDS": 0,
        "NAN_W700_GRIDS": 0,
        "HIGH_NAN_RECORDS": 0,
        "STATUS": "UNKNOWN",
        "ERROR": "",
        "ROW_LOGS": [],
    }

    if not nc_path.exists():
        task_result["MISSING_FILE"] = len(records)
        task_result["STATUS"] = "MISSING_FILE"
        for record in records:
            output_path = build_output_path(record[0], record[3])
            log = empty_row_log(record, output_path, "MISSING_FILE")
            log["ERROR"] = f"Missing ERA5 file: {nc_path}"
            task_result["ROW_LOGS"].append(log)
        return task_result

    pending_records = []
    for record in records:
        output_path = build_output_path(record[0], record[3])
        if SKIP_EXISTING and output_is_complete(output_path):
            task_result["SKIPPED_EXISTING"] += 1
            task_result["ROW_LOGS"].append(
                empty_row_log(record, output_path, "SKIPPED_EXISTING")
            )
        else:
            pending_records.append(record)

    if not pending_records:
        task_result["STATUS"] = "ALL_EXISTING"
        return task_result

    try:
        with xr.open_dataset(nc_path, decode_times=True, mask_and_scale=True, cache=False) as ds:
            missing_coords = {"time", "lat", "lon", "plev"}.difference(ds.coords)
            if missing_coords:
                raise KeyError("NetCDF missing coordinates: " + ", ".join(sorted(missing_coords)))
            if "var135" not in ds:
                raise KeyError("NetCDF missing var135")

            all_lat = np.asarray(ds["lat"].values, dtype=np.float64)
            all_lon = np.asarray(ds["lon"].values, dtype=np.float64)
            plev = np.asarray(ds["plev"].values, dtype=np.float64)

            if all_lat.ndim != 1 or all_lon.ndim != 1:
                raise ValueError("lat and lon must be one-dimensional")
            if len(all_lat) < 2 or len(all_lon) < 2:
                raise ValueError("lat or lon coordinate is too short")
            if not np.all(np.isfinite(all_lat)) or not np.all(np.isfinite(all_lon)):
                raise ValueError("lat or lon contains non-finite values")
            if not (np.all(np.diff(all_lat) > 0) or np.all(np.diff(all_lat) < 0)):
                raise ValueError("lat must be strictly monotonic")
            if not np.all(np.diff(all_lon) > 0):
                raise ValueError("lon must be strictly increasing")
            if plev.size != 1 or not np.isclose(plev[0], EXPECTED_PLEV_PA):
                raise ValueError(f"Expected plev=[{EXPECTED_PLEV_PA}], found {plev.tolist()}")

            d_lat_deg = float(np.median(np.abs(np.diff(all_lat))))
            d_lon_deg = float(np.median(np.diff(all_lon)))
            if not np.allclose(np.abs(np.diff(all_lat)), d_lat_deg, rtol=0.0, atol=1e-6):
                raise ValueError("Latitude grid is not regular")
            if not np.allclose(np.diff(all_lon), d_lon_deg, rtol=0.0, atol=1e-6):
                raise ValueError("Longitude grid is not regular")

            variable = ds["var135"]
            if set(variable.dims) != {"time", "plev", "lat", "lon"}:
                raise ValueError(f"Unexpected var135 dimensions: {variable.dims}")
            variable = variable.transpose("time", "plev", "lat", "lon")

            time_index = ds.indexes.get("time")
            if time_index is None or not time_index.is_unique:
                raise ValueError("ERA5 time coordinate is missing or not unique")

            for record in pending_records:
                row_id, sid, iso_time, rain_time_id, center_lat, center_lon = record
                output_path = build_output_path(row_id, rain_time_id)
                log = empty_row_log(record, output_path, "UNKNOWN")
                try:
                    target_time = pd.Timestamp(iso_time)
                    time_position = int(time_index.get_indexer([target_time])[0])
                    if time_position < 0:
                        raise KeyError(f"Time not found in ERA5 file: {target_time}")

                    w700 = variable.isel(time=time_position, plev=0)
                    result = extract_tc_grid(
                        w700,
                        all_lat,
                        all_lon,
                        d_lat_deg,
                        d_lon_deg,
                        center_lat,
                        center_lon,
                    )
                    if result.empty:
                        raise RuntimeError("Extracted grid is empty")

                    grid_count = len(result)
                    nan_count = int(result["W700"].isna().sum())
                    valid_count = grid_count - nan_count
                    nan_fraction = nan_count / grid_count
                    high_nan = nan_fraction > HIGH_NAN_FRACTION_THRESHOLD

                    write_csv_gzip_atomic(result, output_path)

                    task_result["SUCCESS"] += 1
                    task_result["OUTPUT_GRIDS"] += grid_count
                    task_result["VALID_W700_GRIDS"] += valid_count
                    task_result["NAN_W700_GRIDS"] += nan_count
                    task_result["HIGH_NAN_RECORDS"] += int(high_nan)
                    log.update(
                        {
                            "STATUS": "SUCCESS",
                            "GRID_COUNT": grid_count,
                            "VALID_W700_GRIDS": valid_count,
                            "NAN_W700_GRIDS": nan_count,
                            "NAN_FRACTION": nan_fraction,
                            "HIGH_NAN_FLAG": high_nan,
                        }
                    )
                except Exception as row_error:
                    task_result["FAILED"] += 1
                    error_text = f"ROW_ID={row_id}: {type(row_error).__name__}: {row_error}"
                    task_result["ERROR"] = (
                        task_result["ERROR"] + " | " + error_text
                        if task_result["ERROR"]
                        else error_text
                    )
                    log["STATUS"] = "FAILED"
                    log["ERROR"] = error_text
                task_result["ROW_LOGS"].append(log)

        if task_result["FAILED"]:
            task_result["STATUS"] = "PARTIAL_FAILURE"
        elif task_result["SUCCESS"]:
            task_result["STATUS"] = "SUCCESS"
        else:
            task_result["STATUS"] = "NO_NEW_OUTPUT"

    except Exception as file_error:
        task_result["FAILED"] += len(pending_records)
        task_result["STATUS"] = "FILE_ERROR"
        task_result["ERROR"] = f"{type(file_error).__name__}: {file_error}"
        for record in pending_records:
            output_path = build_output_path(record[0], record[3])
            log = empty_row_log(record, output_path, "FILE_ERROR")
            log["ERROR"] = task_result["ERROR"]
            task_result["ROW_LOGS"].append(log)

    return task_result


# ============================================================
# 6. Logs
# ============================================================

def save_processing_logs(results):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    row_logs = []
    task_rows = []
    for result in results:
        row_logs.extend(result.pop("ROW_LOGS", []))
        task_rows.append(result)

    task_df = pd.DataFrame(task_rows)
    row_df = pd.DataFrame(row_logs, columns=ROW_LOG_COLUMNS)

    task_log_path = LOG_DIR / "processing_month_log.csv"
    row_log_path = LOG_DIR / "processing_row_log.csv"
    error_log_path = LOG_DIR / "processing_error_log.csv"
    high_nan_log_path = LOG_DIR / "high_nan_records.csv"
    summary_path = LOG_DIR / "processing_summary.csv"

    task_df.to_csv(task_log_path, index=False)
    row_df.to_csv(row_log_path, index=False)
    row_df.loc[row_df["STATUS"].isin(["MISSING_FILE", "FILE_ERROR", "FAILED"])].to_csv(
        error_log_path, index=False
    )
    row_df.loc[row_df["HIGH_NAN_FLAG"] == True].to_csv(high_nan_log_path, index=False)

    summary = {
        "MONTH_TASK_COUNT": int(len(task_df)),
        "TC_RECORD_COUNT": int(task_df["TOTAL_RECORDS"].sum()),
        "SUCCESS_COUNT": int(task_df["SUCCESS"].sum()),
        "SKIPPED_EXISTING_COUNT": int(task_df["SKIPPED_EXISTING"].sum()),
        "MISSING_FILE_COUNT": int(task_df["MISSING_FILE"].sum()),
        "FAILED_COUNT": int(task_df["FAILED"].sum()),
        "OUTPUT_GRID_COUNT": int(task_df["OUTPUT_GRIDS"].sum()),
        "VALID_W700_GRID_COUNT": int(task_df["VALID_W700_GRIDS"].sum()),
        "NAN_W700_GRID_COUNT": int(task_df["NAN_W700_GRIDS"].sum()),
        "HIGH_NAN_RECORD_COUNT": int(task_df["HIGH_NAN_RECORDS"].sum()),
        "HIGH_NAN_THRESHOLD": HIGH_NAN_FRACTION_THRESHOLD,
    }
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    return summary, task_log_path, row_log_path, error_log_path, high_nan_log_path, summary_path


# ============================================================
# 7. Main
# ============================================================

def main():
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    df = read_and_validate_base_csv()
    total_tasks = int(df["ISO_TIME"].dt.strftime("%Y%m").nunique())

    print("\n" + "=" * 80)
    print("Starting ERA5 W700 500-km grid extraction")
    print("=" * 80)
    print(f"TC records: {len(df)}")
    print(f"Monthly tasks: {total_tasks}")
    print(f"Worker processes: {MAX_WORKERS}")
    print(f"Extraction radius: {RADIUS_KM:.1f} km")
    print(f"ERA5 directory: {ERA5_BASE_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print("Source variable: var135 at 70000 Pa; output column: W700")
    print("NaN W700 values are retained and never treated as zero")

    results = []
    with Pool(processes=MAX_WORKERS, maxtasksperchild=100) as pool:
        iterator = pool.imap_unordered(
            process_single_month,
            generate_month_tasks(df),
            chunksize=POOL_CHUNKSIZE,
        )
        for result in tqdm(
            iterator,
            total=total_tasks,
            desc="ERA5 monthly tasks",
            unit="month",
        ):
            results.append(result)

    if not results:
        raise RuntimeError("No processing results were produced")

    (
        summary,
        task_log_path,
        row_log_path,
        error_log_path,
        high_nan_log_path,
        summary_path,
    ) = save_processing_logs(results)

    print("\n" + "=" * 80)
    print("Processing completed")
    print("=" * 80)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nMonthly task log: {task_log_path}")
    print(f"Row coverage log: {row_log_path}")
    print(f"Error log: {error_log_path}")
    print(f"High-NaN log: {high_nan_log_path}")
    print(f"Summary log: {summary_path}")
    if TEST_MODE:
        print("\nTEST_MODE is enabled. Check outputs, then set TEST_MODE = False.")


if __name__ == "__main__":
    main()
