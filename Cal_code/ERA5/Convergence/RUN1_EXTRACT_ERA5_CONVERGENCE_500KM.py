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
    PROJECT_ROOT / "Data" / "Processed" / "MSWEP" / "Thresholds"
    / "PRE_DATA_IBT_1982_2024_MSWEP_THRESHOLD_30.csv"
)
CONVERGENCE_INTERMEDIATE_ROOT = (
    PROJECT_ROOT / "Data" / "Intermediate" / "ERA5" / "Convergence"
)
CLEANED_BASE_CSV = CONVERGENCE_INTERMEDIATE_ROOT / (
    "PRE_DATA_IBT_1982_2024_MSWEP_THRESHOLD_30_CLEANED.csv"
)
ERA5_ROOT = PROJECT_ROOT / "Data" / "Raw" / "ERA5" / "PRESS_LEV_HOU"
OUTPUT_ROOT = CONVERGENCE_INTERMEDIATE_ROOT / "Grids_500KM"
OVERALL_LOG_DIR = (
    PROJECT_ROOT / "Results" / "Quality_control" / "ERA5"
    / "Convergence" / "RUN1"
)

PRESSURE_LEVELS_HPA = [
    500, 550, 600, 650, 700, 750, 775, 800,
    825, 850, 875, 900, 925, 950, 975, 1000,
]

# These globals are configured for one pressure level before its Pool starts.
CURRENT_LEVEL_HPA = None
EXPECTED_PLEV_PA = None
ERA5_BASE_DIR = None
OUTPUT_BASE_DIR = None
LOG_DIR = None
DATA_COLUMN = None
CURRENT_FIELD_KIND = None
CURRENT_FIELD_LABEL = None
FILE_PREFIX = None
SOURCE_VARIABLE = None
HAS_PLEV = None

START_YEAR = 1982
END_YEAR = 2024

EARTH_RADIUS_KM = 6371.0
RADIUS_KM = 500.0
# Each worker can open a roughly 500 MB monthly file. Eight workers avoid
# excessive memory use and disk contention on most servers. Increase only
# after checking RAM and storage throughput.
MAX_WORKERS = 8
POOL_CHUNKSIZE = 1
GZIP_COMPRESS_LEVEL = 1
SKIP_EXISTING = True
STRICT_EXISTING_CHECK = False

# First run: process only the first 500 eligible TC records.
# Set to False after checking the test outputs and logs.
TEST_MODE = False
TEST_MAX_ROWS = 500

# Only flags records in the audit log; it does not remove any grid cells.
HIGH_NAN_FRACTION_THRESHOLD = 0.10

OUTPUT_COLUMNS = ["LAT", "LON", "FIELD", "DIST_KM", "AREA_KM2"]

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
    "VALID_FIELD_GRIDS",
    "NAN_FIELD_GRIDS",
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


def configure_level(level_hpa):
    global CURRENT_LEVEL_HPA
    global EXPECTED_PLEV_PA
    global ERA5_BASE_DIR
    global OUTPUT_BASE_DIR
    global LOG_DIR
    global DATA_COLUMN
    global OUTPUT_COLUMNS
    global CURRENT_FIELD_KIND
    global CURRENT_FIELD_LABEL
    global FILE_PREFIX
    global SOURCE_VARIABLE
    global HAS_PLEV

    CURRENT_LEVEL_HPA = int(level_hpa)
    EXPECTED_PLEV_PA = float(CURRENT_LEVEL_HPA * 100)
    ERA5_BASE_DIR = ERA5_ROOT / f"div{CURRENT_LEVEL_HPA}_1980_2025_nc"
    OUTPUT_BASE_DIR = OUTPUT_ROOT / f"CONV{CURRENT_LEVEL_HPA}"
    LOG_DIR = OVERALL_LOG_DIR / CURRENT_FIELD_LABEL
    DATA_COLUMN = f"CONV{CURRENT_LEVEL_HPA}"
    CURRENT_FIELD_KIND = "CONV"
    CURRENT_FIELD_LABEL = DATA_COLUMN
    FILE_PREFIX = f"div{CURRENT_LEVEL_HPA}"
    SOURCE_VARIABLE = "var155"
    HAS_PLEV = True
    OUTPUT_COLUMNS = ["LAT", "LON", DATA_COLUMN, "DIST_KM", "AREA_KM2"]


def configure_sp0():
    global CURRENT_LEVEL_HPA
    global EXPECTED_PLEV_PA
    global ERA5_BASE_DIR
    global OUTPUT_BASE_DIR
    global LOG_DIR
    global DATA_COLUMN
    global OUTPUT_COLUMNS
    global CURRENT_FIELD_KIND
    global CURRENT_FIELD_LABEL
    global FILE_PREFIX
    global SOURCE_VARIABLE
    global HAS_PLEV

    CURRENT_LEVEL_HPA = None
    EXPECTED_PLEV_PA = None
    ERA5_BASE_DIR = ERA5_ROOT / "sp0_1980_2025_nc"
    OUTPUT_BASE_DIR = OUTPUT_ROOT / "SP0_500KM"
    LOG_DIR = OVERALL_LOG_DIR / CURRENT_FIELD_LABEL
    DATA_COLUMN = "SP0"
    CURRENT_FIELD_KIND = "SP0"
    CURRENT_FIELD_LABEL = "SP0"
    FILE_PREFIX = "sp0"
    SOURCE_VARIABLE = "var134"
    HAS_PLEV = False
    OUTPUT_COLUMNS = ["LAT", "LON", DATA_COLUMN, "DIST_KM", "AREA_KM2"]


def configure_worker(field_kind, level_hpa):
    if field_kind == "CONV":
        configure_level(level_hpa)
    elif field_kind == "SP0":
        configure_sp0()
    else:
        raise ValueError(f"Unsupported field kind: {field_kind}")


def build_output_path(row_id, rain_time_id):
    output_dir = OUTPUT_BASE_DIR / rain_time_id[:4]
    return output_dir / f"row_{row_id:09d}_{rain_time_id}.csv.gz"


def output_is_complete(output_path):
    if not output_path.exists():
        return False
    try:
        if output_path.stat().st_size <= 0:
            return False
        if not STRICT_EXISTING_CHECK:
            return True
        with gzip.open(output_path, mode="rt", encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")
        return header == OUTPUT_COLUMNS
    except (OSError, EOFError, gzip.BadGzipFile, UnicodeDecodeError):
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


def write_csv_atomic(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    try:
        df.to_csv(
            temp_path,
            index=False,
            date_format="%Y-%m-%d %H:%M:%S",
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
        "VALID_FIELD_GRIDS": 0,
        "NAN_FIELD_GRIDS": 0,
        "NAN_FRACTION": np.nan,
        "HIGH_NAN_FLAG": False,
        "ERROR": "",
    }


# ============================================================
# 3. Read and validate BASE CSV
# ============================================================

def read_and_validate_base_csv():
    print("=" * 80)
    print("Reading, filtering, and validating the MSWEP source CSV")
    print("=" * 80)

    if not BASE_CSV.exists():
        raise FileNotFoundError(f"BASE CSV does not exist: {BASE_CSV}")

    required_columns = [
        "ROW_ID",
        "SID",
        "ISO_TIME",
        "RAIN_TIME_ID",
        "USA_LAT",
        "USA_LON",
        "MSWEP_DIST_30",
    ]
    df = pd.read_csv(
        BASE_CSV,
        usecols=required_columns,
        dtype={"SID": "string", "RAIN_TIME_ID": "string"},
        low_memory=False,
    )

    required = {
        "ROW_ID",
        "SID",
        "ISO_TIME",
        "RAIN_TIME_ID",
        "USA_LAT",
        "USA_LON",
        "MSWEP_DIST_30",
    }
    missing = required.difference(df.columns)
    if missing:
        raise KeyError("BASE CSV is missing: " + ", ".join(sorted(missing)))

    original_count = len(df)

    row_id_numeric = pd.to_numeric(df["ROW_ID"], errors="coerce")
    invalid_row_id = (
        row_id_numeric.isna()
        | ~np.isfinite(row_id_numeric.to_numpy(dtype=np.float64))
        | (row_id_numeric < 0)
        | (row_id_numeric % 1 != 0)
    )
    if invalid_row_id.any():
        examples = df.loc[invalid_row_id, ["ROW_ID", "SID", "ISO_TIME"]].head(20)
        raise ValueError(
            f"Invalid source ROW_ID values: count={int(invalid_row_id.sum())}; "
            f"examples={examples.to_dict('records')}"
        )
    df["ROW_ID"] = row_id_numeric.astype(np.int64)
    if df["ROW_ID"].duplicated().any():
        examples = df.loc[
            df["ROW_ID"].duplicated(keep=False), "ROW_ID"
        ].head(20).tolist()
        raise ValueError(f"Duplicate source ROW_ID values: {examples}")

    df["MSWEP_DIST_30"] = pd.to_numeric(df["MSWEP_DIST_30"], errors="coerce")
    valid_mswep_distance = np.isfinite(
        df["MSWEP_DIST_30"].to_numpy(dtype=np.float64)
    )
    removed_missing_mswep = int((~valid_mswep_distance).sum())
    df = df.loc[valid_mswep_distance].copy().reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No rows have a finite MSWEP_DIST_30 value")
    if (df["MSWEP_DIST_30"] < 0.0).any():
        examples = df.loc[
            df["MSWEP_DIST_30"] < 0.0,
            ["ROW_ID", "SID", "ISO_TIME", "MSWEP_DIST_30"],
        ].head(20).to_dict("records")
        raise ValueError(f"Negative MSWEP_DIST_30 values were found: {examples}")

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

    write_csv_atomic(df, CLEANED_BASE_CSV)

    df.attrs["source_filter_stats"] = {
        "SOURCE_PATH": str(BASE_CSV),
        "CLEANED_PATH": str(CLEANED_BASE_CSV),
        "SOURCE_ROW_COUNT": original_count,
        "REMOVED_MISSING_OR_NONFINITE_MSWEP_DIST_30": removed_missing_mswep,
        "CLEANED_ROW_COUNT": eligible_count,
        "START_YEAR": START_YEAR,
        "END_YEAR": END_YEAR,
    }

    print(f"Original MSWEP records: {original_count}")
    print(f"Rows removed for missing/non-finite MSWEP_DIST_30: {removed_missing_mswep}")
    print(f"Eligible {START_YEAR}-{END_YEAR} cleaned records: {eligible_count}")
    print(f"Eligible time range: {df['ISO_TIME'].min()} to {df['ISO_TIME'].max()}")
    print(f"Cleaned source table: {CLEANED_BASE_CSV}")

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
    field_data,
    all_lat,
    all_lon,
    d_lat_deg,
    d_lon_deg,
    all_area_by_lat,
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

    field_subset = np.asarray(
        field_data.isel(lat=lat_indices, lon=lon_indices).load().values,
        dtype=np.float32,
    )
    expected_shape = (len(candidate_lat), len(candidate_lon))
    if field_subset.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected {DATA_COLUMN} shape {field_subset.shape}; "
            f"expected {expected_shape}"
        )

    lat_grid = np.broadcast_to(candidate_lat[:, None], distance_grid.shape)
    lon_grid = np.broadcast_to(candidate_lon[None, :], distance_grid.shape)
    area_by_lat = all_area_by_lat[lat_indices]
    area_grid = np.broadcast_to(area_by_lat[:, None], distance_grid.shape)

    # Keep ERA5's 0-360 coordinates for cyclic selection and distance
    # calculations, but write longitude in the same -180 to 180 convention
    # as the validated TC centre longitude. This remains safe at the
    # international date line because selection and distance were calculated
    # with wrapped angular differences before this output-only conversion.
    output_lon = ((lon_grid[radius_mask] + 180.0) % 360.0) - 180.0

    result = pd.DataFrame(
        {
            "LAT": lat_grid[radius_mask].astype(np.float32),
            "LON": output_lon.astype(np.float32),
            DATA_COLUMN: field_subset[radius_mask].astype(np.float32),
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
    nc_path = ERA5_BASE_DIR / f"{FILE_PREFIX}_{year_month}.nc"

    task_result = {
        "YEAR_MONTH": year_month,
        "NC_PATH": str(nc_path),
        "TOTAL_RECORDS": len(records),
        "SUCCESS": 0,
        "SKIPPED_EXISTING": 0,
        "MISSING_FILE": 0,
        "FAILED": 0,
        "OUTPUT_GRIDS": 0,
        "VALID_FIELD_GRIDS": 0,
        "NAN_FIELD_GRIDS": 0,
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
            required_coords = {"time", "lat", "lon"}
            if HAS_PLEV:
                required_coords.add("plev")
            missing_coords = required_coords.difference(ds.coords)
            if missing_coords:
                raise KeyError("NetCDF missing coordinates: " + ", ".join(sorted(missing_coords)))
            if SOURCE_VARIABLE not in ds:
                raise KeyError(f"NetCDF missing {SOURCE_VARIABLE}")

            all_lat = np.asarray(ds["lat"].values, dtype=np.float64)
            all_lon = np.asarray(ds["lon"].values, dtype=np.float64)

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
            if HAS_PLEV:
                plev = np.asarray(ds["plev"].values, dtype=np.float64)
                if plev.size != 1 or not np.isclose(plev[0], EXPECTED_PLEV_PA):
                    raise ValueError(
                        f"Expected plev=[{EXPECTED_PLEV_PA}], found {plev.tolist()}"
                    )

            d_lat_deg = float(np.median(np.abs(np.diff(all_lat))))
            d_lon_deg = float(np.median(np.diff(all_lon)))
            if not np.allclose(np.abs(np.diff(all_lat)), d_lat_deg, rtol=0.0, atol=1e-6):
                raise ValueError("Latitude grid is not regular")
            if not np.allclose(np.diff(all_lon), d_lon_deg, rtol=0.0, atol=1e-6):
                raise ValueError("Longitude grid is not regular")

            # Cell area depends only on latitude and grid spacing, so compute it
            # once per monthly file rather than once per TC record.
            all_area_by_lat = calculate_grid_area_km2(
                all_lat, d_lat_deg, d_lon_deg
            )

            variable = ds[SOURCE_VARIABLE]
            expected_dims = (
                {"time", "plev", "lat", "lon"}
                if HAS_PLEV
                else {"time", "lat", "lon"}
            )
            if set(variable.dims) != expected_dims:
                raise ValueError(
                    f"Unexpected {SOURCE_VARIABLE} dimensions: {variable.dims}; "
                    f"expected {sorted(expected_dims)}"
                )
            if HAS_PLEV:
                variable = variable.transpose("time", "plev", "lat", "lon")
            else:
                variable = variable.transpose("time", "lat", "lon")

            time_index = ds.indexes.get("time")
            if time_index is None or not time_index.is_unique:
                raise ValueError("ERA5 time coordinate is missing or not unique")

            # Resolve all requested times in one vectorized lookup.
            target_times = pd.DatetimeIndex(
                [pd.Timestamp(record[2]) for record in pending_records]
            )
            time_positions = time_index.get_indexer(target_times)

            for record, time_position in zip(pending_records, time_positions):
                row_id, sid, iso_time, rain_time_id, center_lat, center_lon = record
                output_path = build_output_path(row_id, rain_time_id)
                log = empty_row_log(record, output_path, "UNKNOWN")
                try:
                    target_time = pd.Timestamp(iso_time)
                    time_position = int(time_position)
                    if time_position < 0:
                        raise KeyError(f"Time not found in ERA5 file: {target_time}")

                    if HAS_PLEV:
                        field_data = variable.isel(time=time_position, plev=0)
                    else:
                        field_data = variable.isel(time=time_position)

                    # ERA5 var155 is horizontal divergence. Store its negative
                    # so positive values consistently mean convergence.
                    if CURRENT_FIELD_KIND == "CONV":
                        field_data = -field_data
                    result = extract_tc_grid(
                        field_data,
                        all_lat,
                        all_lon,
                        d_lat_deg,
                        d_lon_deg,
                        all_area_by_lat,
                        center_lat,
                        center_lon,
                    )
                    if result.empty:
                        raise RuntimeError("Extracted grid is empty")

                    grid_count = len(result)
                    nan_count = int(result[DATA_COLUMN].isna().sum())
                    valid_count = grid_count - nan_count
                    nan_fraction = nan_count / grid_count
                    high_nan = nan_fraction > HIGH_NAN_FRACTION_THRESHOLD

                    write_csv_gzip_atomic(result, output_path)

                    task_result["SUCCESS"] += 1
                    task_result["OUTPUT_GRIDS"] += grid_count
                    task_result["VALID_FIELD_GRIDS"] += valid_count
                    task_result["NAN_FIELD_GRIDS"] += nan_count
                    task_result["HIGH_NAN_RECORDS"] += int(high_nan)
                    log.update(
                        {
                            "STATUS": "SUCCESS",
                            "GRID_COUNT": grid_count,
                            "VALID_FIELD_GRIDS": valid_count,
                            "NAN_FIELD_GRIDS": nan_count,
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
        "VALID_FIELD_GRID_COUNT": int(task_df["VALID_FIELD_GRIDS"].sum()),
        "NAN_FIELD_GRID_COUNT": int(task_df["NAN_FIELD_GRIDS"].sum()),
        "HIGH_NAN_RECORD_COUNT": int(task_df["HIGH_NAN_RECORDS"].sum()),
        "HIGH_NAN_THRESHOLD": HIGH_NAN_FRACTION_THRESHOLD,
    }
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    return summary, task_log_path, row_log_path, error_log_path, high_nan_log_path, summary_path


# ============================================================
# 7. Main
# ============================================================

def process_configured_field(base_df):
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    df = base_df
    total_tasks = int(df["ISO_TIME"].dt.strftime("%Y%m").nunique())

    print("\n" + "=" * 80)
    print(f"Starting ERA5 {CURRENT_FIELD_LABEL} 500-km grid extraction")
    print("=" * 80)
    print(f"TC records: {len(df)}")
    print(f"Monthly tasks: {total_tasks}")
    print(f"Worker processes: {MAX_WORKERS}")
    print(f"Extraction radius: {RADIUS_KM:.1f} km")
    print(f"ERA5 directory: {ERA5_BASE_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    if HAS_PLEV:
        print(
            f"Source variable: {SOURCE_VARIABLE} at {EXPECTED_PLEV_PA:.0f} Pa; "
            f"output column: {DATA_COLUMN}"
        )
    else:
        print(
            f"Source variable: {SOURCE_VARIABLE} with no pressure dimension; "
            f"output column: {DATA_COLUMN}"
        )
    print(f"NaN {DATA_COLUMN} values are retained and never treated as zero")

    results = []
    with Pool(
        processes=MAX_WORKERS,
        initializer=configure_worker,
        initargs=(CURRENT_FIELD_KIND, CURRENT_LEVEL_HPA),
        maxtasksperchild=100,
    ) as pool:
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

    return summary


def save_overall_level_log(level_results):
    OVERALL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    overall_path = OVERALL_LOG_DIR / "level_processing_log.csv"
    overall_df = pd.DataFrame(level_results)
    temp_path = overall_path.with_name(overall_path.name + f".tmp.{os.getpid()}")
    try:
        overall_df.to_csv(temp_path, index=False)
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise RuntimeError(f"Overall temporary log is missing or empty: {temp_path}")
        os.replace(temp_path, overall_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
    return overall_path


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OVERALL_LOG_DIR.mkdir(parents=True, exist_ok=True)

    base_df = read_and_validate_base_csv()
    source_filter_log = OVERALL_LOG_DIR / "source_filter_summary.csv"
    write_csv_atomic(
        pd.DataFrame([base_df.attrs["source_filter_stats"]]),
        source_filter_log,
    )
    level_results = []

    print("\n" + "=" * 80)
    print("Starting sequential multi-level ERA5 convergence extraction")
    print("=" * 80)
    print("Pressure levels (hPa): " + ", ".join(map(str, PRESSURE_LEVELS_HPA)))
    print("Levels are processed sequentially; monthly tasks are parallel within a level")
    print("Convergence is calculated as -1 * ERA5 var155 divergence")
    print(f"Source filter log: {source_filter_log}")

    for level_index, level_hpa in enumerate(PRESSURE_LEVELS_HPA, start=1):
        configure_level(level_hpa)

        print("\n" + "#" * 80)
        print(
            f"Pressure level {level_index}/{len(PRESSURE_LEVELS_HPA)}: "
            f"CONV{level_hpa}"
        )
        print("#" * 80)

        summary = process_configured_field(base_df)
        level_record = {
            "FIELD": f"CONV{level_hpa}",
            "LEVEL_HPA": level_hpa,
            "INPUT_DIRECTORY": str(ERA5_BASE_DIR),
            "OUTPUT_DIRECTORY": str(OUTPUT_BASE_DIR),
            **summary,
        }
        level_results.append(level_record)
        overall_path = save_overall_level_log(level_results)

        error_count = summary["MISSING_FILE_COUNT"] + summary["FAILED_COUNT"]
        if error_count > 0:
            raise RuntimeError(
                f"CONV{level_hpa} completed with {error_count} file/record errors. "
                f"Processing stopped before the next level. Review {LOG_DIR}."
            )

        print(f"CONV{level_hpa} completed without errors")
        print(f"Overall level log: {overall_path}")

    configure_sp0()

    print("\n" + "#" * 80)
    print("Final field: SP0 surface pressure")
    print("#" * 80)

    sp0_summary = process_configured_field(base_df)
    sp0_record = {
        "FIELD": "SP0",
        "LEVEL_HPA": np.nan,
        "INPUT_DIRECTORY": str(ERA5_BASE_DIR),
        "OUTPUT_DIRECTORY": str(OUTPUT_BASE_DIR),
        **sp0_summary,
    }
    level_results.append(sp0_record)
    overall_path = save_overall_level_log(level_results)

    sp0_error_count = (
        sp0_summary["MISSING_FILE_COUNT"] + sp0_summary["FAILED_COUNT"]
    )
    if sp0_error_count > 0:
        raise RuntimeError(
            f"SP0 completed with {sp0_error_count} file/record errors. "
            f"Review {LOG_DIR}."
        )

    print("SP0 completed without errors")
    print(f"Overall level log: {overall_path}")

    print("\n" + "=" * 80)
    print("All convergence pressure levels and SP0 completed successfully")
    print("=" * 80)
    print(f"Fields completed: {len(level_results)}")
    print(f"Overall level log: {OVERALL_LOG_DIR / 'level_processing_log.csv'}")


if __name__ == "__main__":
    main()
