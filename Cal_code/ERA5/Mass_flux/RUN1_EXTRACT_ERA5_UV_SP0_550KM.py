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
SOURCE_CSV = (
    PROJECT_ROOT / "Data" / "Processed" / "MSWEP" / "Thresholds"
    / "PRE_DATA_IBT_1982_2024_MSWEP_THRESHOLD_30.csv"
)

ERA5_ROOT = PROJECT_ROOT / "Data" / "Raw" / "ERA5" / "PRESS_LEV_HOU"
OUTPUT_ROOT = (
    PROJECT_ROOT / "Data" / "Intermediate" / "ERA5"
    / "Mass_flux" / "UV_550KM"
)
OVERALL_LOG_DIR = OUTPUT_ROOT / "_logs"

PRESSURE_LEVELS_HPA = [
    500, 550, 600, 650, 700, 750, 775, 800,
    825, 850, 875, 900, 925, 950, 975, 1000,
]

START_YEAR = 1982
END_YEAR = 2024

EARTH_RADIUS_KM = 6371.0
RADIUS_KM = 550.0

MAX_WORKERS = 24
POOL_CHUNKSIZE = 1
GZIP_COMPRESS_LEVEL = 1
SKIP_EXISTING = True
STRICT_EXISTING_CHECK = False

TEST_MODE = False
TEST_MAX_ROWS = 500

HIGH_NAN_FRACTION_THRESHOLD = 0.10

# These globals are configured before each field-specific Pool starts.
CURRENT_FIELD_KIND = None
CURRENT_FIELD_LABEL = None
CURRENT_LEVEL_HPA = None
EXPECTED_PLEV_PA = None
U_BASE_DIR = None
V_BASE_DIR = None
SP0_BASE_DIR = None
OUTPUT_BASE_DIR = None
LOG_DIR = None
OUTPUT_COLUMNS = None


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
    "VALID_DATA_GRIDS",
    "NAN_ANY_DATA_GRIDS",
    "NAN_FRACTION",
    "HIGH_NAN_FLAG",
    "ERROR",
]


# ============================================================
# 2. Configuration helpers
# ============================================================

def configure_uv_level(level_hpa):
    global CURRENT_FIELD_KIND
    global CURRENT_FIELD_LABEL
    global CURRENT_LEVEL_HPA
    global EXPECTED_PLEV_PA
    global U_BASE_DIR
    global V_BASE_DIR
    global SP0_BASE_DIR
    global OUTPUT_BASE_DIR
    global LOG_DIR
    global OUTPUT_COLUMNS

    CURRENT_LEVEL_HPA = int(level_hpa)
    EXPECTED_PLEV_PA = float(CURRENT_LEVEL_HPA * 100)
    CURRENT_FIELD_KIND = "UV"
    CURRENT_FIELD_LABEL = f"UV{CURRENT_LEVEL_HPA}"

    U_BASE_DIR = ERA5_ROOT / f"u{CURRENT_LEVEL_HPA}_1980_2025_nc"
    V_BASE_DIR = ERA5_ROOT / f"v{CURRENT_LEVEL_HPA}_1980_2025_nc"
    SP0_BASE_DIR = None

    OUTPUT_BASE_DIR = OUTPUT_ROOT / CURRENT_FIELD_LABEL
    LOG_DIR = OUTPUT_BASE_DIR / "_logs"
    OUTPUT_COLUMNS = [
        "LAT",
        "LON",
        f"U{CURRENT_LEVEL_HPA}",
        f"V{CURRENT_LEVEL_HPA}",
        "DIST_KM",
        "AREA_KM2",
    ]


def configure_sp0():
    global CURRENT_FIELD_KIND
    global CURRENT_FIELD_LABEL
    global CURRENT_LEVEL_HPA
    global EXPECTED_PLEV_PA
    global U_BASE_DIR
    global V_BASE_DIR
    global SP0_BASE_DIR
    global OUTPUT_BASE_DIR
    global LOG_DIR
    global OUTPUT_COLUMNS

    CURRENT_FIELD_KIND = "SP0"
    CURRENT_FIELD_LABEL = "SP0"
    CURRENT_LEVEL_HPA = None
    EXPECTED_PLEV_PA = None

    U_BASE_DIR = None
    V_BASE_DIR = None
    SP0_BASE_DIR = ERA5_ROOT / "sp0_1980_2025_nc"

    OUTPUT_BASE_DIR = OUTPUT_ROOT / "SP0_550KM"
    LOG_DIR = OUTPUT_BASE_DIR / "_logs"
    OUTPUT_COLUMNS = ["LAT", "LON", "SP0", "DIST_KM", "AREA_KM2"]


def configure_worker(field_kind, level_hpa):
    if field_kind == "UV":
        configure_uv_level(level_hpa)
    elif field_kind == "SP0":
        configure_sp0()
    else:
        raise ValueError(f"Unsupported field kind: {field_kind}")


# ============================================================
# 3. General helpers
# ============================================================

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
        df.to_csv(temp_path, index=False)
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
        "VALID_DATA_GRIDS": 0,
        "NAN_ANY_DATA_GRIDS": 0,
        "NAN_FRACTION": np.nan,
        "HIGH_NAN_FLAG": False,
        "ERROR": "",
    }


# ============================================================
# 4. Read and validate the cleaned MSWEP source table
# ============================================================

def read_and_validate_source_csv():
    print("=" * 80)
    print("Reading, filtering, and validating the MSWEP threshold-30 CSV")
    print("=" * 80)

    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Source CSV does not exist: {SOURCE_CSV}")

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
        SOURCE_CSV,
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
        raise KeyError("Source CSV is missing: " + ", ".join(sorted(missing)))

    row_id_numeric = pd.to_numeric(df["ROW_ID"], errors="coerce")
    invalid_row_id = (
        row_id_numeric.isna()
        | ~np.isfinite(row_id_numeric.to_numpy(dtype=np.float64))
        | (row_id_numeric < 0)
        | (row_id_numeric % 1 != 0)
    )
    if invalid_row_id.any():
        raise ValueError(f"Invalid ROW_ID values: {int(invalid_row_id.sum())}")

    df["ROW_ID"] = row_id_numeric.astype(np.int64)
    if df["ROW_ID"].duplicated().any():
        examples = df.loc[
            df["ROW_ID"].duplicated(keep=False), "ROW_ID"
        ].head(20).tolist()
        raise ValueError(f"Duplicate ROW_ID values: {examples}")

    df["MSWEP_DIST_30"] = pd.to_numeric(df["MSWEP_DIST_30"], errors="coerce")
    valid_mswep = np.isfinite(df["MSWEP_DIST_30"].to_numpy(dtype=np.float64))
    removed_missing_mswep = int((~valid_mswep).sum())
    df = df.loc[valid_mswep].copy().reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No rows have a finite MSWEP_DIST_30 value")
    if (df["MSWEP_DIST_30"] < 0.0).any():
        raise ValueError("Cleaned source contains negative MSWEP_DIST_30 values")

    df["USA_LAT"] = pd.to_numeric(df["USA_LAT"], errors="coerce")
    df["USA_LON"] = pd.to_numeric(df["USA_LON"], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")

    invalid_required = (
        df["SID"].isna()
        | df["ISO_TIME"].isna()
        | df["RAIN_TIME_ID"].isna()
        | df["USA_LAT"].isna()
        | df["USA_LON"].isna()
    )
    if invalid_required.any():
        examples = df.loc[
            invalid_required,
            ["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID", "USA_LAT", "USA_LON"],
        ].head(20).to_dict("records")
        raise ValueError(
            f"Invalid required source values: count={int(invalid_required.sum())}; "
            f"examples={examples}"
        )

    invalid_lat = ~df["USA_LAT"].between(-90.0, 90.0)
    if invalid_lat.any():
        raise ValueError(f"Out-of-range latitudes: {int(invalid_lat.sum())}")

    df["USA_LON"] = ((df["USA_LON"] + 180.0) % 360.0) - 180.0

    valid_rain_id = df["RAIN_TIME_ID"].str.fullmatch(r"\d{7}\.\d{2}", na=False)
    if not valid_rain_id.all():
        raise ValueError(f"Invalid RAIN_TIME_ID values: {int((~valid_rain_id).sum())}")

    expected_rain_id = df["ISO_TIME"].dt.strftime("%Y%j.%H")
    mismatch = expected_rain_id != df["RAIN_TIME_ID"]
    if mismatch.any():
        examples = df.loc[
            mismatch, ["ROW_ID", "ISO_TIME", "RAIN_TIME_ID"]
        ].head(20).to_dict("records")
        raise ValueError(
            f"ISO_TIME and RAIN_TIME_ID mismatch: count={int(mismatch.sum())}; "
            f"examples={examples}"
        )

    strict_3hour = (
        df["ISO_TIME"].dt.hour.isin(range(0, 24, 3))
        & (df["ISO_TIME"].dt.minute == 0)
        & (df["ISO_TIME"].dt.second == 0)
        & (df["ISO_TIME"].dt.microsecond == 0)
    )
    if not strict_3hour.all():
        raise ValueError(
            f"Source contains non-exact ERA5 3-hour timestamps: "
            f"{int((~strict_3hour).sum())}"
        )

    outside_years = ~df["ISO_TIME"].dt.year.between(
        START_YEAR, END_YEAR, inclusive="both"
    )
    if outside_years.any():
        raise ValueError(
            f"Source rows outside {START_YEAR}-{END_YEAR}: "
            f"{int(outside_years.sum())}"
        )

    duplicate_sid_time = df.duplicated(subset=["SID", "ISO_TIME"], keep=False)
    if duplicate_sid_time.any():
        raise ValueError(
            f"Duplicate SID + ISO_TIME records: {int(duplicate_sid_time.sum())}"
        )

    df = df.reset_index(drop=True)

    print(f"Rows removed for invalid MSWEP_DIST_30: {removed_missing_mswep}")
    print(f"Valid source records: {len(df)}")
    print(f"ROW_ID range: {df['ROW_ID'].min()} to {df['ROW_ID'].max()}")
    print(f"Time range: {df['ISO_TIME'].min()} to {df['ISO_TIME'].max()}")

    if TEST_MODE:
        df = df.iloc[: min(TEST_MAX_ROWS, len(df))].copy()
        print(f"TEST_MODE: processing the first {len(df)} records")
    else:
        print("TEST_MODE disabled: processing all valid R30 records")

    return df


# ============================================================
# 5. Monthly task generation
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


# ============================================================
# 6. Dataset validation
# ============================================================

def validate_regular_grid(all_lat, all_lon):
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

    d_lat_deg = float(np.median(np.abs(np.diff(all_lat))))
    d_lon_deg = float(np.median(np.diff(all_lon)))

    if not np.allclose(
        np.abs(np.diff(all_lat)), d_lat_deg, rtol=0.0, atol=1e-6
    ):
        raise ValueError("Latitude grid is not regular")
    if not np.allclose(np.diff(all_lon), d_lon_deg, rtol=0.0, atol=1e-6):
        raise ValueError("Longitude grid is not regular")

    return d_lat_deg, d_lon_deg


def validate_pressure_dataset(ds, variable_name, expected_plev_pa):
    missing_coords = {"time", "lat", "lon", "plev"}.difference(ds.coords)
    if missing_coords:
        raise KeyError(
            "NetCDF missing coordinates: " + ", ".join(sorted(missing_coords))
        )
    if variable_name not in ds:
        raise KeyError(f"NetCDF missing {variable_name}")

    plev = np.asarray(ds["plev"].values, dtype=np.float64)
    if plev.size != 1 or not np.isclose(plev[0], expected_plev_pa):
        raise ValueError(
            f"Expected plev=[{expected_plev_pa}], found {plev.tolist()}"
        )

    variable = ds[variable_name]
    expected_dims = {"time", "plev", "lat", "lon"}
    if set(variable.dims) != expected_dims:
        raise ValueError(
            f"Unexpected {variable_name} dimensions: {variable.dims}"
        )

    all_lat = np.asarray(ds["lat"].values, dtype=np.float64)
    all_lon = np.asarray(ds["lon"].values, dtype=np.float64)
    d_lat_deg, d_lon_deg = validate_regular_grid(all_lat, all_lon)

    variable = variable.transpose("time", "plev", "lat", "lon")

    time_index = ds.indexes.get("time")
    if time_index is None or not time_index.is_unique:
        raise ValueError("ERA5 time coordinate is missing or not unique")

    return variable, all_lat, all_lon, d_lat_deg, d_lon_deg, time_index


def validate_sp0_dataset(ds):
    missing_coords = {"time", "lat", "lon"}.difference(ds.coords)
    if missing_coords:
        raise KeyError(
            "NetCDF missing coordinates: " + ", ".join(sorted(missing_coords))
        )
    if "var134" not in ds:
        raise KeyError("NetCDF missing var134")

    variable = ds["var134"]
    expected_dims = {"time", "lat", "lon"}
    if set(variable.dims) != expected_dims:
        raise ValueError(f"Unexpected var134 dimensions: {variable.dims}")

    all_lat = np.asarray(ds["lat"].values, dtype=np.float64)
    all_lon = np.asarray(ds["lon"].values, dtype=np.float64)
    d_lat_deg, d_lon_deg = validate_regular_grid(all_lat, all_lon)

    variable = variable.transpose("time", "lat", "lon")

    time_index = ds.indexes.get("time")
    if time_index is None or not time_index.is_unique:
        raise ValueError("ERA5 time coordinate is missing or not unique")

    return variable, all_lat, all_lon, d_lat_deg, d_lon_deg, time_index


def ensure_uv_alignment(
    u_lat,
    u_lon,
    u_dlat,
    u_dlon,
    u_time_index,
    v_lat,
    v_lon,
    v_dlat,
    v_dlon,
    v_time_index,
):
    if not np.array_equal(u_lat, v_lat):
        raise ValueError("U and V latitude coordinates do not match")
    if not np.array_equal(u_lon, v_lon):
        raise ValueError("U and V longitude coordinates do not match")
    if not np.isclose(u_dlat, v_dlat, rtol=0.0, atol=1e-12):
        raise ValueError("U and V latitude spacing does not match")
    if not np.isclose(u_dlon, v_dlon, rtol=0.0, atol=1e-12):
        raise ValueError("U and V longitude spacing does not match")
    if not u_time_index.equals(v_time_index):
        raise ValueError("U and V time coordinates do not match")


# ============================================================
# 7. Spatial extraction
# ============================================================

def build_spatial_selection(
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

    lat_indices = np.flatnonzero(
        np.abs(all_lat - center_lat) <= lat_margin_deg
    )
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
        candidate_lat,
        candidate_lon,
        center_lat,
        center_lon_360,
    )
    radius_mask = distance_grid <= RADIUS_KM

    if not radius_mask.any():
        raise RuntimeError(f"No grid-cell centres within {RADIUS_KM} km")

    lat_grid = np.broadcast_to(candidate_lat[:, None], distance_grid.shape)
    lon_grid = np.broadcast_to(candidate_lon[None, :], distance_grid.shape)
    area_by_lat = all_area_by_lat[lat_indices]
    area_grid = np.broadcast_to(area_by_lat[:, None], distance_grid.shape)

    output_lon = ((lon_grid[radius_mask] + 180.0) % 360.0) - 180.0

    return {
        "lat_indices": lat_indices,
        "lon_indices": lon_indices,
        "radius_mask": radius_mask,
        "lat_output": lat_grid[radius_mask].astype(np.float32),
        "lon_output": output_lon.astype(np.float32),
        "distance_output": distance_grid[radius_mask].astype(np.float32),
        "area_output": area_grid[radius_mask].astype(np.float32),
        "candidate_shape": (len(candidate_lat), len(candidate_lon)),
    }


def extract_uv_grid(
    u_field,
    v_field,
    all_lat,
    all_lon,
    d_lat_deg,
    d_lon_deg,
    all_area_by_lat,
    center_lat,
    center_lon,
):
    selection = build_spatial_selection(
        all_lat,
        all_lon,
        d_lat_deg,
        d_lon_deg,
        all_area_by_lat,
        center_lat,
        center_lon,
    )

    lat_indices = selection["lat_indices"]
    lon_indices = selection["lon_indices"]
    radius_mask = selection["radius_mask"]
    expected_shape = selection["candidate_shape"]

    u_subset = np.asarray(
        u_field.isel(lat=lat_indices, lon=lon_indices).load().values,
        dtype=np.float32,
    )
    v_subset = np.asarray(
        v_field.isel(lat=lat_indices, lon=lon_indices).load().values,
        dtype=np.float32,
    )

    if u_subset.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected U{CURRENT_LEVEL_HPA} shape {u_subset.shape}; "
            f"expected {expected_shape}"
        )
    if v_subset.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected V{CURRENT_LEVEL_HPA} shape {v_subset.shape}; "
            f"expected {expected_shape}"
        )

    result = pd.DataFrame(
        {
            "LAT": selection["lat_output"],
            "LON": selection["lon_output"],
            f"U{CURRENT_LEVEL_HPA}": u_subset[radius_mask],
            f"V{CURRENT_LEVEL_HPA}": v_subset[radius_mask],
            "DIST_KM": selection["distance_output"],
            "AREA_KM2": selection["area_output"],
        }
    )
    return result.sort_values("DIST_KM", kind="mergesort").reset_index(drop=True)


def extract_sp0_grid(
    sp0_field,
    all_lat,
    all_lon,
    d_lat_deg,
    d_lon_deg,
    all_area_by_lat,
    center_lat,
    center_lon,
):
    selection = build_spatial_selection(
        all_lat,
        all_lon,
        d_lat_deg,
        d_lon_deg,
        all_area_by_lat,
        center_lat,
        center_lon,
    )

    lat_indices = selection["lat_indices"]
    lon_indices = selection["lon_indices"]
    radius_mask = selection["radius_mask"]
    expected_shape = selection["candidate_shape"]

    sp0_subset = np.asarray(
        sp0_field.isel(lat=lat_indices, lon=lon_indices).load().values,
        dtype=np.float32,
    )

    if sp0_subset.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected SP0 shape {sp0_subset.shape}; expected {expected_shape}"
        )

    result = pd.DataFrame(
        {
            "LAT": selection["lat_output"],
            "LON": selection["lon_output"],
            "SP0": sp0_subset[radius_mask],
            "DIST_KM": selection["distance_output"],
            "AREA_KM2": selection["area_output"],
        }
    )
    return result.sort_values("DIST_KM", kind="mergesort").reset_index(drop=True)


# ============================================================
# 8. Process one monthly task
# ============================================================

def process_single_month(task):
    year_month, records = task

    if CURRENT_FIELD_KIND == "UV":
        u_path = U_BASE_DIR / f"u{CURRENT_LEVEL_HPA}_{year_month}.nc"
        v_path = V_BASE_DIR / f"v{CURRENT_LEVEL_HPA}_{year_month}.nc"
        input_paths = f"{u_path} | {v_path}"
        required_paths = [u_path, v_path]
    else:
        sp0_path = SP0_BASE_DIR / f"sp0_{year_month}.nc"
        input_paths = str(sp0_path)
        required_paths = [sp0_path]

    task_result = {
        "FIELD": CURRENT_FIELD_LABEL,
        "YEAR_MONTH": year_month,
        "INPUT_PATHS": input_paths,
        "TOTAL_RECORDS": len(records),
        "SUCCESS": 0,
        "SKIPPED_EXISTING": 0,
        "MISSING_FILE": 0,
        "FAILED": 0,
        "OUTPUT_GRIDS": 0,
        "VALID_DATA_GRIDS": 0,
        "NAN_ANY_DATA_GRIDS": 0,
        "HIGH_NAN_RECORDS": 0,
        "STATUS": "UNKNOWN",
        "ERROR": "",
        "ROW_LOGS": [],
    }

    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        task_result["MISSING_FILE"] = len(records)
        task_result["STATUS"] = "MISSING_FILE"
        missing_text = "Missing ERA5 file(s): " + ", ".join(map(str, missing_paths))

        for record in records:
            output_path = build_output_path(record[0], record[3])
            log = empty_row_log(record, output_path, "MISSING_FILE")
            log["ERROR"] = missing_text
            task_result["ROW_LOGS"].append(log)

        task_result["ERROR"] = missing_text
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
        if CURRENT_FIELD_KIND == "UV":
            with xr.open_dataset(
                u_path,
                decode_times=True,
                mask_and_scale=True,
                cache=False,
            ) as ds_u, xr.open_dataset(
                v_path,
                decode_times=True,
                mask_and_scale=True,
                cache=False,
            ) as ds_v:
                (
                    u_variable,
                    u_lat,
                    u_lon,
                    u_dlat,
                    u_dlon,
                    u_time_index,
                ) = validate_pressure_dataset(ds_u, "var131", EXPECTED_PLEV_PA)

                (
                    v_variable,
                    v_lat,
                    v_lon,
                    v_dlat,
                    v_dlon,
                    v_time_index,
                ) = validate_pressure_dataset(ds_v, "var132", EXPECTED_PLEV_PA)

                ensure_uv_alignment(
                    u_lat,
                    u_lon,
                    u_dlat,
                    u_dlon,
                    u_time_index,
                    v_lat,
                    v_lon,
                    v_dlat,
                    v_dlon,
                    v_time_index,
                )

                all_area_by_lat = calculate_grid_area_km2(
                    u_lat, u_dlat, u_dlon
                )

                process_pending_records(
                    pending_records=pending_records,
                    task_result=task_result,
                    time_index=u_time_index,
                    all_lat=u_lat,
                    all_lon=u_lon,
                    d_lat_deg=u_dlat,
                    d_lon_deg=u_dlon,
                    all_area_by_lat=all_area_by_lat,
                    u_variable=u_variable,
                    v_variable=v_variable,
                    sp0_variable=None,
                )

        else:
            with xr.open_dataset(
                sp0_path,
                decode_times=True,
                mask_and_scale=True,
                cache=False,
            ) as ds_sp0:
                (
                    sp0_variable,
                    all_lat,
                    all_lon,
                    d_lat_deg,
                    d_lon_deg,
                    time_index,
                ) = validate_sp0_dataset(ds_sp0)

                all_area_by_lat = calculate_grid_area_km2(
                    all_lat, d_lat_deg, d_lon_deg
                )

                process_pending_records(
                    pending_records=pending_records,
                    task_result=task_result,
                    time_index=time_index,
                    all_lat=all_lat,
                    all_lon=all_lon,
                    d_lat_deg=d_lat_deg,
                    d_lon_deg=d_lon_deg,
                    all_area_by_lat=all_area_by_lat,
                    u_variable=None,
                    v_variable=None,
                    sp0_variable=sp0_variable,
                )

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

        existing_logged_ids = {
            int(log["ROW_ID"]) for log in task_result["ROW_LOGS"]
        }

        for record in pending_records:
            if int(record[0]) in existing_logged_ids:
                continue

            output_path = build_output_path(record[0], record[3])
            log = empty_row_log(record, output_path, "FILE_ERROR")
            log["ERROR"] = task_result["ERROR"]
            task_result["ROW_LOGS"].append(log)

    return task_result


def process_pending_records(
    pending_records,
    task_result,
    time_index,
    all_lat,
    all_lon,
    d_lat_deg,
    d_lon_deg,
    all_area_by_lat,
    u_variable,
    v_variable,
    sp0_variable,
):
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

            if CURRENT_FIELD_KIND == "UV":
                u_field = u_variable.isel(time=time_position, plev=0)
                v_field = v_variable.isel(time=time_position, plev=0)

                result = extract_uv_grid(
                    u_field,
                    v_field,
                    all_lat,
                    all_lon,
                    d_lat_deg,
                    d_lon_deg,
                    all_area_by_lat,
                    center_lat,
                    center_lon,
                )
                data_columns = [
                    f"U{CURRENT_LEVEL_HPA}",
                    f"V{CURRENT_LEVEL_HPA}",
                ]

            else:
                sp0_field = sp0_variable.isel(time=time_position)

                result = extract_sp0_grid(
                    sp0_field,
                    all_lat,
                    all_lon,
                    d_lat_deg,
                    d_lon_deg,
                    all_area_by_lat,
                    center_lat,
                    center_lon,
                )
                data_columns = ["SP0"]

            if result.empty:
                raise RuntimeError("Extracted grid is empty")

            grid_count = len(result)
            nan_any = result[data_columns].isna().any(axis=1)
            nan_count = int(nan_any.sum())
            valid_count = grid_count - nan_count
            nan_fraction = nan_count / grid_count
            high_nan = nan_fraction > HIGH_NAN_FRACTION_THRESHOLD

            write_csv_gzip_atomic(result, output_path)

            task_result["SUCCESS"] += 1
            task_result["OUTPUT_GRIDS"] += grid_count
            task_result["VALID_DATA_GRIDS"] += valid_count
            task_result["NAN_ANY_DATA_GRIDS"] += nan_count
            task_result["HIGH_NAN_RECORDS"] += int(high_nan)

            log.update(
                {
                    "STATUS": "SUCCESS",
                    "GRID_COUNT": grid_count,
                    "VALID_DATA_GRIDS": valid_count,
                    "NAN_ANY_DATA_GRIDS": nan_count,
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


# ============================================================
# 9. Logs
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

    write_csv_atomic(task_df, task_log_path)
    write_csv_atomic(row_df, row_log_path)

    error_df = row_df.loc[
        row_df["STATUS"].isin(["MISSING_FILE", "FILE_ERROR", "FAILED"])
    ]
    write_csv_atomic(error_df, error_log_path)

    high_nan_df = row_df.loc[row_df["HIGH_NAN_FLAG"] == True]
    write_csv_atomic(high_nan_df, high_nan_log_path)

    summary = {
        "FIELD": CURRENT_FIELD_LABEL,
        "MONTH_TASK_COUNT": int(len(task_df)),
        "TC_RECORD_COUNT": int(task_df["TOTAL_RECORDS"].sum()),
        "SUCCESS_COUNT": int(task_df["SUCCESS"].sum()),
        "SKIPPED_EXISTING_COUNT": int(task_df["SKIPPED_EXISTING"].sum()),
        "MISSING_FILE_COUNT": int(task_df["MISSING_FILE"].sum()),
        "FAILED_COUNT": int(task_df["FAILED"].sum()),
        "OUTPUT_GRID_COUNT": int(task_df["OUTPUT_GRIDS"].sum()),
        "VALID_DATA_GRID_COUNT": int(task_df["VALID_DATA_GRIDS"].sum()),
        "NAN_ANY_DATA_GRID_COUNT": int(task_df["NAN_ANY_DATA_GRIDS"].sum()),
        "HIGH_NAN_RECORD_COUNT": int(task_df["HIGH_NAN_RECORDS"].sum()),
        "HIGH_NAN_THRESHOLD": HIGH_NAN_FRACTION_THRESHOLD,
    }
    write_csv_atomic(pd.DataFrame([summary]), summary_path)

    return (
        summary,
        task_log_path,
        row_log_path,
        error_log_path,
        high_nan_log_path,
        summary_path,
    )


def save_overall_field_log(field_results):
    OVERALL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    overall_path = OVERALL_LOG_DIR / "field_processing_log.csv"
    write_csv_atomic(pd.DataFrame(field_results), overall_path)
    return overall_path


# ============================================================
# 10. Process one configured field
# ============================================================

def process_configured_field(source_df):
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    total_tasks = int(source_df["ISO_TIME"].dt.strftime("%Y%m").nunique())

    print("\n" + "=" * 80)
    print(f"Starting ERA5 {CURRENT_FIELD_LABEL} {RADIUS_KM:.0f}-km grid extraction")
    print("=" * 80)
    print(f"TC records: {len(source_df)}")
    print(f"Monthly tasks: {total_tasks}")
    print(f"Worker processes: {MAX_WORKERS}")
    print(f"Extraction radius: {RADIUS_KM:.1f} km")

    if CURRENT_FIELD_KIND == "UV":
        print(f"U directory: {U_BASE_DIR}")
        print(f"V directory: {V_BASE_DIR}")
        print(
            f"Source variables: var131 and var132 at "
            f"{EXPECTED_PLEV_PA:.0f} Pa"
        )
    else:
        print(f"SP0 directory: {SP0_BASE_DIR}")
        print("Source variable: var134 with no pressure dimension")

    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print("NaN values are retained and never treated as zero")

    results = []

    with Pool(
        processes=MAX_WORKERS,
        initializer=configure_worker,
        initargs=(CURRENT_FIELD_KIND, CURRENT_LEVEL_HPA),
        maxtasksperchild=100,
    ) as pool:
        iterator = pool.imap_unordered(
            process_single_month,
            generate_month_tasks(source_df),
            chunksize=POOL_CHUNKSIZE,
        )

        for result in tqdm(
            iterator,
            total=total_tasks,
            desc=f"{CURRENT_FIELD_LABEL} monthly tasks",
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
    print(f"{CURRENT_FIELD_LABEL} processing completed")
    print("=" * 80)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nMonthly task log: {task_log_path}")
    print(f"Row coverage log: {row_log_path}")
    print(f"Error log: {error_log_path}")
    print(f"High-NaN log: {high_nan_log_path}")
    print(f"Summary log: {summary_path}")

    return summary


# ============================================================
# 11. Main
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OVERALL_LOG_DIR.mkdir(parents=True, exist_ok=True)

    source_df = read_and_validate_source_csv()
    field_results = []

    print("\n" + "=" * 80)
    print("Starting sequential multi-level ERA5 U/V and SP0 extraction")
    print("=" * 80)
    print("Pressure levels (hPa): " + ", ".join(map(str, PRESSURE_LEVELS_HPA)))
    print("U and V are written together for each pressure level")
    print("Fields are processed sequentially; monthly tasks are parallel")
    print(f"Extraction radius: {RADIUS_KM:.1f} km")

    for field_index, level_hpa in enumerate(PRESSURE_LEVELS_HPA, start=1):
        configure_uv_level(level_hpa)

        print("\n" + "#" * 80)
        print(
            f"Pressure field {field_index}/{len(PRESSURE_LEVELS_HPA)}: "
            f"UV{level_hpa}"
        )
        print("#" * 80)

        summary = process_configured_field(source_df)
        field_record = {
            "FIELD": f"UV{level_hpa}",
            "LEVEL_HPA": level_hpa,
            "U_INPUT_DIRECTORY": str(U_BASE_DIR),
            "V_INPUT_DIRECTORY": str(V_BASE_DIR),
            "OUTPUT_DIRECTORY": str(OUTPUT_BASE_DIR),
            **summary,
        }
        field_results.append(field_record)
        overall_path = save_overall_field_log(field_results)

        error_count = summary["MISSING_FILE_COUNT"] + summary["FAILED_COUNT"]
        if error_count > 0:
            raise RuntimeError(
                f"UV{level_hpa} completed with {error_count} errors. "
                f"Processing stopped before the next field. Review {LOG_DIR}."
            )

        print(f"UV{level_hpa} completed without errors")
        print(f"Overall field log: {overall_path}")

    configure_sp0()

    print("\n" + "#" * 80)
    print("Final field: SP0 surface pressure")
    print("#" * 80)

    sp0_summary = process_configured_field(source_df)
    sp0_record = {
        "FIELD": "SP0",
        "LEVEL_HPA": np.nan,
        "U_INPUT_DIRECTORY": "",
        "V_INPUT_DIRECTORY": "",
        "SP0_INPUT_DIRECTORY": str(SP0_BASE_DIR),
        "OUTPUT_DIRECTORY": str(OUTPUT_BASE_DIR),
        **sp0_summary,
    }
    field_results.append(sp0_record)
    overall_path = save_overall_field_log(field_results)

    sp0_error_count = (
        sp0_summary["MISSING_FILE_COUNT"] + sp0_summary["FAILED_COUNT"]
    )
    if sp0_error_count > 0:
        raise RuntimeError(
            f"SP0 completed with {sp0_error_count} errors. Review {LOG_DIR}."
        )

    print("SP0 completed without errors")
    print(f"Overall field log: {overall_path}")

    print("\n" + "=" * 80)
    print("All U/V pressure levels and SP0 completed successfully")
    print("=" * 80)
    print(f"Fields completed: {len(field_results)}")
    print(f"Overall field log: {overall_path}")


if __name__ == "__main__":
    main()
