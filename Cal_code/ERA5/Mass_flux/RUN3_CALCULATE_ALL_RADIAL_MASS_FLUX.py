#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Calculate three TC-relative radial mass-flux products.

Products
--------
1. 500 hPa to local surface pressure (PB): 200- and 500-km rings.
2. Fixed 500 to 700 hPa: 300- and 500-km rings. This product does not
   apply an SP0 terrain mask; every grid cell with all five pressure levels
   available is retained, consistently with the convergence calculation.
3. 700 hPa to local surface pressure (PB): 100- and 500-km rings.

Every ring is 25 km wide and includes both boundaries:
[target radius - 12.5 km, target radius + 12.5 km]. Positive flux denotes
inward mass transport toward the TC centre. Annular net influx is calculated
as outer-ring inward flux minus inner-ring inward flux. The fixed 500-700-hPa
product retains the established five-level, no-SP0 calculation.
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
    PROJECT_ROOT / "Data" / "Processed" / "MSWEP" / "Thresholds"
    / "PRE_DATA_IBT_1982_2024_MSWEP_THRESHOLD_30.csv"
)

ERA5_ROOT = (
    PROJECT_ROOT / "Data" / "Intermediate" / "ERA5"
    / "Mass_flux" / "UV_550KM"
)
OUTPUT_ROOT = (
    PROJECT_ROOT / "Data" / "Processed" / "ERA5"
    / "Mass_flux" / "Endpoints"
)
# Use a dedicated cache so results produced with half-open rings or an older
# fixed-layer definition can never be reused by this closed-ring run.
ROW_RESULT_DIR = (
    PROJECT_ROOT / ".work" / "ERA5" / "Mass_flux" / "RUN3"
    / "ROW_RESULTS_ALL_PRODUCTS_CLOSED_RINGS_V1"
)
LOG_DIR = PROJECT_ROOT / "Results" / "Quality_control" / "ERA5" / "Mass_flux" / "RUN3"

FINAL_OUTPUT_CSV = OUTPUT_ROOT / (
    "PRE_DATA_IBT_1982_2024_MSWEP30_"
    "RADIAL_MASS_FLUX_500PB_500_700_700PB_CLOSED_RINGS.csv"
)

PRESSURE_LEVELS_HPA = np.array(
    [
        500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 775.0,
        800.0, 825.0, 850.0, 875.0, 900.0, 925.0, 950.0,
        975.0, 1000.0,
    ],
    dtype=np.float64,
)

RING_HALF_WIDTH_KM = 12.5
TARGET_RADII_KM = (100.0, 200.0, 300.0, 500.0)

GRAVITY_MS2 = 9.80665

START_YEAR = 1982
END_YEAR = 2024
MAX_WORKERS = 24
POOL_CHUNKSIZE = 1
GZIP_COMPRESS_LEVEL = 1

SKIP_EXISTING = True
STRICT_EXISTING_CHECK = False

TEST_MODE = False
TEST_MAX_ROWS = 500


PRODUCTS = (
    {
        "name": "500_PB",
        "top_hpa": 500.0,
        "inner_radius_km": 200.0,
        "outer_radius_km": 500.0,
        "surface_bottom": True,
    },
    {
        "name": "500_700",
        "top_hpa": 500.0,
        "bottom_hpa": 700.0,
        "inner_radius_km": 300.0,
        "outer_radius_km": 500.0,
        "surface_bottom": False,
    },
    {
        "name": "700_PB",
        "top_hpa": 700.0,
        "inner_radius_km": 100.0,
        "outer_radius_km": 500.0,
        "surface_bottom": True,
    },
)


BASE_RESULT_COLUMNS = [
    "ROW_ID",
    "SID",
    "ISO_TIME",
    "RAIN_TIME_ID",
    "FLUX_STATUS",
    "FLUX_ERROR",
]


# ============================================================
# 2. Output-column definitions
# ============================================================

def ring_prefix(radius_km, product_name):
    return f"F{int(radius_km)}_{product_name}"


def ring_result_columns(radius_km, product_name, surface_bottom):
    prefix = ring_prefix(radius_km, product_name)
    columns = [
        f"{prefix}_KG_S",
        f"{prefix}_COLUMN_TRANSPORT_MEAN_KG_M_S",
        f"{prefix}_TOTAL_AREA_KM2",
        f"{prefix}_VALID_AREA_KM2",
        f"{prefix}_VALID_AREA_FRACTION",
        f"{prefix}_TOTAL_GRID_COUNT",
        f"{prefix}_VALID_GRID_COUNT",
    ]
    if surface_bottom:
        columns.extend(
            [
                f"{prefix}_SP0_LE_TOP_AREA_FRACTION",
                f"{prefix}_SP0_GT_1000_AREA_FRACTION",
                f"{prefix}_MEAN_SP0_HPA",
                f"{prefix}_MEAN_INTEGRATION_DEPTH_HPA",
            ]
        )
    else:
        columns.append(f"{prefix}_SP0_LT_700_AREA_FRACTION")
    return columns


def build_result_columns():
    columns = BASE_RESULT_COLUMNS.copy()
    for product in PRODUCTS:
        inner = product["inner_radius_km"]
        outer = product["outer_radius_km"]
        name = product["name"]
        columns.extend(
            ring_result_columns(inner, name, product["surface_bottom"])
        )
        columns.extend(
            ring_result_columns(outer, name, product["surface_bottom"])
        )
        columns.append(
            f"F{int(outer)}_MINUS_F{int(inner)}_{name}_KG_S"
        )
    return columns


RESULT_COLUMNS = build_result_columns()


# ============================================================
# 3. Path and file helpers
# ============================================================

def normalize_rain_time_id(value):
    """Return RAIN_TIME_ID in fixed YYYYDDD.HH form."""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        raise ValueError("RAIN_TIME_ID is empty")
    if "." in text:
        left, right = text.split(".", maxsplit=1)
        return f"{left}.{right[:2].ljust(2, '0')}"
    if len(text) == 9 and text.isdigit():
        return f"{text[:7]}.{text[7:]}"
    raise ValueError(f"Invalid RAIN_TIME_ID: {value!r}")


def row_grid_path(base_dir, row_id, rain_time_id):
    return (
        base_dir
        / rain_time_id[:4]
        / f"row_{int(row_id):09d}_{rain_time_id}.csv.gz"
    )


def inflow_path(level_hpa, row_id, rain_time_id):
    return row_grid_path(
        ERA5_ROOT / f"INFLOW_{int(level_hpa)}",
        row_id,
        rain_time_id,
    )


def sp0_path(row_id, rain_time_id):
    return row_grid_path(
        ERA5_ROOT / "SP0_550KM",
        row_id,
        rain_time_id,
    )


def result_path(row_id, rain_time_id):
    return row_grid_path(ROW_RESULT_DIR, row_id, rain_time_id)


def result_is_complete(path):
    if not path.exists() or path.stat().st_size <= 0:
        return False
    if not STRICT_EXISTING_CHECK:
        return True
    try:
        with gzip.open(path, mode="rt", encoding="utf-8") as handle:
            columns = handle.readline().strip().split(",")
        return set(RESULT_COLUMNS).issubset(columns)
    except (OSError, EOFError, gzip.BadGzipFile, UnicodeDecodeError):
        return False


def write_gzip_csv_atomic(frame, path):
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
            frame.to_csv(
                handle,
                index=False,
                columns=RESULT_COLUMNS,
                float_format="%.10g",
            )
        if not temporary.exists() or temporary.stat().st_size <= 0:
            raise RuntimeError(f"Temporary output is missing: {temporary}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def write_csv_atomic(frame, path):
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


def require_columns(frame, required, path):
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} is missing columns: {missing}")


def normalize_sp0_to_pa(values):
    array = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    finite = array[np.isfinite(array)]
    if finite.size and np.nanmedian(finite) < 2000.0:
        array *= 100.0
    return array


# ============================================================
# 4. Source-table preparation
# ============================================================

def read_source_records():
    """Read source keys and retain records with completed RUN2 inputs."""
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Source table does not exist: {SOURCE_CSV}")

    header = pd.read_csv(SOURCE_CSV, nrows=0).columns.tolist()
    required = ["SID", "ISO_TIME", "RAIN_TIME_ID"]
    missing = [column for column in required if column not in header]
    if missing:
        raise KeyError(f"Source table is missing columns: {missing}")

    usecols = required.copy()
    if "ROW_ID" in header:
        usecols.insert(0, "ROW_ID")

    frame = pd.read_csv(
        SOURCE_CSV,
        usecols=usecols,
        dtype={"SID": "string", "RAIN_TIME_ID": "string"},
        low_memory=False,
    )
    if "ROW_ID" not in frame.columns:
        frame.insert(0, "ROW_ID", np.arange(len(frame), dtype=np.int64))

    frame["ROW_ID"] = pd.to_numeric(frame["ROW_ID"], errors="coerce")
    frame["ISO_TIME"] = pd.to_datetime(frame["ISO_TIME"], errors="coerce")
    if frame["ROW_ID"].isna().any() or frame["ISO_TIME"].isna().any():
        raise ValueError("Source table contains invalid ROW_ID or ISO_TIME")

    frame["ROW_ID"] = frame["ROW_ID"].astype(np.int64)
    if frame["ROW_ID"].duplicated().any():
        raise ValueError("Source table contains duplicate ROW_ID values")

    frame = frame.loc[
        frame["ISO_TIME"].dt.year.between(START_YEAR, END_YEAR)
    ].copy()
    frame["RAIN_TIME_ID"] = frame["RAIN_TIME_ID"].map(
        normalize_rain_time_id
    )
    frame["YEAR_MONTH"] = frame["ISO_TIME"].dt.strftime("%Y%m")

    # RUN2 has one fewer record than RUN1 because one TC has no translation
    # velocity. INFLOW_500 is used as the authoritative RUN2 inventory.
    available = np.fromiter(
        (
            inflow_path(500, row.ROW_ID, row.RAIN_TIME_ID).is_file()
            for row in frame.itertuples(index=False)
        ),
        dtype=bool,
        count=len(frame),
    )
    unavailable = frame.loc[~available].copy()
    frame = frame.loc[available].copy()

    if TEST_MODE:
        frame = frame.head(TEST_MAX_ROWS).copy()

    return frame, unavailable


def build_monthly_tasks(frame):
    tasks = []
    columns = ["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID"]
    for year_month, group in frame.groupby("YEAR_MONTH", sort=True):
        records = [
            (
                int(row[0]),
                str(row[1]),
                pd.Timestamp(row[2]).isoformat(),
                str(row[3]),
            )
            for row in group[columns].itertuples(index=False, name=None)
        ]
        tasks.append((str(year_month), records))
    return tasks


# ============================================================
# 5. Read and align one TC-centred grid
# ============================================================

def coordinates_match(base_lat, base_lon, frame, path):
    lat = pd.to_numeric(frame["LAT"], errors="coerce").to_numpy(np.float64)
    lon = pd.to_numeric(frame["LON"], errors="coerce").to_numpy(np.float64)
    if len(lat) != len(base_lat):
        raise ValueError(f"Grid length differs in {path}")
    if not np.allclose(lat, base_lat, rtol=0.0, atol=1.0e-7, equal_nan=False):
        raise ValueError(f"LAT grid differs in {path}")
    if not np.allclose(lon, base_lon, rtol=0.0, atol=1.0e-7, equal_nan=False):
        raise ValueError(f"LON grid differs in {path}")


def read_aligned_fields(row_id, rain_time_id):
    """Read all radial-wind levels and SP0 in the common RUN1 grid order."""
    base_path = inflow_path(500, row_id, rain_time_id)
    base = pd.read_csv(base_path, low_memory=False)
    require_columns(
        base,
        ["LAT", "LON", "DIST_KM", "AREA_KM2", "RADIAL_VEL"],
        base_path,
    )

    base_lat = pd.to_numeric(base["LAT"], errors="coerce").to_numpy(np.float64)
    base_lon = pd.to_numeric(base["LON"], errors="coerce").to_numpy(np.float64)
    distance = pd.to_numeric(
        base["DIST_KM"], errors="coerce"
    ).to_numpy(np.float64)
    area = pd.to_numeric(
        base["AREA_KM2"], errors="coerce"
    ).to_numpy(np.float64)

    radial = np.full(
        (len(base), len(PRESSURE_LEVELS_HPA)),
        np.nan,
        dtype=np.float64,
    )
    radial[:, 0] = pd.to_numeric(
        base["RADIAL_VEL"], errors="coerce"
    ).to_numpy(np.float64)

    for level_index, level_hpa in enumerate(PRESSURE_LEVELS_HPA[1:], start=1):
        path = inflow_path(level_hpa, row_id, rain_time_id)
        level = pd.read_csv(
            path,
            usecols=["LAT", "LON", "RADIAL_VEL"],
            low_memory=False,
        )
        coordinates_match(base_lat, base_lon, level, path)
        radial[:, level_index] = pd.to_numeric(
            level["RADIAL_VEL"], errors="coerce"
        ).to_numpy(np.float64)

    surface_path = sp0_path(row_id, rain_time_id)
    surface = pd.read_csv(
        surface_path,
        usecols=["LAT", "LON", "SP0"],
        low_memory=False,
    )
    coordinates_match(base_lat, base_lon, surface, surface_path)
    sp0_pa = normalize_sp0_to_pa(surface["SP0"])

    return {
        "distance_km": distance,
        "area_km2": area,
        "radial_ms": radial,
        "sp0_pa": sp0_pa,
    }


# ============================================================
# 6. Vertical integration
# ============================================================

def integrate_to_surface(radial_profiles, sp0_pa, top_hpa):
    """Integrate radial velocity from top_hpa to local PB for every grid."""
    output = np.full(len(radial_profiles), np.nan, dtype=np.float64)
    effective_bottom = np.full(len(radial_profiles), np.nan, dtype=np.float64)

    start_index = int(np.where(PRESSURE_LEVELS_HPA == top_hpa)[0][0])
    levels = PRESSURE_LEVELS_HPA[start_index:]
    profiles = radial_profiles[:, start_index:]

    for index in range(len(profiles)):
        surface_pressure = sp0_pa[index]
        profile = profiles[index]

        if not np.isfinite(surface_pressure) or surface_pressure <= top_hpa * 100.0:
            continue
        if not np.isfinite(profile).all():
            continue

        bottom_hpa = min(surface_pressure / 100.0, 1000.0)
        interior = levels < bottom_hpa
        integration_levels = np.append(levels[interior], bottom_hpa)
        bottom_velocity = np.interp(bottom_hpa, levels, profile)
        integration_velocity = np.append(profile[interior], bottom_velocity)

        output[index] = np.trapz(
            integration_velocity,
            x=integration_levels * 100.0,
        ) / GRAVITY_MS2
        effective_bottom[index] = bottom_hpa

    return output, effective_bottom


def integrate_fixed_500_700(radial_profiles):
    """Integrate the fixed 500-700-hPa layer without an SP0 terrain mask."""
    mask = (
        (PRESSURE_LEVELS_HPA >= 500.0)
        & (PRESSURE_LEVELS_HPA <= 700.0)
    )
    layer = radial_profiles[:, mask]
    levels_pa = PRESSURE_LEVELS_HPA[mask] * 100.0

    # Match the fixed-layer convergence workflow: retain a grid cell whenever
    # all five pressure-level values (500, 550, 600, 650, and 700 hPa) exist.
    valid = np.isfinite(layer).all(axis=1)
    output = np.full(len(layer), np.nan, dtype=np.float64)
    if valid.any():
        output[valid] = np.trapz(layer[valid], x=levels_pa, axis=1) / GRAVITY_MS2
    return output


# ============================================================
# 7. Ring flux calculation
# ============================================================

def empty_ring_result(radius_km, product_name, surface_bottom):
    prefix = ring_prefix(radius_km, product_name)
    result = {
        f"{prefix}_KG_S": np.nan,
        f"{prefix}_COLUMN_TRANSPORT_MEAN_KG_M_S": np.nan,
        f"{prefix}_TOTAL_AREA_KM2": np.nan,
        f"{prefix}_VALID_AREA_KM2": np.nan,
        f"{prefix}_VALID_AREA_FRACTION": np.nan,
        f"{prefix}_TOTAL_GRID_COUNT": 0,
        f"{prefix}_VALID_GRID_COUNT": 0,
    }
    if surface_bottom:
        result.update(
            {
                f"{prefix}_SP0_LE_TOP_AREA_FRACTION": np.nan,
                f"{prefix}_SP0_GT_1000_AREA_FRACTION": np.nan,
                f"{prefix}_MEAN_SP0_HPA": np.nan,
                f"{prefix}_MEAN_INTEGRATION_DEPTH_HPA": np.nan,
            }
        )
    else:
        result[f"{prefix}_SP0_LT_700_AREA_FRACTION"] = np.nan
    return result


def area_fraction(condition, area, finite_area, total_area):
    if not np.isfinite(total_area) or total_area <= 0.0:
        return np.nan
    return float(area[condition & finite_area].sum() / total_area)


def calculate_ring_flux(
    data,
    transport,
    radius_km,
    product,
    effective_bottom_hpa=None,
):
    """Calculate one circumference-integrated radial mass flux."""
    product_name = product["name"]
    surface_bottom = product["surface_bottom"]
    prefix = ring_prefix(radius_km, product_name)
    result = empty_ring_result(radius_km, product_name, surface_bottom)

    lower = radius_km - RING_HALF_WIDTH_KM
    upper = radius_km + RING_HALF_WIDTH_KM
    # Both radial boundaries are included for every product.
    ring = (
        np.isfinite(data["distance_km"])
        & (data["distance_km"] >= lower)
        & (data["distance_km"] <= upper)
    )
    if not ring.any():
        return result, f"No grid cells in the {radius_km:g}-km ring"

    area = data["area_km2"][ring]
    ring_transport = transport[ring]
    sp0_pa = data["sp0_pa"][ring]
    finite_area = np.isfinite(area) & (area > 0.0)
    total_area = float(area[finite_area].sum())
    valid = finite_area & np.isfinite(ring_transport)
    valid_area = float(area[valid].sum())
    valid_fraction = valid_area / total_area if total_area > 0.0 else np.nan

    result.update(
        {
            f"{prefix}_TOTAL_AREA_KM2": total_area,
            f"{prefix}_VALID_AREA_KM2": valid_area,
            f"{prefix}_VALID_AREA_FRACTION": valid_fraction,
            f"{prefix}_TOTAL_GRID_COUNT": int(ring.sum()),
            f"{prefix}_VALID_GRID_COUNT": int(valid.sum()),
        }
    )

    if surface_bottom:
        top_hpa = product["top_hpa"]
        result[f"{prefix}_SP0_LE_TOP_AREA_FRACTION"] = area_fraction(
            np.isfinite(sp0_pa) & (sp0_pa <= top_hpa * 100.0),
            area,
            finite_area,
            total_area,
        )
        result[f"{prefix}_SP0_GT_1000_AREA_FRACTION"] = area_fraction(
            np.isfinite(sp0_pa) & (sp0_pa > 100000.0),
            area,
            finite_area,
            total_area,
        )
        if valid.any():
            weights = area[valid]
            result[f"{prefix}_MEAN_SP0_HPA"] = np.average(
                sp0_pa[valid] / 100.0,
                weights=weights,
            )
            ring_bottom = effective_bottom_hpa[ring]
            result[f"{prefix}_MEAN_INTEGRATION_DEPTH_HPA"] = np.average(
                ring_bottom[valid] - top_hpa,
                weights=weights,
            )
    else:
        result[f"{prefix}_SP0_LT_700_AREA_FRACTION"] = area_fraction(
            np.isfinite(sp0_pa) & (sp0_pa < 70000.0),
            area,
            finite_area,
            total_area,
        )

    # Do not reject a ring based on a prescribed coverage threshold. The
    # coverage diagnostics are retained in the output so they can be examined
    # or filtered later if required.
    if not valid.any():
        return result, (
            f"No valid vertically integrated grid cells in the "
            f"{radius_km:g}-km {product_name} ring"
        )

    mean_transport = float(np.average(ring_transport[valid], weights=area[valid]))
    circumference_m = 2.0 * np.pi * radius_km * 1000.0
    result[f"{prefix}_COLUMN_TRANSPORT_MEAN_KG_M_S"] = mean_transport
    result[f"{prefix}_KG_S"] = mean_transport * circumference_m
    return result, ""


# ============================================================
# 8. One-record calculation
# ============================================================

def calculate_one_record(record):
    row_id, sid, iso_time, rain_time_id = record
    data = read_aligned_fields(row_id, rain_time_id)

    transport_500_pb, bottom_500_pb = integrate_to_surface(
        data["radial_ms"], data["sp0_pa"], 500.0
    )
    transport_500_700 = integrate_fixed_500_700(data["radial_ms"])
    transport_700_pb, bottom_700_pb = integrate_to_surface(
        data["radial_ms"], data["sp0_pa"], 700.0
    )

    transports = {
        "500_PB": (transport_500_pb, bottom_500_pb),
        "500_700": (transport_500_700, None),
        "700_PB": (transport_700_pb, bottom_700_pb),
    }

    result = {
        "ROW_ID": row_id,
        "SID": sid,
        "ISO_TIME": iso_time,
        "RAIN_TIME_ID": rain_time_id,
    }
    errors = []

    for product in PRODUCTS:
        name = product["name"]
        inner = product["inner_radius_km"]
        outer = product["outer_radius_km"]
        transport, effective_bottom = transports[name]

        inner_result, inner_error = calculate_ring_flux(
            data, transport, inner, product, effective_bottom
        )
        outer_result, outer_error = calculate_ring_flux(
            data, transport, outer, product, effective_bottom
        )
        result.update(inner_result)
        result.update(outer_result)

        inner_flux = inner_result[f"{ring_prefix(inner, name)}_KG_S"]
        outer_flux = outer_result[f"{ring_prefix(outer, name)}_KG_S"]
        difference_column = (
            f"F{int(outer)}_MINUS_F{int(inner)}_{name}_KG_S"
        )
        result[difference_column] = (
            outer_flux - inner_flux
            if np.isfinite(outer_flux) and np.isfinite(inner_flux)
            else np.nan
        )

        if inner_error:
            errors.append(inner_error)
        if outer_error:
            errors.append(outer_error)

    result["FLUX_STATUS"] = "OK" if not errors else "INSUFFICIENT_VALID_COVERAGE"
    result["FLUX_ERROR"] = " | ".join(errors)
    return result


# ============================================================
# 9. Monthly worker
# ============================================================

def process_month(task):
    year_month, records = task
    processed = skipped = failed = 0
    errors = []

    for record in records:
        row_id, sid, iso_time, rain_time_id = record
        target = result_path(row_id, rain_time_id)

        if SKIP_EXISTING and result_is_complete(target):
            skipped += 1
            continue

        try:
            result = calculate_one_record(record)
            write_gzip_csv_atomic(pd.DataFrame([result]), target)
            processed += 1
        except Exception as error:
            failed += 1
            errors.append(
                {
                    "YEAR_MONTH": year_month,
                    "ROW_ID": row_id,
                    "SID": sid,
                    "ISO_TIME": iso_time,
                    "RAIN_TIME_ID": rain_time_id,
                    "ERROR": f"{type(error).__name__}: {error}",
                }
            )

    return {
        "YEAR_MONTH": year_month,
        "RECORD_COUNT": len(records),
        "PROCESSED_COUNT": processed,
        "SKIPPED_COUNT": skipped,
        "FAILED_COUNT": failed,
        "ERRORS": errors,
    }


# ============================================================
# 10. Result collection and final merge
# ============================================================

def collect_row_results():
    files = sorted(ROW_RESULT_DIR.glob("*/*.csv.gz"))
    frames = []

    for path in tqdm(files, desc="Collecting RUN3 results", unit="file"):
        try:
            frame = pd.read_csv(path, low_memory=False)
            if set(RESULT_COLUMNS).issubset(frame.columns):
                frames.append(frame[RESULT_COLUMNS])
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    result = pd.concat(frames, ignore_index=True)
    return result.drop_duplicates("ROW_ID", keep="last")


def build_final_table(results):
    """Merge valid RUN3 results back into the complete threshold-30 table."""
    source = pd.read_csv(SOURCE_CSV, low_memory=False)
    if "ROW_ID" not in source.columns:
        source.insert(0, "ROW_ID", np.arange(len(source), dtype=np.int64))
    source["ROW_ID"] = pd.to_numeric(source["ROW_ID"], errors="raise").astype(np.int64)

    result_values = results.drop(
        columns=["SID", "ISO_TIME", "RAIN_TIME_ID"],
        errors="ignore",
    )
    overlap = [
        column for column in result_values.columns
        if column != "ROW_ID" and column in source.columns
    ]
    if overlap:
        source = source.drop(columns=overlap)

    final = source.merge(
        result_values,
        on="ROW_ID",
        how="left",
        validate="one_to_one",
    )
    write_csv_atomic(final, FINAL_OUTPUT_CSV)
    return final


# ============================================================
# 11. Main program
# ============================================================

def main():
    print("=" * 96)
    print("Multi-layer TC-relative radial mass-flux calculation")
    print("=" * 96)
    print(f"Source table: {SOURCE_CSV}")
    print(f"ERA5 input root: {ERA5_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print("Ring width: target radius +/- 12.5 km")
    print("500-PB: 200- and 500-km rings")
    print("500-700 hPa: 300- and 500-km rings; fixed layer without SP0 masking")
    print("700-PB: 100- and 500-km rings")
    print("No minimum valid-area fraction is imposed")
    print("Positive flux: inward transport toward the TC centre")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ROW_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    frame, unavailable = read_source_records()
    tasks = build_monthly_tasks(frame)
    write_csv_atomic(unavailable, LOG_DIR / "records_without_run2_inflow500.csv")

    print("\n" + "=" * 96)
    print("Input inventory")
    print("=" * 96)
    print(f"Eligible RUN3 records: {len(frame):,}")
    print(f"Records without RUN2 INFLOW_500: {len(unavailable):,}")
    print(f"Monthly tasks: {len(tasks):,}")
    print(f"Worker processes: {min(MAX_WORKERS, len(tasks))}")

    monthly_summaries = []
    all_errors = []
    with Pool(
        processes=min(MAX_WORKERS, len(tasks)),
        maxtasksperchild=12,
    ) as pool:
        iterator = pool.imap_unordered(
            process_month,
            tasks,
            chunksize=POOL_CHUNKSIZE,
        )
        for summary in tqdm(
            iterator,
            total=len(tasks),
            desc="RUN3 radial mass flux",
            unit="month",
        ):
            all_errors.extend(summary.pop("ERRORS"))
            monthly_summaries.append(summary)

    monthly_log = pd.DataFrame(monthly_summaries).sort_values("YEAR_MONTH")
    error_log = pd.DataFrame(
        all_errors,
        columns=[
            "YEAR_MONTH", "ROW_ID", "SID", "ISO_TIME",
            "RAIN_TIME_ID", "ERROR",
        ],
    )
    write_csv_atomic(monthly_log, LOG_DIR / "processing_month_log.csv")
    write_csv_atomic(error_log, LOG_DIR / "processing_error_log.csv")

    total_processed = int(monthly_log["PROCESSED_COUNT"].sum())
    total_skipped = int(monthly_log["SKIPPED_COUNT"].sum())
    total_failed = int(monthly_log["FAILED_COUNT"].sum())

    print("\n" + "=" * 96)
    print("Collecting row results and building the final table")
    print("=" * 96)
    results = collect_row_results()
    final = build_final_table(results)

    summary = pd.DataFrame(
        [
            {"ITEM": "ELIGIBLE_RUN3_RECORDS", "VALUE": len(frame)},
            {"ITEM": "NEW_RESULTS", "VALUE": total_processed},
            {"ITEM": "REUSED_RESULTS", "VALUE": total_skipped},
            {"ITEM": "FAILED_RESULTS", "VALUE": total_failed},
            {"ITEM": "COLLECTED_RESULTS", "VALUE": len(results)},
            {"ITEM": "FINAL_TABLE_ROWS", "VALUE": len(final)},
            {"ITEM": "FINAL_TABLE_COLUMNS", "VALUE": len(final.columns)},
        ]
    )
    write_csv_atomic(summary, LOG_DIR / "processing_summary.csv")

    print("\n" + "=" * 96)
    print("RUN3 completed")
    print("=" * 96)
    print(f"New row results: {total_processed:,}")
    print(f"Existing row results reused: {total_skipped:,}")
    print(f"Failed records: {total_failed:,}")
    print(f"Collected complete results: {len(results):,}")
    print(f"Final table rows: {len(final):,}")
    print(f"Final output: {FINAL_OUTPUT_CSV}")
    print(f"Logs: {LOG_DIR}")

    if total_failed:
        raise RuntimeError(
            f"RUN3 finished with {total_failed} failed records; review {LOG_DIR}"
        )


if __name__ == "__main__":
    main()
