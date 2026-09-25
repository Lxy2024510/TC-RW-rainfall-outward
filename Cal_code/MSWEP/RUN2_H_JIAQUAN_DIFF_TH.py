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

# The script is expected at TC-RW-V1/Cal_code/MSWEP/.
# TC_RW_PROJECT_ROOT can override automatic project-root detection.
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(
    os.environ.get("TC_RW_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))
).expanduser().resolve()

BASE_CSV = (
    PROJECT_ROOT / "Data" / "Processed" / "IBTrACS"
    / "PRE_DATA_IBT_1982_2024_BASE.csv"
)

MSWEP_GRID_DIR = (
    PROJECT_ROOT / "Data" / "Intermediate" / "MSWEP" / "Grids_500KM"
)

YEAR_CACHE_DIR = (
    PROJECT_ROOT / ".work" / "MSWEP" / "RUN2" / "year_cache_v3"
)

LOG_DIR = (
    PROJECT_ROOT / "Results" / "Quality_control" / "MSWEP" / "RUN2"
)

FINAL_OUTPUT_DIR = (
    PROJECT_ROOT / "Data" / "Processed" / "MSWEP" / "Thresholds"
)


# ============================================================
# 2. Processing parameters
# ============================================================

MAX_WORKERS = min(
    24,
    cpu_count(),
)

POOL_CHUNKSIZE = 1

THRESHOLDS = [
    10,
    20,
    30,
    40,
]

# Reuse valid annual cache files when rerunning the program.
REUSE_YEAR_CACHE = True

# Fail before final merging if any grid file cannot be processed.
FAIL_ON_FILE_ERROR = True

EXPECTED_RADIUS_KM = 500.0
DISTANCE_TOLERANCE_KM = 0.01


# ============================================================
# 3. File-name pattern
# ============================================================

GRID_FILE_PATTERN = re.compile(
    r"^row_(\d+)_"
    r"(\d{7}\.\d{2})"
    r"\.csv\.gz$"
)


# ============================================================
# 4. Input grid columns
# ============================================================

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
# 5. Distance-zone definitions
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


# ============================================================
# 6. General helper functions
# ============================================================

def threshold_label(threshold):
    """
    Convert a numeric threshold to its column-name representation.

    Examples
    --------
    1.5 -> "1.5"
    10  -> "10"
    """
    threshold = float(threshold)

    if threshold.is_integer():
        return str(int(threshold))

    return str(threshold)


def build_threshold_columns(threshold):
    """
    Return all summary columns for one precipitation threshold.
    """
    label = threshold_label(
        threshold
    )

    columns = [
        "ROW_ID",
        "RAIN_POINT",
        f"RAIN_POINT_{label}",
    ]

    for zone_name in ALL_ZONE_NAMES:
        columns.append(
            f"MSWEP_AREA_{label}_{zone_name}"
        )

    for zone_name in ALL_ZONE_NAMES:
        columns.append(
            f"MSWEP_VOLUME_{label}_{zone_name}"
        )

    columns.append(
        f"MSWEP_DIST_{label}"
    )

    return columns


def build_all_summary_columns():
    """
    Return the complete annual-cache column list.
    """
    columns = [
        "ROW_ID",
        "RAIN_POINT",
    ]

    for threshold in THRESHOLDS:
        label = threshold_label(
            threshold
        )

        columns.append(
            f"RAIN_POINT_{label}"
        )

        for zone_name in ALL_ZONE_NAMES:
            columns.append(
                f"MSWEP_AREA_{label}_{zone_name}"
            )

        for zone_name in ALL_ZONE_NAMES:
            columns.append(
                f"MSWEP_VOLUME_{label}_{zone_name}"
            )

        columns.append(
            f"MSWEP_DIST_{label}"
        )

    return columns


ALL_SUMMARY_COLUMNS = (
    build_all_summary_columns()
)


def parse_grid_file_name(file_path):
    """
    Extract ROW_ID and RAIN_TIME_ID from a grid file name.

    Example
    -------
    row_000000000_1982001.00.csv.gz
    """
    match = GRID_FILE_PATTERN.fullmatch(
        file_path.name
    )

    if match is None:
        raise ValueError(
            "Unexpected grid file name: "
            f"{file_path.name}"
        )

    row_id = int(
        match.group(1)
    )

    rain_time_id = match.group(2)

    return row_id, rain_time_id


def write_gzip_csv_atomic(
    df,
    output_path,
):
    """
    Write a gzip CSV atomically.

    The formal output appears only after the temporary file has
    been written successfully.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = output_path.with_name(
        output_path.name
        + f".tmp.{os.getpid()}"
    )

    try:
        with gzip.open(
            temp_path,
            mode="wt",
            encoding="utf-8",
            newline="",
            compresslevel=1,
        ) as gzip_file:

            df.to_csv(
                gzip_file,
                index=False,
            )

        if not temp_path.exists():
            raise RuntimeError(
                "Temporary file was not created: "
                f"{temp_path}"
            )

        if temp_path.stat().st_size == 0:
            raise RuntimeError(
                "Temporary file is empty: "
                f"{temp_path}"
            )

        os.replace(
            temp_path,
            output_path,
        )

    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def write_csv_atomic(
    df,
    output_path,
    date_format=None,
):
    """
    Write an uncompressed CSV atomically.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = output_path.with_name(
        output_path.name
        + f".tmp.{os.getpid()}"
    )

    try:
        df.to_csv(
            temp_path,
            index=False,
            date_format=date_format,
        )

        if not temp_path.exists():
            raise RuntimeError(
                "Temporary file was not created: "
                f"{temp_path}"
            )

        if temp_path.stat().st_size == 0:
            raise RuntimeError(
                "Temporary file is empty: "
                f"{temp_path}"
            )

        os.replace(
            temp_path,
            output_path,
        )

    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


# ============================================================
# 7. Derive quadrant-mean wind radii
# ============================================================

R34_QUADRANT_COLUMNS = [
    "USA_R34_NE",
    "USA_R34_SE",
    "USA_R34_SW",
    "USA_R34_NW",
]

R64_QUADRANT_COLUMNS = [
    "USA_R64_NE",
    "USA_R64_SE",
    "USA_R64_SW",
    "USA_R64_NW",
]

NAUTICAL_MILE_TO_KM = 1.852


def add_quadrant_mean_wind_radii(base_df):
    """
    Calculate strict four-quadrant R34 and R64 mean radii.

    IBTrACS wind radii are stored in nautical miles. Zero is retained as a
    physically valid quadrant radius; only missing values are invalid. R64 is
    additionally restricted to records with USA_WIND >= 64 kt.
    """
    base_df = base_df.copy()
    radius_columns = R34_QUADRANT_COLUMNS + R64_QUADRANT_COLUMNS

    for column in ["USA_WIND"] + radius_columns:
        base_df[column] = pd.to_numeric(base_df[column], errors="coerce")

    r34_valid = base_df[R34_QUADRANT_COLUMNS].notna().all(axis=1)
    r64_valid = (
        base_df[R64_QUADRANT_COLUMNS].notna().all(axis=1)
        & base_df["USA_WIND"].ge(64.0)
    )

    base_df["USA_R34_ALL_QUADRANTS_VALID"] = r34_valid.astype(np.int8)
    base_df["USA_R64_ALL_QUADRANTS_VALID"] = r64_valid.astype(np.int8)

    base_df["USA_R34_MEAN_NMI"] = np.where(
        r34_valid,
        base_df[R34_QUADRANT_COLUMNS].mean(axis=1),
        np.nan,
    )
    base_df["USA_R64_MEAN_NMI"] = np.where(
        r64_valid,
        base_df[R64_QUADRANT_COLUMNS].mean(axis=1),
        np.nan,
    )
    base_df["USA_R34_MEAN_KM"] = (
        base_df["USA_R34_MEAN_NMI"] * NAUTICAL_MILE_TO_KM
    )
    base_df["USA_R64_MEAN_KM"] = (
        base_df["USA_R64_MEAN_NMI"] * NAUTICAL_MILE_TO_KM
    )

    print(
        "R34 records with four non-missing quadrants: "
        f"{int(r34_valid.sum()):,}"
    )
    print(
        "R64 records with USA_WIND >= 64 kt and four non-missing quadrants: "
        f"{int(r64_valid.sum()):,}"
    )
    return base_df


# ============================================================
# 8. Read and validate the BASE table
# ============================================================

def read_base_table():
    """
    Read the BASE CSV and create its zero-based ROW_ID.
    """
    print("=" * 80)
    print("Reading the BASE table")
    print("=" * 80)

    if not BASE_CSV.exists():
        raise FileNotFoundError(
            f"BASE CSV does not exist: {BASE_CSV}"
        )

    base_df = pd.read_csv(
        BASE_CSV,
        dtype={
            "SID": "string",
            "RAIN_TIME_ID": "string",
        },
        low_memory=False,
    )

    required_columns = {
        "SID",
        "ISO_TIME",
        "RAIN_TIME_ID",
        "USA_WIND",
        "USA_R34_NE",
        "USA_R34_SE",
        "USA_R34_SW",
        "USA_R34_NW",
        "USA_R64_NE",
        "USA_R64_SE",
        "USA_R64_SW",
        "USA_R64_NW",
    }

    missing_columns = (
        required_columns.difference(
            base_df.columns
        )
    )

    if missing_columns:
        raise KeyError(
            "BASE CSV is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    base_df = base_df.reset_index(
        drop=True
    )

    base_df.insert(
        0,
        "ROW_ID",
        np.arange(
            len(base_df),
            dtype=np.int64,
        ),
    )

    base_df["ISO_TIME"] = pd.to_datetime(
        base_df["ISO_TIME"],
        errors="coerce",
    )

    invalid_time_mask = (
        base_df["ISO_TIME"].isna()
    )

    if invalid_time_mask.any():
        raise ValueError(
            "BASE CSV contains invalid ISO_TIME values. "
            f"Count: {int(invalid_time_mask.sum())}"
        )

    invalid_rain_id_mask = (
        ~base_df["RAIN_TIME_ID"]
        .str.fullmatch(
            r"\d{7}\.\d{2}",
            na=False,
        )
    )

    if invalid_rain_id_mask.any():
        examples = (
            base_df.loc[
                invalid_rain_id_mask,
                [
                    "ROW_ID",
                    "ISO_TIME",
                    "RAIN_TIME_ID",
                ],
            ]
            .head(20)
            .to_dict("records")
        )

        raise ValueError(
            "BASE CSV contains invalid RAIN_TIME_ID values. "
            f"Count: {int(invalid_rain_id_mask.sum())}. "
            f"Examples: {examples}"
        )

    base_df["YEAR"] = (
        base_df["ISO_TIME"]
        .dt.year
        .astype(np.int16)
    )

    base_df = add_quadrant_mean_wind_radii(base_df)

    print(
        f"BASE rows: {len(base_df)}"
    )

    print(
        f"ROW_ID range: "
        f"{base_df['ROW_ID'].min()} to "
        f"{base_df['ROW_ID'].max()}"
    )

    print(
        f"Year range: "
        f"{base_df['YEAR'].min()} to "
        f"{base_df['YEAR'].max()}"
    )

    return base_df


# ============================================================
# 8. Validate year directories and construct tasks
# ============================================================

def build_year_tasks(base_df):
    """
    Build one processing task per calendar year.
    """
    year_tasks = []

    base_years = sorted(
        base_df["YEAR"]
        .astype(int)
        .unique()
        .tolist()
    )

    for year in base_years:
        year_dir = (
            MSWEP_GRID_DIR
            / str(year)
        )

        if not year_dir.is_dir():
            raise FileNotFoundError(
                f"MSWEP year directory is missing: {year_dir}"
            )

        expected_row_ids = (
            base_df.loc[
                base_df["YEAR"] == year,
                "ROW_ID",
            ]
            .astype(np.int64)
            .tolist()
        )

        year_tasks.append(
            (
                int(year),
                str(year_dir),
                expected_row_ids,
            )
        )

    print(
        f"Year tasks: {len(year_tasks)}"
    )

    print(
        f"Worker processes: {MAX_WORKERS}"
    )

    return year_tasks


# ============================================================
# 9. Distance-zone masks
# ============================================================

def create_distance_zone_masks(distance):
    """
    Create mutually exclusive distance-zone masks.

    The interval convention is:

        [0, 100]
        (100, 200]
        (200, 300]
        (300, 400]
        (400, 500]
    """
    zone_masks = {
        "0_100": (
            (distance >= 0.0)
            & (distance <= 100.0)
        ),
        "100_200": (
            (distance > 100.0)
            & (distance <= 200.0)
        ),
        "200_300": (
            (distance > 200.0)
            & (distance <= 300.0)
        ),
        "300_400": (
            (distance > 300.0)
            & (distance <= 400.0)
        ),
        "400_500": (
            (distance > 400.0)
            & (
                distance
                <= (
                    EXPECTED_RADIUS_KM
                    + DISTANCE_TOLERANCE_KM
                )
            )
        ),
    }

    return zone_masks


# ============================================================
# 10. Summarise one grid file
# ============================================================

def summarise_grid_file(
    file_path,
    expected_year,
):
    """
    Calculate all threshold metrics for one TC grid file.
    """
    row_id, rain_time_id = (
        parse_grid_file_name(
            file_path
        )
    )

    file_year = int(
        rain_time_id[:4]
    )

    if file_year != expected_year:
        raise ValueError(
            "File year does not match its directory: "
            f"file={file_path.name}, "
            f"directory_year={expected_year}"
        )

    grid_df = pd.read_csv(
        file_path,
        usecols=GRID_COLUMNS,
        dtype=GRID_DTYPES,
        low_memory=False,
    )

    if grid_df.empty:
        raise ValueError(
            f"Grid file is empty: {file_path}"
        )

    precipitation = (
        grid_df["PRECIP_3H"]
        .to_numpy(
            dtype=np.float32,
            copy=False,
        )
    )

    distance = (
        grid_df["DIST_KM"]
        .to_numpy(
            dtype=np.float32,
            copy=False,
        )
    )

    area = (
        grid_df["AREA_KM2"]
        .to_numpy(
            dtype=np.float32,
            copy=False,
        )
    )

    volume = (
        grid_df["RAIN_VOLUME_M3"]
        .to_numpy(
            dtype=np.float64,
            copy=False,
        )
    )

    finite_mask = (
        np.isfinite(precipitation)
        & np.isfinite(distance)
        & np.isfinite(area)
        & np.isfinite(volume)
    )

    if not finite_mask.all():
        raise ValueError(
            "Non-finite grid values were found in "
            f"{file_path}. Count: "
            f"{int((~finite_mask).sum())}"
        )

    if np.any(precipitation < 0.0):
        raise ValueError(
            "Negative precipitation values were found in "
            f"{file_path}."
        )

    if np.any(area <= 0.0):
        raise ValueError(
            "Non-positive grid areas were found in "
            f"{file_path}."
        )

    if np.any(volume < 0.0):
        raise ValueError(
            "Negative rain volumes were found in "
            f"{file_path}."
        )

    if np.any(distance < 0.0):
        raise ValueError(
            "Negative distances were found in "
            f"{file_path}."
        )

    max_distance = float(
        np.max(distance)
    )

    if (
        max_distance
        > (
            EXPECTED_RADIUS_KM
            + DISTANCE_TOLERANCE_KM
        )
    ):
        raise ValueError(
            "A distance exceeds the expected 500-km radius "
            f"in {file_path}: {max_distance}"
        )

    zone_masks = (
        create_distance_zone_masks(
            distance
        )
    )

    record = {
        "ROW_ID": row_id,
        "RAIN_POINT": int(
            precipitation.size
        ),
    }

    # Evaluate all thresholds once. Reusing this matrix avoids rebuilding
    # the same masks for every distance band and every statistic.
    threshold_values = np.asarray(
        THRESHOLDS,
        dtype=np.float32,
    )
    threshold_masks = (
        precipitation[:, None]
        >= threshold_values[None, :]
    )
    rain_point_counts = threshold_masks.sum(
        axis=0,
        dtype=np.int64,
    )

    area_by_zone = {}
    volume_by_zone = {}

    for zone_name in BASE_ZONE_NAMES:
        zone_mask = zone_masks[zone_name]
        zone_threshold_masks = threshold_masks[zone_mask]

        area_by_zone[zone_name] = np.sum(
            np.where(
                zone_threshold_masks,
                area[zone_mask, None],
                0.0,
            ),
            axis=0,
            dtype=np.float64,
        )

        volume_by_zone[zone_name] = np.sum(
            np.where(
                zone_threshold_masks,
                volume[zone_mask, None],
                0.0,
            ),
            axis=0,
            dtype=np.float64,
        )

    area_by_zone["0_200"] = (
        area_by_zone["0_100"]
        + area_by_zone["100_200"]
    )
    area_by_zone["200_500"] = (
        area_by_zone["200_300"]
        + area_by_zone["300_400"]
        + area_by_zone["400_500"]
    )
    area_by_zone["0_500"] = (
        area_by_zone["0_200"]
        + area_by_zone["200_500"]
    )

    volume_by_zone["0_200"] = (
        volume_by_zone["0_100"]
        + volume_by_zone["100_200"]
    )
    volume_by_zone["200_500"] = (
        volume_by_zone["200_300"]
        + volume_by_zone["300_400"]
        + volume_by_zone["400_500"]
    )
    volume_by_zone["0_500"] = (
        volume_by_zone["0_200"]
        + volume_by_zone["200_500"]
    )

    weighted_numerators = np.sum(
        np.where(
            threshold_masks,
            (volume * distance)[:, None],
            0.0,
        ),
        axis=0,
        dtype=np.float64,
    )

    for threshold_index, threshold in enumerate(THRESHOLDS):
        label = threshold_label(
            threshold
        )

        record[
            f"RAIN_POINT_{label}"
        ] = int(
            rain_point_counts[threshold_index]
        )

        for zone_name in ALL_ZONE_NAMES:
            record[
                f"MSWEP_AREA_{label}_{zone_name}"
            ] = float(
                area_by_zone[zone_name][threshold_index]
            )

        for zone_name in ALL_ZONE_NAMES:
            record[
                f"MSWEP_VOLUME_{label}_{zone_name}"
            ] = float(
                volume_by_zone[zone_name][threshold_index]
            )

        total_selected_volume = (
            volume_by_zone["0_500"][threshold_index]
        )

        if total_selected_volume > 0.0:
            weighted_distance = float(
                weighted_numerators[threshold_index]
                / total_selected_volume
            )
        else:
            weighted_distance = np.nan

        record[
            f"MSWEP_DIST_{label}"
        ] = weighted_distance

    return record


# ============================================================
# 11. Validate an annual cache
# ============================================================

def validate_year_cache(
    cache_path,
    expected_row_ids,
):
    """
    Validate an existing annual cache before reusing it.
    """
    if not cache_path.exists():
        return False

    if cache_path.stat().st_size == 0:
        return False

    try:
        cache_columns = pd.read_csv(
            cache_path,
            nrows=0,
        ).columns.tolist()

        if cache_columns != (
            ALL_SUMMARY_COLUMNS
        ):
            return False

        cache_ids = pd.read_csv(
            cache_path,
            usecols=["ROW_ID"],
            dtype={"ROW_ID": np.int64},
        )["ROW_ID"].to_numpy(
            dtype=np.int64,
            copy=False,
        )

        if np.unique(cache_ids).size != cache_ids.size:
            return False

        expected_ids = np.asarray(
            expected_row_ids,
            dtype=np.int64,
        )

        if not np.array_equal(
            np.sort(cache_ids),
            np.sort(expected_ids),
        ):
            return False

        return True

    except Exception:
        return False


# ============================================================
# 12. Process one year
# ============================================================

def process_year_task(task):
    """
    Process all TC grid files in one year directory.
    """
    (
        year,
        year_dir_text,
        expected_row_ids,
    ) = task

    year_dir = Path(
        year_dir_text
    )

    cache_path = (
        YEAR_CACHE_DIR
        / f"summary_{year}.csv.gz"
    )

    error_path = (
        LOG_DIR
        / f"errors_{year}.csv"
    )

    result = {
        "YEAR": year,
        "EXPECTED_FILES": len(
            expected_row_ids
        ),
        "FOUND_FILES": 0,
        "PROCESSED_FILES": 0,
        "CACHE_REUSED": 0,
        "ERROR_COUNT": 0,
        "STATUS": "UNKNOWN",
        "CACHE_PATH": str(
            cache_path
        ),
        "ERROR_PATH": str(
            error_path
        ),
    }

    if (
        REUSE_YEAR_CACHE
        and validate_year_cache(
            cache_path,
            expected_row_ids,
        )
    ):
        result["FOUND_FILES"] = len(
            expected_row_ids
        )

        result["PROCESSED_FILES"] = len(
            expected_row_ids
        )

        result["CACHE_REUSED"] = 1
        result["STATUS"] = "CACHE_REUSED"

        return result

    grid_files = list(
        year_dir.glob("row_*.csv.gz")
    )

    result["FOUND_FILES"] = len(
        grid_files
    )

    expected_id_set = set(
        int(value)
        for value in expected_row_ids
    )

    file_map = {}
    errors = []

    for file_path in grid_files:
        try:
            row_id, _ = (
                parse_grid_file_name(
                    file_path
                )
            )

            if row_id in file_map:
                errors.append(
                    {
                        "YEAR": year,
                        "ROW_ID": row_id,
                        "FILE_PATH": str(
                            file_path
                        ),
                        "ERROR_TYPE": (
                            "DUPLICATE_ROW_ID"
                        ),
                        "ERROR_MESSAGE": (
                            "More than one grid file "
                            "uses this ROW_ID."
                        ),
                    }
                )
            else:
                file_map[row_id] = (
                    file_path
                )

        except Exception as error:
            errors.append(
                {
                    "YEAR": year,
                    "ROW_ID": np.nan,
                    "FILE_PATH": str(
                        file_path
                    ),
                    "ERROR_TYPE": type(
                        error
                    ).__name__,
                    "ERROR_MESSAGE": str(
                        error
                    ),
                }
            )

    found_id_set = set(
        file_map.keys()
    )

    missing_ids = sorted(
        expected_id_set
        - found_id_set
    )

    unexpected_ids = sorted(
        found_id_set
        - expected_id_set
    )

    for row_id in missing_ids:
        errors.append(
            {
                "YEAR": year,
                "ROW_ID": row_id,
                "FILE_PATH": "",
                "ERROR_TYPE": (
                    "MISSING_GRID_FILE"
                ),
                "ERROR_MESSAGE": (
                    "No grid file was found for "
                    "this BASE ROW_ID."
                ),
            }
        )

    for row_id in unexpected_ids:
        errors.append(
            {
                "YEAR": year,
                "ROW_ID": row_id,
                "FILE_PATH": str(
                    file_map[row_id]
                ),
                "ERROR_TYPE": (
                    "UNEXPECTED_ROW_ID"
                ),
                "ERROR_MESSAGE": (
                    "Grid file ROW_ID is not expected "
                    "for this year."
                ),
            }
        )

    records = []

    valid_row_ids = sorted(
        expected_id_set
        & found_id_set
    )

    for row_id in valid_row_ids:
        file_path = file_map[
            row_id
        ]

        try:
            record = summarise_grid_file(
                file_path,
                year,
            )

            records.append(
                record
            )

        except Exception as error:
            errors.append(
                {
                    "YEAR": year,
                    "ROW_ID": row_id,
                    "FILE_PATH": str(
                        file_path
                    ),
                    "ERROR_TYPE": type(
                        error
                    ).__name__,
                    "ERROR_MESSAGE": str(
                        error
                    ),
                }
            )

    result["PROCESSED_FILES"] = len(
        records
    )

    result["ERROR_COUNT"] = len(
        errors
    )

    if errors:
        error_df = pd.DataFrame(
            errors
        )

        write_csv_atomic(
            error_df,
            error_path,
        )

        result["STATUS"] = "FAILED"

        return result

    summary_df = pd.DataFrame(
        records
    )

    if summary_df.empty:
        result["ERROR_COUNT"] = 1
        result["STATUS"] = "FAILED"

        error_df = pd.DataFrame(
            [
                {
                    "YEAR": year,
                    "ROW_ID": np.nan,
                    "FILE_PATH": str(
                        year_dir
                    ),
                    "ERROR_TYPE": (
                        "EMPTY_YEAR_RESULT"
                    ),
                    "ERROR_MESSAGE": (
                        "No annual summary records "
                        "were generated."
                    ),
                }
            ]
        )

        write_csv_atomic(
            error_df,
            error_path,
        )

        return result

    summary_df = summary_df[
        ALL_SUMMARY_COLUMNS
    ]

    summary_df = summary_df.sort_values(
        "ROW_ID"
    ).reset_index(
        drop=True
    )

    if summary_df["ROW_ID"].duplicated().any():
        result["ERROR_COUNT"] = 1
        result["STATUS"] = "FAILED"

        return result

    actual_id_set = set(
        summary_df["ROW_ID"]
        .astype(np.int64)
        .tolist()
    )

    if actual_id_set != expected_id_set:
        result["ERROR_COUNT"] = 1
        result["STATUS"] = "FAILED"

        return result

    write_gzip_csv_atomic(
        summary_df,
        cache_path,
    )

    result["STATUS"] = "SUCCESS"

    return result


# ============================================================
# 13. Combine annual caches
# ============================================================

def combine_year_caches(
    year_tasks,
    expected_total_rows,
):
    """
    Combine all validated annual caches into one summary table.
    """
    annual_frames = []

    for (
        year,
        _,
        expected_row_ids,
    ) in tqdm(
        year_tasks,
        desc="Reading annual caches",
        unit="year",
    ):
        cache_path = (
            YEAR_CACHE_DIR
            / f"summary_{year}.csv.gz"
        )

        if not validate_year_cache(
            cache_path,
            expected_row_ids,
        ):
            raise RuntimeError(
                "Annual cache validation failed: "
                f"{cache_path}"
            )

        annual_df = pd.read_csv(
            cache_path,
            low_memory=False,
        )

        annual_frames.append(
            annual_df
        )

    summary_df = pd.concat(
        annual_frames,
        axis=0,
        ignore_index=True,
    )

    summary_df["ROW_ID"] = pd.to_numeric(
        summary_df["ROW_ID"],
        errors="raise",
    ).astype(
        np.int64
    )

    summary_df = summary_df.sort_values(
        "ROW_ID"
    ).reset_index(
        drop=True
    )

    if len(summary_df) != expected_total_rows:
        raise RuntimeError(
            "Combined summary row count does not match BASE: "
            f"summary={len(summary_df)}, "
            f"base={expected_total_rows}"
        )

    if summary_df["ROW_ID"].duplicated().any():
        duplicate_ids = (
            summary_df.loc[
                summary_df[
                    "ROW_ID"
                ].duplicated(
                    keep=False
                ),
                "ROW_ID",
            ]
            .head(20)
            .tolist()
        )

        raise RuntimeError(
            "Duplicate ROW_ID values were found in "
            f"the combined summary: {duplicate_ids}"
        )

    expected_ids = np.arange(
        expected_total_rows,
        dtype=np.int64,
    )

    actual_ids = summary_df[
        "ROW_ID"
    ].to_numpy(
        dtype=np.int64
    )

    if not np.array_equal(
        actual_ids,
        expected_ids,
    ):
        raise RuntimeError(
            "Combined summary ROW_ID values do not "
            "fully cover the BASE table."
        )

    return summary_df


# ============================================================
# 14. Create final threshold-specific BASE tables
# ============================================================

def create_final_threshold_tables(
    base_df,
    summary_df,
):
    """
    Create one BASE-wide output table for each threshold.
    """
    output_paths = []

    base_output_df = base_df.drop(
        columns=["YEAR"],
        errors="ignore",
    ).copy()

    for threshold in THRESHOLDS:
        label = threshold_label(
            threshold
        )

        metric_columns = (
            build_threshold_columns(
                threshold
            )
        )

        threshold_summary = summary_df[
            metric_columns
        ]

        summary_row_ids = threshold_summary[
            "ROW_ID"
        ].to_numpy(dtype=np.int64, copy=False)
        base_row_ids = base_output_df[
            "ROW_ID"
        ].to_numpy(dtype=np.int64, copy=False)

        if not np.array_equal(summary_row_ids, base_row_ids):
            raise RuntimeError(
                "Summary rows are not aligned with the BASE table "
                f"for threshold {label}."
            )

        final_df = base_output_df.copy()
        for column in metric_columns:
            if column != "ROW_ID":
                final_df[column] = threshold_summary[
                    column
                ].to_numpy(copy=False)

        if len(final_df) != len(
            base_output_df
        ):
            raise RuntimeError(
                "Final merged row count changed for "
                f"threshold {label}."
            )

        metric_only_columns = [
            column
            for column in metric_columns
            if column != "ROW_ID"
        ]

        completely_missing_mask = (
            final_df[
                metric_only_columns
            ]
            .isna()
            .all(axis=1)
        )

        if completely_missing_mask.any():
            missing_ids = (
                final_df.loc[
                    completely_missing_mask,
                    "ROW_ID",
                ]
                .head(20)
                .tolist()
            )

            raise RuntimeError(
                "Some BASE rows have no summary metrics "
                f"for threshold {label}: {missing_ids}"
            )

        output_path = (
            FINAL_OUTPUT_DIR
            / (
                "PRE_DATA_IBT_1982_2024_"
                f"MSWEP_THRESHOLD_{label}.csv"
            )
        )

        print(
            f"Writing threshold {label}: "
            f"{output_path}"
        )

        write_csv_atomic(
            final_df,
            output_path,
            date_format="%Y-%m-%d %H:%M:%S",
        )

        output_paths.append(
            output_path
        )

        del final_df

    return output_paths


# ============================================================
# 15. Save the overall processing log
# ============================================================

def save_overall_log(
    processing_results,
):
    """
    Save an overall year-level processing log.
    """
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_df = pd.DataFrame(
        processing_results
    )

    log_df = log_df.sort_values(
        "YEAR"
    ).reset_index(
        drop=True
    )

    log_path = (
        LOG_DIR
        / "year_processing_log.csv"
    )

    write_csv_atomic(
        log_df,
        log_path,
    )

    return log_df, log_path


# ============================================================
# 16. Main program
# ============================================================

def main():
    YEAR_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FINAL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_df = read_base_table()

    year_tasks = build_year_tasks(
        base_df
    )

    print("\n" + "=" * 80)
    print("Starting threshold summary calculation")
    print("=" * 80)

    print(
        "Thresholds (mm/3h): "
        + ", ".join(
            threshold_label(value)
            for value in THRESHOLDS
        )
    )

    print(
        f"Worker processes: {MAX_WORKERS}"
    )

    print(
        f"Grid input directory: {MSWEP_GRID_DIR}"
    )

    print(
        f"Annual cache directory: {YEAR_CACHE_DIR}"
    )

    processing_results = []

    with Pool(
        processes=MAX_WORKERS,
        maxtasksperchild=2,
    ) as pool:

        result_iterator = (
            pool.imap_unordered(
                process_year_task,
                year_tasks,
                chunksize=POOL_CHUNKSIZE,
            )
        )

        for result in tqdm(
            result_iterator,
            total=len(year_tasks),
            desc="Processing years",
            unit="year",
        ):
            processing_results.append(
                result
            )

    (
        processing_log_df,
        processing_log_path,
    ) = save_overall_log(
        processing_results
    )

    failed_years = (
        processing_log_df.loc[
            processing_log_df[
                "STATUS"
            ] == "FAILED",
            "YEAR",
        ]
        .astype(int)
        .tolist()
    )

    total_errors = int(
        processing_log_df[
            "ERROR_COUNT"
        ].sum()
    )

    print("\n" + "=" * 80)
    print("Year processing summary")
    print("=" * 80)

    print(
        f"Years processed: "
        f"{len(processing_log_df)}"
    )

    print(
        f"Expected grid files: "
        f"{int(processing_log_df['EXPECTED_FILES'].sum())}"
    )

    print(
        f"Found grid files: "
        f"{int(processing_log_df['FOUND_FILES'].sum())}"
    )

    print(
        f"Successfully summarised files: "
        f"{int(processing_log_df['PROCESSED_FILES'].sum())}"
    )

    print(
        f"Reused annual caches: "
        f"{int(processing_log_df['CACHE_REUSED'].sum())}"
    )

    print(
        f"Errors: {total_errors}"
    )

    print(
        f"Year processing log: "
        f"{processing_log_path}"
    )

    if failed_years:
        print(
            "Failed years: "
            + ", ".join(
                str(year)
                for year in failed_years
            )
        )

    if total_errors > 0 and FAIL_ON_FILE_ERROR:
        raise RuntimeError(
            "Grid-file errors were detected. "
            "Final threshold tables were not created. "
            f"Review the logs in {LOG_DIR}."
        )

    print("\n" + "=" * 80)
    print("Combining annual summary caches")
    print("=" * 80)

    summary_df = combine_year_caches(
        year_tasks,
        expected_total_rows=len(
            base_df
        ),
    )

    print(
        f"Combined summary rows: "
        f"{len(summary_df)}"
    )

    print(
        f"Combined summary columns: "
        f"{len(summary_df.columns)}"
    )

    print("\n" + "=" * 80)
    print("Creating threshold-specific BASE tables")
    print("=" * 80)

    output_paths = (
        create_final_threshold_tables(
            base_df,
            summary_df,
        )
    )

    print("\n" + "=" * 80)
    print("Processing completed successfully")
    print("=" * 80)

    print(
        f"Final output files: "
        f"{len(output_paths)}"
    )

    for output_path in output_paths:
        print(
            f"  {output_path}"
        )


if __name__ == "__main__":
    main()
