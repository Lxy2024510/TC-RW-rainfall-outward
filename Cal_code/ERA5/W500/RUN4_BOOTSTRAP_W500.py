import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. Configuration
# ============================================================

BOOTSTRAP_COUNT = 5000
BOOTSTRAP_BATCH_SIZE = 32
RANDOM_SEED = 20260812
CI_LOWER_PERCENTILE = 2.5
CI_UPPER_PERCENTILE = 97.5
SMALL_SAMPLE_THRESHOLD = 30

START_YEAR = 1982
END_YEAR = 2024
TIME_INTERVAL_HOURS = 24
COAST_DISTANCE_THRESHOLD_KM = 500.0

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(
    os.environ.get("TC_RW_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))
).expanduser().resolve()
W500_ROOT = PROJECT_ROOT / "Data" / "Processed" / "ERA5" / "W500"
INPUT_CSV = (
    W500_ROOT / "Windows" / "24H"
    / "PRE_DATA_IBT_1982_2024_MSWEP30_W500_24H_SLIDING_CLEANED.csv"
)

OUTPUT_ROOT = W500_ROOT / "Bootstrap"
OUTPUT_CSV = OUTPUT_ROOT / "W500_REVERSED_1982_2024_MSWEP30_BOOTSTRAP_LONG.csv"
LOG_DIR = (
    PROJECT_ROOT / "Results" / "Quality_control" / "ERA5" / "W500" / "RUN4"
)
PROCESSING_SUMMARY_CSV = LOG_DIR / "bootstrap_processing_summary.csv"
GROUP_COUNT_CSV = LOG_DIR / "bootstrap_group_counts.csv"


# ============================================================
# 2. Distance zones and fixed intensity groups
# ============================================================

DISTANCE_ZONES = [
    "0_100",
    "100_200",
    "200_300",
    "300_400",
    "400_500",
    "0_200",
    "200_500",
    "0_500",
]

DISTANCE_ZONE_LABELS = {
    "0_100": "0-100 km",
    "100_200": "100-200 km",
    "200_300": "200-300 km",
    "300_400": "300-400 km",
    "400_500": "400-500 km",
    "0_200": "0-200 km",
    "200_500": "200-500 km",
    "0_500": "0-500 km",
}

METRIC_COLUMNS = [
    f"DIFF_W500_REVERSED_{zone}" for zone in DISTANCE_ZONES
]

SPATIAL_TYPES = ["OPEN_OCEAN", "NEARSHORE"]

# These are fixed reference thresholds. They are not recalculated from
# the 1982-2024 MSWEP sample.
INTENSITY_GROUPS = [
    {
        "order": 1,
        "code": "LE_5TH",
        "label": "<=5th",
        "range": "<= -30 kt",
        "mask": lambda values: values <= -30.0,
    },
    {
        "order": 2,
        "code": "P05_P25",
        "label": "(5th,25th]",
        "range": "(-30, -8] kt",
        "mask": lambda values: (values > -30.0) & (values <= -8.0),
    },
    {
        "order": 3,
        "code": "P25_P50",
        "label": "(25th,50th)",
        "range": "(-8, 0) kt",
        "mask": lambda values: (values > -8.0) & (values < 0.0),
    },
    {
        "order": 4,
        "code": "EQ_50TH",
        "label": "=50th",
        "range": "0 kt",
        "mask": lambda values: values == 0.0,
    },
    {
        "order": 5,
        "code": "P50_P75",
        "label": "(50th,75th)",
        "range": "(0, 12) kt",
        "mask": lambda values: (values > 0.0) & (values < 12.0),
    },
    {
        "order": 6,
        "code": "P75_P95",
        "label": "[75th,95th)",
        "range": "[12, 30) kt",
        "mask": lambda values: (values >= 12.0) & (values < 30.0),
    },
    {
        "order": 7,
        "code": "GE_95TH",
        "label": ">=95th",
        "range": ">= 30 kt",
        "mask": lambda values: values >= 30.0,
    },
]


# ============================================================
# 3. General helpers
# ============================================================

def write_csv_atomic(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        output_path.name + f".tmp.{os.getpid()}"
    )
    try:
        df.to_csv(temporary_path, index=False)
        if not temporary_path.exists() or temporary_path.stat().st_size == 0:
            raise RuntimeError(f"Temporary output missing or empty: {temporary_path}")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def validate_required_columns(df, required_columns):
    missing = set(required_columns).difference(df.columns)
    if missing:
        raise KeyError(
            "Input table is missing required columns: "
            + ", ".join(sorted(missing))
        )


def coerce_boolean(series, column_name):
    """Strictly parse common CSV boolean representations."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    parsed = normalized.map(mapping)
    invalid = parsed.isna()
    if invalid.any():
        examples = series.loc[invalid].head(20).tolist()
        raise ValueError(
            f"Invalid boolean values in {column_name}: "
            f"count={int(invalid.sum())}; examples={examples}"
        )
    return parsed.astype(bool)


# ============================================================
# 4. Read and validate input
# ============================================================

def read_and_validate_input():
    print("=" * 100)
    print("Reading and validating the MSWEP30 + reversed-W500 24-hour table")
    print("=" * 100)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {INPUT_CSV}")

    required_columns = [
        "SID",
        "ROW_ID_S",
        "ROW_ID_E",
        "ISO_TIME_S",
        "ISO_TIME_E",
        "DIFF_USA_WIND",
        "CAL_DIST_REAL_S",
        "CAL_DIST_REAL_E",
        "MSWEP_DIST_30_S",
        "MSWEP_DIST_30_E",
        "RAIN_POINT_30_S",
        "RAIN_POINT_30_E",
        "MSWEP_RAIN30_S_FLAG",
        "MSWEP_RAIN30_E_FLAG",
        "MSWEP_ENDPOINTS_VALID_FLAG",
        "COUNT_ET_24H",
        "COUNT_ER_24H",
        "OBS_COUNT_24H",
        "COMPLETE_24H_FLAG",
    ] + METRIC_COLUMNS

    df = pd.read_csv(INPUT_CSV, dtype={"SID": "string"}, low_memory=False)
    validate_required_columns(df, required_columns)
    input_rows = len(df)

    print(f"Input file: {INPUT_CSV}")
    print(f"Input windows: {input_rows}")
    print(f"Input columns: {len(df.columns)}")

    if df["SID"].isna().any() or df["SID"].str.strip().eq("").any():
        raise ValueError("Input contains missing or blank SID values")
    df["SID"] = df["SID"].str.strip()

    df["ISO_TIME_S"] = pd.to_datetime(df["ISO_TIME_S"], errors="coerce")
    df["ISO_TIME_E"] = pd.to_datetime(df["ISO_TIME_E"], errors="coerce")
    invalid_time = df["ISO_TIME_S"].isna() | df["ISO_TIME_E"].isna()
    if invalid_time.any():
        raise ValueError(f"Invalid pair times: {int(invalid_time.sum())}")

    interval_hours = (
        (df["ISO_TIME_E"] - df["ISO_TIME_S"]).dt.total_seconds() / 3600.0
    )
    invalid_interval = ~np.isclose(
        interval_hours,
        float(TIME_INTERVAL_HOURS),
        rtol=0.0,
        atol=1e-9,
    )
    if invalid_interval.any():
        raise ValueError(f"Non-24-hour windows: {int(invalid_interval.sum())}")

    year_mask = (
        df["ISO_TIME_S"].dt.year.between(START_YEAR, END_YEAR, inclusive="both")
        & df["ISO_TIME_E"].dt.year.between(
            START_YEAR, END_YEAR, inclusive="both"
        )
    )
    rows_removed_by_year = int((~year_mask).sum())
    df = df.loc[year_mask].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f"No windows have both endpoints in {START_YEAR}-{END_YEAR}"
        )

    for column in ["ROW_ID_S", "ROW_ID_E"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    invalid_id = df["ROW_ID_S"].isna() | df["ROW_ID_E"].isna()
    if invalid_id.any():
        raise ValueError(f"Invalid ROW_ID values: {int(invalid_id.sum())}")
    df["ROW_ID_S"] = df["ROW_ID_S"].astype(np.int64)
    df["ROW_ID_E"] = df["ROW_ID_E"].astype(np.int64)

    duplicate_pair = df.duplicated(
        ["SID", "ROW_ID_S", "ROW_ID_E"], keep=False
    )
    if duplicate_pair.any():
        examples = df.loc[
            duplicate_pair,
            ["SID", "ROW_ID_S", "ROW_ID_E", "ISO_TIME_S", "ISO_TIME_E"],
        ].head(20).to_dict("records")
        raise ValueError(
            "Duplicate pair keys: "
            f"count={int(duplicate_pair.sum())}; examples={examples}"
        )

    numeric_columns = [
        "DIFF_USA_WIND",
        "CAL_DIST_REAL_S",
        "CAL_DIST_REAL_E",
        "MSWEP_DIST_30_S",
        "MSWEP_DIST_30_E",
        "RAIN_POINT_30_S",
        "RAIN_POINT_30_E",
        "COUNT_ET_24H",
        "COUNT_ER_24H",
        "OBS_COUNT_24H",
    ] + METRIC_COLUMNS

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

    if df[numeric_columns].isna().any().any():
        invalid_by_column = df[numeric_columns].isna().sum()
        invalid_by_column = invalid_by_column[invalid_by_column > 0]
        raise ValueError(
            "Input contains missing or non-finite required numeric values: "
            + "; ".join(
                f"{column}={int(count)}"
                for column, count in invalid_by_column.items()
            )
        )

    for column in [
        "MSWEP_RAIN30_S_FLAG",
        "MSWEP_RAIN30_E_FLAG",
        "MSWEP_ENDPOINTS_VALID_FLAG",
        "COMPLETE_24H_FLAG",
    ]:
        df[column] = coerce_boolean(df[column], column)

    invalid_mswep_endpoint = (
        df["MSWEP_DIST_30_S"].lt(0)
        | df["MSWEP_DIST_30_E"].lt(0)
        | df["RAIN_POINT_30_S"].le(0)
        | df["RAIN_POINT_30_E"].le(0)
        | ~df["MSWEP_RAIN30_S_FLAG"]
        | ~df["MSWEP_RAIN30_E_FLAG"]
        | ~df["MSWEP_ENDPOINTS_VALID_FLAG"]
    )
    if invalid_mswep_endpoint.any():
        raise ValueError(
            "Scheme-A input contains invalid MSWEP endpoint windows: "
            f"{int(invalid_mswep_endpoint.sum())}"
        )

    for column in [
        "RAIN_POINT_30_S",
        "RAIN_POINT_30_E",
        "COUNT_ET_24H",
        "COUNT_ER_24H",
        "OBS_COUNT_24H",
    ]:
        invalid_integer = (
            df[column].lt(0)
            | ~np.isclose(df[column], np.round(df[column]), rtol=0.0, atol=1e-9)
        )
        if invalid_integer.any():
            raise ValueError(
                f"Invalid nonnegative integer values in {column}: "
                f"{int(invalid_integer.sum())}"
            )
        df[column] = np.round(df[column]).astype(np.int64)

    invalid_obs_count = (df["OBS_COUNT_24H"] < 2) | (df["OBS_COUNT_24H"] > 9)
    if invalid_obs_count.any():
        raise ValueError(
            "OBS_COUNT_24H must be between 2 and 9: "
            f"{int(invalid_obs_count.sum())}"
        )

    completeness_mismatch = (
        df["COMPLETE_24H_FLAG"] != df["OBS_COUNT_24H"].eq(9)
    )
    if completeness_mismatch.any():
        raise ValueError(
            "COMPLETE_24H_FLAG disagrees with OBS_COUNT_24H: "
            f"{int(completeness_mismatch.sum())}"
        )

    if df["COUNT_ET_24H"].gt(df["OBS_COUNT_24H"]).any():
        raise ValueError("COUNT_ET_24H exceeds OBS_COUNT_24H")
    if df["COUNT_ER_24H"].gt(df["OBS_COUNT_24H"]).any():
        raise ValueError("COUNT_ER_24H exceeds OBS_COUNT_24H")

    complete_count = int(df["COMPLETE_24H_FLAG"].sum())
    incomplete_count = int((~df["COMPLETE_24H_FLAG"]).sum())
    et_window_count = int(df["COUNT_ET_24H"].gt(0).sum())
    er_window_count = int(df["COUNT_ER_24H"].gt(0).sum())

    validation_stats = {
        "INPUT_COMPLETE_WINDOWS": complete_count,
        "INPUT_INCOMPLETE_WINDOWS": incomplete_count,
        "INPUT_ET_WINDOWS": et_window_count,
        "INPUT_ER_WINDOWS": er_window_count,
        "INPUT_UNIQUE_SID_COUNT": int(df["SID"].nunique()),
    }

    print(
        f"Analysis period: {START_YEAR}-{END_YEAR}; "
        f"retained={len(df)}, removed={rows_removed_by_year}"
    )
    print(f"Complete windows: {complete_count}")
    print(f"Incomplete windows retained: {incomplete_count}")
    print(f"Windows containing ET: {et_window_count}")
    print(f"Windows containing ER: {er_window_count}")
    print(f"Unique SIDs: {validation_stats['INPUT_UNIQUE_SID_COUNT']}")
    print("[PASS] Input validation completed")
    return df, input_rows, rows_removed_by_year, validation_stats


# ============================================================
# 5. Spatial classification and ET filtering
# ============================================================

def create_spatial_subsets(df):
    open_ocean = (
        df["CAL_DIST_REAL_S"].gt(COAST_DISTANCE_THRESHOLD_KM)
        & df["CAL_DIST_REAL_E"].gt(COAST_DISTANCE_THRESHOLD_KM)
    )
    nearshore = (
        df["CAL_DIST_REAL_S"].le(COAST_DISTANCE_THRESHOLD_KM)
        & df["CAL_DIST_REAL_E"].le(COAST_DISTANCE_THRESHOLD_KM)
    )
    mixed = ~open_ocean & ~nearshore

    subsets = {
        "OPEN_OCEAN": df.loc[open_ocean].copy(),
        "NEARSHORE": df.loc[nearshore].copy(),
    }
    mixed_df = df.loc[mixed].copy()

    if sum(len(part) for part in subsets.values()) + len(mixed_df) != len(df):
        raise RuntimeError("Spatial categories do not fully cover input")

    mixed_et_count = int(mixed_df["COUNT_ET_24H"].gt(0).sum())
    mixed_non_et_count = len(mixed_df) - mixed_et_count

    print("Spatial classification:")
    print(f"  OPEN_OCEAN before ET filtering: {len(subsets['OPEN_OCEAN'])}")
    print(f"  NEARSHORE before ET filtering: {len(subsets['NEARSHORE'])}")
    print(f"  MIXED excluded: {len(mixed_df)}")
    print(f"    MIXED containing ET: {mixed_et_count}")
    print(f"    MIXED non-ET: {mixed_non_et_count}")

    mixed_stats = {
        "MIXED_EXCLUDED_WINDOWS": len(mixed_df),
        "MIXED_ET_WINDOWS": mixed_et_count,
        "MIXED_NON_ET_WINDOWS": mixed_non_et_count,
    }
    return subsets, mixed_stats


def apply_non_et_filter(spatial_subsets):
    filtered = {}
    summary = {}
    for spatial_type in SPATIAL_TYPES:
        subset = spatial_subsets[spatial_type]
        filtered_df = subset.loc[subset["COUNT_ET_24H"].eq(0)].copy()
        removed = len(subset) - len(filtered_df)
        filtered[spatial_type] = filtered_df
        summary[spatial_type] = {
            "before_et": len(subset),
            "after_et": len(filtered_df),
            "et_removed": removed,
        }
        print(
            f"{spatial_type}: before ET={len(subset)}, "
            f"after ET={len(filtered_df)}, ET removed={removed}"
        )
    return filtered, summary


def validate_intensity_groups(subset_df, spatial_type):
    wind = subset_df["DIFF_USA_WIND"].to_numpy(dtype=np.float64, copy=False)
    membership = np.zeros(len(subset_df), dtype=np.int8)
    counts = {}

    for group in INTENSITY_GROUPS:
        mask = group["mask"](wind)
        membership += mask.astype(np.int8)
        counts[group["code"]] = int(mask.sum())

    invalid = membership != 1
    if invalid.any():
        examples = subset_df.loc[
            invalid, ["SID", "ROW_ID_S", "ROW_ID_E", "DIFF_USA_WIND"]
        ].head(20).to_dict("records")
        raise ValueError(
            f"{spatial_type}: invalid intensity membership for "
            f"{int(invalid.sum())} windows; examples={examples}"
        )
    if sum(counts.values()) != len(subset_df):
        raise RuntimeError("Intensity group counts do not sum to subset size")

    print(f"[PASS] {spatial_type} fixed-threshold intensity groups")
    return counts


# ============================================================
# 6. Window-level bootstrap
# ============================================================

def calculate_bootstrap_means(values, bootstrap_count, random_seed):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Bootstrap input must be two-dimensional")

    sample_size, variable_count = values.shape
    if sample_size == 0:
        return np.empty((0, variable_count), dtype=np.float64)

    rng = np.random.default_rng(random_seed)
    means = np.empty((bootstrap_count, variable_count), dtype=np.float64)
    completed = 0

    while completed < bootstrap_count:
        batch_size = min(BOOTSTRAP_BATCH_SIZE, bootstrap_count - completed)
        indices = rng.integers(
            low=0,
            high=sample_size,
            size=(batch_size, sample_size),
            endpoint=False,
        )
        # The same resampled window indices are used for all eight zones.
        means[completed:completed + batch_size] = values[indices].mean(
            axis=1,
            dtype=np.float64,
        )
        completed += batch_size

    return means


def build_group_seed(spatial_type, group_order):
    spatial_offset = {"OPEN_OCEAN": 0, "NEARSHORE": 100000}
    return RANDOM_SEED + spatial_offset[spatial_type] + int(group_order)


def summarise_spatial_type(subset_df, spatial_type, input_window_count):
    wind = subset_df["DIFF_USA_WIND"].to_numpy(dtype=np.float64, copy=False)
    output_records = []
    group_count_records = []

    print(f"\nCalculating {spatial_type} W500 window-level Bootstrap statistics")

    for group in INTENSITY_GROUPS:
        group_mask = group["mask"](wind)
        group_df = subset_df.loc[group_mask].copy()
        group_count = len(group_df)

        # Upstream processing requires every W500 metric to be valid. Any
        # invalid value here is therefore treated as an error, not discarded.
        invalid_metric = ~group_df[METRIC_COLUMNS].notna().all(axis=1)
        if invalid_metric.any():
            examples = group_df.loc[
                invalid_metric,
                ["SID", "ROW_ID_S", "ROW_ID_E"] + METRIC_COLUMNS,
            ].head(10).to_dict("records")
            raise ValueError(
                f"{spatial_type}/{group['code']} contains invalid W500 metrics: "
                f"count={int(invalid_metric.sum())}; examples={examples}"
            )

        valid_count = group_count
        unique_sid_count = int(group_df["SID"].nunique())
        windows_per_sid_mean = (
            float(group_count / unique_sid_count) if unique_sid_count else np.nan
        )
        small_sample_flag = valid_count < SMALL_SAMPLE_THRESHOLD
        group_seed = build_group_seed(spatial_type, group["order"])

        print(
            f"  {group['label']}: windows={group_count}, "
            f"unique SID={unique_sid_count}, small-sample={small_sample_flag}"
        )

        if valid_count > 0:
            values = group_df[METRIC_COLUMNS].to_numpy(
                dtype=np.float64,
                copy=True,
            )
            original_means = values.mean(axis=0, dtype=np.float64)
            bootstrap_means = calculate_bootstrap_means(
                values,
                BOOTSTRAP_COUNT,
                group_seed,
            )
            ci_lower = np.percentile(
                bootstrap_means, CI_LOWER_PERCENTILE, axis=0
            )
            ci_upper = np.percentile(
                bootstrap_means, CI_UPPER_PERCENTILE, axis=0
            )
            standard_error = bootstrap_means.std(axis=0, ddof=1)
        else:
            original_means = np.full(len(METRIC_COLUMNS), np.nan)
            ci_lower = np.full(len(METRIC_COLUMNS), np.nan)
            ci_upper = np.full(len(METRIC_COLUMNS), np.nan)
            standard_error = np.full(len(METRIC_COLUMNS), np.nan)

        group_count_records.append(
            {
                "PRECIPITATION_DATASET": "MSWEP",
                "RAINFALL_THRESHOLD": 30,
                "PAIR_SELECTION": "BOTH_ENDPOINTS_HAVE_RAIN30",
                "SPATIAL_TYPE": spatial_type,
                "INTENSITY_BOUNDARY_TYPE": "FIXED_REFERENCE_THRESHOLDS",
                "INTENSITY_GROUP_ORDER": group["order"],
                "INTENSITY_GROUP_CODE": group["code"],
                "INTENSITY_GROUP_LABEL": group["label"],
                "WIND_CHANGE_RANGE": group["range"],
                "GROUP_WINDOW_COUNT": group_count,
                "COMMON_VALID_WINDOW_COUNT": valid_count,
                "INVALID_WINDOW_COUNT": 0,
                "UNIQUE_SID_COUNT": unique_sid_count,
                "WINDOWS_PER_SID_MEAN": windows_per_sid_mean,
                "SMALL_SAMPLE_THRESHOLD": SMALL_SAMPLE_THRESHOLD,
                "SMALL_SAMPLE_FLAG": small_sample_flag,
                "RANDOM_SEED": group_seed,
            }
        )

        for zone_order, (zone, source_column) in enumerate(
            zip(DISTANCE_ZONES, METRIC_COLUMNS), start=1
        ):
            index = zone_order - 1
            output_records.append(
                {
                    "START_YEAR": START_YEAR,
                    "END_YEAR": END_YEAR,
                    "PRECIPITATION_DATASET": "MSWEP",
                    "RAINFALL_THRESHOLD": 30,
                    "PAIR_SELECTION": "BOTH_ENDPOINTS_HAVE_RAIN30",
                    "INPUT_WINDOW_COUNT": input_window_count,
                    "SPATIAL_TYPE": spatial_type,
                    "COAST_DISTANCE_RULE": (
                        ">500 km at S and E"
                        if spatial_type == "OPEN_OCEAN"
                        else "<=500 km at S and E"
                    ),
                    "ET_RULE": "COUNT_ET_24H == 0",
                    "INCOMPLETE_WINDOW_RULE": "RETAINED",
                    "METRIC_TYPE": "W500_REVERSED_CHANGE",
                    "METRIC_DESCRIPTION": (
                        "24-hour E-minus-S change in area-weighted W500 after "
                        "multiplication by -1; positive means upward motion increased"
                    ),
                    "UNIT": "same as source W500",
                    "INTENSITY_BOUNDARY_TYPE": "FIXED_REFERENCE_THRESHOLDS",
                    "INTENSITY_GROUP_ORDER": group["order"],
                    "INTENSITY_GROUP_CODE": group["code"],
                    "INTENSITY_GROUP_LABEL": group["label"],
                    "WIND_CHANGE_RANGE": group["range"],
                    "DISTANCE_ZONE_ORDER": zone_order,
                    "DISTANCE_ZONE": zone,
                    "DISTANCE_ZONE_LABEL": DISTANCE_ZONE_LABELS[zone],
                    "SOURCE_COLUMN": source_column,
                    "GROUP_WINDOW_COUNT": group_count,
                    "COMMON_VALID_WINDOW_COUNT": valid_count,
                    "INVALID_WINDOW_COUNT": 0,
                    "UNIQUE_SID_COUNT": unique_sid_count,
                    "WINDOWS_PER_SID_MEAN": windows_per_sid_mean,
                    "SMALL_SAMPLE_THRESHOLD": SMALL_SAMPLE_THRESHOLD,
                    "SMALL_SAMPLE_FLAG": small_sample_flag,
                    "MEAN_CHANGE": float(original_means[index]),
                    "BOOTSTRAP_CI_LOWER": float(ci_lower[index]),
                    "BOOTSTRAP_CI_UPPER": float(ci_upper[index]),
                    "BOOTSTRAP_STANDARD_ERROR": float(standard_error[index]),
                    "BOOTSTRAP_COUNT": BOOTSTRAP_COUNT,
                    "BOOTSTRAP_METHOD": "WINDOW_LEVEL_PERCENTILE",
                    "BOOTSTRAP_UNIT": "24-hour window",
                    "RANDOM_SEED": group_seed,
                }
            )

    result_df = pd.DataFrame(output_records)
    group_count_df = pd.DataFrame(group_count_records)

    expected_rows = len(INTENSITY_GROUPS) * len(DISTANCE_ZONES)
    if len(result_df) != expected_rows:
        raise RuntimeError(
            f"{spatial_type}: output rows={len(result_df)}, "
            f"expected={expected_rows}"
        )

    result_df = result_df.sort_values(
        ["INTENSITY_GROUP_ORDER", "DISTANCE_ZONE_ORDER"]
    ).reset_index(drop=True)
    return result_df, group_count_df


# ============================================================
# 7. Main
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Starting MSWEP reversed-W500 window-level Bootstrap processing")
    print("=" * 100)
    print(f"Input: {INPUT_CSV}")
    print(f"Analysis period: {START_YEAR}-{END_YEAR}")
    print(f"Bootstrap samples per group: {BOOTSTRAP_COUNT}")
    print("Bootstrap method: window-level percentile Bootstrap")
    print("Spatial types: OPEN_OCEAN and NEARSHORE; MIXED is excluded")
    print("ET rule: windows with COUNT_ET_24H > 0 are excluded")
    print("Incomplete 24-hour windows are retained")
    print("Intensity groups use fixed reference thresholds")
    print("Metric: eight DIFF_W500_REVERSED distance-zone variables")

    (
        df,
        input_rows,
        rows_removed_by_year,
        validation_stats,
    ) = read_and_validate_input()

    spatial_subsets, mixed_stats = create_spatial_subsets(df)
    filtered_subsets, et_summary = apply_non_et_filter(spatial_subsets)

    result_frames = []
    group_count_frames = []
    intensity_count_text = {}

    for spatial_type in SPATIAL_TYPES:
        subset_df = filtered_subsets[spatial_type]
        group_counts = validate_intensity_groups(subset_df, spatial_type)
        intensity_count_text[spatial_type] = ";".join(
            f"{code}={count}" for code, count in group_counts.items()
        )

        result_df, group_count_df = summarise_spatial_type(
            subset_df,
            spatial_type,
            input_window_count=len(df),
        )
        result_frames.append(result_df)
        group_count_frames.append(group_count_df)

    final_result = pd.concat(result_frames, ignore_index=True)
    final_result = final_result.sort_values(
        ["SPATIAL_TYPE", "INTENSITY_GROUP_ORDER", "DISTANCE_ZONE_ORDER"]
    ).reset_index(drop=True)

    expected_total_rows = (
        len(SPATIAL_TYPES) * len(INTENSITY_GROUPS) * len(DISTANCE_ZONES)
    )
    if len(final_result) != expected_total_rows:
        raise RuntimeError(
            f"Final output rows={len(final_result)}, "
            f"expected={expected_total_rows}"
        )

    group_count_output = pd.concat(group_count_frames, ignore_index=True)
    group_count_output = group_count_output.sort_values(
        ["SPATIAL_TYPE", "INTENSITY_GROUP_ORDER"]
    ).reset_index(drop=True)

    analyzed_windows = sum(len(df_) for df_ in filtered_subsets.values())
    total_et_excluded_after_spatial = sum(
        et_summary[spatial_type]["et_removed"] for spatial_type in SPATIAL_TYPES
    )

    accounting_total = (
        analyzed_windows
        + total_et_excluded_after_spatial
        + mixed_stats["MIXED_EXCLUDED_WINDOWS"]
    )
    if accounting_total != len(df):
        raise RuntimeError(
            "Sample accounting failed: "
            f"accounted={accounting_total}, input={len(df)}"
        )

    summary_record = {
        "INPUT_PATH": str(INPUT_CSV),
        "PRECIPITATION_DATASET": "MSWEP",
        "RAINFALL_THRESHOLD": 30,
        "PAIR_SELECTION": "BOTH_ENDPOINTS_HAVE_RAIN30",
        "START_YEAR": START_YEAR,
        "END_YEAR": END_YEAR,
        "INPUT_WINDOWS_BEFORE_YEAR_FILTER": input_rows,
        "YEAR_FILTER_REMOVED_WINDOWS": rows_removed_by_year,
        "INPUT_WINDOWS_AFTER_YEAR_FILTER": len(df),
        **validation_stats,
        "OPEN_OCEAN_WINDOWS_BEFORE_ET": len(spatial_subsets["OPEN_OCEAN"]),
        "NEARSHORE_WINDOWS_BEFORE_ET": len(spatial_subsets["NEARSHORE"]),
        **mixed_stats,
        "OPEN_OCEAN_NON_ET_WINDOWS": len(filtered_subsets["OPEN_OCEAN"]),
        "NEARSHORE_NON_ET_WINDOWS": len(filtered_subsets["NEARSHORE"]),
        "OPEN_OCEAN_ET_REMOVED": et_summary["OPEN_OCEAN"]["et_removed"],
        "NEARSHORE_ET_REMOVED": et_summary["NEARSHORE"]["et_removed"],
        "TOTAL_ET_REMOVED_AFTER_SPATIAL": total_et_excluded_after_spatial,
        "ANALYZED_WINDOWS": analyzed_windows,
        "OPEN_OCEAN_INTENSITY_COUNTS": intensity_count_text["OPEN_OCEAN"],
        "NEARSHORE_INTENSITY_COUNTS": intensity_count_text["NEARSHORE"],
        "INTENSITY_BOUNDARY_TYPE": "FIXED_REFERENCE_THRESHOLDS",
        "BOOTSTRAP_COUNT": BOOTSTRAP_COUNT,
        "BOOTSTRAP_METHOD": "WINDOW_LEVEL_PERCENTILE",
        "BOOTSTRAP_UNIT": "24-hour window",
        "INCOMPLETE_WINDOWS_RETAINED": True,
        "OUTPUT_ROWS": len(final_result),
        "OUTPUT_PATH": str(OUTPUT_CSV),
    }
    summary_df = pd.DataFrame([summary_record])

    write_csv_atomic(final_result, OUTPUT_CSV)
    write_csv_atomic(group_count_output, GROUP_COUNT_CSV)
    write_csv_atomic(summary_df, PROCESSING_SUMMARY_CSV)

    print("\n" + "=" * 100)
    print("Processing completed successfully")
    print("=" * 100)
    print(f"Input windows after year filter: {len(df)}")
    print(f"MIXED windows excluded: {mixed_stats['MIXED_EXCLUDED_WINDOWS']}")
    print(f"ET windows removed after spatial classification: {total_et_excluded_after_spatial}")
    print(f"OPEN_OCEAN non-ET windows: {len(filtered_subsets['OPEN_OCEAN'])}")
    print(f"NEARSHORE non-ET windows: {len(filtered_subsets['NEARSHORE'])}")
    print(f"Total analyzed windows: {analyzed_windows}")
    print(f"Bootstrap result rows: {len(final_result)}")
    print(f"Bootstrap result: {OUTPUT_CSV}")
    print(f"Group counts: {GROUP_COUNT_CSV}")
    print(f"Processing summary: {PROCESSING_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
