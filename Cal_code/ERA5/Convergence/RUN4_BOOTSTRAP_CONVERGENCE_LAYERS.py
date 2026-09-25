#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Bootstrap 24-hour multi-layer convergence changes.

The analysis uses exact 24-hour Scheme-A endpoint pairs produced by RUN3.
Windows are divided into open-ocean and nearshore subsets using both endpoint
distances to land. Mixed coastal-transition windows and windows containing ET
records are excluded. Incomplete 24-hour windows and low-valid-area windows
are retained.

Seven fixed wind-change groups, three vertical products, and seven radial
zones are summarized with a window-level percentile bootstrap. Positive
metric values mean that area-weighted convergence strengthened from the
start endpoint to the end endpoint.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# 1. Configuration and paths
# ============================================================================

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
CONVERGENCE_ROOT = (
    PROJECT_ROOT / "Data" / "Processed" / "ERA5" / "Convergence"
)
INPUT_CSV = (
    CONVERGENCE_ROOT / "Windows" / "24H"
    / "PRE_DATA_IBT_1982_2024_MSWEP30_CONVERGENCE_LAYERS_"
      "24H_SLIDING_CLEANED.csv"
)

OUTPUT_ROOT = CONVERGENCE_ROOT / "Bootstrap"
OUTPUT_CSV = OUTPUT_ROOT / (
    "CONVERGENCE_LAYERS_1982_2024_MSWEP30_BOOTSTRAP_LONG.csv"
)
LOG_DIR = (
    PROJECT_ROOT / "Results" / "Quality_control" / "ERA5"
    / "Convergence" / "RUN4"
)
PROCESSING_SUMMARY_CSV = LOG_DIR / "bootstrap_processing_summary.csv"
GROUP_COUNT_CSV = LOG_DIR / "bootstrap_group_counts.csv"
SMALL_SAMPLE_CSV = LOG_DIR / "small_sample_groups.csv"


# ============================================================================
# 2. Radial zones and fixed wind-change groups
# ============================================================================

DISTANCE_ZONES = [
    "R000_100",
    "R100_200",
    "R200_300",
    "R300_400",
    "R400_500",
    "R000_200",
    "R200_500",
]

DISTANCE_ZONE_LABELS = {
    "R000_100": "0-100 km",
    "R100_200": "100-200 km",
    "R200_300": "200-300 km",
    "R300_400": "300-400 km",
    "R400_500": "400-500 km",
    "R000_200": "0-200 km",
    "R200_500": "200-500 km",
}

VERTICAL_PRODUCTS = ["500_PB", "700_PB", "500_700"]

VERTICAL_PRODUCT_LABELS = {
    "500_PB": "500 hPa to discrete PB",
    "700_PB": "700 hPa to discrete PB",
    "500_700": "500-700 hPa",
}

METRIC_SPECS = [
    (product, zone, f"DIFF_CONV_AW_{product}_{zone}")
    for product in VERTICAL_PRODUCTS
    for zone in DISTANCE_ZONES
]
METRIC_COLUMNS = [source_column for _, _, source_column in METRIC_SPECS]
SPATIAL_TYPES = ["OPEN_OCEAN", "NEARSHORE"]

# These fixed reference thresholds are intentionally not recalculated from
# the current FUHE sample.
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


# ============================================================================
# 3. General helpers
# ============================================================================

def write_csv_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        output_path.name + f".tmp.{os.getpid()}"
    )
    try:
        frame.to_csv(temporary_path, index=False)
        if not temporary_path.is_file() or temporary_path.stat().st_size <= 0:
            raise RuntimeError(f"Temporary output is missing or empty: {temporary_path}")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def require_columns(frame: pd.DataFrame, required_columns: list[str]) -> None:
    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        raise KeyError(f"Input table is missing required columns: {missing}")


def coerce_boolean(series: pd.Series, column_name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = series.astype("string").str.strip().str.lower()
    parsed = normalized.map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    invalid = parsed.isna()
    if invalid.any():
        examples = series.loc[invalid].head(20).tolist()
        raise ValueError(
            f"Invalid boolean values in {column_name}: "
            f"count={int(invalid.sum())}; examples={examples}"
        )
    return parsed.astype(bool)


def validate_nonnegative_integer(frame: pd.DataFrame, column: str) -> None:
    invalid = frame[column].lt(0.0) | ~np.isclose(
        frame[column], np.round(frame[column]), rtol=0.0, atol=1.0e-9
    )
    if invalid.any():
        raise ValueError(
            f"Invalid nonnegative integer values in {column}: "
            f"{int(invalid.sum())}"
        )
    frame[column] = np.round(frame[column]).astype(np.int64)


# ============================================================================
# 4. Read and validate the RUN3 pair table
# ============================================================================

def read_and_validate_input() -> tuple[
    pd.DataFrame, int, int, dict[str, object]
]:
    print("=" * 104)
    print("Reading and validating the MSWEP30 multi-layer convergence table")
    print("=" * 104)

    if not INPUT_CSV.is_file():
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
        "FUHE_ALL_ZONES_VALID_FLAG",
    ] + METRIC_COLUMNS

    frame = pd.read_csv(INPUT_CSV, dtype={"SID": "string"}, low_memory=False)
    require_columns(frame, required_columns)
    input_rows = len(frame)

    print(f"Input file: {INPUT_CSV}")
    print(f"Input windows: {input_rows}")
    print(f"Input columns: {len(frame.columns)}")

    missing_sid = frame["SID"].isna() | frame["SID"].str.strip().eq("")
    if missing_sid.any():
        raise ValueError(f"Missing or blank SID values: {int(missing_sid.sum())}")
    frame["SID"] = frame["SID"].str.strip()

    frame["ISO_TIME_S"] = pd.to_datetime(frame["ISO_TIME_S"], errors="coerce")
    frame["ISO_TIME_E"] = pd.to_datetime(frame["ISO_TIME_E"], errors="coerce")
    invalid_time = frame["ISO_TIME_S"].isna() | frame["ISO_TIME_E"].isna()
    if invalid_time.any():
        raise ValueError(f"Invalid pair timestamps: {int(invalid_time.sum())}")

    interval_hours = (
        (frame["ISO_TIME_E"] - frame["ISO_TIME_S"])
        .dt.total_seconds()
        .div(3600.0)
    )
    invalid_interval = ~np.isclose(
        interval_hours,
        float(TIME_INTERVAL_HOURS),
        rtol=0.0,
        atol=1.0e-9,
    )
    if invalid_interval.any():
        raise ValueError(f"Non-24-hour windows: {int(invalid_interval.sum())}")

    year_mask = (
        frame["ISO_TIME_S"].dt.year.between(
            START_YEAR, END_YEAR, inclusive="both"
        )
        & frame["ISO_TIME_E"].dt.year.between(
            START_YEAR, END_YEAR, inclusive="both"
        )
    )
    rows_removed_by_year = int((~year_mask).sum())
    frame = frame.loc[year_mask].copy().reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(
            f"No windows have both endpoints in {START_YEAR}-{END_YEAR}"
        )

    for column in ["ROW_ID_S", "ROW_ID_E"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        invalid = (
            frame[column].isna()
            | frame[column].lt(0.0)
            | frame[column].mod(1.0).ne(0.0)
        )
        if invalid.any():
            raise ValueError(f"Invalid {column} values: {int(invalid.sum())}")
        frame[column] = frame[column].astype(np.int64)

    duplicate_pair = frame.duplicated(
        ["SID", "ROW_ID_S", "ROW_ID_E"], keep=False
    )
    if duplicate_pair.any():
        examples = frame.loc[
            duplicate_pair,
            ["SID", "ROW_ID_S", "ROW_ID_E", "ISO_TIME_S", "ISO_TIME_E"],
        ].head(20).to_dict("records")
        raise ValueError(
            f"Duplicate pair keys: count={int(duplicate_pair.sum())}; "
            f"examples={examples}"
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
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[numeric_columns] = frame[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    if frame[numeric_columns].isna().any().any():
        invalid_counts = frame[numeric_columns].isna().sum()
        invalid_counts = invalid_counts[invalid_counts.gt(0)]
        details = "; ".join(
            f"{column}={int(count)}" for column, count in invalid_counts.items()
        )
        raise ValueError(f"Missing or non-finite required numeric values: {details}")

    boolean_columns = [
        "MSWEP_RAIN30_S_FLAG",
        "MSWEP_RAIN30_E_FLAG",
        "MSWEP_ENDPOINTS_VALID_FLAG",
        "COMPLETE_24H_FLAG",
        "FUHE_ALL_ZONES_VALID_FLAG",
    ]
    for column in boolean_columns:
        frame[column] = coerce_boolean(frame[column], column)

    invalid_endpoints = (
        frame["MSWEP_DIST_30_S"].lt(0.0)
        | frame["MSWEP_DIST_30_E"].lt(0.0)
        | frame["RAIN_POINT_30_S"].le(0.0)
        | frame["RAIN_POINT_30_E"].le(0.0)
        | ~frame["MSWEP_RAIN30_S_FLAG"]
        | ~frame["MSWEP_RAIN30_E_FLAG"]
        | ~frame["MSWEP_ENDPOINTS_VALID_FLAG"]
    )
    if invalid_endpoints.any():
        raise ValueError(
            "Scheme-A input contains invalid MSWEP endpoints: "
            f"{int(invalid_endpoints.sum())}"
        )

    for column in [
        "RAIN_POINT_30_S",
        "RAIN_POINT_30_E",
        "COUNT_ET_24H",
        "COUNT_ER_24H",
        "OBS_COUNT_24H",
    ]:
        validate_nonnegative_integer(frame, column)

    invalid_obs = frame["OBS_COUNT_24H"].lt(2) | frame["OBS_COUNT_24H"].gt(9)
    if invalid_obs.any():
        raise ValueError(
            f"OBS_COUNT_24H must be between 2 and 9: {int(invalid_obs.sum())}"
        )
    completeness_mismatch = frame["COMPLETE_24H_FLAG"].ne(
        frame["OBS_COUNT_24H"].eq(9)
    )
    if completeness_mismatch.any():
        raise ValueError(
            "COMPLETE_24H_FLAG disagrees with OBS_COUNT_24H: "
            f"{int(completeness_mismatch.sum())}"
        )
    if frame["COUNT_ET_24H"].gt(frame["OBS_COUNT_24H"]).any():
        raise ValueError("COUNT_ET_24H exceeds OBS_COUNT_24H")
    if frame["COUNT_ER_24H"].gt(frame["OBS_COUNT_24H"]).any():
        raise ValueError("COUNT_ER_24H exceeds OBS_COUNT_24H")

    validation_stats = {
        "INPUT_COMPLETE_WINDOWS": int(frame["COMPLETE_24H_FLAG"].sum()),
        "INPUT_INCOMPLETE_WINDOWS": int((~frame["COMPLETE_24H_FLAG"]).sum()),
        "INPUT_ET_WINDOWS": int(frame["COUNT_ET_24H"].gt(0).sum()),
        "INPUT_ER_WINDOWS": int(frame["COUNT_ER_24H"].gt(0).sum()),
        "INPUT_LOW_FUHE_AREA_WINDOWS": int(
            (~frame["FUHE_ALL_ZONES_VALID_FLAG"]).sum()
        ),
        "INPUT_UNIQUE_SID_COUNT": int(frame["SID"].nunique()),
    }

    print(
        f"Analysis period: {START_YEAR}-{END_YEAR}; "
        f"retained={len(frame)}, removed={rows_removed_by_year}"
    )
    print(f"Complete windows: {validation_stats['INPUT_COMPLETE_WINDOWS']}")
    print(
        "Incomplete windows retained: "
        f"{validation_stats['INPUT_INCOMPLETE_WINDOWS']}"
    )
    print(f"Windows containing ET: {validation_stats['INPUT_ET_WINDOWS']}")
    print(f"Windows containing ER: {validation_stats['INPUT_ER_WINDOWS']}")
    print(
        "Low-valid-area windows retained: "
        f"{validation_stats['INPUT_LOW_FUHE_AREA_WINDOWS']}"
    )
    print(f"Unique SIDs: {validation_stats['INPUT_UNIQUE_SID_COUNT']}")
    print("[PASS] Input validation completed")
    return frame, input_rows, rows_removed_by_year, validation_stats


# ============================================================================
# 5. Spatial classification and ET filtering
# ============================================================================

def create_spatial_subsets(
    frame: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, int]]:
    open_ocean_mask = (
        frame["CAL_DIST_REAL_S"].gt(COAST_DISTANCE_THRESHOLD_KM)
        & frame["CAL_DIST_REAL_E"].gt(COAST_DISTANCE_THRESHOLD_KM)
    )
    nearshore_mask = (
        frame["CAL_DIST_REAL_S"].le(COAST_DISTANCE_THRESHOLD_KM)
        & frame["CAL_DIST_REAL_E"].le(COAST_DISTANCE_THRESHOLD_KM)
    )
    mixed_mask = ~open_ocean_mask & ~nearshore_mask

    subsets = {
        "OPEN_OCEAN": frame.loc[open_ocean_mask].copy(),
        "NEARSHORE": frame.loc[nearshore_mask].copy(),
    }
    mixed = frame.loc[mixed_mask].copy()

    accounted = sum(len(subset) for subset in subsets.values()) + len(mixed)
    if accounted != len(frame):
        raise RuntimeError(
            f"Spatial classification accounting failed: {accounted} != {len(frame)}"
        )

    mixed_stats = {
        "MIXED_EXCLUDED_WINDOWS": len(mixed),
        "MIXED_ET_WINDOWS": int(mixed["COUNT_ET_24H"].gt(0).sum()),
        "MIXED_NON_ET_WINDOWS": int(mixed["COUNT_ET_24H"].eq(0).sum()),
        "MIXED_LOW_FUHE_AREA_WINDOWS": int(
            (~mixed["FUHE_ALL_ZONES_VALID_FLAG"]).sum()
        ),
    }

    print("Spatial classification:")
    print(f"  OPEN_OCEAN before ET filtering: {len(subsets['OPEN_OCEAN'])}")
    print(f"  NEARSHORE before ET filtering: {len(subsets['NEARSHORE'])}")
    print(f"  MIXED excluded: {len(mixed)}")
    print(f"    MIXED containing ET: {mixed_stats['MIXED_ET_WINDOWS']}")
    print(f"    MIXED non-ET: {mixed_stats['MIXED_NON_ET_WINDOWS']}")
    return subsets, mixed, mixed_stats


def apply_non_et_filter(
    spatial_subsets: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, int]]]:
    filtered: dict[str, pd.DataFrame] = {}
    summary: dict[str, dict[str, int]] = {}

    for spatial_type in SPATIAL_TYPES:
        subset = spatial_subsets[spatial_type]
        kept = subset.loc[subset["COUNT_ET_24H"].eq(0)].copy()
        removed = len(subset) - len(kept)
        filtered[spatial_type] = kept
        summary[spatial_type] = {
            "before_et": len(subset),
            "after_et": len(kept),
            "et_removed": removed,
            "low_area_retained": int(
                (~kept["FUHE_ALL_ZONES_VALID_FLAG"]).sum()
            ),
            "incomplete_retained": int((~kept["COMPLETE_24H_FLAG"]).sum()),
        }
        print(
            f"{spatial_type}: before ET={len(subset)}, after ET={len(kept)}, "
            f"ET removed={removed}, low-area retained="
            f"{summary[spatial_type]['low_area_retained']}"
        )
    return filtered, summary


def validate_intensity_groups(
    subset: pd.DataFrame,
    spatial_type: str,
) -> dict[str, int]:
    wind_change = subset["DIFF_USA_WIND"].to_numpy(
        dtype=np.float64, copy=False
    )
    membership_count = np.zeros(len(subset), dtype=np.int8)
    counts: dict[str, int] = {}

    for group in INTENSITY_GROUPS:
        mask = group["mask"](wind_change)
        membership_count += mask.astype(np.int8)
        counts[group["code"]] = int(mask.sum())

    invalid = membership_count != 1
    if invalid.any():
        examples = subset.loc[
            invalid, ["SID", "ROW_ID_S", "ROW_ID_E", "DIFF_USA_WIND"]
        ].head(20).to_dict("records")
        raise ValueError(
            f"{spatial_type} has {int(invalid.sum())} windows with invalid "
            f"intensity-group membership: {examples}"
        )
    if sum(counts.values()) != len(subset):
        raise RuntimeError(
            f"{spatial_type} intensity-group counts do not sum to subset size"
        )
    print(f"[PASS] {spatial_type} fixed wind-change groups")
    return counts


# ============================================================================
# 6. Window-level bootstrap
# ============================================================================

def calculate_bootstrap_means(
    values: np.ndarray,
    bootstrap_count: int,
    random_seed: int,
) -> np.ndarray:
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
        # Identical resampled window indices are used for all seven zones.
        means[completed : completed + batch_size] = values[indices].mean(
            axis=1, dtype=np.float64
        )
        completed += batch_size

    return means


def build_group_seed(spatial_type: str, group_order: int) -> int:
    spatial_offset = {"OPEN_OCEAN": 0, "NEARSHORE": 100000}
    return RANDOM_SEED + spatial_offset[spatial_type] + int(group_order)


def summarize_spatial_type(
    subset: pd.DataFrame,
    spatial_type: str,
    input_window_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wind_change = subset["DIFF_USA_WIND"].to_numpy(
        dtype=np.float64, copy=False
    )
    output_records: list[dict[str, object]] = []
    group_count_records: list[dict[str, object]] = []

    print(f"\nCalculating {spatial_type} FUHE window-level bootstrap statistics")

    for group in INTENSITY_GROUPS:
        group_mask = group["mask"](wind_change)
        group_frame = subset.loc[group_mask].copy()
        group_count = len(group_frame)

        invalid_metric = ~group_frame[METRIC_COLUMNS].notna().all(axis=1)
        if invalid_metric.any():
            examples = group_frame.loc[
                invalid_metric,
                ["SID", "ROW_ID_S", "ROW_ID_E"] + METRIC_COLUMNS,
            ].head(10).to_dict("records")
            raise ValueError(
                f"{spatial_type}/{group['code']} has invalid convergence metrics: "
                f"count={int(invalid_metric.sum())}; examples={examples}"
            )

        unique_sid_count = int(group_frame["SID"].nunique())
        windows_per_sid_mean = (
            float(group_count / unique_sid_count) if unique_sid_count else np.nan
        )
        small_sample_flag = group_count < SMALL_SAMPLE_THRESHOLD
        group_seed = build_group_seed(spatial_type, group["order"])
        incomplete_count = int((~group_frame["COMPLETE_24H_FLAG"]).sum())
        low_area_count = int(
            (~group_frame["FUHE_ALL_ZONES_VALID_FLAG"]).sum()
        )

        print(
            f"  {group['label']}: windows={group_count}, "
            f"unique SID={unique_sid_count}, incomplete={incomplete_count}, "
            f"low-area={low_area_count}, small-sample={small_sample_flag}"
        )

        if group_count > 0:
            values = group_frame[METRIC_COLUMNS].to_numpy(
                dtype=np.float64, copy=True
            )
            original_means = values.mean(axis=0, dtype=np.float64)
            bootstrap_means = calculate_bootstrap_means(
                values, BOOTSTRAP_COUNT, group_seed
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
                "COMMON_VALID_WINDOW_COUNT": group_count,
                "INVALID_WINDOW_COUNT": 0,
                "INCOMPLETE_WINDOW_COUNT_RETAINED": incomplete_count,
                "LOW_FUHE_AREA_WINDOW_COUNT_RETAINED": low_area_count,
                "UNIQUE_SID_COUNT": unique_sid_count,
                "WINDOWS_PER_SID_MEAN": windows_per_sid_mean,
                "SMALL_SAMPLE_THRESHOLD": SMALL_SAMPLE_THRESHOLD,
                "SMALL_SAMPLE_FLAG": small_sample_flag,
                "RANDOM_SEED": group_seed,
            }
        )

        for index, (product, zone, source_column) in enumerate(METRIC_SPECS):
            zone_order = DISTANCE_ZONES.index(zone) + 1
            product_order = VERTICAL_PRODUCTS.index(product) + 1
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
                    "LOW_FUHE_AREA_RULE": "RETAINED",
                    "VERTICAL_PRODUCT_ORDER": product_order,
                    "VERTICAL_PRODUCT": product,
                    "VERTICAL_PRODUCT_LABEL": VERTICAL_PRODUCT_LABELS[product],
                    "METRIC_TYPE": f"{product}_CONVERGENCE_CHANGE",
                    "METRIC_DESCRIPTION": (
                        "24-hour E-minus-S change in area-weighted mean "
                        f"{VERTICAL_PRODUCT_LABELS[product]} convergence; "
                        "positive means "
                        "convergence strengthened"
                    ),
                    "UNIT": "s^-1",
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
                    "COMMON_VALID_WINDOW_COUNT": group_count,
                    "INVALID_WINDOW_COUNT": 0,
                    "INCOMPLETE_WINDOW_COUNT_RETAINED": incomplete_count,
                    "LOW_FUHE_AREA_WINDOW_COUNT_RETAINED": low_area_count,
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

    result = pd.DataFrame(output_records)
    group_counts = pd.DataFrame(group_count_records)
    expected_rows = (
        len(INTENSITY_GROUPS)
        * len(VERTICAL_PRODUCTS)
        * len(DISTANCE_ZONES)
    )
    if len(result) != expected_rows:
        raise RuntimeError(
            f"{spatial_type} output has {len(result)} rows; expected {expected_rows}"
        )
    result = result.sort_values(
        [
            "INTENSITY_GROUP_ORDER",
            "VERTICAL_PRODUCT_ORDER",
            "DISTANCE_ZONE_ORDER",
        ]
    ).reset_index(drop=True)
    return result, group_counts


# ============================================================================
# 7. Main program
# ============================================================================

def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 104)
    print("MSWEP30 multi-layer convergence-change window-level bootstrap")
    print("=" * 104)
    print(f"Input: {INPUT_CSV}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Analysis period: {START_YEAR}-{END_YEAR}")
    print(f"Bootstrap samples per group: {BOOTSTRAP_COUNT}")
    print("Bootstrap method: window-level percentile bootstrap")
    print("Spatial subsets: OPEN_OCEAN and NEARSHORE; MIXED is excluded")
    print("ET rule: COUNT_ET_24H > 0 is excluded")
    print("Incomplete windows: retained")
    print("Low-valid-area windows: retained")
    print("Wind-change groups: seven fixed reference thresholds")
    print("Vertical products: " + ", ".join(VERTICAL_PRODUCTS))
    print("Metric: 21 DIFF_CONV_AW product-by-radial-zone variables")
    print("=" * 104)

    frame, input_rows, year_removed, validation_stats = read_and_validate_input()
    spatial_subsets, _, mixed_stats = create_spatial_subsets(frame)
    filtered_subsets, et_summary = apply_non_et_filter(spatial_subsets)

    result_frames: list[pd.DataFrame] = []
    group_count_frames: list[pd.DataFrame] = []
    intensity_count_text: dict[str, str] = {}

    for spatial_type in SPATIAL_TYPES:
        subset = filtered_subsets[spatial_type]
        group_counts = validate_intensity_groups(subset, spatial_type)
        intensity_count_text[spatial_type] = ";".join(
            f"{code}={count}" for code, count in group_counts.items()
        )
        result, count_table = summarize_spatial_type(
            subset, spatial_type, input_window_count=len(frame)
        )
        result_frames.append(result)
        group_count_frames.append(count_table)

    final_result = pd.concat(result_frames, ignore_index=True).sort_values(
        [
            "SPATIAL_TYPE",
            "INTENSITY_GROUP_ORDER",
            "VERTICAL_PRODUCT_ORDER",
            "DISTANCE_ZONE_ORDER",
        ]
    ).reset_index(drop=True)
    expected_total_rows = (
        len(SPATIAL_TYPES) * len(INTENSITY_GROUPS) * len(DISTANCE_ZONES)
        * len(VERTICAL_PRODUCTS)
    )
    if len(final_result) != expected_total_rows:
        raise RuntimeError(
            f"Final output has {len(final_result)} rows; "
            f"expected {expected_total_rows}"
        )

    group_count_output = pd.concat(
        group_count_frames, ignore_index=True
    ).sort_values(
        ["SPATIAL_TYPE", "INTENSITY_GROUP_ORDER"]
    ).reset_index(drop=True)

    analyzed_windows = sum(len(subset) for subset in filtered_subsets.values())
    total_et_excluded_after_spatial = sum(
        et_summary[spatial_type]["et_removed"] for spatial_type in SPATIAL_TYPES
    )
    accounting_total = (
        analyzed_windows
        + total_et_excluded_after_spatial
        + mixed_stats["MIXED_EXCLUDED_WINDOWS"]
    )
    if accounting_total != len(frame):
        raise RuntimeError(
            f"Sample accounting failed: accounted={accounting_total}, "
            f"input={len(frame)}"
        )

    analyzed_low_area = sum(
        et_summary[spatial_type]["low_area_retained"]
        for spatial_type in SPATIAL_TYPES
    )
    analyzed_incomplete = sum(
        et_summary[spatial_type]["incomplete_retained"]
        for spatial_type in SPATIAL_TYPES
    )

    summary_record = {
        "INPUT_PATH": str(INPUT_CSV),
        "PRECIPITATION_DATASET": "MSWEP",
        "RAINFALL_THRESHOLD": 30,
        "PAIR_SELECTION": "BOTH_ENDPOINTS_HAVE_RAIN30",
        "START_YEAR": START_YEAR,
        "END_YEAR": END_YEAR,
        "INPUT_WINDOWS_BEFORE_YEAR_FILTER": input_rows,
        "YEAR_FILTER_REMOVED_WINDOWS": year_removed,
        "INPUT_WINDOWS_AFTER_YEAR_FILTER": len(frame),
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
        "ANALYZED_INCOMPLETE_WINDOWS_RETAINED": analyzed_incomplete,
        "ANALYZED_LOW_FUHE_AREA_WINDOWS_RETAINED": analyzed_low_area,
        "OPEN_OCEAN_INTENSITY_COUNTS": intensity_count_text["OPEN_OCEAN"],
        "NEARSHORE_INTENSITY_COUNTS": intensity_count_text["NEARSHORE"],
        "INTENSITY_BOUNDARY_TYPE": "FIXED_REFERENCE_THRESHOLDS",
        "BOOTSTRAP_COUNT": BOOTSTRAP_COUNT,
        "BOOTSTRAP_METHOD": "WINDOW_LEVEL_PERCENTILE",
        "BOOTSTRAP_UNIT": "24-hour window",
        "INCOMPLETE_WINDOWS_RETAINED": True,
        "LOW_FUHE_AREA_WINDOWS_RETAINED": True,
        "OUTPUT_ROWS": len(final_result),
        "OUTPUT_PATH": str(OUTPUT_CSV),
    }

    small_samples = group_count_output.loc[
        group_count_output["SMALL_SAMPLE_FLAG"]
    ].copy()

    write_csv_atomic(final_result, OUTPUT_CSV)
    write_csv_atomic(group_count_output, GROUP_COUNT_CSV)
    write_csv_atomic(pd.DataFrame([summary_record]), PROCESSING_SUMMARY_CSV)
    write_csv_atomic(small_samples, SMALL_SAMPLE_CSV)

    print()
    print("=" * 104)
    print("Processing completed successfully")
    print("=" * 104)
    print(f"Input windows after year filter: {len(frame)}")
    print(f"MIXED windows excluded: {mixed_stats['MIXED_EXCLUDED_WINDOWS']}")
    print(f"ET windows removed after spatial classification: {total_et_excluded_after_spatial}")
    print(f"OPEN_OCEAN non-ET windows: {len(filtered_subsets['OPEN_OCEAN'])}")
    print(f"NEARSHORE non-ET windows: {len(filtered_subsets['NEARSHORE'])}")
    print(f"Incomplete analyzed windows retained: {analyzed_incomplete}")
    print(f"Low-area analyzed windows retained: {analyzed_low_area}")
    print(f"Total analyzed windows: {analyzed_windows}")
    print(f"Small-sample groups: {len(small_samples)}")
    print(f"Bootstrap result rows: {len(final_result)}")
    print(f"Bootstrap result: {OUTPUT_CSV}")
    print(f"Group counts: {GROUP_COUNT_CSV}")
    print(f"Small-sample groups: {SMALL_SAMPLE_CSV}")
    print(f"Processing summary: {PROCESSING_SUMMARY_CSV}")
    print("=" * 104)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        raise
