#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build exact 24-hour MSWEP30 + multi-layer convergence endpoint pairs.

The FUHE table supplies the start and end endpoint metrics. The uncleaned
MSWEP threshold-30 table supplies the complete 3-hour storm timeline used to
count observations, threshold-rainfall occurrences, ET records, and ER
records inside each inclusive 24-hour window.

Pairing follows Scheme A:

* The end time must equal the start time plus exactly 24 hours for the same
  SID.
* Both endpoints must have threshold-30 rainfall and a finite MSWEP_DIST_30.
* Missing intermediate observations do not remove a pair.
* Intermediate non-rainfall records do not remove a pair.
* ET and ER records do not remove a pair.
* Low FUHE valid-area coverage does not remove a pair; it is flagged only.

The three vertical products are 500-PB, 700-PB, and 500-700 hPa. Positive
CONV_AW and positive DIFF_CONV_AW represent convergence and strengthening
convergence, respectively.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# 1. Configuration and paths
# ============================================================================

START_YEAR = 1982
END_YEAR = 2024
TIME_INTERVAL_HOURS = 24
EXPECTED_OBSERVATIONS_24H = 9
VALID_3H_HOURS = {0, 3, 6, 9, 12, 15, 18, 21}
FUHE_VALID_AREA_THRESHOLD = 0.90

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(
    os.environ.get("TC_RW_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))
).expanduser().resolve()
CONVERGENCE_ROOT = (
    PROJECT_ROOT / "Data" / "Processed" / "ERA5" / "Convergence"
)
FUHE_INPUT = (
    CONVERGENCE_ROOT / "Endpoints"
    / "PRE_DATA_IBT_1982_2024_MSWEP_THRESHOLD_30_VALID_DIST30_"
      "FUHE_CONVERGENCE_LAYERS.csv"
)
TIMELINE_INPUT = (
    PROJECT_ROOT / "Data" / "Processed" / "MSWEP" / "Thresholds"
    / "PRE_DATA_IBT_1982_2024_MSWEP_THRESHOLD_30.csv"
)

OUTPUT_DIR = CONVERGENCE_ROOT / "Windows" / "24H"
OUTPUT_CSV = OUTPUT_DIR / (
    "PRE_DATA_IBT_1982_2024_MSWEP30_CONVERGENCE_LAYERS_"
    "24H_SLIDING_CLEANED.csv"
)
LOG_DIR = (
    PROJECT_ROOT / "Results" / "Quality_control" / "ERA5"
    / "Convergence" / "RUN3"
)
SUMMARY_CSV = LOG_DIR / "processing_summary.csv"
INCOMPLETE_PAIRS_CSV = LOG_DIR / "incomplete_24h_pairs.csv"
LOW_AREA_PAIRS_CSV = LOG_DIR / "low_fuhe_area_pairs.csv"


# ============================================================================
# 2. Columns
# ============================================================================

ZONES = [
    "R000_100",
    "R100_200",
    "R200_300",
    "R300_400",
    "R400_500",
    "R000_200",
    "R200_500",
]

PRODUCTS = ["500_PB", "700_PB", "500_700"]

COMMON_RADIAL_METRICS = [
    "TOTAL_GRID_COUNT",
    "TOTAL_AREA_KM2",
    "PS_HPA_AW",
    "PB_HPA_AW",
    "PS_MINUS_PB_AW",
]

PRODUCT_RADIAL_METRICS = [
    "VALID_GRID_COUNT",
    "VALID_AREA_KM2",
    "VALID_AREA_FRACTION",
    "DIV_AW",
    "CONV_AW",
    "CONVERGENCE_AREA_KM2",
    "CONVERGENCE_AREA_FRACTION",
    "VALID_DP_AW",
]

COMMON_RADIAL_COLUMNS = [
    f"{metric}_{zone}"
    for zone in ZONES
    for metric in COMMON_RADIAL_METRICS
]

PRODUCT_RADIAL_COLUMNS = [
    f"{metric}_{product}_{zone}"
    for zone in ZONES
    for product in PRODUCTS
    for metric in PRODUCT_RADIAL_METRICS
]

RADIAL_COLUMNS = COMMON_RADIAL_COLUMNS + PRODUCT_RADIAL_COLUMNS

COMMON_TARGET_COLUMNS = [
    "ROW_ID",
    "USA_LAT",
    "USA_LON",
    "USA_WIND",
    "USA_SSHS",
    "ISO_TIME",
    "RAIN_TIME_ID",
    "CAL_DIST_REAL",
    "DIST2LAND",
    "NATURE",
    "BASIN",
    "MSWEP_DIST_30",
    "RAIN_POINT_30",
]

OPTIONAL_ID_COLUMNS = ["NAME"]
TARGET_COLUMNS = COMMON_TARGET_COLUMNS + RADIAL_COLUMNS

BASE_DIFFERENCE_COLUMNS = [
    "USA_WIND",
    "CAL_DIST_REAL",
    "DIST2LAND",
    "MSWEP_DIST_30",
]

COMMON_RADIAL_DIFFERENCE_METRICS = [
    "PS_HPA_AW",
    "PB_HPA_AW",
    "PS_MINUS_PB_AW",
]

PRODUCT_RADIAL_DIFFERENCE_METRICS = [
    "DIV_AW",
    "CONV_AW",
    "CONVERGENCE_AREA_FRACTION",
    "VALID_DP_AW",
    "VALID_AREA_FRACTION",
]

COMMON_RADIAL_DIFFERENCE_COLUMNS = [
    f"{metric}_{zone}"
    for zone in ZONES
    for metric in COMMON_RADIAL_DIFFERENCE_METRICS
]

PRODUCT_RADIAL_DIFFERENCE_COLUMNS = [
    f"{metric}_{product}_{zone}"
    for zone in ZONES
    for product in PRODUCTS
    for metric in PRODUCT_RADIAL_DIFFERENCE_METRICS
]

RADIAL_DIFFERENCE_COLUMNS = (
    COMMON_RADIAL_DIFFERENCE_COLUMNS + PRODUCT_RADIAL_DIFFERENCE_COLUMNS
)

DIFFERENCE_SOURCE_COLUMNS = BASE_DIFFERENCE_COLUMNS + RADIAL_DIFFERENCE_COLUMNS

TIMELINE_READ_COLUMNS = [
    "ROW_ID",
    "SID",
    "ISO_TIME",
    "RAIN_TIME_ID",
    "NATURE",
    "RAIN_POINT_30",
    "MSWEP_DIST_30",
]

RAIN_TIME_PATTERN = re.compile(r"^\d{7}\.\d{2}$")


# ============================================================================
# 3. Generic validation and writing helpers
# ============================================================================

def write_csv_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        output_path.name + f".tmp.{os.getpid()}"
    )
    try:
        frame.to_csv(
            temporary_path,
            index=False,
            date_format="%Y-%m-%d %H:%M:%S",
        )
        if not temporary_path.is_file() or temporary_path.stat().st_size <= 0:
            raise RuntimeError(f"Temporary output is missing or empty: {temporary_path}")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def require_columns(
    frame: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{table_name} is missing required columns: {missing}")


def validate_row_id(frame: pd.DataFrame, table_name: str) -> None:
    numeric = pd.to_numeric(frame["ROW_ID"], errors="coerce")
    invalid = numeric.isna() | ~np.isfinite(numeric.to_numpy(dtype=np.float64))
    invalid |= numeric.lt(0.0) | numeric.mod(1.0).ne(0.0)
    if invalid.any():
        examples = frame.loc[invalid, "ROW_ID"].head(20).tolist()
        raise ValueError(
            f"{table_name} has {int(invalid.sum())} invalid ROW_ID values: {examples}"
        )
    frame["ROW_ID"] = numeric.astype(np.int64)
    if frame["ROW_ID"].duplicated().any():
        examples = frame.loc[
            frame["ROW_ID"].duplicated(keep=False), "ROW_ID"
        ].head(20).tolist()
        raise ValueError(f"{table_name} has duplicate ROW_ID values: {examples}")


def validate_time_fields(frame: pd.DataFrame, table_name: str) -> None:
    frame["ISO_TIME"] = pd.to_datetime(frame["ISO_TIME"], errors="coerce")
    if frame["ISO_TIME"].isna().any():
        raise ValueError(
            f"{table_name} has invalid ISO_TIME values: "
            f"{int(frame['ISO_TIME'].isna().sum())}"
        )

    strict_3hour = (
        frame["ISO_TIME"].dt.hour.isin(VALID_3H_HOURS)
        & frame["ISO_TIME"].dt.minute.eq(0)
        & frame["ISO_TIME"].dt.second.eq(0)
        & frame["ISO_TIME"].dt.microsecond.eq(0)
    )
    if not strict_3hour.all():
        examples = frame.loc[
            ~strict_3hour, ["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID"]
        ].head(20).to_dict("records")
        raise ValueError(
            f"{table_name} has non-exact 3-hour timestamps: "
            f"count={int((~strict_3hour).sum())}; examples={examples}"
        )

    rain_time_id = frame["RAIN_TIME_ID"].astype("string").str.strip()
    valid_format = rain_time_id.map(
        lambda value: bool(RAIN_TIME_PATTERN.fullmatch(str(value)))
        if pd.notna(value)
        else False
    )
    if not valid_format.all():
        examples = frame.loc[
            ~valid_format, ["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID"]
        ].head(20).to_dict("records")
        raise ValueError(
            f"{table_name} has invalid RAIN_TIME_ID values: "
            f"count={int((~valid_format).sum())}; examples={examples}"
        )

    expected = frame["ISO_TIME"].dt.strftime("%Y%j.%H")
    mismatch = expected.ne(rain_time_id)
    if mismatch.any():
        examples = (
            frame.loc[mismatch, ["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID"]]
            .assign(EXPECTED_RAIN_TIME_ID=expected[mismatch].values)
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            f"{table_name} has ISO_TIME/RAIN_TIME_ID mismatches: "
            f"count={int(mismatch.sum())}; examples={examples}"
        )
    frame["RAIN_TIME_ID"] = rain_time_id


def validate_unique_storm_time(frame: pd.DataFrame, table_name: str) -> None:
    duplicated = frame.duplicated(["SID", "ISO_TIME"], keep=False)
    if duplicated.any():
        examples = frame.loc[
            duplicated, ["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID"]
        ].head(20).to_dict("records")
        raise ValueError(
            f"{table_name} has duplicate SID/ISO_TIME rows: "
            f"count={int(duplicated.sum())}; examples={examples}"
        )


# ============================================================================
# 4. Read and validate the complete timeline
# ============================================================================

def read_timeline() -> tuple[pd.DataFrame, dict[str, object]]:
    if not TIMELINE_INPUT.is_file():
        raise FileNotFoundError(f"Timeline input does not exist: {TIMELINE_INPUT}")

    timeline = pd.read_csv(
        TIMELINE_INPUT,
        usecols=TIMELINE_READ_COLUMNS,
        dtype={"SID": "string", "RAIN_TIME_ID": "string"},
        low_memory=False,
    )
    require_columns(timeline, TIMELINE_READ_COLUMNS, "Timeline table")
    validate_row_id(timeline, "Timeline table")
    validate_time_fields(timeline, "Timeline table")

    in_years = timeline["ISO_TIME"].dt.year.between(
        START_YEAR, END_YEAR, inclusive="both"
    )
    timeline = timeline.loc[in_years].copy().reset_index(drop=True)
    if timeline.empty:
        raise RuntimeError(f"No timeline records remain in {START_YEAR}-{END_YEAR}")
    validate_unique_storm_time(timeline, "Filtered timeline table")

    timeline["RAIN_POINT_30"] = pd.to_numeric(
        timeline["RAIN_POINT_30"], errors="coerce"
    )
    timeline["MSWEP_DIST_30"] = pd.to_numeric(
        timeline["MSWEP_DIST_30"], errors="coerce"
    )
    if timeline["RAIN_POINT_30"].isna().any():
        raise ValueError("Timeline table contains invalid RAIN_POINT_30 values")
    if (timeline["RAIN_POINT_30"] < 0.0).any():
        raise ValueError("Timeline table contains negative RAIN_POINT_30 values")

    integer_points = np.isclose(
        timeline["RAIN_POINT_30"],
        np.round(timeline["RAIN_POINT_30"]),
        rtol=0.0,
        atol=1.0e-9,
    )
    if not integer_points.all():
        raise ValueError(
            "Timeline RAIN_POINT_30 contains non-integer values: "
            f"{int((~integer_points).sum())}"
        )
    timeline["RAIN_POINT_30"] = np.round(
        timeline["RAIN_POINT_30"]
    ).astype(np.int64)

    distance_available = timeline["MSWEP_DIST_30"].notna()
    zero_with_distance = timeline["RAIN_POINT_30"].eq(0) & distance_available
    positive_without_distance = timeline["RAIN_POINT_30"].gt(0) & ~distance_available
    if zero_with_distance.any():
        raise ValueError(
            "Timeline MSWEP_DIST_30 exists when RAIN_POINT_30 is zero: "
            f"{int(zero_with_distance.sum())}"
        )
    if positive_without_distance.any():
        raise ValueError(
            "Timeline MSWEP_DIST_30 is missing when RAIN_POINT_30 is positive: "
            f"{int(positive_without_distance.sum())}"
        )
    if timeline.loc[distance_available, "MSWEP_DIST_30"].lt(0.0).any():
        raise ValueError("Timeline table contains negative MSWEP_DIST_30 values")

    stats = {
        "TIMELINE_INPUT_ROWS": len(timeline),
        "TIMELINE_RAIN30_ROWS": int(timeline["RAIN_POINT_30"].gt(0).sum()),
        "TIMELINE_NO_RAIN30_ROWS": int(timeline["RAIN_POINT_30"].eq(0).sum()),
    }
    return timeline, stats


# ============================================================================
# 5. Read, validate, and identity-check the FUHE endpoint table
# ============================================================================

def read_fuhe_source(timeline: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    if not FUHE_INPUT.is_file():
        raise FileNotFoundError(f"FUHE input does not exist: {FUHE_INPUT}")

    fuhe = pd.read_csv(
        FUHE_INPUT,
        dtype={"SID": "string", "NAME": "string", "RAIN_TIME_ID": "string"},
        low_memory=False,
    )
    require_columns(
        fuhe,
        ["SID"] + TARGET_COLUMNS,
        "FUHE endpoint table",
    )
    validate_row_id(fuhe, "FUHE endpoint table")
    validate_time_fields(fuhe, "FUHE endpoint table")

    in_years = fuhe["ISO_TIME"].dt.year.between(
        START_YEAR, END_YEAR, inclusive="both"
    )
    fuhe = fuhe.loc[in_years].copy().reset_index(drop=True)
    if fuhe.empty:
        raise RuntimeError(f"No FUHE records remain in {START_YEAR}-{END_YEAR}")
    validate_unique_storm_time(fuhe, "Filtered FUHE endpoint table")

    numeric_columns = list(
        dict.fromkeys(
            [
                "USA_LAT",
                "USA_LON",
                "USA_WIND",
                "USA_SSHS",
                "CAL_DIST_REAL",
                "DIST2LAND",
                "MSWEP_DIST_30",
                "RAIN_POINT_30",
            ]
            + RADIAL_COLUMNS
        )
    )
    for column in numeric_columns:
        fuhe[column] = pd.to_numeric(fuhe[column], errors="coerce")
    fuhe[numeric_columns] = fuhe[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    required_finite = [
        "USA_LAT",
        "USA_LON",
        "USA_WIND",
        "USA_SSHS",
        "CAL_DIST_REAL",
        "DIST2LAND",
        "MSWEP_DIST_30",
        "RAIN_POINT_30",
    ]
    invalid_required = fuhe[required_finite].isna().any(axis=1)
    if invalid_required.any():
        examples = fuhe.loc[
            invalid_required,
            ["ROW_ID", "SID", "ISO_TIME"] + required_finite,
        ].head(10).to_dict("records")
        raise ValueError(
            "FUHE endpoint table contains missing required values: "
            f"count={int(invalid_required.sum())}; examples={examples}"
        )
    if (fuhe["RAIN_POINT_30"] <= 0.0).any():
        raise ValueError(
            "FUHE endpoint table contains rows without threshold-30 rainfall: "
            f"{int((fuhe['RAIN_POINT_30'] <= 0.0).sum())}"
        )
    if (fuhe["MSWEP_DIST_30"] < 0.0).any():
        raise ValueError("FUHE endpoint table contains negative MSWEP_DIST_30 values")

    for product in PRODUCTS:
        for zone in ZONES:
            fraction_column = f"VALID_AREA_FRACTION_{product}_{zone}"
            finite = fuhe[fraction_column].notna()
            out_of_range = finite & ~fuhe[fraction_column].between(0.0, 1.0)
            if out_of_range.any():
                raise ValueError(
                    f"{fraction_column} contains values outside [0, 1]"
                )

    identity = fuhe[["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID"]].merge(
        timeline[["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID"]],
        on="ROW_ID",
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("_FUHE", "_TIMELINE"),
    )
    unmatched = identity["_merge"].ne("both")
    disagreement = identity["_merge"].eq("both") & (
        identity["SID_FUHE"].astype("string").ne(
            identity["SID_TIMELINE"].astype("string")
        )
        | identity["ISO_TIME_FUHE"].ne(identity["ISO_TIME_TIMELINE"])
        | identity["RAIN_TIME_ID_FUHE"].astype("string").ne(
            identity["RAIN_TIME_ID_TIMELINE"].astype("string")
        )
    )
    if unmatched.any() or disagreement.any():
        bad = identity.loc[unmatched | disagreement].copy()
        path = LOG_DIR / "fuhe_timeline_identity_errors.csv"
        write_csv_atomic(bad, path)
        raise ValueError(
            "FUHE/timeline identity validation failed: "
            f"unmatched={int(unmatched.sum())}, disagreement={int(disagreement.sum())}; "
            f"review {path}"
        )

    stats = {
        "FUHE_INPUT_ROWS": len(fuhe),
        "FUHE_LOW_AREA_SOURCE_ROWS": int(
            np.logical_or.reduce(
                [
                    fuhe[f"VALID_AREA_FRACTION_{product}_{zone}"]
                    .lt(FUHE_VALID_AREA_THRESHOLD)
                    .fillna(True)
                    .to_numpy()
                    for product in PRODUCTS
                    for zone in ZONES
                ]
            ).sum()
        ),
    }
    return fuhe, stats


# ============================================================================
# 6. Prepare complete-timeline cumulative counters
# ============================================================================

def prepare_timeline_counters(timeline: pd.DataFrame) -> pd.DataFrame:
    working = timeline.sort_values(["SID", "ISO_TIME", "ROW_ID"]).reset_index(
        drop=True
    )
    normalized_nature = (
        working["NATURE"].fillna("").astype(str).str.strip().str.upper()
    )
    working["IS_ET"] = normalized_nature.eq("ET").astype(np.int8)
    working["IS_ER"] = normalized_nature.eq("ER").astype(np.int8)
    working["IS_MSWEP_RAIN30"] = working["RAIN_POINT_30"].gt(0).astype(np.int8)
    working["IS_OBSERVATION"] = np.int8(1)

    group = working.groupby("SID", sort=False)
    working["ET_CUM"] = group["IS_ET"].cumsum().astype(np.int32)
    working["ER_CUM"] = group["IS_ER"].cumsum().astype(np.int32)
    working["RAIN30_CUM"] = group["IS_MSWEP_RAIN30"].cumsum().astype(np.int32)
    working["OBS_CUM"] = group["IS_OBSERVATION"].cumsum().astype(np.int32)

    return working[
        [
            "ROW_ID",
            "SID",
            "ISO_TIME",
            "IS_ET",
            "IS_ER",
            "IS_MSWEP_RAIN30",
            "ET_CUM",
            "ER_CUM",
            "RAIN30_CUM",
            "OBS_CUM",
        ]
    ].copy()


# ============================================================================
# 7. Build exact 24-hour FUHE endpoint pairs
# ============================================================================

def build_exact_pairs(
    fuhe: pd.DataFrame,
    timeline_counters: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    source = fuhe.sort_values(["SID", "ISO_TIME", "ROW_ID"]).reset_index(drop=True)
    id_columns = [column for column in OPTIONAL_ID_COLUMNS if column in source.columns]

    start = source[["SID"] + id_columns + TARGET_COLUMNS].copy()
    start["TIME_TARGET_E"] = start["ISO_TIME"] + pd.Timedelta(
        hours=TIME_INTERVAL_HOURS
    )
    start = start.rename(columns={column: f"{column}_S" for column in TARGET_COLUMNS})

    end = source[["SID"] + TARGET_COLUMNS].copy()
    end = end.rename(columns={column: f"{column}_E" for column in TARGET_COLUMNS})

    pairs = start.merge(
        end,
        left_on=["SID", "TIME_TARGET_E"],
        right_on=["SID", "ISO_TIME_E"],
        how="inner",
        validate="one_to_one",
        sort=False,
    ).drop(columns=["TIME_TARGET_E"])

    if pairs.empty:
        raise RuntimeError("No exact 24-hour FUHE endpoint pairs were found")

    interval_hours = (
        (pairs["ISO_TIME_E"] - pairs["ISO_TIME_S"])
        .dt.total_seconds()
        .div(3600.0)
    )
    if not np.isclose(
        interval_hours,
        float(TIME_INTERVAL_HOURS),
        rtol=0.0,
        atol=1.0e-9,
    ).all():
        raise RuntimeError("A non-24-hour endpoint pair was generated")

    counter_columns = [
        "IS_ET",
        "IS_ER",
        "IS_MSWEP_RAIN30",
        "ET_CUM",
        "ER_CUM",
        "RAIN30_CUM",
        "OBS_CUM",
    ]
    start_counter = timeline_counters[["ROW_ID"] + counter_columns].rename(
        columns={
            "ROW_ID": "ROW_ID_S",
            **{column: f"{column}_S" for column in counter_columns},
        }
    )
    end_counter = timeline_counters[["ROW_ID"] + counter_columns].rename(
        columns={
            "ROW_ID": "ROW_ID_E",
            **{column: f"{column}_E" for column in counter_columns},
        }
    )
    pairs = pairs.merge(
        start_counter,
        on="ROW_ID_S",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    pairs = pairs.merge(
        end_counter,
        on="ROW_ID_E",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    required_counters = [
        f"{column}_{endpoint}"
        for column in counter_columns
        for endpoint in ("S", "E")
    ]
    if pairs[required_counters].isna().any().any():
        raise RuntimeError("A FUHE endpoint is missing complete-timeline counters")

    pairs["COUNT_ET_24H"] = (
        pairs["ET_CUM_E"] - pairs["ET_CUM_S"] + pairs["IS_ET_S"]
    ).astype(np.int16)
    pairs["COUNT_ER_24H"] = (
        pairs["ER_CUM_E"] - pairs["ER_CUM_S"] + pairs["IS_ER_S"]
    ).astype(np.int16)
    pairs["COUNT_MSWEP_RAIN30_24H"] = (
        pairs["RAIN30_CUM_E"]
        - pairs["RAIN30_CUM_S"]
        + pairs["IS_MSWEP_RAIN30_S"]
    ).astype(np.int16)
    pairs["OBS_COUNT_24H"] = (
        pairs["OBS_CUM_E"] - pairs["OBS_CUM_S"] + 1
    ).astype(np.int16)

    pairs["COMPLETE_24H_FLAG"] = pairs["OBS_COUNT_24H"].eq(
        EXPECTED_OBSERVATIONS_24H
    )
    pairs["ANY_MSWEP_RAIN30_24H_FLAG"] = pairs[
        "COUNT_MSWEP_RAIN30_24H"
    ].gt(0)
    pairs["ALL_MSWEP_RAIN30_24H_FLAG"] = pairs[
        "COUNT_MSWEP_RAIN30_24H"
    ].eq(EXPECTED_OBSERVATIONS_24H)
    pairs["MSWEP_RAIN30_S_FLAG"] = pairs["RAIN_POINT_30_S"].gt(0)
    pairs["MSWEP_RAIN30_E_FLAG"] = pairs["RAIN_POINT_30_E"].gt(0)
    pairs["MSWEP_ENDPOINTS_VALID_FLAG"] = (
        pairs["MSWEP_RAIN30_S_FLAG"] & pairs["MSWEP_RAIN30_E_FLAG"]
    )

    # Scheme A is explicit even though every FUHE source row should already
    # satisfy the endpoint threshold condition.
    exact_pairs_before_endpoint_filter = len(pairs)
    pairs = pairs.loc[pairs["MSWEP_ENDPOINTS_VALID_FLAG"]].copy()
    endpoint_valid_pairs = len(pairs)

    if pairs[["MSWEP_DIST_30_S", "MSWEP_DIST_30_E"]].isna().any().any():
        raise RuntimeError("A retained pair has a missing endpoint MSWEP_DIST_30")

    for product in PRODUCTS:
        for zone in ZONES:
            start_valid = pairs[
                f"VALID_AREA_FRACTION_{product}_{zone}_S"
            ].ge(FUHE_VALID_AREA_THRESHOLD)
            end_valid = pairs[
                f"VALID_AREA_FRACTION_{product}_{zone}_E"
            ].ge(FUHE_VALID_AREA_THRESHOLD)
            prefix = f"FUHE_AREA_VALID_{product}_{zone}"
            pairs[f"{prefix}_S_FLAG"] = start_valid
            pairs[f"{prefix}_E_FLAG"] = end_valid
            pairs[f"{prefix}_PAIR_FLAG"] = start_valid & end_valid

    pair_area_flags = [
        f"FUHE_AREA_VALID_{product}_{zone}_PAIR_FLAG"
        for product in PRODUCTS
        for zone in ZONES
    ]
    pairs["FUHE_ALL_ZONES_VALID_FLAG"] = pairs[pair_area_flags].all(axis=1)

    drop_counter_columns = required_counters
    pairs = pairs.drop(columns=drop_counter_columns)

    if pairs.duplicated(["SID", "ROW_ID_S", "ROW_ID_E"], keep=False).any():
        raise RuntimeError("Duplicate exact 24-hour pair keys were generated")

    stats = {
        "ALL_EXACT_24H_PAIRS_BEFORE_ENDPOINT_FILTER": exact_pairs_before_endpoint_filter,
        "PAIRS_REMOVED_WITHOUT_BOTH_RAIN30_ENDPOINTS": (
            exact_pairs_before_endpoint_filter - endpoint_valid_pairs
        ),
        "ENDPOINT_VALID_24H_PAIRS": endpoint_valid_pairs,
    }
    return pairs, id_columns, stats


# ============================================================================
# 8. Calculate E-minus-S differences and arrange final columns
# ============================================================================

def calculate_differences_and_arrange(
    pairs: pd.DataFrame,
    id_columns: list[str],
) -> pd.DataFrame:
    difference_data: dict[str, object] = {}
    difference_data["DIFF_ISO_TIME"] = (
        (pairs["ISO_TIME_E"] - pairs["ISO_TIME_S"])
        .dt.total_seconds()
        .div(3600.0)
    )

    for column in DIFFERENCE_SOURCE_COLUMNS:
        difference_data[f"DIFF_{column}"] = (
            pairs[f"{column}_E"].to_numpy()
            - pairs[f"{column}_S"].to_numpy()
        )

    difference_data["DIFF_CAL_DIST_REAL_ABS"] = np.abs(
        difference_data["DIFF_CAL_DIST_REAL"]
    )
    difference_data["DIFF_DIST2LAND_ABS"] = np.abs(
        difference_data["DIFF_DIST2LAND"]
    )

    # Add all difference columns in one operation. Repeated column insertion
    # fragments a wide DataFrame and produces pandas PerformanceWarning.
    pairs = pd.concat(
        [pairs, pd.DataFrame(difference_data, index=pairs.index)],
        axis=1,
        copy=False,
    )

    start_columns = [f"{column}_S" for column in TARGET_COLUMNS]
    end_columns = [f"{column}_E" for column in TARGET_COLUMNS]

    ordered_differences = [
        "DIFF_USA_WIND",
        "DIFF_ISO_TIME",
        "DIFF_CAL_DIST_REAL",
        "DIFF_CAL_DIST_REAL_ABS",
        "DIFF_DIST2LAND",
        "DIFF_DIST2LAND_ABS",
        "DIFF_MSWEP_DIST_30",
    ]
    for zone in ZONES:
        ordered_differences.extend(
            f"DIFF_{metric}_{zone}"
            for metric in COMMON_RADIAL_DIFFERENCE_METRICS
        )
        for product in PRODUCTS:
            ordered_differences.extend(
                f"DIFF_{metric}_{product}_{zone}"
                for metric in PRODUCT_RADIAL_DIFFERENCE_METRICS
            )

    endpoint_quality_columns = [
        "MSWEP_RAIN30_S_FLAG",
        "MSWEP_RAIN30_E_FLAG",
        "MSWEP_ENDPOINTS_VALID_FLAG",
        "COUNT_MSWEP_RAIN30_24H",
        "ANY_MSWEP_RAIN30_24H_FLAG",
        "ALL_MSWEP_RAIN30_24H_FLAG",
        "COUNT_ET_24H",
        "COUNT_ER_24H",
        "OBS_COUNT_24H",
        "COMPLETE_24H_FLAG",
    ]
    area_quality_columns = []
    for product in PRODUCTS:
        for zone in ZONES:
            prefix = f"FUHE_AREA_VALID_{product}_{zone}"
            area_quality_columns.extend(
                [
                    f"{prefix}_S_FLAG",
                    f"{prefix}_E_FLAG",
                    f"{prefix}_PAIR_FLAG",
                ]
            )
    area_quality_columns.append("FUHE_ALL_ZONES_VALID_FLAG")

    final_columns = (
        ["SID"]
        + id_columns
        + start_columns
        + end_columns
        + ordered_differences
        + endpoint_quality_columns
        + area_quality_columns
    )
    require_columns(pairs, final_columns, "Final paired table")
    return pairs[final_columns].sort_values(
        ["ROW_ID_S", "ROW_ID_E"]
    ).reset_index(drop=True)


# ============================================================================
# 9. Logs and summary
# ============================================================================

def build_log_extract(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "SID",
        *[column for column in OPTIONAL_ID_COLUMNS if column in frame.columns],
        "ROW_ID_S",
        "ROW_ID_E",
        "ISO_TIME_S",
        "ISO_TIME_E",
        "RAIN_TIME_ID_S",
        "RAIN_TIME_ID_E",
        "OBS_COUNT_24H",
        "COMPLETE_24H_FLAG",
        "COUNT_MSWEP_RAIN30_24H",
        "COUNT_ET_24H",
        "COUNT_ER_24H",
        "FUHE_ALL_ZONES_VALID_FLAG",
    ]
    columns.extend(
        f"FUHE_AREA_VALID_{product}_{zone}_PAIR_FLAG"
        for product in PRODUCTS
        for zone in ZONES
    )
    return frame[columns].copy()


# ============================================================================
# 10. Main program
# ============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("MSWEP30 + multi-layer convergence exact 24-hour endpoint pairing")
    print("=" * 96)
    print(f"FUHE endpoint input: {FUHE_INPUT}")
    print(f"Complete timeline input: {TIMELINE_INPUT}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Years: {START_YEAR}-{END_YEAR}")
    print("Pairing: same SID and end time exactly start time + 24 hours")
    print("Scheme A: both endpoints must have threshold-30 rainfall")
    print("Incomplete windows: retained and flagged")
    print("ET/ER windows: retained and counted")
    print("Low FUHE area coverage: retained and flagged")
    print("Vertical products: " + ", ".join(PRODUCTS))
    print("Radial zones: " + ", ".join(ZONES))
    print("=" * 96)

    timeline, stats = read_timeline()
    fuhe, fuhe_stats = read_fuhe_source(timeline)
    stats.update(fuhe_stats)

    timeline_counters = prepare_timeline_counters(timeline)
    pairs, id_columns, pairing_stats = build_exact_pairs(
        fuhe, timeline_counters
    )
    stats.update(pairing_stats)
    final = calculate_differences_and_arrange(pairs, id_columns)

    incomplete = final.loc[~final["COMPLETE_24H_FLAG"]].copy()
    low_area = final.loc[~final["FUHE_ALL_ZONES_VALID_FLAG"]].copy()

    stats.update(
        {
            "FINAL_24H_PAIRS": len(final),
            "COMPLETE_24H_PAIRS": int(final["COMPLETE_24H_FLAG"].sum()),
            "INCOMPLETE_24H_PAIRS": len(incomplete),
            "MIN_OBS_COUNT_24H": (
                int(final["OBS_COUNT_24H"].min()) if len(final) else np.nan
            ),
            "MAX_OBS_COUNT_24H": (
                int(final["OBS_COUNT_24H"].max()) if len(final) else np.nan
            ),
            "ET_PAIR_COUNT": int(final["COUNT_ET_24H"].gt(0).sum()),
            "ER_PAIR_COUNT": int(final["COUNT_ER_24H"].gt(0).sum()),
            "ALL_RAIN30_24H_PAIR_COUNT": int(
                final["ALL_MSWEP_RAIN30_24H_FLAG"].sum()
            ),
            "LOW_FUHE_AREA_PAIR_COUNT": len(low_area),
            "FUHE_VALID_AREA_THRESHOLD": FUHE_VALID_AREA_THRESHOLD,
            "FINAL_COLUMN_COUNT": len(final.columns),
            "OUTPUT_PATH": str(OUTPUT_CSV),
        }
    )

    print(f"Writing final paired table: {OUTPUT_CSV}")
    write_csv_atomic(final, OUTPUT_CSV)
    write_csv_atomic(pd.DataFrame([stats]), SUMMARY_CSV)
    write_csv_atomic(build_log_extract(incomplete), INCOMPLETE_PAIRS_CSV)
    write_csv_atomic(build_log_extract(low_area), LOW_AREA_PAIRS_CSV)

    print()
    print("=" * 96)
    print("Processing completed successfully")
    print("=" * 96)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"Final output: {OUTPUT_CSV}")
    print(f"Processing summary: {SUMMARY_CSV}")
    print(f"Incomplete-pair log: {INCOMPLETE_PAIRS_CSV}")
    print(f"Low-area-pair log: {LOW_AREA_PAIRS_CSV}")
    print("=" * 96)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        raise
