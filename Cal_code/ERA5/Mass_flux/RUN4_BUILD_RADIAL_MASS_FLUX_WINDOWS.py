#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build exact 24-hour radial mass-flux windows without bootstrapping.

The input is the closed-ring endpoint table produced by
RUN3_CALCULATE_ALL_RADIAL_MASS_FLUX.py. This script:

1. pairs records from the same SID exactly 24 hours apart;
2. records, but does not require, whether all nine standard 3-hourly records
   from T+0 through T+24 are present;
3. requires valid R30 and all nine mass-flux metrics at both endpoints;
4. calculates every change as end minus start;
5. excludes windows containing ET and coastal-transition windows;
6. classifies open-ocean and near-coast windows into RW, Steady, and RI;
7. saves window tables and simple descriptive statistics only.

No bootstrap is performed in this script.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. Paths and analysis settings
# =============================================================================

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(
    os.environ.get("TC_RW_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))
).expanduser().resolve()
MASS_FLUX_ROOT = (
    PROJECT_ROOT / "Data" / "Processed" / "ERA5" / "Mass_flux"
)
INPUT_CSV = (
    MASS_FLUX_ROOT / "Endpoints"
    / "PRE_DATA_IBT_1982_2024_MSWEP30_"
      "RADIAL_MASS_FLUX_500PB_500_700_700PB_CLOSED_RINGS.csv"
)

OUTPUT_DIR = MASS_FLUX_ROOT / "Windows" / "24H"
ALL_PAIRS_CSV = OUTPUT_DIR / (
    "PRE_DATA_IBT_1982_2024_MSWEP30_RADIAL_MASS_FLUX_"
    "ALL_EXACT_24H_WINDOWS.csv"
)
CLEANED_PAIRS_CSV = OUTPUT_DIR / (
    "PRE_DATA_IBT_1982_2024_MSWEP30_RADIAL_MASS_FLUX_"
    "CLEANED_ENDPOINT_VALID_24H_WINDOWS.csv"
)
ANALYSIS_WINDOWS_CSV = OUTPUT_DIR / (
    "PRE_DATA_IBT_1982_2024_MSWEP30_RADIAL_MASS_FLUX_"
    "NON_ET_REGIONAL_RW_STEADY_RI_WINDOWS.csv"
)
REMOVED_PAIRS_CSV = OUTPUT_DIR / "REMOVED_24H_WINDOWS.csv"
GROUP_SUMMARY_CSV = OUTPUT_DIR / "RW_STEADY_RI_DESCRIPTIVE_STATISTICS.csv"
PROCESSING_SUMMARY_CSV = OUTPUT_DIR / "processing_summary.csv"

START_YEAR = 1982
END_YEAR = 2024
COAST_DISTANCE_KM = 500.0
RW_THRESHOLD_KT = -30.0
RI_THRESHOLD_KT = 30.0
EXPECTED_OBSERVATIONS_24H = 9

# False retains exact 24-hour endpoint pairs even when one or more intermediate
# 3-hourly records are absent. COMPLETE_24H_FLAG and OBS_COUNT_24H remain in
# every output table for auditing and sensitivity tests.
REQUIRE_COMPLETE_24H_WINDOW = False

VALID_FLUX_STATUS = {"OK", "OK_COVERAGE_WARNING"}


# =============================================================================
# 2. Mass-flux variables
# =============================================================================

METRICS = [
    {
        "product_order": 1,
        "product": "500_PB",
        "product_label": "500 hPa-PB",
        "component_order": 1,
        "component": "INNER_BOUNDARY",
        "component_label": "200 km",
        "endpoint": "F200_500_PB_KG_S",
    },
    {
        "product_order": 1,
        "product": "500_PB",
        "product_label": "500 hPa-PB",
        "component_order": 2,
        "component": "NET_INFLUX",
        "component_label": "200-500 km net flux",
        "endpoint": "F500_MINUS_F200_500_PB_KG_S",
    },
    {
        "product_order": 1,
        "product": "500_PB",
        "product_label": "500 hPa-PB",
        "component_order": 3,
        "component": "OUTER_BOUNDARY",
        "component_label": "500 km",
        "endpoint": "F500_500_PB_KG_S",
    },
    {
        "product_order": 2,
        "product": "500_700",
        "product_label": "500-700 hPa",
        "component_order": 1,
        "component": "INNER_BOUNDARY",
        "component_label": "300 km",
        "endpoint": "F300_500_700_KG_S",
    },
    {
        "product_order": 2,
        "product": "500_700",
        "product_label": "500-700 hPa",
        "component_order": 2,
        "component": "NET_INFLUX",
        "component_label": "300-500 km net flux",
        "endpoint": "F500_MINUS_F300_500_700_KG_S",
    },
    {
        "product_order": 2,
        "product": "500_700",
        "product_label": "500-700 hPa",
        "component_order": 3,
        "component": "OUTER_BOUNDARY",
        "component_label": "500 km",
        "endpoint": "F500_500_700_KG_S",
    },
    {
        "product_order": 3,
        "product": "700_PB",
        "product_label": "700 hPa-PB",
        "component_order": 1,
        "component": "INNER_BOUNDARY",
        "component_label": "100 km",
        "endpoint": "F100_700_PB_KG_S",
    },
    {
        "product_order": 3,
        "product": "700_PB",
        "product_label": "700 hPa-PB",
        "component_order": 2,
        "component": "NET_INFLUX",
        "component_label": "100-500 km net flux",
        "endpoint": "F500_MINUS_F100_700_PB_KG_S",
    },
    {
        "product_order": 3,
        "product": "700_PB",
        "product_label": "700 hPa-PB",
        "component_order": 3,
        "component": "OUTER_BOUNDARY",
        "component_label": "500 km",
        "endpoint": "F500_700_PB_KG_S",
    },
]

for metric in METRICS:
    metric["change"] = f"DIFF_{metric['endpoint']}"

ENDPOINT_METRICS = [metric["endpoint"] for metric in METRICS]
CHANGE_METRICS = [metric["change"] for metric in METRICS]


# =============================================================================
# 3. General helpers
# =============================================================================

def print_section(title):
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)


def write_csv_atomic(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def require_columns(frame, columns, label):
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")


def all_finite(frame, columns):
    return np.isfinite(frame[columns].to_numpy(np.float64)).all(axis=1)


# =============================================================================
# 4. Read the endpoint table
# =============================================================================

def read_endpoint_table():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Input table does not exist: {INPUT_CSV}")

    frame = pd.read_csv(INPUT_CSV, low_memory=False)
    if "ROW_ID" not in frame.columns:
        frame.insert(0, "ROW_ID", np.arange(len(frame), dtype=np.int64))

    required = [
        "ROW_ID",
        "SID",
        "ISO_TIME",
        "RAIN_TIME_ID",
        "NATURE",
        "USA_WIND",
        "CAL_DIST_REAL",
        "MSWEP_DIST_30",
        "RAIN_POINT_30",
        "FLUX_STATUS",
        *ENDPOINT_METRICS,
    ]
    require_columns(frame, required, "RUN3 endpoint table")

    frame["ROW_ID"] = pd.to_numeric(frame["ROW_ID"], errors="coerce")
    frame["ISO_TIME"] = pd.to_datetime(frame["ISO_TIME"], errors="coerce")
    numeric_columns = [
        "USA_WIND",
        "CAL_DIST_REAL",
        "MSWEP_DIST_30",
        "RAIN_POINT_30",
        *ENDPOINT_METRICS,
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame["ROW_ID"].isna().any() or frame["ISO_TIME"].isna().any():
        raise ValueError("Input contains invalid ROW_ID or ISO_TIME")
    frame["ROW_ID"] = frame["ROW_ID"].astype(np.int64)
    if frame["ROW_ID"].duplicated().any():
        raise ValueError("Input contains duplicate ROW_ID values")
    if frame.duplicated(["SID", "ISO_TIME"]).any():
        raise ValueError("Input contains duplicate (SID, ISO_TIME) keys")

    frame = frame.loc[
        frame["ISO_TIME"].dt.year.between(START_YEAR, END_YEAR)
    ].sort_values(["SID", "ISO_TIME", "ROW_ID"]).reset_index(drop=True)
    return frame


# =============================================================================
# 5. Timeline counters and exact 24-hour pairing
# =============================================================================

def prepare_timeline(frame):
    timeline = frame[
        ["ROW_ID", "SID", "ISO_TIME", "NATURE", "RAIN_POINT_30"]
    ].copy()
    nature = timeline["NATURE"].fillna("").astype(str).str.strip().str.upper()
    timeline["IS_ET"] = nature.eq("ET").astype(np.int8)
    timeline["IS_RAIN30"] = timeline["RAIN_POINT_30"].gt(0).astype(np.int8)
    timeline["OBS"] = np.int8(1)
    for source, target in [
        ("IS_ET", "ET_CUM"),
        ("IS_RAIN30", "RAIN30_CUM"),
        ("OBS", "OBS_CUM"),
    ]:
        timeline[target] = timeline.groupby("SID", sort=False)[source].cumsum()
    return timeline


def build_exact_pairs(frame):
    endpoint_columns = [
        "ROW_ID",
        "ISO_TIME",
        "RAIN_TIME_ID",
        "USA_WIND",
        "CAL_DIST_REAL",
        "MSWEP_DIST_30",
        "RAIN_POINT_30",
        "FLUX_STATUS",
        *ENDPOINT_METRICS,
    ]

    start = frame[["SID", *endpoint_columns]].copy()
    start["TARGET_TIME_E"] = start["ISO_TIME"] + pd.Timedelta(hours=24)
    start = start.rename(
        columns={column: f"{column}_S" for column in endpoint_columns}
    )
    end = frame[["SID", *endpoint_columns]].copy().rename(
        columns={column: f"{column}_E" for column in endpoint_columns}
    )
    pairs = start.merge(
        end,
        left_on=["SID", "TARGET_TIME_E"],
        right_on=["SID", "ISO_TIME_E"],
        how="inner",
        validate="one_to_one",
        sort=False,
    ).drop(columns="TARGET_TIME_E")

    timeline = prepare_timeline(frame)
    counter_columns = [
        "ROW_ID",
        "IS_ET",
        "IS_RAIN30",
        "ET_CUM",
        "RAIN30_CUM",
        "OBS_CUM",
    ]
    start_counter = timeline[counter_columns].rename(
        columns={column: f"{column}_S" for column in counter_columns}
    )
    end_counter = timeline[counter_columns].rename(
        columns={column: f"{column}_E" for column in counter_columns}
    )
    pairs = pairs.merge(start_counter, on="ROW_ID_S", validate="many_to_one")
    pairs = pairs.merge(end_counter, on="ROW_ID_E", validate="many_to_one")

    pairs["COUNT_ET_24H"] = (
        pairs["ET_CUM_E"] - pairs["ET_CUM_S"] + pairs["IS_ET_S"]
    ).astype(np.int16)
    pairs["COUNT_RAIN30_24H"] = (
        pairs["RAIN30_CUM_E"]
        - pairs["RAIN30_CUM_S"]
        + pairs["IS_RAIN30_S"]
    ).astype(np.int16)
    pairs["OBS_COUNT_24H"] = (
        pairs["OBS_CUM_E"] - pairs["OBS_CUM_S"] + 1
    ).astype(np.int16)
    pairs["COMPLETE_24H_FLAG"] = pairs["OBS_COUNT_24H"].eq(
        EXPECTED_OBSERVATIONS_24H
    )

    pairs["DIFF_USA_WIND"] = pairs["USA_WIND_E"] - pairs["USA_WIND_S"]
    pairs["DIFF_CAL_DIST_REAL"] = (
        pairs["CAL_DIST_REAL_E"] - pairs["CAL_DIST_REAL_S"]
    )
    pairs["DIFF_MSWEP_DIST_30"] = (
        pairs["MSWEP_DIST_30_E"] - pairs["MSWEP_DIST_30_S"]
    )
    for metric in METRICS:
        source = metric["endpoint"]
        pairs[metric["change"]] = pairs[f"{source}_E"] - pairs[f"{source}_S"]

    internal = [
        f"{column}_{endpoint}"
        for column in [
            "IS_ET", "IS_RAIN30", "ET_CUM", "RAIN30_CUM", "OBS_CUM"
        ]
        for endpoint in ("S", "E")
    ]
    pairs = pairs.drop(columns=internal)
    return pairs.sort_values(["SID", "ISO_TIME_S"]).reset_index(drop=True)


# =============================================================================
# 6. Strict cleaning and group classification
# =============================================================================

def clean_and_classify(pairs):
    endpoint_numeric = [
        "USA_WIND_S",
        "USA_WIND_E",
        "CAL_DIST_REAL_S",
        "CAL_DIST_REAL_E",
        "MSWEP_DIST_30_S",
        "MSWEP_DIST_30_E",
        *[
            f"{metric}_{endpoint}"
            for metric in ENDPOINT_METRICS
            for endpoint in ("S", "E")
        ],
    ]
    endpoint_valid = (
        pairs["FLUX_STATUS_S"].isin(VALID_FLUX_STATUS)
        & pairs["FLUX_STATUS_E"].isin(VALID_FLUX_STATUS)
        & all_finite(pairs, endpoint_numeric)
        & pairs["RAIN_POINT_30_S"].gt(0)
        & pairs["RAIN_POINT_30_E"].gt(0)
    )
    if REQUIRE_COMPLETE_24H_WINDOW:
        valid = endpoint_valid & pairs["COMPLETE_24H_FLAG"]
    else:
        valid = endpoint_valid

    cleaned = pairs.loc[valid].copy()
    removed = pairs.loc[~valid].copy()

    open_ocean = (
        cleaned["CAL_DIST_REAL_S"].gt(COAST_DISTANCE_KM)
        & cleaned["CAL_DIST_REAL_E"].gt(COAST_DISTANCE_KM)
    )
    near_coast = (
        cleaned["CAL_DIST_REAL_S"].le(COAST_DISTANCE_KM)
        & cleaned["CAL_DIST_REAL_E"].le(COAST_DISTANCE_KM)
    )
    cleaned["SPATIAL_TYPE"] = "MIXED"
    cleaned.loc[open_ocean, "SPATIAL_TYPE"] = "OPEN_OCEAN"
    cleaned.loc[near_coast, "SPATIAL_TYPE"] = "NEAR_COAST"

    analysis = cleaned.loc[
        cleaned["SPATIAL_TYPE"].ne("MIXED")
        & cleaned["COUNT_ET_24H"].eq(0)
    ].copy()
    analysis["INTENSITY_GROUP"] = "OTHER"
    analysis.loc[
        analysis["DIFF_USA_WIND"].le(RW_THRESHOLD_KT), "INTENSITY_GROUP"
    ] = "RW"
    analysis.loc[
        analysis["DIFF_USA_WIND"].eq(0.0), "INTENSITY_GROUP"
    ] = "STEADY"
    analysis.loc[
        analysis["DIFF_USA_WIND"].ge(RI_THRESHOLD_KT), "INTENSITY_GROUP"
    ] = "RI"

    target = analysis.loc[
        analysis["INTENSITY_GROUP"].isin(["RW", "STEADY", "RI"])
    ].copy()
    return cleaned, removed, analysis, target


# =============================================================================
# 7. Descriptive statistics without bootstrap
# =============================================================================

def summarize_groups(target):
    rows = []
    spatial_order = ["OPEN_OCEAN", "NEAR_COAST"]
    group_order = ["RW", "STEADY", "RI"]

    for spatial_index, spatial_type in enumerate(spatial_order, start=1):
        for group_index, group_name in enumerate(group_order, start=1):
            group = target.loc[
                target["SPATIAL_TYPE"].eq(spatial_type)
                & target["INTENSITY_GROUP"].eq(group_name)
            ]
            if group.empty:
                raise ValueError(f"No samples for {spatial_type} / {group_name}")

            for metric in METRICS:
                values = group[metric["change"]].to_numpy(np.float64)
                rows.append(
                    {
                        "SPATIAL_ORDER": spatial_index,
                        "SPATIAL_TYPE": spatial_type,
                        "INTENSITY_GROUP_ORDER": group_index,
                        "INTENSITY_GROUP": group_name,
                        "PRODUCT_ORDER": metric["product_order"],
                        "PRODUCT": metric["product"],
                        "PRODUCT_LABEL": metric["product_label"],
                        "COMPONENT_ORDER": metric["component_order"],
                        "COMPONENT": metric["component"],
                        "COMPONENT_LABEL": metric["component_label"],
                        "CHANGE_COLUMN": metric["change"],
                        "CHANGE_DEFINITION": "END_MINUS_START",
                        "UNIT": "kg s^-1",
                        "WINDOW_COUNT": len(group),
                        "UNIQUE_SID_COUNT": int(group["SID"].nunique()),
                        "MEAN_CHANGE": float(values.mean()),
                        "MEDIAN_CHANGE": float(np.median(values)),
                        "STANDARD_DEVIATION": float(values.std(ddof=1)),
                        "STANDARD_ERROR": float(values.std(ddof=1) / np.sqrt(len(values))),
                    }
                )

    return pd.DataFrame(rows).sort_values(
        [
            "SPATIAL_ORDER",
            "INTENSITY_GROUP_ORDER",
            "PRODUCT_ORDER",
            "COMPONENT_ORDER",
        ]
    ).reset_index(drop=True)


# =============================================================================
# 8. Main program
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_section("Closed-ring radial mass-flux exact 24-hour processing")
    print(f"Input: {INPUT_CSV}")
    print("Change definition: end minus start")
    print(
        "Complete-window rule: "
        + (
            "all nine standard 3-hour records are required"
            if REQUIRE_COMPLETE_24H_WINDOW
            else "incomplete intermediate records are retained and flagged"
        )
    )
    print("Spatial groups: open ocean and near coast; mixed is excluded")
    print("Intensity groups: RW, Steady, and RI")
    print("Bootstrap: not performed")

    endpoint = read_endpoint_table()
    pairs = build_exact_pairs(endpoint)
    cleaned, removed, analysis, target = clean_and_classify(pairs)
    summary = summarize_groups(target)

    write_csv_atomic(pairs, ALL_PAIRS_CSV)
    write_csv_atomic(cleaned, CLEANED_PAIRS_CSV)
    write_csv_atomic(removed, REMOVED_PAIRS_CSV)
    write_csv_atomic(target, ANALYSIS_WINDOWS_CSV)
    write_csv_atomic(summary, GROUP_SUMMARY_CSV)

    counts = (
        target.groupby(["SPATIAL_TYPE", "INTENSITY_GROUP"], observed=True)
        .agg(WINDOW_COUNT=("SID", "size"), UNIQUE_SID_COUNT=("SID", "nunique"))
        .reset_index()
    )
    processing_summary = pd.DataFrame(
        [
            {
                "INPUT_ENDPOINT_ROWS": len(endpoint),
                "ALL_EXACT_24H_PAIRS": len(pairs),
                "ENDPOINT_VALID_CLEANED_PAIRS": len(cleaned),
                "REMOVED_PAIRS": len(removed),
                "REQUIRE_COMPLETE_24H_WINDOW": REQUIRE_COMPLETE_24H_WINDOW,
                "INCOMPLETE_PAIRS_BEFORE_CLEANING": int(
                    (~pairs["COMPLETE_24H_FLAG"]).sum()
                ),
                "INCOMPLETE_PAIRS_RETAINED": int(
                    (~cleaned["COMPLETE_24H_FLAG"]).sum()
                ),
                "MIXED_WINDOWS": int(cleaned["SPATIAL_TYPE"].eq("MIXED").sum()),
                "ET_WINDOWS_AFTER_SPATIAL_FILTER": int(
                    cleaned["SPATIAL_TYPE"].ne("MIXED")
                    .mul(cleaned["COUNT_ET_24H"].gt(0))
                    .sum()
                ),
                "RW_STEADY_RI_ANALYSIS_WINDOWS": len(target),
                "DESCRIPTIVE_STATISTIC_ROWS": len(summary),
            }
        ]
    )
    write_csv_atomic(processing_summary, PROCESSING_SUMMARY_CSV)

    print_section("Processing completed successfully")
    print(f"Input endpoint rows: {len(endpoint):,}")
    print(f"All exact 24-hour pairs: {len(pairs):,}")
    print(f"Endpoint-valid cleaned pairs: {len(cleaned):,}")
    print(
        "Incomplete pairs retained: "
        f"{int((~cleaned['COMPLETE_24H_FLAG']).sum()):,}"
    )
    print(f"Removed pairs: {len(removed):,}")
    print("\nFinal group counts:")
    print(counts.to_string(index=False))
    print(f"\nAnalysis windows: {ANALYSIS_WINDOWS_CSV}")
    print(f"Descriptive statistics: {GROUP_SUMMARY_CSV}")
    print(f"Processing summary: {PROCESSING_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
