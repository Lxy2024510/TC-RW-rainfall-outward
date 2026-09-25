#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Bootstrap closed-ring radial mass-flux changes from the RUN4 windows.

RUN4 has already performed exact 24-hour endpoint pairing, endpoint validation,
ET exclusion, spatial classification, and RW/Steady/RI classification. Windows
with missing intermediate 3-hourly records are retained and flagged. This
script does not repeat those operations.

For each spatial region and intensity group, exact 24-hour endpoint windows
are sampled with replacement. Windows with missing intermediate 3-hourly
records remain included and are reported for auditing. Each replicate has the
same number of windows as the original group. The same sampled indices are
applied to all nine metrics within a group, preserving their within-window
covariance. Percentile 95% confidence intervals are calculated from 5,000
bootstrap means.
"""

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd


# =============================================================================
# 1. Paths and bootstrap settings
# =============================================================================

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(
    os.environ.get("TC_RW_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))
).expanduser().resolve()
MASS_FLUX_ROOT = (
    PROJECT_ROOT / "Data" / "Processed" / "ERA5" / "Mass_flux"
)
INPUT_CSV = (
    MASS_FLUX_ROOT / "Windows" / "24H"
    / "PRE_DATA_IBT_1982_2024_MSWEP30_RADIAL_MASS_FLUX_"
      "NON_ET_REGIONAL_RW_STEADY_RI_WINDOWS.csv"
)

OUTPUT_DIR = MASS_FLUX_ROOT / "Bootstrap"
OUTPUT_CSV = OUTPUT_DIR / (
    "RADIAL_MASS_FLUX_RW_STEADY_RI_1982_2024_"
    "MSWEP30_BOOTSTRAP_LONG.csv"
)
GROUP_COUNTS_CSV = OUTPUT_DIR / "bootstrap_group_counts.csv"
PROCESSING_SUMMARY_CSV = OUTPUT_DIR / "processing_summary.csv"
SMALL_SAMPLE_GROUPS_CSV = OUTPUT_DIR / "small_sample_groups.csv"

BOOTSTRAP_COUNT = 5_000
BOOTSTRAP_BATCH_SIZE = 32
CONFIDENCE_LEVEL = 0.95
CI_LOWER_PERCENTILE = 2.5
CI_UPPER_PERCENTILE = 97.5
RANDOM_SEED = 20260924
SMALL_SAMPLE_THRESHOLD = 30

SPATIAL_ORDER = ["OPEN_OCEAN", "NEAR_COAST"]
INTENSITY_ORDER = ["RW", "STEADY", "RI"]


# =============================================================================
# 2. Variables
# =============================================================================

METRICS = [
    {
        "product_order": 1,
        "product": "500_PB",
        "product_label": "500 hPa-PB",
        "component_order": 1,
        "component": "INNER_BOUNDARY",
        "component_label": "200 km",
        "change": "DIFF_F200_500_PB_KG_S",
    },
    {
        "product_order": 1,
        "product": "500_PB",
        "product_label": "500 hPa-PB",
        "component_order": 2,
        "component": "NET_INFLUX",
        "component_label": "200-500 km net flux",
        "change": "DIFF_F500_MINUS_F200_500_PB_KG_S",
    },
    {
        "product_order": 1,
        "product": "500_PB",
        "product_label": "500 hPa-PB",
        "component_order": 3,
        "component": "OUTER_BOUNDARY",
        "component_label": "500 km",
        "change": "DIFF_F500_500_PB_KG_S",
    },
    {
        "product_order": 2,
        "product": "500_700",
        "product_label": "500-700 hPa",
        "component_order": 1,
        "component": "INNER_BOUNDARY",
        "component_label": "300 km",
        "change": "DIFF_F300_500_700_KG_S",
    },
    {
        "product_order": 2,
        "product": "500_700",
        "product_label": "500-700 hPa",
        "component_order": 2,
        "component": "NET_INFLUX",
        "component_label": "300-500 km net flux",
        "change": "DIFF_F500_MINUS_F300_500_700_KG_S",
    },
    {
        "product_order": 2,
        "product": "500_700",
        "product_label": "500-700 hPa",
        "component_order": 3,
        "component": "OUTER_BOUNDARY",
        "component_label": "500 km",
        "change": "DIFF_F500_500_700_KG_S",
    },
    {
        "product_order": 3,
        "product": "700_PB",
        "product_label": "700 hPa-PB",
        "component_order": 1,
        "component": "INNER_BOUNDARY",
        "component_label": "100 km",
        "change": "DIFF_F100_700_PB_KG_S",
    },
    {
        "product_order": 3,
        "product": "700_PB",
        "product_label": "700 hPa-PB",
        "component_order": 2,
        "component": "NET_INFLUX",
        "component_label": "100-500 km net flux",
        "change": "DIFF_F500_MINUS_F100_700_PB_KG_S",
    },
    {
        "product_order": 3,
        "product": "700_PB",
        "product_label": "700 hPa-PB",
        "component_order": 3,
        "component": "OUTER_BOUNDARY",
        "component_label": "500 km",
        "change": "DIFF_F500_700_PB_KG_S",
    },
]

CHANGE_COLUMNS = [metric["change"] for metric in METRICS]


# =============================================================================
# 3. Helpers
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


def bootstrap_mean_matrix(values, seed):
    """Return bootstrap means for all metrics using shared sampled indices."""
    sample_count, metric_count = values.shape
    if sample_count == 0:
        raise ValueError("Cannot bootstrap an empty sample")

    rng = np.random.default_rng(seed)
    output = np.empty((BOOTSTRAP_COUNT, metric_count), dtype=np.float64)
    completed = 0
    while completed < BOOTSTRAP_COUNT:
        batch_size = min(BOOTSTRAP_BATCH_SIZE, BOOTSTRAP_COUNT - completed)
        indices = rng.integers(
            low=0,
            high=sample_count,
            size=(batch_size, sample_count),
        )
        output[completed:completed + batch_size] = values[indices].mean(axis=1)
        completed += batch_size
    return output


# =============================================================================
# 4. Read and validate RUN4 windows
# =============================================================================

def read_analysis_windows():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"RUN4 window table does not exist: {INPUT_CSV}")

    frame = pd.read_csv(INPUT_CSV, low_memory=False)
    required = [
        "SID",
        "SPATIAL_TYPE",
        "INTENSITY_GROUP",
        "COUNT_ET_24H",
        "COMPLETE_24H_FLAG",
        *CHANGE_COLUMNS,
    ]
    require_columns(frame, required, "RUN4 analysis-window table")

    for column in CHANGE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if not frame["SPATIAL_TYPE"].isin(SPATIAL_ORDER).all():
        unexpected = sorted(
            frame.loc[
                ~frame["SPATIAL_TYPE"].isin(SPATIAL_ORDER), "SPATIAL_TYPE"
            ].dropna().unique()
        )
        raise ValueError(f"Unexpected SPATIAL_TYPE values: {unexpected}")
    if not frame["INTENSITY_GROUP"].isin(INTENSITY_ORDER).all():
        unexpected = sorted(
            frame.loc[
                ~frame["INTENSITY_GROUP"].isin(INTENSITY_ORDER),
                "INTENSITY_GROUP",
            ].dropna().unique()
        )
        raise ValueError(f"Unexpected INTENSITY_GROUP values: {unexpected}")
    if not frame["COUNT_ET_24H"].eq(0).all():
        raise ValueError("RUN4 analysis input contains ET windows")
    # Incomplete intermediate records are intentionally retained. The start
    # and end timestamps remain exactly 24 hours apart, and both endpoints
    # have already passed RUN4 validity checks.
    complete_text = (
        frame["COMPLETE_24H_FLAG"].astype(str).str.strip().str.upper()
    )
    invalid_complete = ~complete_text.isin(["TRUE", "FALSE"])
    if invalid_complete.any():
        examples = frame.loc[
            invalid_complete, "COMPLETE_24H_FLAG"
        ].head(10).tolist()
        raise ValueError(
            "Invalid COMPLETE_24H_FLAG values in RUN4 input: "
            f"{examples}"
        )
    frame["COMPLETE_24H_FLAG"] = complete_text.eq("TRUE")
    if not np.isfinite(frame[CHANGE_COLUMNS].to_numpy(np.float64)).all():
        raise ValueError("RUN4 analysis input contains invalid mass-flux changes")
    return frame


# =============================================================================
# 5. Bootstrap all six groups
# =============================================================================

def calculate_bootstrap(frame):
    result_rows = []
    count_rows = []

    for spatial_index, spatial_type in enumerate(SPATIAL_ORDER, start=1):
        for group_index, intensity_group in enumerate(INTENSITY_ORDER, start=1):
            group = frame.loc[
                frame["SPATIAL_TYPE"].eq(spatial_type)
                & frame["INTENSITY_GROUP"].eq(intensity_group)
            ].copy()
            if group.empty:
                raise ValueError(f"No windows for {spatial_type} / {intensity_group}")

            values = group[CHANGE_COLUMNS].to_numpy(np.float64)
            window_count = len(group)
            sid_count = int(group["SID"].nunique())
            incomplete_count = int((~group["COMPLETE_24H_FLAG"]).sum())
            small_sample = window_count < SMALL_SAMPLE_THRESHOLD
            seed = RANDOM_SEED + spatial_index * 1000 + group_index
            replicates = bootstrap_mean_matrix(values, seed)

            observed_mean = values.mean(axis=0)
            observed_median = np.median(values, axis=0)
            ci_lower = np.percentile(
                replicates, CI_LOWER_PERCENTILE, axis=0
            )
            ci_upper = np.percentile(
                replicates, CI_UPPER_PERCENTILE, axis=0
            )
            bootstrap_se = replicates.std(axis=0, ddof=1)

            count_rows.append(
                {
                    "SPATIAL_ORDER": spatial_index,
                    "SPATIAL_TYPE": spatial_type,
                    "INTENSITY_GROUP_ORDER": group_index,
                    "INTENSITY_GROUP": intensity_group,
                    "WINDOW_COUNT": window_count,
                    "UNIQUE_SID_COUNT": sid_count,
                    "INCOMPLETE_WINDOW_COUNT_RETAINED": incomplete_count,
                    "SMALL_SAMPLE_THRESHOLD": SMALL_SAMPLE_THRESHOLD,
                    "SMALL_SAMPLE_FLAG": small_sample,
                    "RANDOM_SEED": seed,
                }
            )

            print(
                f"{spatial_type:11s} / {intensity_group:6s}: "
                f"windows={window_count:,}, SIDs={sid_count:,}, "
                f"incomplete retained={incomplete_count:,}, seed={seed}"
            )

            for metric_index, metric in enumerate(METRICS):
                lower = float(ci_lower[metric_index])
                upper = float(ci_upper[metric_index])
                result_rows.append(
                    {
                        "START_YEAR": 1982,
                        "END_YEAR": 2024,
                        "PRECIPITATION_DATASET": "MSWEP",
                        "RAINFALL_THRESHOLD_MM_3H": 30,
                        "RING_INTERVAL": "CLOSED",
                        "SPATIAL_ORDER": spatial_index,
                        "SPATIAL_TYPE": spatial_type,
                        "COAST_DISTANCE_RULE": (
                            ">500 km at S and E"
                            if spatial_type == "OPEN_OCEAN"
                            else "<=500 km at S and E"
                        ),
                        "INTENSITY_GROUP_ORDER": group_index,
                        "INTENSITY_GROUP": intensity_group,
                        "INTENSITY_RULE": {
                            "RW": "DIFF_USA_WIND <= -30 kt",
                            "STEADY": "DIFF_USA_WIND == 0 kt",
                            "RI": "DIFF_USA_WIND >= 30 kt",
                        }[intensity_group],
                        "PRODUCT_ORDER": metric["product_order"],
                        "PRODUCT": metric["product"],
                        "PRODUCT_LABEL": metric["product_label"],
                        "COMPONENT_ORDER": metric["component_order"],
                        "COMPONENT": metric["component"],
                        "COMPONENT_LABEL": metric["component_label"],
                        "CHANGE_COLUMN": metric["change"],
                        "CHANGE_DEFINITION": "END_MINUS_START",
                        "POSITIVE_MEANING": "INWARD_MASS_FLUX_INCREASED",
                        "UNIT": "kg s^-1",
                        "WINDOW_COUNT": window_count,
                        "UNIQUE_SID_COUNT": sid_count,
                        "INCOMPLETE_WINDOW_COUNT_RETAINED": incomplete_count,
                        "OBSERVED_MEAN": float(observed_mean[metric_index]),
                        "OBSERVED_MEDIAN": float(observed_median[metric_index]),
                        "BOOTSTRAP_CI_LOWER": lower,
                        "BOOTSTRAP_CI_UPPER": upper,
                        "BOOTSTRAP_CI_EXCLUDES_ZERO": lower > 0.0 or upper < 0.0,
                        "BOOTSTRAP_STANDARD_ERROR": float(
                            bootstrap_se[metric_index]
                        ),
                        "BOOTSTRAP_COUNT": BOOTSTRAP_COUNT,
                        "BOOTSTRAP_METHOD": "WINDOW_LEVEL_PERCENTILE",
                        "BOOTSTRAP_UNIT": "exact 24-hour endpoint window",
                        "CONFIDENCE_LEVEL": CONFIDENCE_LEVEL,
                        "RANDOM_SEED": seed,
                        "SMALL_SAMPLE_FLAG": small_sample,
                    }
                )

    result = pd.DataFrame(result_rows).sort_values(
        [
            "SPATIAL_ORDER",
            "INTENSITY_GROUP_ORDER",
            "PRODUCT_ORDER",
            "COMPONENT_ORDER",
        ]
    ).reset_index(drop=True)
    counts = pd.DataFrame(count_rows).sort_values(
        ["SPATIAL_ORDER", "INTENSITY_GROUP_ORDER"]
    ).reset_index(drop=True)
    return result, counts


# =============================================================================
# 6. Main program
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_section("Closed-ring radial mass-flux window-level bootstrap")
    print(f"Input: {INPUT_CSV}")
    print("Groups: open ocean / near coast x RW / Steady / RI")
    print(f"Metrics per group: {len(METRICS)}")
    print(f"Bootstrap repetitions: {BOOTSTRAP_COUNT:,}")
    print("Bootstrap unit: exact 24-hour endpoint window")
    print("Incomplete intermediate records: retained and flagged")
    print("Confidence interval: percentile 95%")

    frame = read_analysis_windows()
    result, counts = calculate_bootstrap(frame)
    small_samples = counts.loc[counts["SMALL_SAMPLE_FLAG"]].copy()

    expected_rows = len(SPATIAL_ORDER) * len(INTENSITY_ORDER) * len(METRICS)
    if len(result) != expected_rows:
        raise RuntimeError(
            f"Bootstrap output has {len(result)} rows; expected {expected_rows}"
        )

    write_csv_atomic(result, OUTPUT_CSV)
    write_csv_atomic(counts, GROUP_COUNTS_CSV)
    write_csv_atomic(small_samples, SMALL_SAMPLE_GROUPS_CSV)

    processing_summary = pd.DataFrame(
        [
            {
                "INPUT_PATH": str(INPUT_CSV),
                "INPUT_WINDOW_COUNT": len(frame),
                "INPUT_UNIQUE_SID_COUNT": frame["SID"].nunique(),
                "INPUT_INCOMPLETE_WINDOW_COUNT_RETAINED": int(
                    (~frame["COMPLETE_24H_FLAG"]).sum()
                ),
                "SPATIAL_GROUP_COUNT": len(SPATIAL_ORDER),
                "INTENSITY_GROUP_COUNT": len(INTENSITY_ORDER),
                "METRIC_COUNT": len(METRICS),
                "BOOTSTRAP_COUNT": BOOTSTRAP_COUNT,
                "BOOTSTRAP_METHOD": "WINDOW_LEVEL_PERCENTILE",
                "OUTPUT_ROW_COUNT": len(result),
                "SMALL_SAMPLE_GROUP_COUNT": len(small_samples),
                "OUTPUT_PATH": str(OUTPUT_CSV),
            }
        ]
    )
    write_csv_atomic(processing_summary, PROCESSING_SUMMARY_CSV)

    print_section("Processing completed successfully")
    print(f"Input windows: {len(frame):,}")
    print(f"Bootstrap output rows: {len(result):,}")
    print(f"Small-sample groups: {len(small_samples):,}")
    print(f"Bootstrap result: {OUTPUT_CSV}")
    print(f"Group counts: {GROUP_COUNTS_CSV}")
    print(f"Processing summary: {PROCESSING_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
