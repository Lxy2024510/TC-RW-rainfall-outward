import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. Configuration
# ============================================================

THRESHOLDS = [
    1.5,
    10,
    20,
    30,
    40,
    50,
]

TIME_INTERVAL_HOURS = 24
EXPECTED_INPUT_ROWS = 116162

VALID_3H_HOURS = {
    0,
    3,
    6,
    9,
    12,
    15,
    18,
    21,
}


# ============================================================
# 2. Paths
# ============================================================

# The script is expected at TC-RW-V1/Cal_code/IMERG/.
# TC_RW_PROJECT_ROOT can override automatic project-root detection.
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(
    os.environ.get("TC_RW_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))
).expanduser().resolve()

IMERG_PROCESSED_ROOT = PROJECT_ROOT / "Data" / "Processed" / "IMERG"
INPUT_DIR = IMERG_PROCESSED_ROOT / "Thresholds"
OUTPUT_DIR = IMERG_PROCESSED_ROOT / "Windows" / "24H"
LOG_DIR = (
    PROJECT_ROOT / "Results" / "Quality_control" / "IMERG" / "RUN3"
)
THRESHOLD_SUMMARY_CSV = LOG_DIR / "threshold_processing_summary.csv"


# ============================================================
# 3. Distance zones and retained fields
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
]

# Coverage/QC fields are retained at S and E. They do not affect pairing,
# CLEANED selection, differences, or ET counts.
OPTIONAL_COVERAGE_COLUMNS = [
    "RAIN_POINT",
    "VALID_RAIN_POINT",
    "NAN_RAIN_POINT",
    "TOTAL_AREA_KM2",
    "VALID_AREA_KM2",
    "NAN_AREA_KM2",
    "VALID_AREA_FRACTION",
    "HAS_NAN_PRECIP",
]


# ============================================================
# 4. General helpers
# ============================================================

def threshold_label(threshold):
    value = float(threshold)
    return str(int(value)) if value.is_integer() else format(value, "g")


def build_input_path(threshold):
    label = threshold_label(threshold)
    return INPUT_DIR / f"PRE_DATA_IBT_1998_2024_IMERG_THRESHOLD_{label}.csv"


def build_output_paths(threshold):
    label = threshold_label(threshold)
    all_path = OUTPUT_DIR / (
        f"PRE_DATA_IBT_1998_2024_IMERG_TH{label}_24H_SLIDING_ET_ALL.csv"
    )
    cleaned_path = OUTPUT_DIR / (
        f"PRE_DATA_IBT_1998_2024_IMERG_TH{label}_24H_SLIDING_ET_CLEANED.csv"
    )
    return all_path, cleaned_path


def build_threshold_columns(threshold):
    label = threshold_label(threshold)
    return {
        "label": label,
        "rain_point": f"RAIN_POINT_{label}",
        "area": [f"IMERG_AREA_{label}_{zone}" for zone in DISTANCE_ZONES],
        "volume": [f"IMERG_VOLUME_{label}_{zone}" for zone in DISTANCE_ZONES],
        "distance": f"IMERG_DIST_{label}",
    }


def write_csv_atomic(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        output_path.name + f".tmp.{os.getpid()}"
    )
    try:
        df.to_csv(
            temporary_path,
            index=False,
            date_format="%Y-%m-%d %H:%M:%S",
        )
        if not temporary_path.exists() or temporary_path.stat().st_size == 0:
            raise RuntimeError(f"Temporary output missing or empty: {temporary_path}")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def validate_required_columns(df, required_columns, label):
    missing = set(required_columns).difference(df.columns)
    if missing:
        raise KeyError(
            f"Threshold {label} input is missing required columns: "
            + ", ".join(sorted(missing))
        )


# ============================================================
# 5. Read and validate one threshold table
# ============================================================

def read_and_validate_threshold_table(threshold):
    config = build_threshold_columns(threshold)
    label = config["label"]
    input_path = build_input_path(threshold)

    print("\n" + "=" * 90)
    print(f"Reading and validating threshold {label}")
    print("=" * 90)

    if not input_path.exists():
        raise FileNotFoundError(f"Threshold input does not exist: {input_path}")

    df = pd.read_csv(
        input_path,
        dtype={
            "SID": "string",
            "NAME": "string",
            "RAIN_TIME_ID": "string",
        },
        low_memory=False,
    )

    coverage_columns = [
        column for column in OPTIONAL_COVERAGE_COLUMNS if column in df.columns
    ]
    threshold_columns = (
        [config["rain_point"]]
        + config["area"]
        + config["volume"]
        + [config["distance"]]
    )
    required_columns = ["SID", "NAME"] + COMMON_TARGET_COLUMNS + threshold_columns
    validate_required_columns(df, required_columns, label)

    print(f"Input file: {input_path}")
    print(f"Input rows: {len(df)}")
    print(f"Input columns: {len(df.columns)}")

    if EXPECTED_INPUT_ROWS is not None and len(df) != EXPECTED_INPUT_ROWS:
        raise ValueError(
            f"Threshold {label} row count mismatch: found={len(df)}, "
            f"expected={EXPECTED_INPUT_ROWS}"
        )

    df["ROW_ID"] = pd.to_numeric(df["ROW_ID"], errors="coerce")
    if df["ROW_ID"].isna().any():
        raise ValueError(
            f"Threshold {label} has invalid ROW_ID values: "
            f"{int(df['ROW_ID'].isna().sum())}"
        )
    df["ROW_ID"] = df["ROW_ID"].astype(np.int64)
    if df["ROW_ID"].duplicated().any():
        examples = df.loc[
            df["ROW_ID"].duplicated(keep=False), "ROW_ID"
        ].head(20).tolist()
        raise ValueError(f"Threshold {label} has duplicate ROW_ID values: {examples}")

    # ROW_ID values are original BASE row numbers and are intentionally not
    # renumbered after filtering 1998-2024. No 0..N-1 continuity is required.
    if (df["ROW_ID"] < 0).any():
        raise ValueError(f"Threshold {label} contains negative ROW_ID values")

    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    if df["ISO_TIME"].isna().any():
        raise ValueError(
            f"Threshold {label} has invalid ISO_TIME values: "
            f"{int(df['ISO_TIME'].isna().sum())}"
        )

    year_mask = df["ISO_TIME"].dt.year.between(1998, 2024, inclusive="both")
    if not year_mask.all():
        examples = df.loc[
            ~year_mask, ["ROW_ID", "SID", "ISO_TIME"]
        ].head(20).to_dict("records")
        raise ValueError(f"Threshold {label} has rows outside 1998-2024: {examples}")

    strict_3hour = (
        df["ISO_TIME"].dt.hour.isin(VALID_3H_HOURS)
        & (df["ISO_TIME"].dt.minute == 0)
        & (df["ISO_TIME"].dt.second == 0)
        & (df["ISO_TIME"].dt.microsecond == 0)
    )
    if not strict_3hour.all():
        examples = df.loc[
            ~strict_3hour, ["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID"]
        ].head(20).to_dict("records")
        raise ValueError(
            f"Threshold {label} has non-exact 3-hour timestamps: "
            f"count={int((~strict_3hour).sum())}; examples={examples}"
        )

    valid_rain_id = df["RAIN_TIME_ID"].str.fullmatch(r"\d{7}\.\d{2}", na=False)
    if not valid_rain_id.all():
        raise ValueError(
            f"Threshold {label} has invalid RAIN_TIME_ID values: "
            f"{int((~valid_rain_id).sum())}"
        )

    expected_rain_id = df["ISO_TIME"].dt.strftime("%Y%j.%H")
    mismatch = expected_rain_id != df["RAIN_TIME_ID"]
    if mismatch.any():
        examples = (
            df.loc[mismatch, ["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID"]]
            .assign(EXPECTED_RAIN_TIME_ID=expected_rain_id[mismatch].values)
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            f"Threshold {label} ISO_TIME/RAIN_TIME_ID mismatch: "
            f"count={int(mismatch.sum())}; examples={examples}"
        )

    if df.duplicated(["SID", "ISO_TIME"], keep=False).any():
        raise ValueError(f"Threshold {label} has duplicate (SID, ISO_TIME) keys")
    if df.duplicated(["SID", "RAIN_TIME_ID"], keep=False).any():
        raise ValueError(f"Threshold {label} has duplicate (SID, RAIN_TIME_ID) keys")

    numeric_columns = list(dict.fromkeys(
        [
            "USA_LAT",
            "USA_LON",
            "USA_WIND",
            "USA_SSHS",
            "CAL_DIST_REAL",
            "DIST2LAND",
        ]
        + [column for column in coverage_columns if column != "HAS_NAN_PRECIP"]
        + threshold_columns
    ))

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

    required_numeric = [
        "USA_LAT",
        "USA_LON",
        "USA_WIND",
        "USA_SSHS",
        "CAL_DIST_REAL",
        "DIST2LAND",
        config["rain_point"],
    ] + config["area"] + config["volume"]

    invalid_required = df[required_numeric].isna().any(axis=1)
    if invalid_required.any():
        examples = df.loc[
            invalid_required,
            ["ROW_ID", "SID", "ISO_TIME"] + required_numeric,
        ].head(10).to_dict("records")
        raise ValueError(
            f"Threshold {label} has missing required numeric metrics: "
            f"count={int(invalid_required.sum())}; examples={examples}"
        )

    if (df[config["rain_point"]] < 0).any():
        raise ValueError(f"Threshold {label} has negative rain-point counts")
    for column in config["area"] + config["volume"]:
        if (df[column] < 0).any():
            raise ValueError(f"Threshold {label} has negative values in {column}")

    # Distance may be NaN only when total selected threshold volume is zero.
    total_volume_column = f"IMERG_VOLUME_{label}_0_500"
    distance = df[config["distance"]]
    positive_volume_missing_distance = (df[total_volume_column] > 0) & distance.isna()
    zero_volume_with_distance = (df[total_volume_column] <= 0) & distance.notna()
    if positive_volume_missing_distance.any():
        raise ValueError(
            f"Threshold {label}: positive volume with missing distance; "
            f"count={int(positive_volume_missing_distance.sum())}"
        )
    if zero_volume_with_distance.any():
        raise ValueError(
            f"Threshold {label}: zero volume with defined distance; "
            f"count={int(zero_volume_with_distance.sum())}"
        )

    print(f"[PASS] Threshold {label} validation completed")
    return df, config, coverage_columns, input_path


# ============================================================
# 6. Validate combined-zone identities
# ============================================================

def validate_combined_zones(df, config):
    label = config["label"]
    print(f"Validating combined distance zones for threshold {label} ...")

    for metric_name, columns in [("AREA", config["area"]), ("VOLUME", config["volume"])]:
        by_zone = {zone: column for zone, column in zip(DISTANCE_ZONES, columns)}
        checks = [
            ("0_200", df[by_zone["0_100"]] + df[by_zone["100_200"]]),
            (
                "200_500",
                df[by_zone["200_300"]]
                + df[by_zone["300_400"]]
                + df[by_zone["400_500"]],
            ),
            ("0_500", df[by_zone["0_200"]] + df[by_zone["200_500"]]),
        ]

        for combined_zone, expected in checks:
            actual = df[by_zone[combined_zone]]
            consistent = np.isclose(
                actual,
                expected,
                rtol=1e-9,
                atol=1e-6,
                equal_nan=False,
            )
            if not consistent.all():
                maximum_error = (actual - expected).abs().max()
                raise ValueError(
                    f"Threshold {label} {metric_name} {combined_zone} identity "
                    f"failed: invalid={int((~consistent).sum())}; "
                    f"max_error={maximum_error}"
                )

    print(f"[PASS] Threshold {label} combined-zone validation completed")


# ============================================================
# 7. Build exact 24-hour endpoint pairs
# ============================================================

def build_24h_pairs(df, config, coverage_columns):
    label = config["label"]
    print(f"Building exact 24-hour endpoint pairs for threshold {label} ...")

    working_df = df.sort_values(["SID", "ISO_TIME", "ROW_ID"]).reset_index(drop=True)
    working_df["IS_ET"] = (
        working_df["NATURE"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("ET")
        .astype(np.int8)
    )

    # These cumulative columns are only used to describe the closed interval.
    # They do not alter endpoint matching.
    group = working_df.groupby("SID", sort=False)
    working_df["ET_CUM"] = group["IS_ET"].cumsum().astype(np.int32)
    working_df["OBS_CUM"] = group.cumcount().add(1).astype(np.int32)

    threshold_target_columns = (
        [config["rain_point"]]
        + config["area"]
        + config["volume"]
        + [config["distance"]]
    )
    target_columns = list(dict.fromkeys(
        COMMON_TARGET_COLUMNS + coverage_columns + threshold_target_columns
    ))

    start_df = working_df[
        ["SID", "NAME"] + target_columns + ["ET_CUM", "IS_ET", "OBS_CUM"]
    ].copy()
    start_df["TIME_TARGET_E"] = start_df["ISO_TIME"] + pd.Timedelta(
        hours=TIME_INTERVAL_HOURS
    )
    start_df = start_df.rename(columns={
        column: f"{column}_S"
        for column in target_columns + ["ET_CUM", "IS_ET", "OBS_CUM"]
    })

    end_df = working_df[
        ["SID"] + target_columns + ["ET_CUM", "OBS_CUM"]
    ].copy()
    end_df = end_df.rename(columns={
        column: f"{column}_E"
        for column in target_columns + ["ET_CUM", "OBS_CUM"]
    })

    # This is the only pairing rule: same SID and E == S + exactly 24 hours.
    # No requirement is imposed on intermediate 3-hour records.
    paired_df = pd.merge(
        start_df,
        end_df,
        left_on=["SID", "TIME_TARGET_E"],
        right_on=["SID", "ISO_TIME_E"],
        how="inner",
        validate="one_to_one",
        sort=False,
    ).drop(columns=["TIME_TARGET_E"])

    paired_df["COUNT_ET_24H"] = (
        paired_df["ET_CUM_E"]
        - paired_df["ET_CUM_S"]
        + paired_df["IS_ET_S"]
    ).astype(np.int16)

    paired_df["OBS_COUNT_24H"] = (
        paired_df["OBS_CUM_E"]
        - paired_df["OBS_CUM_S"]
        + 1
    ).astype(np.int16)

    paired_df["COMPLETE_24H_FLAG"] = (
        paired_df["OBS_COUNT_24H"] == 9
    )

    paired_df = paired_df.drop(columns=[
        "ET_CUM_S",
        "ET_CUM_E",
        "IS_ET_S",
        "OBS_CUM_S",
        "OBS_CUM_E",
    ])

    print(f"Exact 24-hour pairs for threshold {label}: {len(paired_df)}")
    return paired_df, target_columns


# ============================================================
# 8. E-minus-S differences
# ============================================================

def calculate_differences(paired_df, config):
    label = config["label"]
    paired_df["DIFF_ISO_TIME"] = (
        (paired_df["ISO_TIME_E"] - paired_df["ISO_TIME_S"])
        .dt.total_seconds()
        / 3600.0
    )

    difference_source_columns = (
        [
            "USA_WIND",
            "CAL_DIST_REAL",
            "DIST2LAND",
            config["rain_point"],
        ]
        + config["area"]
        + config["volume"]
        + [config["distance"]]
    )

    for column in difference_source_columns:
        paired_df[f"DIFF_{column}"] = (
            paired_df[f"{column}_E"] - paired_df[f"{column}_S"]
        )

    paired_df["DIFF_CAL_DIST_REAL_ABS"] = paired_df["DIFF_CAL_DIST_REAL"].abs()
    paired_df["DIFF_DIST2LAND_ABS"] = paired_df["DIFF_DIST2LAND"].abs()

    invalid_interval = ~np.isclose(
        paired_df["DIFF_ISO_TIME"],
        float(TIME_INTERVAL_HOURS),
        rtol=0.0,
        atol=1e-9,
    )
    if invalid_interval.any():
        raise RuntimeError(
            f"Threshold {label} produced non-24-hour endpoint pairs: "
            f"{int(invalid_interval.sum())}"
        )
    return paired_df, difference_source_columns


# ============================================================
# 9. Arrange final columns
# ============================================================

def arrange_final_columns(paired_df, target_columns, difference_source_columns):
    start_columns = [f"{column}_S" for column in target_columns]
    end_columns = [f"{column}_E" for column in target_columns]
    difference_columns = [f"DIFF_{column}" for column in difference_source_columns]

    ordered_differences = [
        "DIFF_USA_WIND",
        "DIFF_ISO_TIME",
        "DIFF_CAL_DIST_REAL",
        "DIFF_CAL_DIST_REAL_ABS",
        "DIFF_DIST2LAND",
        "DIFF_DIST2LAND_ABS",
    ]
    ordered_differences.extend([
        column
        for column in difference_columns
        if column not in {
            "DIFF_USA_WIND",
            "DIFF_CAL_DIST_REAL",
            "DIFF_DIST2LAND",
        }
    ])

    final_columns = (
        ["SID", "NAME"]
        + start_columns
        + end_columns
        + ordered_differences
        + ["COUNT_ET_24H", "OBS_COUNT_24H", "COMPLETE_24H_FLAG"]
    )
    missing = set(final_columns).difference(paired_df.columns)
    if missing:
        raise KeyError("Final paired output is missing: " + ", ".join(sorted(missing)))
    return paired_df[final_columns].copy()


# ============================================================
# 10. Cross-threshold source and pair skeleton validation
# ============================================================

def build_source_skeleton(df):
    columns = ["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID"]
    return df[columns].sort_values("ROW_ID").reset_index(drop=True)


def compare_source_skeletons(reference, current, label):
    if len(reference) != len(current):
        raise RuntimeError(
            f"Threshold {label} source row count differs from first threshold"
        )
    columns = ["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID"]
    if not np.array_equal(
        reference[columns].astype(str).to_numpy(),
        current[columns].astype(str).to_numpy(),
    ):
        raise RuntimeError(
            f"Threshold {label} source skeleton differs from first threshold"
        )


def build_pair_skeleton(final_df):
    columns = [
        "SID",
        "ROW_ID_S",
        "ROW_ID_E",
        "ISO_TIME_S",
        "ISO_TIME_E",
        "COUNT_ET_24H",
        "OBS_COUNT_24H",
        "COMPLETE_24H_FLAG",
    ]
    skeleton = final_df[columns].sort_values(
        ["ROW_ID_S", "ROW_ID_E"]
    ).reset_index(drop=True)
    if skeleton.duplicated(["SID", "ROW_ID_S", "ROW_ID_E"], keep=False).any():
        raise RuntimeError("Duplicate 24-hour pair keys were found")
    return skeleton


def compare_pair_skeletons(reference, current, label):
    if len(reference) != len(current):
        raise RuntimeError(
            f"Threshold {label} pair count differs from first threshold: "
            f"current={len(current)}, reference={len(reference)}"
        )
    columns = list(reference.columns)
    if not np.array_equal(
        reference[columns].astype(str).to_numpy(),
        current[columns].astype(str).to_numpy(),
    ):
        raise RuntimeError(
            f"Threshold {label} pair skeleton/ET/observation counts differ "
            "from first threshold"
        )


# ============================================================
# 11. Process one threshold
# ============================================================

def process_threshold(threshold, reference_source_skeleton, reference_pair_skeleton):
    source_df, config, coverage_columns, input_path = (
        read_and_validate_threshold_table(threshold)
    )

    source_skeleton = build_source_skeleton(source_df)
    if reference_source_skeleton is None:
        reference_source_skeleton = source_skeleton.copy()
    else:
        compare_source_skeletons(
            reference_source_skeleton,
            source_skeleton,
            config["label"],
        )

    validate_combined_zones(source_df, config)
    paired_df, target_columns = build_24h_pairs(
        source_df, config, coverage_columns
    )
    paired_df, difference_source_columns = calculate_differences(
        paired_df, config
    )

    final_all_df = arrange_final_columns(
        paired_df, target_columns, difference_source_columns
    ).sort_values(["ROW_ID_S", "ROW_ID_E"]).reset_index(drop=True)

    current_pair_skeleton = build_pair_skeleton(final_all_df)
    if reference_pair_skeleton is None:
        reference_pair_skeleton = current_pair_skeleton.copy()
    else:
        compare_pair_skeletons(
            reference_pair_skeleton,
            current_pair_skeleton,
            config["label"],
        )

    distance_start = f"{config['distance']}_S"
    distance_end = f"{config['distance']}_E"
    start_valid = final_all_df[distance_start].notna()
    end_valid = final_all_df[distance_end].notna()

    # The only CLEANED rule is that the threshold-specific distance must be
    # defined at both endpoints. Source NaN coverage and intermediate record
    # completeness do not participate in this selection.
    cleaned_mask = start_valid & end_valid
    final_cleaned_df = final_all_df.loc[cleaned_mask].copy().reset_index(drop=True)

    both_distance_nan = int((~start_valid & ~end_valid).sum())
    start_only_nan = int((~start_valid & end_valid).sum())
    end_only_nan = int((start_valid & ~end_valid).sum())

    all_path, cleaned_path = build_output_paths(threshold)
    print(f"Writing threshold {config['label']} ALL: {all_path}")
    write_csv_atomic(final_all_df, all_path)
    print(f"Writing threshold {config['label']} CLEANED: {cleaned_path}")
    write_csv_atomic(final_cleaned_df, cleaned_path)

    summary_record = {
        "THRESHOLD": float(threshold),
        "THRESHOLD_LABEL": config["label"],
        "INPUT_PATH": str(input_path),
        "INPUT_ROWS": len(source_df),
        "ALL_24H_PAIRS": len(final_all_df),
        "COMPLETE_24H_PAIRS": int(final_all_df["COMPLETE_24H_FLAG"].sum()),
        "INCOMPLETE_24H_PAIRS": int((~final_all_df["COMPLETE_24H_FLAG"]).sum()),
        "MIN_OBS_COUNT_24H": (
            int(final_all_df["OBS_COUNT_24H"].min()) if len(final_all_df) else np.nan
        ),
        "MAX_OBS_COUNT_24H": (
            int(final_all_df["OBS_COUNT_24H"].max()) if len(final_all_df) else np.nan
        ),
        "START_DIST_NAN": int((~start_valid).sum()),
        "END_DIST_NAN": int((~end_valid).sum()),
        "BOTH_DIST_NAN": both_distance_nan,
        "START_ONLY_NAN": start_only_nan,
        "END_ONLY_NAN": end_only_nan,
        "CLEANED_PAIRS": len(final_cleaned_df),
        "REMOVED_PAIRS": len(final_all_df) - len(final_cleaned_df),
        "CLEANED_RETENTION_RATE": (
            len(final_cleaned_df) / len(final_all_df) if len(final_all_df) else np.nan
        ),
        "ET_PAIR_COUNT_ALL": int((final_all_df["COUNT_ET_24H"] > 0).sum()),
        "ET_PAIR_COUNT_CLEANED": int(
            (final_cleaned_df["COUNT_ET_24H"] > 0).sum()
        ),
        "SOURCE_NAN_ENDPOINT_PAIRS_ALL": int(
            (
                (final_all_df.get("NAN_RAIN_POINT_S", 0) > 0)
                | (final_all_df.get("NAN_RAIN_POINT_E", 0) > 0)
            ).sum()
        ) if "NAN_RAIN_POINT_S" in final_all_df.columns else 0,
        "OUTPUT_ALL_PATH": str(all_path),
        "OUTPUT_CLEANED_PATH": str(cleaned_path),
    }

    print(
        f"Threshold {config['label']} completed: ALL={len(final_all_df)}, "
        f"CLEANED={len(final_cleaned_df)}, "
        f"removed={summary_record['REMOVED_PAIRS']}"
    )
    return (
        reference_source_skeleton,
        reference_pair_skeleton,
        summary_record,
    )


# ============================================================
# 12. Main
# ============================================================

def main():
    if not THRESHOLDS:
        raise ValueError("THRESHOLDS must contain at least one value")

    labels = [threshold_label(value) for value in THRESHOLDS]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate normalized thresholds: {labels}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("Starting batch 24-hour IMERG processing")
    print("=" * 90)
    print("Thresholds: " + ", ".join(labels))
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Pairing rule: same SID and E timestamp exactly S + 24 hours")
    print("Intermediate completeness is descriptive only and never filters pairs")

    reference_source_skeleton = None
    reference_pair_skeleton = None
    summary_records = []

    for threshold in THRESHOLDS:
        (
            reference_source_skeleton,
            reference_pair_skeleton,
            summary_record,
        ) = process_threshold(
            threshold,
            reference_source_skeleton,
            reference_pair_skeleton,
        )
        summary_records.append(summary_record)

    summary_df = pd.DataFrame(summary_records)
    write_csv_atomic(summary_df, THRESHOLD_SUMMARY_CSV)

    print("\n" + "=" * 90)
    print("Batch processing completed successfully")
    print("=" * 90)
    display_columns = [
        "THRESHOLD_LABEL",
        "INPUT_ROWS",
        "ALL_24H_PAIRS",
        "COMPLETE_24H_PAIRS",
        "INCOMPLETE_24H_PAIRS",
        "START_DIST_NAN",
        "END_DIST_NAN",
        "BOTH_DIST_NAN",
        "START_ONLY_NAN",
        "END_ONLY_NAN",
        "CLEANED_PAIRS",
        "REMOVED_PAIRS",
        "CLEANED_RETENTION_RATE",
        "ET_PAIR_COUNT_ALL",
        "ET_PAIR_COUNT_CLEANED",
        "SOURCE_NAN_ENDPOINT_PAIRS_ALL",
    ]
    print(summary_df[display_columns].to_string(index=False))
    print(f"\nThreshold summary: {THRESHOLD_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
