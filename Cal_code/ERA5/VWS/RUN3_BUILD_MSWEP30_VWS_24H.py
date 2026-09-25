import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. Configuration and paths
# ============================================================

START_YEAR = 1982
END_YEAR = 2024
TIME_INTERVAL_HOURS = 24

VALID_3H_HOURS = {0, 3, 6, 9, 12, 15, 18, 21}

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(
    os.environ.get("TC_RW_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))
).expanduser().resolve()
VWS_ROOT = PROJECT_ROOT / "Data" / "Processed" / "ERA5" / "VWS"
VWS_INPUT = (
    VWS_ROOT / "Endpoints"
    / "PRE_DATA_IBT_1982_2024_VWS_200_800KM.csv"
)
MSWEP30_INPUT = (
    PROJECT_ROOT / "Data" / "Processed" / "MSWEP" / "Thresholds"
    / "PRE_DATA_IBT_1982_2024_MSWEP_THRESHOLD_30.csv"
)

OUTPUT_DIR = VWS_ROOT / "Windows" / "24H"
OUTPUT_CSV = OUTPUT_DIR / (
    "PRE_DATA_IBT_1982_2024_MSWEP30_VWS_200_800KM_24H_SLIDING_CLEANED.csv"
)
LOG_DIR = (
    PROJECT_ROOT / "Results" / "Quality_control" / "ERA5" / "VWS" / "RUN3"
)
SUMMARY_CSV = LOG_DIR / "processing_summary.csv"


# ============================================================
# 2. Columns
# ============================================================

VWS_SOURCE_COLUMN = "VWS_AWMEAN_200_800"
VWS_POINT_COLUMN = "VWS_VALID_POINT_200_800"
VWS_AREA_COLUMN = "VWS_VALID_AREA_200_800"

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

OPTIONAL_ID_COLUMNS = ["NAME"]
VWS_QC_COLUMNS = [VWS_POINT_COLUMN, VWS_AREA_COLUMN]

DIFFERENCE_SOURCE_COLUMNS = [
    "USA_WIND",
    "CAL_DIST_REAL",
    "DIST2LAND",
    "MSWEP_DIST_30",
    VWS_SOURCE_COLUMN,
]


# ============================================================
# 3. Helpers
# ============================================================

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


def require_columns(df, columns, table_name):
    missing = set(columns).difference(df.columns)
    if missing:
        raise KeyError(
            f"{table_name} is missing required columns: "
            + ", ".join(sorted(missing))
        )


def validate_row_id(df, table_name):
    df["ROW_ID"] = pd.to_numeric(df["ROW_ID"], errors="coerce")
    if df["ROW_ID"].isna().any():
        raise ValueError(
            f"{table_name} contains invalid ROW_ID values: "
            f"{int(df['ROW_ID'].isna().sum())}"
        )
    df["ROW_ID"] = df["ROW_ID"].astype(np.int64)
    if (df["ROW_ID"] < 0).any():
        raise ValueError(f"{table_name} contains negative ROW_ID values")
    if df["ROW_ID"].duplicated().any():
        examples = df.loc[
            df["ROW_ID"].duplicated(keep=False), "ROW_ID"
        ].head(20).tolist()
        raise ValueError(f"{table_name} contains duplicate ROW_ID values: {examples}")


def validate_time_fields(df):
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    if df["ISO_TIME"].isna().any():
        raise ValueError(
            "Merged table contains invalid ISO_TIME values: "
            f"{int(df['ISO_TIME'].isna().sum())}"
        )

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
            "Merged table contains non-exact 3-hour timestamps: "
            f"count={int((~strict_3hour).sum())}; examples={examples}"
        )

    valid_rain_id = df["RAIN_TIME_ID"].astype("string").str.fullmatch(
        r"\d{7}\.\d{2}", na=False
    )
    if not valid_rain_id.all():
        raise ValueError(
            "Merged table contains invalid RAIN_TIME_ID values: "
            f"{int((~valid_rain_id).sum())}"
        )

    expected_rain_id = df["ISO_TIME"].dt.strftime("%Y%j.%H")
    mismatch = expected_rain_id != df["RAIN_TIME_ID"].astype(str)
    if mismatch.any():
        examples = (
            df.loc[mismatch, ["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID"]]
            .assign(EXPECTED_RAIN_TIME_ID=expected_rain_id[mismatch].values)
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            "ISO_TIME and RAIN_TIME_ID mismatch: "
            f"count={int(mismatch.sum())}; examples={examples}"
        )


# ============================================================
# 4. Read, merge, and validate the complete source timeline
# ============================================================

def read_and_prepare_source():
    print("=" * 90)
    print("Reading VWS and MSWEP threshold-30 tables")
    print("=" * 90)

    if not VWS_INPUT.exists():
        raise FileNotFoundError(f"VWS input does not exist: {VWS_INPUT}")
    if not MSWEP30_INPUT.exists():
        raise FileNotFoundError(f"MSWEP threshold-30 input does not exist: {MSWEP30_INPUT}")

    vws_df = pd.read_csv(
        VWS_INPUT,
        dtype={"SID": "string", "NAME": "string", "RAIN_TIME_ID": "string"},
        low_memory=False,
    )
    require_columns(
        vws_df,
        ["SID", VWS_SOURCE_COLUMN] + COMMON_TARGET_COLUMNS + VWS_QC_COLUMNS,
        "VWS table",
    )
    validate_row_id(vws_df, "VWS table")
    vws_input_rows = len(vws_df)

    mswep_df = pd.read_csv(
        MSWEP30_INPUT,
        usecols=["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID", "MSWEP_DIST_30"],
        dtype={"SID": "string", "RAIN_TIME_ID": "string"},
        low_memory=False,
    )
    validate_row_id(mswep_df, "MSWEP threshold-30 table")
    mswep_input_rows = len(mswep_df)

    # Rename comparison columns so the merge can verify that ROW_ID points to
    # the same storm and timestamp in both source tables.
    mswep_df = mswep_df.rename(
        columns={
            "SID": "MSWEP_SID_CHECK",
            "ISO_TIME": "MSWEP_ISO_TIME_CHECK",
            "RAIN_TIME_ID": "MSWEP_RAIN_TIME_ID_CHECK",
        }
    )

    merged_df = vws_df.merge(
        mswep_df,
        on="ROW_ID",
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )

    unmatched_mswep = merged_df["_merge"] != "both"
    merged_df["ISO_TIME"] = pd.to_datetime(merged_df["ISO_TIME"], errors="coerce")
    if merged_df["ISO_TIME"].isna().any():
        raise ValueError("VWS table contains invalid ISO_TIME values")

    year_mask = merged_df["ISO_TIME"].dt.year.between(
        START_YEAR, END_YEAR, inclusive="both"
    )
    in_range_unmatched = year_mask & unmatched_mswep
    if in_range_unmatched.any():
        examples = merged_df.loc[
            in_range_unmatched, ["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID"]
        ].head(20).to_dict("records")
        raise ValueError(
            f"Some {START_YEAR}-{END_YEAR} VWS rows have no MSWEP "
            "threshold-30 match: "
            f"count={int(in_range_unmatched.sum())}; examples={examples}"
        )

    source_df = merged_df.loc[year_mask].copy()
    if source_df.empty:
        raise RuntimeError(f"No source rows remain in {START_YEAR}-{END_YEAR}")

    mswep_check_time = pd.to_datetime(
        source_df["MSWEP_ISO_TIME_CHECK"], errors="coerce"
    )
    identity_mismatch = (
        source_df["SID"].astype("string")
        != source_df["MSWEP_SID_CHECK"].astype("string")
    ) | (
        source_df["ISO_TIME"] != mswep_check_time
    ) | (
        source_df["RAIN_TIME_ID"].astype("string")
        != source_df["MSWEP_RAIN_TIME_ID_CHECK"].astype("string")
    )
    if identity_mismatch.any():
        examples = source_df.loc[
            identity_mismatch,
            [
                "ROW_ID",
                "SID",
                "MSWEP_SID_CHECK",
                "ISO_TIME",
                "MSWEP_ISO_TIME_CHECK",
                "RAIN_TIME_ID",
                "MSWEP_RAIN_TIME_ID_CHECK",
            ],
        ].head(20).to_dict("records")
        raise ValueError(
            "VWS and MSWEP identities disagree for matching ROW_ID values: "
            f"count={int(identity_mismatch.sum())}; examples={examples}"
        )

    source_df = source_df.drop(
        columns=[
            "_merge",
            "MSWEP_SID_CHECK",
            "MSWEP_ISO_TIME_CHECK",
            "MSWEP_RAIN_TIME_ID_CHECK",
        ]
    )
    validate_time_fields(source_df)

    if source_df.duplicated(["SID", "ISO_TIME"], keep=False).any():
        raise ValueError("Filtered source contains duplicate (SID, ISO_TIME) keys")
    if source_df.duplicated(["SID", "RAIN_TIME_ID"], keep=False).any():
        raise ValueError("Filtered source contains duplicate (SID, RAIN_TIME_ID) keys")

    numeric_columns = list(dict.fromkeys(
        [
            "USA_LAT",
            "USA_LON",
            "USA_WIND",
            "USA_SSHS",
            "CAL_DIST_REAL",
            "DIST2LAND",
            "MSWEP_DIST_30",
            VWS_SOURCE_COLUMN,
            VWS_POINT_COLUMN,
            VWS_AREA_COLUMN,
        ]
    ))
    for column in numeric_columns:
        source_df[column] = pd.to_numeric(source_df[column], errors="coerce")
    source_df[numeric_columns] = source_df[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    required_numeric = [
        "USA_LAT",
        "USA_LON",
        "USA_WIND",
        "USA_SSHS",
        "CAL_DIST_REAL",
        "DIST2LAND",
        VWS_SOURCE_COLUMN,
        VWS_POINT_COLUMN,
        VWS_AREA_COLUMN,
    ]
    invalid_required = source_df[required_numeric].isna().any(axis=1)
    if invalid_required.any():
        examples = source_df.loc[
            invalid_required,
            ["ROW_ID", "SID", "ISO_TIME"] + required_numeric,
        ].head(10).to_dict("records")
        raise ValueError(
            f"{START_YEAR}-{END_YEAR} source contains missing required "
            "VWS/base metrics: "
            f"count={int(invalid_required.sum())}; examples={examples}"
        )

    if (source_df[VWS_AREA_COLUMN] <= 0.0).any():
        raise ValueError("Non-positive VWS valid areas were found")
    if (source_df[VWS_POINT_COLUMN] <= 0).any():
        raise ValueError("Non-positive VWS valid point counts were found")
    if (source_df[VWS_SOURCE_COLUMN] < 0.0).any():
        examples = source_df.loc[
            source_df[VWS_SOURCE_COLUMN] < 0.0,
            ["ROW_ID", "SID", "ISO_TIME", VWS_SOURCE_COLUMN],
        ].head(20).to_dict("records")
        raise ValueError(
            "Negative VWS magnitudes were found: "
            f"count={int((source_df[VWS_SOURCE_COLUMN] < 0.0).sum())}; "
            f"examples={examples}"
        )

    valid_mswep_distance = source_df["MSWEP_DIST_30"].notna()
    if (source_df.loc[valid_mswep_distance, "MSWEP_DIST_30"] < 0.0).any():
        raise ValueError("Negative MSWEP_DIST_30 values were found")

    stats = {
        "VWS_INPUT_ROWS": vws_input_rows,
        "MSWEP_INPUT_ROWS": mswep_input_rows,
        "YEAR_FILTER_START": START_YEAR,
        "YEAR_FILTER_END": END_YEAR,
        "SOURCE_ROWS": len(source_df),
        "ROWS_WITH_VALID_MSWEP_DIST_30": int(valid_mswep_distance.sum()),
        "ROWS_WITH_MISSING_MSWEP_DIST_30": int((~valid_mswep_distance).sum()),
    }

    print(f"VWS input rows: {vws_input_rows}")
    print(f"MSWEP input rows: {mswep_input_rows}")
    print(f"Rows in {START_YEAR}-{END_YEAR}: {len(source_df)}")
    print(
        "Rows with valid MSWEP_DIST_30: "
        f"{int(valid_mswep_distance.sum())}"
    )
    print("MSWEP distance gaps are retained in the complete timeline")
    print("VWS magnitudes are required to be non-negative")
    return source_df, stats


# ============================================================
# 5. Build exact 24-hour endpoint pairs
# ============================================================

def build_24h_pairs(source_df):
    print("\n" + "=" * 90)
    print("Building exact 24-hour endpoint pairs")
    print("=" * 90)

    working_df = source_df.sort_values(
        ["SID", "ISO_TIME", "ROW_ID"]
    ).reset_index(drop=True)

    working_df["IS_ET"] = (
        working_df["NATURE"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("ET")
        .astype(np.int8)
    )

    group = working_df.groupby("SID", sort=False)
    working_df["ET_CUM"] = group["IS_ET"].cumsum().astype(np.int32)
    working_df["OBS_CUM"] = group.cumcount().add(1).astype(np.int32)
    working_df["VWS_VALID"] = (
        np.isfinite(working_df[VWS_SOURCE_COLUMN]).astype(np.int8)
    )
    working_df["VWS_FOR_SUM"] = working_df[VWS_SOURCE_COLUMN].where(
        working_df["VWS_VALID"].eq(1), 0.0
    )
    group = working_df.groupby("SID", sort=False)
    working_df["VWS_COUNT_CUM"] = (
        group["VWS_VALID"].cumsum().astype(np.int32)
    )
    working_df["VWS_SUM_CUM"] = group["VWS_FOR_SUM"].cumsum()

    id_columns = [column for column in OPTIONAL_ID_COLUMNS if column in working_df.columns]
    target_columns = list(dict.fromkeys(
        COMMON_TARGET_COLUMNS
        + VWS_QC_COLUMNS
        + ["MSWEP_DIST_30"]
        + [VWS_SOURCE_COLUMN]
    ))

    start_df = working_df[
        ["SID"]
        + id_columns
        + target_columns
        + [
            "ET_CUM",
            "IS_ET",
            "OBS_CUM",
            "VWS_VALID",
            "VWS_COUNT_CUM",
            "VWS_SUM_CUM",
        ]
    ].copy()
    start_df["TIME_TARGET_E"] = start_df["ISO_TIME"] + pd.Timedelta(
        hours=TIME_INTERVAL_HOURS
    )
    start_df = start_df.rename(
        columns={
            column: f"{column}_S"
            for column in target_columns
            + [
                "ET_CUM",
                "IS_ET",
                "OBS_CUM",
                "VWS_VALID",
                "VWS_COUNT_CUM",
                "VWS_SUM_CUM",
            ]
        }
    )

    end_df = working_df[
        ["SID"]
        + target_columns
        + ["ET_CUM", "OBS_CUM", "VWS_COUNT_CUM", "VWS_SUM_CUM"]
    ].copy()
    end_df = end_df.rename(
        columns={
            column: f"{column}_E"
            for column in target_columns
            + ["ET_CUM", "OBS_CUM", "VWS_COUNT_CUM", "VWS_SUM_CUM"]
        }
    )

    paired_df = pd.merge(
        start_df,
        end_df,
        left_on=["SID", "TIME_TARGET_E"],
        right_on=["SID", "ISO_TIME_E"],
        how="inner",
        validate="one_to_one",
        sort=False,
    ).drop(columns=["TIME_TARGET_E"])

    # Both endpoints are drawn from the complete 1982-2024 source timeline.
    start_in_range = paired_df["ISO_TIME_S"].dt.year.between(
        START_YEAR, END_YEAR, inclusive="both"
    )
    end_in_range = paired_df["ISO_TIME_E"].dt.year.between(
        START_YEAR, END_YEAR, inclusive="both"
    )
    if not (start_in_range & end_in_range).all():
        raise RuntimeError(
            f"A 24-hour pair endpoint lies outside {START_YEAR}-{END_YEAR}"
        )

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
    paired_df["COMPLETE_24H_FLAG"] = paired_df["OBS_COUNT_24H"] == 9

    paired_df["VWS_VALID_COUNT_24H"] = (
        paired_df["VWS_COUNT_CUM_E"]
        - paired_df["VWS_COUNT_CUM_S"]
        + paired_df["VWS_VALID_S"]
    ).astype(np.int16)
    vws_sum_24h = (
        paired_df["VWS_SUM_CUM_E"]
        - paired_df["VWS_SUM_CUM_S"]
        + paired_df[VWS_SOURCE_COLUMN + "_S"].where(
            paired_df["VWS_VALID_S"].eq(1), 0.0
        )
    )
    if (paired_df["VWS_VALID_COUNT_24H"] <= 0).any():
        raise RuntimeError("A 24-hour pair has no valid VWS observations")
    paired_df["MEAN24H_VWS_AWMEAN_200_800"] = (
        vws_sum_24h / paired_df["VWS_VALID_COUNT_24H"]
    )

    if (paired_df["OBS_COUNT_24H"] > 9).any():
        raise RuntimeError("A 24-hour window contains more than nine observations")

    all_exact_pair_count = len(paired_df)
    valid_mswep_endpoints = paired_df[
        ["MSWEP_DIST_30_S", "MSWEP_DIST_30_E"]
    ].notna().all(axis=1)
    removed_missing_mswep_endpoints = int((~valid_mswep_endpoints).sum())
    paired_df = paired_df.loc[valid_mswep_endpoints].copy().reset_index(drop=True)
    if paired_df.empty:
        raise RuntimeError("No 24-hour pairs have valid MSWEP_DIST_30 at both endpoints")

    paired_df = paired_df.drop(
        columns=[
            "ET_CUM_S",
            "ET_CUM_E",
            "IS_ET_S",
            "OBS_CUM_S",
            "OBS_CUM_E",
            "VWS_VALID_S",
            "VWS_COUNT_CUM_S",
            "VWS_COUNT_CUM_E",
            "VWS_SUM_CUM_S",
            "VWS_SUM_CUM_E",
        ]
    )

    if paired_df.duplicated(["SID", "ROW_ID_S", "ROW_ID_E"], keep=False).any():
        raise RuntimeError("Duplicate 24-hour pair keys were generated")

    pairing_stats = {
        "ALL_EXACT_24H_PAIRS_BEFORE_MSWEP_ENDPOINT_FILTER": all_exact_pair_count,
        "PAIRS_REMOVED_MISSING_MSWEP_ENDPOINT": removed_missing_mswep_endpoints,
        "PAIRS_WITH_VALID_MSWEP_ENDPOINTS": len(paired_df),
    }
    print(f"All exact 24-hour pairs: {all_exact_pair_count}")
    print(
        "Pairs removed for a missing MSWEP endpoint: "
        f"{removed_missing_mswep_endpoints}"
    )
    print(f"Pairs retained: {len(paired_df)}")
    return paired_df, target_columns, id_columns, pairing_stats


# ============================================================
# 6. Calculate E-minus-S differences and arrange columns
# ============================================================

def calculate_and_arrange_differences(paired_df, target_columns, id_columns):
    paired_df["DIFF_ISO_TIME"] = (
        (paired_df["ISO_TIME_E"] - paired_df["ISO_TIME_S"])
        .dt.total_seconds()
        / 3600.0
    )

    invalid_interval = ~np.isclose(
        paired_df["DIFF_ISO_TIME"],
        float(TIME_INTERVAL_HOURS),
        rtol=0.0,
        atol=1e-9,
    )
    if invalid_interval.any():
        raise RuntimeError(
            "Non-24-hour endpoint pairs were produced: "
            f"{int(invalid_interval.sum())}"
        )

    for column in DIFFERENCE_SOURCE_COLUMNS:
        paired_df[f"DIFF_{column}"] = (
            paired_df[f"{column}_E"] - paired_df[f"{column}_S"]
        )

    paired_df["DIFF_CAL_DIST_REAL_ABS"] = paired_df["DIFF_CAL_DIST_REAL"].abs()
    paired_df["DIFF_DIST2LAND_ABS"] = paired_df["DIFF_DIST2LAND"].abs()

    # Endpoint filtering was applied after all full-timeline window metrics.
    if paired_df[["MSWEP_DIST_30_S", "MSWEP_DIST_30_E"]].isna().any().any():
        raise RuntimeError("A paired endpoint has missing MSWEP_DIST_30")

    start_columns = [f"{column}_S" for column in target_columns]
    end_columns = [f"{column}_E" for column in target_columns]
    difference_columns = [f"DIFF_{column}" for column in DIFFERENCE_SOURCE_COLUMNS]

    ordered_differences = [
        "DIFF_USA_WIND",
        "DIFF_ISO_TIME",
        "DIFF_CAL_DIST_REAL",
        "DIFF_CAL_DIST_REAL_ABS",
        "DIFF_DIST2LAND",
        "DIFF_DIST2LAND_ABS",
        "DIFF_MSWEP_DIST_30",
        "DIFF_VWS_AWMEAN_200_800",
    ]

    if set(difference_columns).difference(ordered_differences):
        raise RuntimeError("Not all requested difference columns were ordered")

    final_columns = (
        ["SID"]
        + id_columns
        + start_columns
        + end_columns
        + ordered_differences
        + [
            "MEAN24H_VWS_AWMEAN_200_800",
            "VWS_VALID_COUNT_24H",
            "COUNT_ET_24H",
            "OBS_COUNT_24H",
            "COMPLETE_24H_FLAG",
        ]
    )
    missing = set(final_columns).difference(paired_df.columns)
    if missing:
        raise KeyError("Final output is missing columns: " + ", ".join(sorted(missing)))

    return paired_df[final_columns].sort_values(
        ["ROW_ID_S", "ROW_ID_E"]
    ).reset_index(drop=True)


# ============================================================
# 7. Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("Starting MSWEP30 + VWS 24-hour processing")
    print("=" * 90)
    print(f"VWS input: {VWS_INPUT}")
    print(f"MSWEP threshold-30 input: {MSWEP30_INPUT}")
    print(f"Natural-year range: {START_YEAR}-{END_YEAR}")
    print("Pairing: same SID and E timestamp exactly S + 24 hours")
    print("Both pair endpoints must have a valid MSWEP_DIST_30")
    print("VWS 24-hour means use all available full-timeline observations")
    print("VWS magnitudes are required to be non-negative")
    print("VWS E-minus-S differences may be positive, zero, or negative")

    source_df, stats = read_and_prepare_source()
    paired_df, target_columns, id_columns, pairing_stats = build_24h_pairs(source_df)
    stats.update(pairing_stats)
    final_df = calculate_and_arrange_differences(
        paired_df, target_columns, id_columns
    )

    stats.update(
        {
            "ALL_24H_PAIRS": len(final_df),
            "COMPLETE_24H_PAIRS": int(final_df["COMPLETE_24H_FLAG"].sum()),
            "INCOMPLETE_24H_PAIRS": int((~final_df["COMPLETE_24H_FLAG"]).sum()),
            "MIN_OBS_COUNT_24H": (
                int(final_df["OBS_COUNT_24H"].min()) if len(final_df) else np.nan
            ),
            "MAX_OBS_COUNT_24H": (
                int(final_df["OBS_COUNT_24H"].max()) if len(final_df) else np.nan
            ),
            "MIN_VWS_VALID_COUNT_24H": (
                int(final_df["VWS_VALID_COUNT_24H"].min())
                if len(final_df) else np.nan
            ),
            "MAX_VWS_VALID_COUNT_24H": (
                int(final_df["VWS_VALID_COUNT_24H"].max())
                if len(final_df) else np.nan
            ),
            "ET_PAIR_COUNT": int((final_df["COUNT_ET_24H"] > 0).sum()),
            "OUTPUT_PATH": str(OUTPUT_CSV),
        }
    )

    print(f"Writing final 24-hour table: {OUTPUT_CSV}")
    write_csv_atomic(final_df, OUTPUT_CSV)
    write_csv_atomic(pd.DataFrame([stats]), SUMMARY_CSV)

    print("\n" + "=" * 90)
    print("Processing completed successfully")
    print("=" * 90)
    print(f"Complete-timeline source rows: {stats['SOURCE_ROWS']}")
    print(f"24-hour pairs: {stats['ALL_24H_PAIRS']}")
    print(f"Complete 24-hour pairs: {stats['COMPLETE_24H_PAIRS']}")
    print(f"Incomplete 24-hour pairs: {stats['INCOMPLETE_24H_PAIRS']}")
    print(f"ET pairs: {stats['ET_PAIR_COUNT']}")
    print(f"Final output: {OUTPUT_CSV}")
    print(f"Processing summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()

