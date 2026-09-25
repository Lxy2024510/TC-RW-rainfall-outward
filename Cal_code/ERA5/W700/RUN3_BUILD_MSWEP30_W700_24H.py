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
W700_ROOT = PROJECT_ROOT / "Data" / "Processed" / "ERA5" / "W700"
W700_INPUT = W700_ROOT / "Endpoints" / "PRE_DATA_IBT_1982_2024_W700.csv"
MSWEP30_INPUT = (
    PROJECT_ROOT / "Data" / "Processed" / "MSWEP" / "Thresholds"
    / "PRE_DATA_IBT_1982_2024_MSWEP_THRESHOLD_30.csv"
)

OUTPUT_DIR = W700_ROOT / "Windows" / "24H"
OUTPUT_CSV = OUTPUT_DIR / (
    "PRE_DATA_IBT_1982_2024_MSWEP30_W700_24H_SLIDING_CLEANED.csv"
)
LOG_DIR = (
    PROJECT_ROOT / "Results" / "Quality_control" / "ERA5" / "W700" / "RUN3"
)
SUMMARY_CSV = LOG_DIR / "processing_summary.csv"


# ============================================================
# 2. Columns
# ============================================================

W700_ZONES = [
    "0_100",
    "100_200",
    "200_300",
    "300_400",
    "400_500",
    "0_200",
    "200_500",
    "0_500",
]

W700_SOURCE_COLUMNS = [f"W700_AWMEAN_{zone}" for zone in W700_ZONES]
W700_REVERSED_COLUMNS = [f"W700_REVERSED_{zone}" for zone in W700_ZONES]
W700_AREA_COLUMNS = [f"W700_VALID_AREA_{zone}" for zone in W700_ZONES]

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
W700_QC_COLUMNS = ["W700_POINT", "W700_VALID_POINT"] + W700_AREA_COLUMNS

DIFFERENCE_SOURCE_COLUMNS = [
    "USA_WIND",
    "CAL_DIST_REAL",
    "DIST2LAND",
    "MSWEP_DIST_30",
] + W700_REVERSED_COLUMNS


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
# 4. Read, merge, filter, and reverse W700 sign
# ============================================================

def read_and_prepare_source():
    print("=" * 90)
    print("Reading W700 and MSWEP threshold-30 tables")
    print("=" * 90)

    if not W700_INPUT.exists():
        raise FileNotFoundError(f"W700 input does not exist: {W700_INPUT}")
    if not MSWEP30_INPUT.exists():
        raise FileNotFoundError(f"MSWEP threshold-30 input does not exist: {MSWEP30_INPUT}")

    w700_df = pd.read_csv(
        W700_INPUT,
        dtype={"SID": "string", "NAME": "string", "RAIN_TIME_ID": "string"},
        low_memory=False,
    )
    require_columns(
        w700_df,
        ["SID"] + COMMON_TARGET_COLUMNS + W700_SOURCE_COLUMNS + W700_QC_COLUMNS,
        "W700 table",
    )
    validate_row_id(w700_df, "W700 table")
    w700_input_rows = len(w700_df)

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

    merged_df = w700_df.merge(
        mswep_df,
        on="ROW_ID",
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )

    unmatched_mswep = merged_df["_merge"] != "both"
    # Matching is strict for every row in the configured 1982-2024 range.
    merged_df["ISO_TIME"] = pd.to_datetime(merged_df["ISO_TIME"], errors="coerce")
    if merged_df["ISO_TIME"].isna().any():
        raise ValueError("W700 table contains invalid ISO_TIME values")

    year_mask = merged_df["ISO_TIME"].dt.year.between(
        START_YEAR, END_YEAR, inclusive="both"
    )
    in_range_unmatched = year_mask & unmatched_mswep
    if in_range_unmatched.any():
        examples = merged_df.loc[
            in_range_unmatched, ["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID"]
        ].head(20).to_dict("records")
        raise ValueError(
            "Some 1982-2024 W700 rows have no MSWEP threshold-30 match: "
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
            "W700 and MSWEP identities disagree for matching ROW_ID values: "
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
            "W700_POINT",
            "W700_VALID_POINT",
        ]
        + W700_SOURCE_COLUMNS
        + W700_AREA_COLUMNS
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
        "W700_POINT",
        "W700_VALID_POINT",
    ] + W700_SOURCE_COLUMNS + W700_AREA_COLUMNS
    invalid_required = source_df[required_numeric].isna().any(axis=1)
    if invalid_required.any():
        examples = source_df.loc[
            invalid_required,
            ["ROW_ID", "SID", "ISO_TIME"] + required_numeric,
        ].head(10).to_dict("records")
        raise ValueError(
            "1982-2024 source contains missing required W700/base metrics: "
            f"count={int(invalid_required.sum())}; examples={examples}"
        )

    if (source_df[W700_AREA_COLUMNS] < 0.0).any().any():
        raise ValueError("Negative W700 valid areas were found")
    if (source_df[["W700_POINT", "W700_VALID_POINT"]] < 0).any().any():
        raise ValueError("Negative W700 point counts were found")
    if (source_df["W700_VALID_POINT"] > source_df["W700_POINT"]).any():
        raise ValueError("W700_VALID_POINT exceeds W700_POINT")

    before_distance_filter = len(source_df)
    missing_mswep_distance = source_df["MSWEP_DIST_30"].isna()
    removed_missing_distance = int(missing_mswep_distance.sum())
    source_df = source_df.loc[~missing_mswep_distance].copy().reset_index(drop=True)
    if source_df.empty:
        raise RuntimeError("No rows remain after removing missing MSWEP_DIST_30")

    if (source_df["MSWEP_DIST_30"] < 0.0).any():
        raise ValueError("Negative MSWEP_DIST_30 values were found")

    # Reverse the original pressure-vertical-velocity sign. The new variables
    # are positive for upward motion and negative for downward motion.
    for source_column, reversed_column in zip(
        W700_SOURCE_COLUMNS, W700_REVERSED_COLUMNS
    ):
        source_df[reversed_column] = -source_df[source_column]

    # Do not retain ambiguous original-sign W700_AWMEAN columns in the paired
    # output; their explicitly named reversed versions are used instead.
    source_df = source_df.drop(columns=W700_SOURCE_COLUMNS)

    stats = {
        "W700_INPUT_ROWS": w700_input_rows,
        "MSWEP_INPUT_ROWS": mswep_input_rows,
        "YEAR_FILTER_START": START_YEAR,
        "YEAR_FILTER_END": END_YEAR,
        "ROWS_1982_2024_BEFORE_DIST_FILTER": before_distance_filter,
        "ROWS_REMOVED_MISSING_MSWEP_DIST_30": removed_missing_distance,
        "SOURCE_ROWS_AFTER_FILTERS": len(source_df),
    }

    print(f"W700 input rows: {w700_input_rows}")
    print(f"MSWEP input rows: {mswep_input_rows}")
    print(f"Rows in {START_YEAR}-{END_YEAR}: {before_distance_filter}")
    print(f"Rows removed for missing MSWEP_DIST_30: {removed_missing_distance}")
    print(f"Rows retained for pairing: {len(source_df)}")
    print("W700 sign reversed: positive values represent upward motion")
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

    id_columns = [column for column in OPTIONAL_ID_COLUMNS if column in working_df.columns]
    target_columns = list(dict.fromkeys(
        COMMON_TARGET_COLUMNS
        + W700_QC_COLUMNS
        + ["MSWEP_DIST_30"]
        + W700_REVERSED_COLUMNS
    ))

    start_df = working_df[
        ["SID"] + id_columns + target_columns + ["ET_CUM", "IS_ET", "OBS_CUM"]
    ].copy()
    start_df["TIME_TARGET_E"] = start_df["ISO_TIME"] + pd.Timedelta(
        hours=TIME_INTERVAL_HOURS
    )
    start_df = start_df.rename(
        columns={
            column: f"{column}_S"
            for column in target_columns + ["ET_CUM", "IS_ET", "OBS_CUM"]
        }
    )

    end_df = working_df[
        ["SID"] + target_columns + ["ET_CUM", "OBS_CUM"]
    ].copy()
    end_df = end_df.rename(
        columns={
            column: f"{column}_E"
            for column in target_columns + ["ET_CUM", "OBS_CUM"]
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

    # Both endpoints are drawn from the already filtered 1982-2024 source.
    start_in_range = paired_df["ISO_TIME_S"].dt.year.between(
        START_YEAR, END_YEAR, inclusive="both"
    )
    end_in_range = paired_df["ISO_TIME_E"].dt.year.between(
        START_YEAR, END_YEAR, inclusive="both"
    )
    if not (start_in_range & end_in_range).all():
        raise RuntimeError("A 24-hour pair endpoint lies outside 1982-2024")

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

    paired_df = paired_df.drop(
        columns=["ET_CUM_S", "ET_CUM_E", "IS_ET_S", "OBS_CUM_S", "OBS_CUM_E"]
    )

    if paired_df.duplicated(["SID", "ROW_ID_S", "ROW_ID_E"], keep=False).any():
        raise RuntimeError("Duplicate 24-hour pair keys were generated")

    print(f"Exact 24-hour pairs: {len(paired_df)}")
    return paired_df, target_columns, id_columns


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

    # MSWEP_DIST_30 was filtered before pairing, so both endpoints must exist.
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
    ] + [f"DIFF_{column}" for column in W700_REVERSED_COLUMNS]

    if set(difference_columns).difference(ordered_differences):
        raise RuntimeError("Not all requested difference columns were ordered")

    final_columns = (
        ["SID"]
        + id_columns
        + start_columns
        + end_columns
        + ordered_differences
        + ["COUNT_ET_24H", "OBS_COUNT_24H", "COMPLETE_24H_FLAG"]
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
    print("Starting MSWEP30 + reversed-W700 24-hour processing")
    print("=" * 90)
    print(f"W700 input: {W700_INPUT}")
    print(f"MSWEP threshold-30 input: {MSWEP30_INPUT}")
    print(f"Natural-year range: {START_YEAR}-{END_YEAR}")
    print("Pairing: same SID and E timestamp exactly S + 24 hours")
    print("Rows with missing MSWEP_DIST_30 are removed before pairing")
    print("W700 is multiplied by -1 before E-minus-S differences")

    source_df, stats = read_and_prepare_source()
    paired_df, target_columns, id_columns = build_24h_pairs(source_df)
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
    print(f"Source rows after filters: {stats['SOURCE_ROWS_AFTER_FILTERS']}")
    print(f"24-hour pairs: {stats['ALL_24H_PAIRS']}")
    print(f"Complete 24-hour pairs: {stats['COMPLETE_24H_PAIRS']}")
    print(f"Incomplete 24-hour pairs: {stats['INCOMPLETE_24H_PAIRS']}")
    print(f"ET pairs: {stats['ET_PAIR_COUNT']}")
    print(f"Final output: {OUTPUT_CSV}")
    print(f"Processing summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()

