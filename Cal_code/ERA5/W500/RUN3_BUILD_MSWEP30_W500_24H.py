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
W500_ROOT = PROJECT_ROOT / "Data" / "Processed" / "ERA5" / "W500"
W500_INPUT = W500_ROOT / "Endpoints" / "PRE_DATA_IBT_1982_2024_W500.csv"
MSWEP30_INPUT = (
    PROJECT_ROOT / "Data" / "Processed" / "MSWEP" / "Thresholds"
    / "PRE_DATA_IBT_1982_2024_MSWEP_THRESHOLD_30.csv"
)

OUTPUT_DIR = W500_ROOT / "Windows" / "24H"
OUTPUT_CSV = OUTPUT_DIR / (
    "PRE_DATA_IBT_1982_2024_MSWEP30_W500_24H_SLIDING_CLEANED.csv"
)
LOG_DIR = (
    PROJECT_ROOT / "Results" / "Quality_control" / "ERA5" / "W500" / "RUN3"
)
SUMMARY_CSV = LOG_DIR / "processing_summary.csv"


# ============================================================
# 2. Columns
# ============================================================

W500_ZONES = [
    "0_100", "100_200", "200_300", "300_400",
    "400_500", "0_200", "200_500", "0_500",
]

W500_SOURCE_COLUMNS = [f"W500_AWMEAN_{zone}" for zone in W500_ZONES]
W500_REVERSED_COLUMNS = [f"W500_REVERSED_{zone}" for zone in W500_ZONES]
W500_AREA_COLUMNS = [f"W500_VALID_AREA_{zone}" for zone in W500_ZONES]

COMMON_TARGET_COLUMNS = [
    "ROW_ID", "USA_LAT", "USA_LON", "USA_WIND", "USA_SSHS",
    "ISO_TIME", "RAIN_TIME_ID", "CAL_DIST_REAL", "DIST2LAND",
    "NATURE", "BASIN",
]

OPTIONAL_ID_COLUMNS = ["NAME"]
W500_QC_COLUMNS = ["W500_POINT", "W500_VALID_POINT"] + W500_AREA_COLUMNS
MSWEP_COLUMNS = ["MSWEP_DIST_30", "RAIN_POINT_30"]

DIFFERENCE_SOURCE_COLUMNS = [
    "USA_WIND", "CAL_DIST_REAL", "DIST2LAND", "MSWEP_DIST_30",
] + W500_REVERSED_COLUMNS


# ============================================================
# 3. Helpers
# ============================================================

def write_csv_atomic(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
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
        & df["ISO_TIME"].dt.minute.eq(0)
        & df["ISO_TIME"].dt.second.eq(0)
        & df["ISO_TIME"].dt.microsecond.eq(0)
    )
    if not strict_3hour.all():
        examples = df.loc[
            ~strict_3hour, ["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID"]
        ].head(20).to_dict("records")
        raise ValueError(
            "Merged table contains non-exact 3-hour timestamps: "
            f"count={int((~strict_3hour).sum())}; examples={examples}"
        )

    rain_id = df["RAIN_TIME_ID"].astype("string").str.strip()
    valid_rain_id = rain_id.str.fullmatch(r"\d{7}\.\d{2}", na=False)
    if not valid_rain_id.all():
        examples = df.loc[
            ~valid_rain_id, ["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID"]
        ].head(20).to_dict("records")
        raise ValueError(
            "Merged table contains invalid RAIN_TIME_ID values: "
            f"count={int((~valid_rain_id).sum())}; examples={examples}"
        )

    expected_rain_id = df["ISO_TIME"].dt.strftime("%Y%j.%H")
    mismatch = expected_rain_id != rain_id
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
    df["RAIN_TIME_ID"] = rain_id


# ============================================================
# 4. Read, merge, validate, and reverse W500 sign
# ============================================================

def read_and_prepare_source():
    print("=" * 90)
    print("Reading W500 and MSWEP threshold-30 tables")
    print("=" * 90)

    if not W500_INPUT.exists():
        raise FileNotFoundError(f"W500 input does not exist: {W500_INPUT}")
    if not MSWEP30_INPUT.exists():
        raise FileNotFoundError(
            f"MSWEP threshold-30 input does not exist: {MSWEP30_INPUT}"
        )

    w500_df = pd.read_csv(
        W500_INPUT,
        dtype={"SID": "string", "NAME": "string", "RAIN_TIME_ID": "string"},
        low_memory=False,
    )
    require_columns(
        w500_df,
        ["SID"] + COMMON_TARGET_COLUMNS + W500_SOURCE_COLUMNS + W500_QC_COLUMNS,
        "W500 table",
    )
    validate_row_id(w500_df, "W500 table")
    w500_input_rows = len(w500_df)

    mswep_read_columns = [
        "ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID",
        "MSWEP_DIST_30", "RAIN_POINT_30",
    ]
    mswep_df = pd.read_csv(
        MSWEP30_INPUT,
        usecols=mswep_read_columns,
        dtype={"SID": "string", "RAIN_TIME_ID": "string"},
        low_memory=False,
    )
    require_columns(mswep_df, mswep_read_columns, "MSWEP threshold-30 table")
    validate_row_id(mswep_df, "MSWEP threshold-30 table")
    mswep_input_rows = len(mswep_df)

    mswep_df = mswep_df.rename(
        columns={
            "SID": "MSWEP_SID_CHECK",
            "ISO_TIME": "MSWEP_ISO_TIME_CHECK",
            "RAIN_TIME_ID": "MSWEP_RAIN_TIME_ID_CHECK",
        }
    )

    merged_df = w500_df.merge(
        mswep_df,
        on="ROW_ID",
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )

    merged_df["ISO_TIME"] = pd.to_datetime(merged_df["ISO_TIME"], errors="coerce")
    if merged_df["ISO_TIME"].isna().any():
        raise ValueError("W500 table contains invalid ISO_TIME values")

    year_mask = merged_df["ISO_TIME"].dt.year.between(
        START_YEAR, END_YEAR, inclusive="both"
    )
    in_range_unmatched = year_mask & merged_df["_merge"].ne("both")
    if in_range_unmatched.any():
        examples = merged_df.loc[
            in_range_unmatched, ["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID"]
        ].head(20).to_dict("records")
        raise ValueError(
            f"Some {START_YEAR}-{END_YEAR} W500 rows have no MSWEP match: "
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
        source_df["RAIN_TIME_ID"].astype("string").str.strip()
        != source_df["MSWEP_RAIN_TIME_ID_CHECK"].astype("string").str.strip()
    )
    if identity_mismatch.any():
        examples = source_df.loc[
            identity_mismatch,
            [
                "ROW_ID", "SID", "MSWEP_SID_CHECK", "ISO_TIME",
                "MSWEP_ISO_TIME_CHECK", "RAIN_TIME_ID",
                "MSWEP_RAIN_TIME_ID_CHECK",
            ],
        ].head(20).to_dict("records")
        raise ValueError(
            "W500 and MSWEP identities disagree for matching ROW_ID values: "
            f"count={int(identity_mismatch.sum())}; examples={examples}"
        )

    source_df = source_df.drop(
        columns=[
            "_merge", "MSWEP_SID_CHECK", "MSWEP_ISO_TIME_CHECK",
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
            "USA_LAT", "USA_LON", "USA_WIND", "USA_SSHS",
            "CAL_DIST_REAL", "DIST2LAND", "MSWEP_DIST_30",
            "RAIN_POINT_30", "W500_POINT", "W500_VALID_POINT",
        ] + W500_SOURCE_COLUMNS + W500_AREA_COLUMNS
    ))
    for column in numeric_columns:
        source_df[column] = pd.to_numeric(source_df[column], errors="coerce")
    source_df[numeric_columns] = source_df[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    # MSWEP_DIST_30 may be NaN only when RAIN_POINT_30 equals zero.
    required_numeric = [
        "USA_LAT", "USA_LON", "USA_WIND", "USA_SSHS",
        "CAL_DIST_REAL", "DIST2LAND", "RAIN_POINT_30",
        "W500_POINT", "W500_VALID_POINT",
    ] + W500_SOURCE_COLUMNS + W500_AREA_COLUMNS
    invalid_required = source_df[required_numeric].isna().any(axis=1)
    if invalid_required.any():
        examples = source_df.loc[
            invalid_required,
            ["ROW_ID", "SID", "ISO_TIME"] + required_numeric,
        ].head(10).to_dict("records")
        raise ValueError(
            f"{START_YEAR}-{END_YEAR} source contains missing required metrics: "
            f"count={int(invalid_required.sum())}; examples={examples}"
        )

    if (source_df[W500_AREA_COLUMNS] < 0.0).any().any():
        raise ValueError("Negative W500 valid areas were found")
    if (source_df[["W500_POINT", "W500_VALID_POINT"]] < 0).any().any():
        raise ValueError("Negative W500 point counts were found")
    if (source_df["W500_VALID_POINT"] > source_df["W500_POINT"]).any():
        raise ValueError("W500_VALID_POINT exceeds W500_POINT")
    if (source_df["RAIN_POINT_30"] < 0).any():
        raise ValueError("Negative RAIN_POINT_30 values were found")

    non_integer_rain_points = ~np.isclose(
        source_df["RAIN_POINT_30"],
        np.round(source_df["RAIN_POINT_30"]),
        rtol=0.0,
        atol=1e-9,
    )
    if non_integer_rain_points.any():
        raise ValueError(
            "RAIN_POINT_30 contains non-integer values: "
            f"{int(non_integer_rain_points.sum())}"
        )
    source_df["RAIN_POINT_30"] = np.round(
        source_df["RAIN_POINT_30"]
    ).astype(np.int64)

    source_df["HAS_MSWEP_RAIN30"] = source_df["RAIN_POINT_30"].gt(0)
    distance_available = source_df["MSWEP_DIST_30"].notna()

    zero_points_with_distance = source_df["RAIN_POINT_30"].eq(0) & distance_available
    if zero_points_with_distance.any():
        raise ValueError(
            "MSWEP_DIST_30 exists when RAIN_POINT_30 is zero: "
            f"{int(zero_points_with_distance.sum())}"
        )

    positive_points_without_distance = (
        source_df["RAIN_POINT_30"].gt(0) & ~distance_available
    )
    if positive_points_without_distance.any():
        raise ValueError(
            "MSWEP_DIST_30 is missing when RAIN_POINT_30 is positive: "
            f"{int(positive_points_without_distance.sum())}"
        )

    if source_df.loc[distance_available, "MSWEP_DIST_30"].lt(0.0).any():
        raise ValueError("Negative MSWEP_DIST_30 values were found")

    for source_column, reversed_column in zip(
        W500_SOURCE_COLUMNS, W500_REVERSED_COLUMNS
    ):
        source_df[reversed_column] = -source_df[source_column]
    source_df = source_df.drop(columns=W500_SOURCE_COLUMNS)

    rows_with_rain30 = int(source_df["HAS_MSWEP_RAIN30"].sum())
    rows_without_rain30 = int((~source_df["HAS_MSWEP_RAIN30"]).sum())
    stats = {
        "W500_INPUT_ROWS": w500_input_rows,
        "MSWEP_INPUT_ROWS": mswep_input_rows,
        "YEAR_FILTER_START": START_YEAR,
        "YEAR_FILTER_END": END_YEAR,
        "SOURCE_ROWS_1982_2024": len(source_df),
        "ROWS_WITH_MSWEP_RAIN30": rows_with_rain30,
        "ROWS_WITHOUT_MSWEP_RAIN30": rows_without_rain30,
    }

    print(f"W500 input rows: {w500_input_rows}")
    print(f"MSWEP input rows: {mswep_input_rows}")
    print(f"Rows in {START_YEAR}-{END_YEAR}: {len(source_df)}")
    print(f"Rows with MSWEP threshold-30 rainfall: {rows_with_rain30}")
    print(f"Rows without MSWEP threshold-30 rainfall: {rows_without_rain30}")
    print("No rows were removed before 24-hour pairing")
    print("W500 sign reversed: positive values represent upward motion")
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

    normalized_nature = (
        working_df["NATURE"].fillna("").astype(str).str.strip().str.upper()
    )
    working_df["IS_ET"] = normalized_nature.eq("ET").astype(np.int8)
    working_df["IS_ER"] = normalized_nature.eq("ER").astype(np.int8)
    working_df["IS_MSWEP_RAIN30"] = (
        working_df["HAS_MSWEP_RAIN30"].astype(np.int8)
    )

    group = working_df.groupby("SID", sort=False)
    working_df["ET_CUM"] = group["IS_ET"].cumsum().astype(np.int32)
    working_df["ER_CUM"] = group["IS_ER"].cumsum().astype(np.int32)
    working_df["RAIN30_CUM"] = group["IS_MSWEP_RAIN30"].cumsum().astype(np.int32)
    working_df["OBS_CUM"] = group.cumcount().add(1).astype(np.int32)

    id_columns = [c for c in OPTIONAL_ID_COLUMNS if c in working_df.columns]
    target_columns = list(dict.fromkeys(
        COMMON_TARGET_COLUMNS
        + W500_QC_COLUMNS
        + MSWEP_COLUMNS
        + W500_REVERSED_COLUMNS
    ))

    start_extra = [
        "ET_CUM", "ER_CUM", "RAIN30_CUM", "OBS_CUM",
        "IS_ET", "IS_ER", "IS_MSWEP_RAIN30",
    ]
    start_df = working_df[
        ["SID"] + id_columns + target_columns + start_extra
    ].copy()
    start_df["TIME_TARGET_E"] = start_df["ISO_TIME"] + pd.Timedelta(
        hours=TIME_INTERVAL_HOURS
    )
    start_df = start_df.rename(
        columns={c: f"{c}_S" for c in target_columns + start_extra}
    )

    end_extra = ["ET_CUM", "ER_CUM", "RAIN30_CUM", "OBS_CUM"]
    end_df = working_df[["SID"] + target_columns + end_extra].copy()
    end_df = end_df.rename(
        columns={c: f"{c}_E" for c in target_columns + end_extra}
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
        paired_df["ET_CUM_E"] - paired_df["ET_CUM_S"] + paired_df["IS_ET_S"]
    ).astype(np.int16)
    paired_df["COUNT_ER_24H"] = (
        paired_df["ER_CUM_E"] - paired_df["ER_CUM_S"] + paired_df["IS_ER_S"]
    ).astype(np.int16)
    paired_df["COUNT_MSWEP_RAIN30_24H"] = (
        paired_df["RAIN30_CUM_E"]
        - paired_df["RAIN30_CUM_S"]
        + paired_df["IS_MSWEP_RAIN30_S"]
    ).astype(np.int16)
    paired_df["OBS_COUNT_24H"] = (
        paired_df["OBS_CUM_E"] - paired_df["OBS_CUM_S"] + 1
    ).astype(np.int16)

    # Informational flags only; they do not filter samples.
    paired_df["COMPLETE_24H_FLAG"] = paired_df["OBS_COUNT_24H"].eq(9)
    paired_df["ANY_MSWEP_RAIN30_24H_FLAG"] = (
        paired_df["COUNT_MSWEP_RAIN30_24H"].gt(0)
    )
    paired_df["ALL_MSWEP_RAIN30_24H_FLAG"] = (
        paired_df["COUNT_MSWEP_RAIN30_24H"].eq(9)
    )
    paired_df["MSWEP_RAIN30_S_FLAG"] = paired_df["RAIN_POINT_30_S"].gt(0)
    paired_df["MSWEP_RAIN30_E_FLAG"] = paired_df["RAIN_POINT_30_E"].gt(0)
    paired_df["MSWEP_ENDPOINTS_VALID_FLAG"] = (
        paired_df["MSWEP_RAIN30_S_FLAG"] & paired_df["MSWEP_RAIN30_E_FLAG"]
    )

    paired_df = paired_df.drop(
        columns=[
            "ET_CUM_S", "ET_CUM_E", "ER_CUM_S", "ER_CUM_E",
            "RAIN30_CUM_S", "RAIN30_CUM_E", "OBS_CUM_S", "OBS_CUM_E",
            "IS_ET_S", "IS_ER_S", "IS_MSWEP_RAIN30_S",
        ]
    )

    if paired_df.duplicated(["SID", "ROW_ID_S", "ROW_ID_E"], keep=False).any():
        raise RuntimeError("Duplicate 24-hour pair keys were generated")

    all_exact_pairs = len(paired_df)

    # Scheme A: only the two endpoints must have threshold-30 rainfall.
    # Completeness, intermediate rain, ET, and ER do not filter the sample.
    paired_df = paired_df.loc[
        paired_df["MSWEP_ENDPOINTS_VALID_FLAG"]
    ].copy().reset_index(drop=True)

    endpoint_valid_pairs = len(paired_df)
    if paired_df[["MSWEP_DIST_30_S", "MSWEP_DIST_30_E"]].isna().any().any():
        raise RuntimeError(
            "A retained endpoint-valid pair contains missing MSWEP_DIST_30"
        )

    pairing_stats = {
        "ALL_EXACT_24H_PAIRS_BEFORE_ENDPOINT_FILTER": all_exact_pairs,
        "PAIRS_REMOVED_WITHOUT_BOTH_RAIN30_ENDPOINTS": (
            all_exact_pairs - endpoint_valid_pairs
        ),
        "ENDPOINT_VALID_24H_PAIRS": endpoint_valid_pairs,
    }

    print(f"All exact 24-hour pairs before endpoint filter: {all_exact_pairs}")
    print(
        "Pairs removed because S or E has no threshold-30 rainfall: "
        f"{all_exact_pairs - endpoint_valid_pairs}"
    )
    print(f"Scheme-A endpoint-valid 24-hour pairs retained: {endpoint_valid_pairs}")
    print("Incomplete windows are retained; COMPLETE_24H_FLAG is informational only")
    print("ET and ER windows are retained")
    return paired_df, target_columns, id_columns, pairing_stats


# ============================================================
# 6. Calculate E-minus-S differences and arrange columns
# ============================================================

def calculate_and_arrange_differences(paired_df, target_columns, id_columns):
    paired_df["DIFF_ISO_TIME"] = (
        (paired_df["ISO_TIME_E"] - paired_df["ISO_TIME_S"])
        .dt.total_seconds()
        .div(3600.0)
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

    # CAL_DIST_REAL differences are output only and never filter the sample.
    for column in DIFFERENCE_SOURCE_COLUMNS:
        paired_df[f"DIFF_{column}"] = (
            paired_df[f"{column}_E"] - paired_df[f"{column}_S"]
        )

    paired_df["DIFF_CAL_DIST_REAL_ABS"] = paired_df["DIFF_CAL_DIST_REAL"].abs()
    paired_df["DIFF_DIST2LAND_ABS"] = paired_df["DIFF_DIST2LAND"].abs()

    start_columns = [f"{column}_S" for column in target_columns]
    end_columns = [f"{column}_E" for column in target_columns]
    ordered_differences = [
        "DIFF_USA_WIND",
        "DIFF_ISO_TIME",
        "DIFF_CAL_DIST_REAL",
        "DIFF_CAL_DIST_REAL_ABS",
        "DIFF_DIST2LAND",
        "DIFF_DIST2LAND_ABS",
        "DIFF_MSWEP_DIST_30",
    ] + [f"DIFF_{column}" for column in W500_REVERSED_COLUMNS]

    requested = {f"DIFF_{column}" for column in DIFFERENCE_SOURCE_COLUMNS}
    unordered = requested.difference(ordered_differences)
    if unordered:
        raise RuntimeError(
            "Not all requested difference columns were ordered: "
            + ", ".join(sorted(unordered))
        )

    quality_columns = [
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

    final_columns = (
        ["SID"]
        + id_columns
        + start_columns
        + end_columns
        + ordered_differences
        + quality_columns
    )
    missing = set(final_columns).difference(paired_df.columns)
    if missing:
        raise KeyError(
            "Final output is missing columns: " + ", ".join(sorted(missing))
        )

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
    print("Starting MSWEP30 + reversed-W500 24-hour processing")
    print("=" * 90)
    print(f"W500 input: {W500_INPUT}")
    print(f"MSWEP threshold-30 input: {MSWEP30_INPUT}")
    print(f"Natural-year range: {START_YEAR}-{END_YEAR}")
    print("Pairing: same SID and E timestamp exactly S + 24 hours")
    print("Scheme A: both endpoints must have MSWEP threshold-30 rainfall")
    print("Incomplete 24-hour windows are retained")
    print("ET and ER records are retained")
    print("W500 is multiplied by -1 before E-minus-S differences")

    source_df, stats = read_and_prepare_source()
    paired_df, target_columns, id_columns, pairing_stats = build_24h_pairs(
        source_df
    )
    stats.update(pairing_stats)
    final_df = calculate_and_arrange_differences(
        paired_df, target_columns, id_columns
    )

    stats.update(
        {
            "FINAL_24H_PAIRS": len(final_df),
            "COMPLETE_24H_PAIRS": int(final_df["COMPLETE_24H_FLAG"].sum()),
            "INCOMPLETE_24H_PAIRS": int((~final_df["COMPLETE_24H_FLAG"]).sum()),
            "MIN_OBS_COUNT_24H": (
                int(final_df["OBS_COUNT_24H"].min()) if len(final_df) else np.nan
            ),
            "MAX_OBS_COUNT_24H": (
                int(final_df["OBS_COUNT_24H"].max()) if len(final_df) else np.nan
            ),
            "ET_PAIR_COUNT": int(final_df["COUNT_ET_24H"].gt(0).sum()),
            "ER_PAIR_COUNT": int(final_df["COUNT_ER_24H"].gt(0).sum()),
            "ALL_RAIN30_24H_PAIR_COUNT": int(
                final_df["ALL_MSWEP_RAIN30_24H_FLAG"].sum()
            ),
            "OUTPUT_PATH": str(OUTPUT_CSV),
        }
    )

    print(f"Writing final 24-hour table: {OUTPUT_CSV}")
    write_csv_atomic(final_df, OUTPUT_CSV)
    write_csv_atomic(pd.DataFrame([stats]), SUMMARY_CSV)

    print("\n" + "=" * 90)
    print("Processing completed successfully")
    print("=" * 90)
    print(f"Complete source rows: {stats['SOURCE_ROWS_1982_2024']}")
    print(f"Rows with MSWEP threshold-30 rainfall: {stats['ROWS_WITH_MSWEP_RAIN30']}")
    print(f"Rows without threshold-30 rainfall: {stats['ROWS_WITHOUT_MSWEP_RAIN30']}")
    print(
        "All exact pairs before endpoint filter: "
        f"{stats['ALL_EXACT_24H_PAIRS_BEFORE_ENDPOINT_FILTER']}"
    )
    print(f"Final Scheme-A pairs: {stats['FINAL_24H_PAIRS']}")
    print(f"Complete pairs: {stats['COMPLETE_24H_PAIRS']}")
    print(f"Incomplete pairs retained: {stats['INCOMPLETE_24H_PAIRS']}")
    print(f"Pairs containing ET: {stats['ET_PAIR_COUNT']}")
    print(f"Pairs containing ER: {stats['ER_PAIR_COUNT']}")
    print(f"Final output: {OUTPUT_CSV}")
    print(f"Processing summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
