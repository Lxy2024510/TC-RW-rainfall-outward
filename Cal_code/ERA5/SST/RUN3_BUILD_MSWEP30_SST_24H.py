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
SST_ROOT = PROJECT_ROOT / "Data" / "Processed" / "ERA5" / "SST"
SST_INPUT = (
    SST_ROOT / "Endpoints"
    / "PRE_DATA_IBT_1982_2024_SST_200_800KM.csv"
)
MSWEP30_INPUT = (
    PROJECT_ROOT / "Data" / "Processed" / "MSWEP" / "Thresholds"
    / "PRE_DATA_IBT_1982_2024_MSWEP_THRESHOLD_30.csv"
)

OUTPUT_DIR = SST_ROOT / "Windows" / "24H"
OUTPUT_CSV = OUTPUT_DIR / (
    "PRE_DATA_IBT_1982_2024_MSWEP30_SST_200_800KM_24H_SLIDING_CLEANED.csv"
)
LOG_DIR = (
    PROJECT_ROOT / "Results" / "Quality_control" / "ERA5" / "SST" / "RUN3"
)
SUMMARY_CSV = LOG_DIR / "processing_summary.csv"
MISSING_SST_PAIR_LOG = LOG_DIR / "pairs_removed_missing_sst_endpoint.csv"
MISSING_MSWEP_PAIR_LOG = LOG_DIR / "pairs_removed_missing_mswep_endpoint.csv"
NO_VALID_SST_SOURCE_LOG = LOG_DIR / "no_valid_sst_source_records.csv"


# ============================================================
# 2. Columns
# ============================================================

SST_SOURCE_COLUMN = "SST_AWMEAN_200_800"
SST_POINT_COLUMN = "SST_VALID_POINT_200_800"
SST_AREA_COLUMN = "SST_VALID_AREA_200_800"
SST_AREA_FRACTION_COLUMN = "SST_VALID_AREA_FRACTION_200_800"

COMMON_TARGET_COLUMNS = [
    "ROW_ID", "USA_LAT", "USA_LON", "USA_WIND", "USA_SSHS",
    "ISO_TIME", "RAIN_TIME_ID", "CAL_DIST_REAL", "DIST2LAND",
    "NATURE", "BASIN",
]
OPTIONAL_ID_COLUMNS = ["NAME"]
SST_QC_COLUMNS = [
    SST_POINT_COLUMN, SST_AREA_COLUMN, SST_AREA_FRACTION_COLUMN,
]
DIFFERENCE_SOURCE_COLUMNS = [
    "USA_WIND", "CAL_DIST_REAL", "DIST2LAND", "MSWEP_DIST_30",
    SST_SOURCE_COLUMN,
]


# ============================================================
# 3. Helpers
# ============================================================

def write_csv_atomic(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    try:
        df.to_csv(
            temporary_path, index=False, date_format="%Y-%m-%d %H:%M:%S"
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
            f"{table_name} is missing required columns: " + ", ".join(sorted(missing))
        )


def validate_row_id(df, table_name):
    df["ROW_ID"] = pd.to_numeric(df["ROW_ID"], errors="coerce")
    if df["ROW_ID"].isna().any():
        raise ValueError(f"{table_name} contains invalid ROW_ID values")
    df["ROW_ID"] = df["ROW_ID"].astype(np.int64)
    if (df["ROW_ID"] < 0).any() or df["ROW_ID"].duplicated().any():
        raise ValueError(f"{table_name} contains negative or duplicate ROW_ID values")


def validate_time_fields(df):
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    if df["ISO_TIME"].isna().any():
        raise ValueError("Merged table contains invalid ISO_TIME values")
    exact = (
        df["ISO_TIME"].dt.hour.isin(VALID_3H_HOURS)
        & df["ISO_TIME"].dt.minute.eq(0)
        & df["ISO_TIME"].dt.second.eq(0)
        & df["ISO_TIME"].dt.microsecond.eq(0)
    )
    if not exact.all():
        raise ValueError(f"Non-exact 3-hour timestamps: {int((~exact).sum())}")
    rain = df["RAIN_TIME_ID"].astype("string")
    if not rain.str.fullmatch(r"\d{7}\.\d{2}", na=False).all():
        raise ValueError("Invalid RAIN_TIME_ID values")
    expected = df["ISO_TIME"].dt.strftime("%Y%j.%H")
    if (expected != rain.astype(str)).any():
        raise ValueError("ISO_TIME and RAIN_TIME_ID mismatch")


# ============================================================
# 4. Read, merge, and validate complete source timeline
# ============================================================

def read_and_prepare_source():
    print("=" * 90)
    print("Reading SST and MSWEP threshold-30 tables")
    print("=" * 90)
    if not SST_INPUT.exists():
        raise FileNotFoundError(f"SST input does not exist: {SST_INPUT}")
    if not MSWEP30_INPUT.exists():
        raise FileNotFoundError(f"MSWEP input does not exist: {MSWEP30_INPUT}")

    sst_df = pd.read_csv(
        SST_INPUT,
        dtype={"SID": "string", "NAME": "string", "RAIN_TIME_ID": "string"},
        low_memory=False,
    )
    require_columns(
        sst_df,
        ["SID", SST_SOURCE_COLUMN] + COMMON_TARGET_COLUMNS + SST_QC_COLUMNS,
        "SST table",
    )
    validate_row_id(sst_df, "SST table")

    mswep_df = pd.read_csv(
        MSWEP30_INPUT,
        usecols=["ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID", "MSWEP_DIST_30"],
        dtype={"SID": "string", "RAIN_TIME_ID": "string"},
        low_memory=False,
    )
    validate_row_id(mswep_df, "MSWEP table")
    mswep_df = mswep_df.rename(columns={
        "SID": "MSWEP_SID_CHECK",
        "ISO_TIME": "MSWEP_ISO_TIME_CHECK",
        "RAIN_TIME_ID": "MSWEP_RAIN_TIME_ID_CHECK",
    })

    merged = sst_df.merge(
        mswep_df, on="ROW_ID", how="left", validate="one_to_one",
        sort=False, indicator=True,
    )
    merged["ISO_TIME"] = pd.to_datetime(merged["ISO_TIME"], errors="coerce")
    if merged["ISO_TIME"].isna().any():
        raise ValueError("SST table contains invalid ISO_TIME values")
    year_mask = merged["ISO_TIME"].dt.year.between(
        START_YEAR, END_YEAR, inclusive="both"
    )
    unmatched = year_mask & merged["_merge"].ne("both")
    if unmatched.any():
        raise ValueError(
            f"SST rows without MSWEP match: {int(unmatched.sum())}"
        )
    source = merged.loc[year_mask].copy()

    check_time = pd.to_datetime(source["MSWEP_ISO_TIME_CHECK"], errors="coerce")
    mismatch = (
        source["SID"].astype("string").ne(source["MSWEP_SID_CHECK"].astype("string"))
        | source["ISO_TIME"].ne(check_time)
        | source["RAIN_TIME_ID"].astype("string").ne(
            source["MSWEP_RAIN_TIME_ID_CHECK"].astype("string")
        )
    )
    if mismatch.any():
        raise ValueError(f"SST/MSWEP identity mismatches: {int(mismatch.sum())}")
    source = source.drop(columns=[
        "_merge", "MSWEP_SID_CHECK", "MSWEP_ISO_TIME_CHECK",
        "MSWEP_RAIN_TIME_ID_CHECK",
    ])
    validate_time_fields(source)
    if source.duplicated(["SID", "ISO_TIME"]).any():
        raise ValueError("Duplicate (SID, ISO_TIME) source keys")

    numeric = list(dict.fromkeys([
        "USA_LAT", "USA_LON", "USA_WIND", "USA_SSHS", "CAL_DIST_REAL",
        "DIST2LAND", "MSWEP_DIST_30", SST_SOURCE_COLUMN,
    ] + SST_QC_COLUMNS))
    for column in numeric:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source[numeric] = source[numeric].replace([np.inf, -np.inf], np.nan)

    required_base = [
        "USA_LAT", "USA_LON", "USA_WIND", "USA_SSHS",
        "CAL_DIST_REAL", "DIST2LAND",
    ]
    if source[required_base].isna().any(axis=1).any():
        raise ValueError("Missing required BASE metrics")
    if source[SST_QC_COLUMNS].isna().any(axis=1).any():
        raise ValueError("Missing SST QC metrics")
    if (source[SST_POINT_COLUMN] < 0).any() or (source[SST_AREA_COLUMN] < 0).any():
        raise ValueError("Negative SST point counts or areas")
    fraction = source[SST_AREA_FRACTION_COLUMN]
    if ((fraction < 0) | (fraction > 1)).any():
        raise ValueError("SST valid-area fractions outside [0, 1]")

    source["SST_SOURCE_VALID_FLAG"] = (
        source[SST_SOURCE_COLUMN].notna()
        & source[SST_POINT_COLUMN].gt(0)
        & source[SST_AREA_COLUMN].gt(0)
        & source[SST_AREA_FRACTION_COLUMN].gt(0)
    )
    invalid = ~source["SST_SOURCE_VALID_FLAG"]
    inconsistent = invalid & (
        source[SST_SOURCE_COLUMN].notna()
        | source[SST_POINT_COLUMN].ne(0)
        | source[SST_AREA_COLUMN].ne(0)
        | source[SST_AREA_FRACTION_COLUMN].ne(0)
    )
    if inconsistent.any():
        raise ValueError(f"Inconsistent invalid-SST records: {int(inconsistent.sum())}")
    valid_sst = source.loc[source["SST_SOURCE_VALID_FLAG"], SST_SOURCE_COLUMN]
    if ((valid_sst < 150) | (valid_sst > 350)).any():
        raise ValueError("Valid SST means outside 150-350 K")

    no_valid_columns = [
        "ROW_ID", "SID", "ISO_TIME", "RAIN_TIME_ID", "USA_LAT", "USA_LON",
        SST_SOURCE_COLUMN, SST_POINT_COLUMN, SST_AREA_COLUMN,
        SST_AREA_FRACTION_COLUMN, "SST_SOURCE_VALID_FLAG",
    ]
    write_csv_atomic(source.loc[invalid, no_valid_columns], NO_VALID_SST_SOURCE_LOG)

    stats = {
        "SST_INPUT_ROWS": len(sst_df),
        "MSWEP_INPUT_ROWS": len(mswep_df),
        "SOURCE_ROWS": len(source),
        "NO_VALID_SST_SOURCE_ROWS": int(invalid.sum()),
        "VALID_MSWEP_SOURCE_ROWS": int(source["MSWEP_DIST_30"].notna().sum()),
        "MISSING_MSWEP_SOURCE_ROWS": int(source["MSWEP_DIST_30"].isna().sum()),
    }
    print(f"Source rows: {len(source)}")
    print(f"No-valid-SST source rows: {int(invalid.sum())}")
    return source, stats


# ============================================================
# 5. Build exact 24-hour endpoint pairs and exclusion logs
# ============================================================

def build_24h_pairs(source):
    print("\n" + "=" * 90)
    print("Building exact 24-hour endpoint pairs")
    print("=" * 90)
    work = source.sort_values(["SID", "ISO_TIME", "ROW_ID"]).reset_index(drop=True)
    work["IS_ET"] = (
        work["NATURE"].fillna("").astype(str).str.strip().str.upper().eq("ET")
    ).astype(np.int8)
    group = work.groupby("SID", sort=False)
    work["ET_CUM"] = group["IS_ET"].cumsum().astype(np.int32)
    work["OBS_CUM"] = group.cumcount().add(1).astype(np.int32)
    work["SST_VALID_INT"] = work["SST_SOURCE_VALID_FLAG"].astype(np.int8)
    work["SST_FOR_SUM"] = work[SST_SOURCE_COLUMN].where(
        work["SST_SOURCE_VALID_FLAG"], 0.0
    )
    group = work.groupby("SID", sort=False)
    work["SST_COUNT_CUM"] = group["SST_VALID_INT"].cumsum().astype(np.int32)
    work["SST_SUM_CUM"] = group["SST_FOR_SUM"].cumsum()

    ids = [c for c in OPTIONAL_ID_COLUMNS if c in work.columns]
    targets = list(dict.fromkeys(
        COMMON_TARGET_COLUMNS + SST_QC_COLUMNS +
        ["MSWEP_DIST_30", SST_SOURCE_COLUMN, "SST_SOURCE_VALID_FLAG"]
    ))
    internals_s = [
        "ET_CUM", "IS_ET", "OBS_CUM", "SST_VALID_INT",
        "SST_COUNT_CUM", "SST_SUM_CUM",
    ]
    internals_e = ["ET_CUM", "OBS_CUM", "SST_COUNT_CUM", "SST_SUM_CUM"]

    start = work[["SID"] + ids + targets + internals_s].copy()
    start["TIME_TARGET_E"] = start["ISO_TIME"] + pd.Timedelta(hours=24)
    start = start.rename(columns={c: f"{c}_S" for c in targets + internals_s})
    end = work[["SID"] + targets + internals_e].copy()
    end = end.rename(columns={c: f"{c}_E" for c in targets + internals_e})
    pairs = start.merge(
        end, left_on=["SID", "TIME_TARGET_E"],
        right_on=["SID", "ISO_TIME_E"], how="inner",
        validate="one_to_one", sort=False,
    ).drop(columns="TIME_TARGET_E")

    pairs["COUNT_ET_24H"] = (
        pairs["ET_CUM_E"] - pairs["ET_CUM_S"] + pairs["IS_ET_S"]
    ).astype(np.int16)
    pairs["OBS_COUNT_24H"] = (
        pairs["OBS_CUM_E"] - pairs["OBS_CUM_S"] + 1
    ).astype(np.int16)
    if (pairs["OBS_COUNT_24H"] > 9).any():
        raise RuntimeError("A 24-hour window contains more than nine observations")
    pairs["COMPLETE_24H_FLAG"] = pairs["OBS_COUNT_24H"].eq(9)
    pairs["SST_VALID_COUNT_24H"] = (
        pairs["SST_COUNT_CUM_E"] - pairs["SST_COUNT_CUM_S"]
        + pairs["SST_VALID_INT_S"]
    ).astype(np.int16)
    sst_sum = (
        pairs["SST_SUM_CUM_E"] - pairs["SST_SUM_CUM_S"]
        + pairs[SST_SOURCE_COLUMN + "_S"].where(pairs["SST_VALID_INT_S"].eq(1), 0)
    )
    pairs["MEAN24H_SST_AWMEAN_200_800"] = sst_sum / pairs[
        "SST_VALID_COUNT_24H"
    ].replace(0, np.nan)
    pairs["COMPLETE_SST_24H_FLAG"] = (
        pairs["COMPLETE_24H_FLAG"] & pairs["SST_VALID_COUNT_24H"].eq(9)
    )

    pairs["MISSING_MSWEP_START_FLAG"] = pairs["MSWEP_DIST_30_S"].isna()
    pairs["MISSING_MSWEP_END_FLAG"] = pairs["MSWEP_DIST_30_E"].isna()
    pairs["MISSING_MSWEP_ENDPOINT_FLAG"] = (
        pairs["MISSING_MSWEP_START_FLAG"] | pairs["MISSING_MSWEP_END_FLAG"]
    )
    pairs["MISSING_SST_START_FLAG"] = ~pairs["SST_SOURCE_VALID_FLAG_S"]
    pairs["MISSING_SST_END_FLAG"] = ~pairs["SST_SOURCE_VALID_FLAG_E"]
    pairs["MISSING_SST_ENDPOINT_FLAG"] = (
        pairs["MISSING_SST_START_FLAG"] | pairs["MISSING_SST_END_FLAG"]
    )

    log_base = [
        "SID", "ROW_ID_S", "ROW_ID_E", "ISO_TIME_S", "ISO_TIME_E",
        "USA_LAT_S", "USA_LON_S", "USA_LAT_E", "USA_LON_E",
        "MSWEP_DIST_30_S", "MSWEP_DIST_30_E",
        SST_SOURCE_COLUMN + "_S", SST_SOURCE_COLUMN + "_E",
        SST_POINT_COLUMN + "_S", SST_POINT_COLUMN + "_E",
    ]
    sst_flags = [
        "MISSING_SST_START_FLAG", "MISSING_SST_END_FLAG",
        "MISSING_SST_ENDPOINT_FLAG", "MISSING_MSWEP_ENDPOINT_FLAG",
    ]
    mswep_flags = [
        "MISSING_MSWEP_START_FLAG", "MISSING_MSWEP_END_FLAG",
        "MISSING_MSWEP_ENDPOINT_FLAG", "MISSING_SST_ENDPOINT_FLAG",
    ]
    write_csv_atomic(
        pairs.loc[pairs["MISSING_SST_ENDPOINT_FLAG"], log_base + sst_flags],
        MISSING_SST_PAIR_LOG,
    )
    write_csv_atomic(
        pairs.loc[pairs["MISSING_MSWEP_ENDPOINT_FLAG"], log_base + mswep_flags],
        MISSING_MSWEP_PAIR_LOG,
    )

    stats = {
        "ALL_EXACT_24H_PAIRS": len(pairs),
        "PAIRS_MISSING_SST_START": int(pairs["MISSING_SST_START_FLAG"].sum()),
        "PAIRS_MISSING_SST_END": int(pairs["MISSING_SST_END_FLAG"].sum()),
        "PAIRS_MISSING_SST_ANY_ENDPOINT": int(pairs["MISSING_SST_ENDPOINT_FLAG"].sum()),
        "PAIRS_MISSING_SST_BOTH_ENDPOINTS": int((
            pairs["MISSING_SST_START_FLAG"] & pairs["MISSING_SST_END_FLAG"]
        ).sum()),
        "PAIRS_MISSING_MSWEP_ANY_ENDPOINT": int(
            pairs["MISSING_MSWEP_ENDPOINT_FLAG"].sum()
        ),
        "PAIRS_MISSING_BOTH_SST_AND_MSWEP": int((
            pairs["MISSING_SST_ENDPOINT_FLAG"]
            & pairs["MISSING_MSWEP_ENDPOINT_FLAG"]
        ).sum()),
    }
    keep = ~(
        pairs["MISSING_SST_ENDPOINT_FLAG"]
        | pairs["MISSING_MSWEP_ENDPOINT_FLAG"]
    )
    pairs = pairs.loc[keep].copy().reset_index(drop=True)
    if pairs.empty:
        raise RuntimeError("No pairs remain after endpoint filters")
    stats["PAIRS_RETAINED"] = len(pairs)

    drop_internal = [
        "ET_CUM_S", "ET_CUM_E", "IS_ET_S", "OBS_CUM_S", "OBS_CUM_E",
        "SST_VALID_INT_S", "SST_COUNT_CUM_S", "SST_COUNT_CUM_E",
        "SST_SUM_CUM_S", "SST_SUM_CUM_E",
        "MISSING_MSWEP_START_FLAG", "MISSING_MSWEP_END_FLAG",
        "MISSING_MSWEP_ENDPOINT_FLAG", "MISSING_SST_START_FLAG",
        "MISSING_SST_END_FLAG", "MISSING_SST_ENDPOINT_FLAG",
    ]
    pairs = pairs.drop(columns=drop_internal)
    print(f"All exact pairs: {stats['ALL_EXACT_24H_PAIRS']}")
    print(f"Pairs removed for SST endpoint: {stats['PAIRS_MISSING_SST_ANY_ENDPOINT']}")
    print(f"Pairs removed for MSWEP endpoint: {stats['PAIRS_MISSING_MSWEP_ANY_ENDPOINT']}")
    print(f"Pairs retained: {len(pairs)}")
    return pairs, targets, ids, stats


# ============================================================
# 6. Differences and output arrangement
# ============================================================

def calculate_and_arrange(pairs, targets, ids):
    pairs["DIFF_ISO_TIME"] = (
        pairs["ISO_TIME_E"] - pairs["ISO_TIME_S"]
    ).dt.total_seconds() / 3600
    if not np.isclose(pairs["DIFF_ISO_TIME"], 24.0, atol=1e-9, rtol=0).all():
        raise RuntimeError("Non-24-hour pairs were produced")
    for column in DIFFERENCE_SOURCE_COLUMNS:
        pairs[f"DIFF_{column}"] = pairs[f"{column}_E"] - pairs[f"{column}_S"]
    pairs["DIFF_CAL_DIST_REAL_ABS"] = pairs["DIFF_CAL_DIST_REAL"].abs()
    pairs["DIFF_DIST2LAND_ABS"] = pairs["DIFF_DIST2LAND"].abs()

    ordered_diff = [
        "DIFF_USA_WIND", "DIFF_ISO_TIME", "DIFF_CAL_DIST_REAL",
        "DIFF_CAL_DIST_REAL_ABS", "DIFF_DIST2LAND",
        "DIFF_DIST2LAND_ABS", "DIFF_MSWEP_DIST_30",
        "DIFF_SST_AWMEAN_200_800",
    ]
    final_columns = (
        ["SID"] + ids
        + [f"{c}_S" for c in targets]
        + [f"{c}_E" for c in targets]
        + ordered_diff
        + [
            "MEAN24H_SST_AWMEAN_200_800", "SST_VALID_COUNT_24H",
            "COMPLETE_SST_24H_FLAG", "COUNT_ET_24H", "OBS_COUNT_24H",
            "COMPLETE_24H_FLAG",
        ]
    )
    missing = set(final_columns).difference(pairs.columns)
    if missing:
        raise KeyError("Final columns missing: " + ", ".join(sorted(missing)))
    final = pairs[final_columns].sort_values(["ROW_ID_S", "ROW_ID_E"])
    if final.duplicated(["SID", "ROW_ID_S", "ROW_ID_E"]).any():
        raise RuntimeError("Duplicate final pair keys")
    return final.reset_index(drop=True)


# ============================================================
# 7. Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 90)
    print("Starting MSWEP30 + SST 24-hour processing")
    print("=" * 90)
    print(f"SST input: {SST_INPUT}")
    print(f"MSWEP input: {MSWEP30_INPUT}")
    print("Pairing: same SID and E timestamp exactly S + 24 hours")
    print("Both endpoints must have valid MSWEP_DIST_30 and SST")
    print("SST-invalid endpoint pairs are written to a dedicated log")

    source, stats = read_and_prepare_source()
    pairs, targets, ids, pair_stats = build_24h_pairs(source)
    stats.update(pair_stats)
    final = calculate_and_arrange(pairs, targets, ids)
    stats.update({
        "OUTPUT_ROWS": len(final),
        "COMPLETE_24H_PAIRS": int(final["COMPLETE_24H_FLAG"].sum()),
        "INCOMPLETE_24H_PAIRS": int((~final["COMPLETE_24H_FLAG"]).sum()),
        "COMPLETE_SST_24H_PAIRS": int(final["COMPLETE_SST_24H_FLAG"].sum()),
        "INCOMPLETE_SST_24H_PAIRS": int((~final["COMPLETE_SST_24H_FLAG"]).sum()),
        "ET_PAIR_COUNT": int(final["COUNT_ET_24H"].gt(0).sum()),
        "OUTPUT_PATH": str(OUTPUT_CSV),
        "MISSING_SST_PAIR_LOG": str(MISSING_SST_PAIR_LOG),
    })
    write_csv_atomic(final, OUTPUT_CSV)
    write_csv_atomic(pd.DataFrame([stats]), SUMMARY_CSV)

    print("\n" + "=" * 90)
    print("Processing completed successfully")
    print("=" * 90)
    print(f"Source rows: {stats['SOURCE_ROWS']}")
    print(f"No-valid-SST source rows: {stats['NO_VALID_SST_SOURCE_ROWS']}")
    print(f"All exact 24-hour pairs: {stats['ALL_EXACT_24H_PAIRS']}")
    print(f"Pairs removed for SST endpoint: {stats['PAIRS_MISSING_SST_ANY_ENDPOINT']}")
    print(f"Pairs retained: {stats['PAIRS_RETAINED']}")
    print(f"Complete 24-hour pairs: {stats['COMPLETE_24H_PAIRS']}")
    print(f"Complete SST 24-hour pairs: {stats['COMPLETE_SST_24H_PAIRS']}")
    print(f"Final output: {OUTPUT_CSV}")
    print(f"Missing-SST pair log: {MISSING_SST_PAIR_LOG}")
    print(f"Processing summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
