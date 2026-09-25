#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build all MSWEP 12-, 24-, and 36-hour analysis tables.

This program replaces the former RUN3 scripts for window construction,
intensity-change quantiles, translation-speed summaries, and 24-hour overlap
sensitivity. All comments and generated log labels are in English.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. Configuration
# ============================================================

# The script is expected at TC-RW-V1/Cal_code/MSWEP/.
# TC_RW_PROJECT_ROOT can override automatic project-root detection.
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(
    os.environ.get("TC_RW_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))
).expanduser().resolve()

MSWEP_PROCESSED_ROOT = PROJECT_ROOT / "Data" / "Processed" / "MSWEP"
INPUT_DIR = MSWEP_PROCESSED_ROOT / "Thresholds"

WINDOW_THRESHOLDS = {
    12: [30],
    24: [10, 20, 30, 40],
    36: [30],
}

OUTPUT_DIRS = {
    12: MSWEP_PROCESSED_ROOT / "Windows" / "12H",
    24: MSWEP_PROCESSED_ROOT / "Windows" / "24H",
    36: MSWEP_PROCESSED_ROOT / "Windows" / "36H",
}

SPEED_OUTPUT_DIR = MSWEP_PROCESSED_ROOT / "Speed_24H"
OVERLAP_OUTPUT_DIR = (
    MSWEP_PROCESSED_ROOT / "Overlap_sensitivity_24H"
)
RUN3_LOG_DIR = (
    PROJECT_ROOT / "Results" / "Quality_control" / "MSWEP" / "RUN3"
)

EXPECTED_BASE_ROWS = 211857
RW_WIND_CHANGE_LIMIT = -30.0
OVERLAP_CONFIG = {21: 3, 12: 12, 0: 24}
VALID_3H_HOURS = {0, 3, 6, 9, 12, 15, 18, 21}
QUANTILE_LEVELS = [0.05, 0.25, 0.50, 0.75, 0.95]

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

COMMON_COLUMNS = [
    "ROW_ID",
    "USA_LAT",
    "USA_LON",
    "USA_WIND",
    "USA_SSHS",
    "USA_RMW",
    "USA_R34_NE",
    "USA_R34_SE",
    "USA_R34_SW",
    "USA_R34_NW",
    "USA_R64_NE",
    "USA_R64_SE",
    "USA_R64_SW",
    "USA_R64_NW",
    "USA_R34_ALL_QUADRANTS_VALID",
    "USA_R64_ALL_QUADRANTS_VALID",
    "USA_R34_MEAN_NMI",
    "USA_R64_MEAN_NMI",
    "USA_R34_MEAN_KM",
    "USA_R64_MEAN_KM",
    "ISO_TIME",
    "RAIN_TIME_ID",
    "CAL_DIST_REAL",
    "DIST2LAND",
    "NATURE",
    "BASIN",
]

OPTIONAL_COLUMNS = [
    "RAIN_POINT",
    "u_speed",
    "v_speed",
    "STORM_SPEED",
]


# ============================================================
# 2. General helpers
# ============================================================

def threshold_label(threshold: float) -> str:
    value = float(threshold)
    return str(int(value)) if value.is_integer() else format(value, "g")


def input_path(threshold: float) -> Path:
    label = threshold_label(threshold)
    return INPUT_DIR / (
        "PRE_DATA_IBT_1982_2024_"
        f"MSWEP_THRESHOLD_{label}.csv"
    )


def output_paths(window_hours: int, threshold: float) -> tuple[Path, Path]:
    label = threshold_label(threshold)
    output_dir = OUTPUT_DIRS[window_hours]
    stem = (
        "PRE_DATA_IBT_1982_2024_"
        f"MSWEP_TH{label}_{window_hours}H_SLIDING_ET"
    )
    return output_dir / f"{stem}_ALL.csv", output_dir / f"{stem}_CLEANED.csv"


def threshold_columns(threshold: float) -> dict[str, object]:
    label = threshold_label(threshold)
    return {
        "label": label,
        "rain_point": f"RAIN_POINT_{label}",
        "area": [f"MSWEP_AREA_{label}_{zone}" for zone in DISTANCE_ZONES],
        "volume": [f"MSWEP_VOLUME_{label}_{zone}" for zone in DISTANCE_ZONES],
        "distance": f"MSWEP_DIST_{label}",
    }


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        df.to_csv(
            temporary_path,
            index=False,
            date_format="%Y-%m-%d %H:%M:%S",
        )
        if not temporary_path.exists() or temporary_path.stat().st_size == 0:
            raise RuntimeError(f"Temporary output was not created: {temporary_path}")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def validate_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise KeyError("Input is missing required columns: " + ", ".join(missing))


# ============================================================
# 3. Read and validate one threshold table
# ============================================================

def read_threshold_table(threshold: float) -> tuple[pd.DataFrame, dict[str, object]]:
    config = threshold_columns(threshold)
    path = input_path(threshold)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    print("\n" + "=" * 90)
    print(f"Reading threshold {config['label']}")
    print("=" * 90)
    print(f"Input: {path}")

    df = pd.read_csv(
        path,
        dtype={
            "SID": "string",
            "NAME": "string",
            "RAIN_TIME_ID": "string",
            "NATURE": "string",
            "BASIN": "string",
        },
        low_memory=False,
    )

    metric_columns = (
        [config["rain_point"]]
        + list(config["area"])
        + list(config["volume"])
        + [config["distance"]]
    )
    required = ["SID", "NAME"] + COMMON_COLUMNS + metric_columns
    validate_required_columns(df, required)

    if len(df) != EXPECTED_BASE_ROWS:
        raise RuntimeError(
            f"Unexpected source row count: {len(df):,}; "
            f"expected {EXPECTED_BASE_ROWS:,}."
        )

    df["ROW_ID"] = pd.to_numeric(df["ROW_ID"], errors="raise").astype(np.int64)
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    if df["ISO_TIME"].isna().any():
        raise ValueError("Invalid ISO_TIME values were found.")

    exact_3h = (
        df["ISO_TIME"].dt.hour.isin(VALID_3H_HOURS)
        & df["ISO_TIME"].dt.minute.eq(0)
        & df["ISO_TIME"].dt.second.eq(0)
    )
    if not exact_3h.all():
        raise ValueError(
            f"Non-exact 3-hour timestamps found: {int((~exact_3h).sum()):,}."
        )

    if df["ROW_ID"].duplicated().any():
        raise RuntimeError("Duplicate ROW_ID values were found.")
    if df.duplicated(["SID", "ISO_TIME"]).any():
        raise RuntimeError("Duplicate (SID, ISO_TIME) keys were found.")

    required_numeric = [
        "USA_LAT",
        "USA_LON",
        "USA_WIND",
        "USA_SSHS",
        "CAL_DIST_REAL",
        "DIST2LAND",
        config["rain_point"],
    ] + list(config["area"]) + list(config["volume"])
    for column in required_numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[required_numeric].isna().any(axis=None):
        bad = int(df[required_numeric].isna().any(axis=1).sum())
        raise ValueError(f"Missing required numeric values were found in {bad:,} rows.")

    df[config["distance"]] = pd.to_numeric(
        df[config["distance"]], errors="coerce"
    )

    wind_radius_columns = [
        "USA_RMW",
        "USA_R34_NE",
        "USA_R34_SE",
        "USA_R34_SW",
        "USA_R34_NW",
        "USA_R64_NE",
        "USA_R64_SE",
        "USA_R64_SW",
        "USA_R64_NW",
        "USA_R34_ALL_QUADRANTS_VALID",
        "USA_R64_ALL_QUADRANTS_VALID",
        "USA_R34_MEAN_NMI",
        "USA_R64_MEAN_NMI",
        "USA_R34_MEAN_KM",
        "USA_R64_MEAN_KM",
    ]
    for column in wind_radius_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ["u_speed", "v_speed", "STORM_SPEED"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    validate_combined_zones(df, config)
    add_translation_speed(df)
    print(f"Rows validated: {len(df):,}")
    return df, config


def validate_combined_zones(df: pd.DataFrame, config: dict[str, object]) -> None:
    label = str(config["label"])
    for metric in ("AREA", "VOLUME"):
        prefix = f"MSWEP_{metric}_{label}_"
        checks = {
            "0_200": ["0_100", "100_200"],
            "200_500": ["200_300", "300_400", "400_500"],
            "0_500": ["0_100", "100_200", "200_300", "300_400", "400_500"],
        }
        for combined, parts in checks.items():
            expected = df[[prefix + part for part in parts]].sum(axis=1)
            actual = df[prefix + combined]
            if not np.allclose(actual, expected, rtol=1e-10, atol=1e-6):
                raise RuntimeError(
                    f"Inconsistent {metric} combined zone for threshold "
                    f"{label}: {combined}."
                )


def add_translation_speed(df: pd.DataFrame) -> None:
    if "STORM_SPEED" in df.columns and df["STORM_SPEED"].notna().any():
        speed = df["STORM_SPEED"].astype(float)
    elif {"u_speed", "v_speed"}.issubset(df.columns):
        speed = np.hypot(df["u_speed"], df["v_speed"])
    else:
        speed = pd.Series(np.nan, index=df.index, dtype=float)
    df["STORM_SPEED"] = speed


# ============================================================
# 4. Build exact endpoint pairs
# ============================================================

def build_window_pairs(
    source_df: pd.DataFrame,
    config: dict[str, object],
    window_hours: int,
) -> pd.DataFrame:
    working = source_df.sort_values(
        ["SID", "ISO_TIME", "ROW_ID"]
    ).reset_index(drop=True).copy()

    working["IS_ET"] = (
        working["NATURE"].fillna("").str.strip().str.upper().eq("ET").astype(np.int8)
    )
    working["ET_CUM"] = working.groupby("SID", sort=False)["IS_ET"].cumsum()

    working["SPEED_VALID"] = working["STORM_SPEED"].notna().astype(np.int8)
    working["SPEED_FOR_SUM"] = working["STORM_SPEED"].fillna(0.0)
    working["SPEED_CUM_SUM"] = working.groupby("SID", sort=False)[
        "SPEED_FOR_SUM"
    ].cumsum()
    working["SPEED_CUM_COUNT"] = working.groupby("SID", sort=False)[
        "SPEED_VALID"
    ].cumsum()
    working["SPEED_CUM_SUM_BEFORE"] = (
        working["SPEED_CUM_SUM"] - working["SPEED_FOR_SUM"]
    )
    working["SPEED_CUM_COUNT_BEFORE"] = (
        working["SPEED_CUM_COUNT"] - working["SPEED_VALID"]
    )

    metric_columns = (
        [config["rain_point"]]
        + list(config["area"])
        + list(config["volume"])
        + [config["distance"]]
    )
    optional = [column for column in OPTIONAL_COLUMNS if column in working.columns]
    target_columns = list(dict.fromkeys(COMMON_COLUMNS + optional + metric_columns))

    start_internal = [
        "ET_CUM",
        "IS_ET",
        "SPEED_CUM_SUM_BEFORE",
        "SPEED_CUM_COUNT_BEFORE",
    ]
    end_internal = ["ET_CUM", "SPEED_CUM_SUM", "SPEED_CUM_COUNT"]

    start = working[["SID", "NAME"] + target_columns + start_internal].copy()
    start["TIME_TARGET_E"] = start["ISO_TIME"] + pd.Timedelta(hours=window_hours)
    start = start.rename(
        columns={column: f"{column}_S" for column in target_columns + start_internal}
    )

    end = working[["SID"] + target_columns + end_internal].copy()
    end = end.rename(
        columns={column: f"{column}_E" for column in target_columns + end_internal}
    )

    pairs = pd.merge(
        start,
        end,
        left_on=["SID", "TIME_TARGET_E"],
        right_on=["SID", "ISO_TIME_E"],
        how="inner",
        validate="one_to_one",
        sort=False,
    ).drop(columns=["TIME_TARGET_E"])

    et_column = f"COUNT_ET_{window_hours}H"
    pairs[et_column] = (
        pairs["ET_CUM_E"] - pairs["ET_CUM_S"] + pairs["IS_ET_S"]
    ).astype(np.int16)

    speed_count_column = f"COUNT_STORM_SPEED_{window_hours}H"
    speed_mean_column = f"MEAN_STORM_SPEED_{window_hours}H"
    speed_sum = pairs["SPEED_CUM_SUM_E"] - pairs["SPEED_CUM_SUM_BEFORE_S"]
    pairs[speed_count_column] = (
        pairs["SPEED_CUM_COUNT_E"] - pairs["SPEED_CUM_COUNT_BEFORE_S"]
    ).astype(np.int16)
    pairs[speed_mean_column] = np.where(
        pairs[speed_count_column].gt(0),
        speed_sum / pairs[speed_count_column],
        np.nan,
    )

    pairs["DIFF_ISO_TIME"] = (
        (pairs["ISO_TIME_E"] - pairs["ISO_TIME_S"])
        .dt.total_seconds()
        .div(3600.0)
    )
    if not np.isclose(pairs["DIFF_ISO_TIME"], float(window_hours)).all():
        raise RuntimeError(f"A non-{window_hours}-hour pair was produced.")

    difference_sources = (
        [
            "USA_WIND",
            "USA_RMW",
            "USA_R34_MEAN_KM",
            "USA_R64_MEAN_KM",
            "STORM_SPEED",
            "CAL_DIST_REAL",
            "DIST2LAND",
            config["rain_point"],
        ]
        + list(config["area"])
        + list(config["volume"])
        + [config["distance"]]
    )
    for column in difference_sources:
        pairs[f"DIFF_{column}"] = pairs[f"{column}_E"] - pairs[f"{column}_S"]
    pairs["DIFF_CAL_DIST_REAL_ABS"] = pairs["DIFF_CAL_DIST_REAL"].abs()
    pairs["DIFF_DIST2LAND_ABS"] = pairs["DIFF_DIST2LAND"].abs()

    start_columns = [f"{column}_S" for column in target_columns]
    end_columns = [f"{column}_E" for column in target_columns]
    differences = [f"DIFF_{column}" for column in difference_sources]
    leading_differences = [
        "DIFF_USA_WIND",
        "DIFF_STORM_SPEED",
        speed_mean_column,
        speed_count_column,
        "DIFF_ISO_TIME",
        "DIFF_CAL_DIST_REAL",
        "DIFF_CAL_DIST_REAL_ABS",
        "DIFF_DIST2LAND",
        "DIFF_DIST2LAND_ABS",
    ]
    excluded = {
        "DIFF_USA_WIND",
        "DIFF_STORM_SPEED",
        "DIFF_CAL_DIST_REAL",
        "DIFF_DIST2LAND",
    }
    ordered = (
        ["SID", "NAME"]
        + start_columns
        + end_columns
        + leading_differences
        + [column for column in differences if column not in excluded]
        + [et_column]
    )
    final = pairs[ordered].sort_values(["ROW_ID_S", "ROW_ID_E"]).reset_index(drop=True)
    if final.duplicated(["SID", "ROW_ID_S", "ROW_ID_E"]).any():
        raise RuntimeError(f"Duplicate {window_hours}-hour pair keys were found.")
    return final


# ============================================================
# 5. Save ALL/CLEANED data and summaries
# ============================================================

def save_window_outputs(
    all_df: pd.DataFrame,
    config: dict[str, object],
    threshold: float,
    window_hours: int,
) -> dict[str, object]:
    distance_start = f"{config['distance']}_S"
    distance_end = f"{config['distance']}_E"
    valid_start = all_df[distance_start].notna()
    valid_end = all_df[distance_end].notna()
    cleaned_df = all_df.loc[valid_start & valid_end].copy().reset_index(drop=True)

    all_path, cleaned_path = output_paths(window_hours, threshold)
    print(f"Writing ALL table: {all_path}")
    write_csv_atomic(all_df, all_path)
    print(f"Writing CLEANED table: {cleaned_path}")
    write_csv_atomic(cleaned_df, cleaned_path)

    if window_hours == 24 and float(threshold) == 30.0:
        speed_all = SPEED_OUTPUT_DIR / (
            "PRE_DATA_IBT_1982_2024_MSWEP_TH30_24H_SLIDING_ET_SPEED_ALL.csv"
        )
        speed_cleaned = SPEED_OUTPUT_DIR / (
            "PRE_DATA_IBT_1982_2024_MSWEP_TH30_24H_SLIDING_ET_SPEED_CLEANED.csv"
        )
        write_csv_atomic(all_df, speed_all)
        write_csv_atomic(cleaned_df, speed_cleaned)

    et_column = f"COUNT_ET_{window_hours}H"
    return {
        "WINDOW_HOURS": window_hours,
        "THRESHOLD": float(threshold),
        "THRESHOLD_LABEL": config["label"],
        "INPUT_PATH": str(input_path(threshold)),
        "INPUT_ROWS": EXPECTED_BASE_ROWS,
        "ALL_PAIRS": len(all_df),
        "START_DIST_NAN": int((~valid_start).sum()),
        "END_DIST_NAN": int((~valid_end).sum()),
        "BOTH_DIST_NAN": int((~valid_start & ~valid_end).sum()),
        "START_ONLY_NAN": int((~valid_start & valid_end).sum()),
        "END_ONLY_NAN": int((valid_start & ~valid_end).sum()),
        "CLEANED_PAIRS": len(cleaned_df),
        "REMOVED_PAIRS": len(all_df) - len(cleaned_df),
        "CLEANED_RETENTION_RATE": len(cleaned_df) / len(all_df) if len(all_df) else np.nan,
        "ET_PAIR_COUNT_ALL": int(all_df[et_column].gt(0).sum()),
        "ET_PAIR_COUNT_CLEANED": int(cleaned_df[et_column].gt(0).sum()),
        "OUTPUT_ALL_PATH": str(all_path),
        "OUTPUT_CLEANED_PATH": str(cleaned_path),
    }


def save_intensity_quantiles(window_results: dict[int, pd.DataFrame]) -> None:
    records = []
    for window_hours in sorted(window_results):
        wind = window_results[window_hours]["DIFF_USA_WIND"]
        quantiles = wind.quantile(QUANTILE_LEVELS)
        record = {
            "WINDOW_HOURS": window_hours,
            "TOTAL_ROWS": len(wind),
            "MISSING_VALUES": int(wind.isna().sum()),
        }
        for level, value in quantiles.items():
            record[f"Q{int(round(level * 100)):02d}"] = float(value)
        records.append(record)
    table = pd.DataFrame(records)
    path = RUN3_LOG_DIR / "RUN3_INTENSITY_CHANGE_QUANTILES.csv"
    write_csv_atomic(table, path)
    print("\nIntensity-change quantiles:")
    print(table.to_string(index=False))
    print(f"Quantile table: {path}")


# ============================================================
# 6. 24-hour RW overlap sensitivity
# ============================================================

def select_dynamic_overlap(candidates: pd.DataFrame, step_hours: int) -> pd.DataFrame:
    selected_indices = []
    step = pd.Timedelta(hours=step_hours)
    for _, group in candidates.groupby("SID", sort=False):
        group = group.sort_values(["ISO_TIME_S", "ROW_ID_S"])
        next_allowed = None
        for index, row in group.iterrows():
            if next_allowed is None or row["ISO_TIME_S"] >= next_allowed:
                selected_indices.append(index)
                next_allowed = row["ISO_TIME_S"] + step
    return candidates.loc[selected_indices].sort_values(
        ["ROW_ID_S", "ROW_ID_E"]
    ).reset_index(drop=True)


def save_overlap_sensitivity(all_24h_th30: pd.DataFrame) -> None:
    candidates = all_24h_th30.loc[
        all_24h_th30["DIFF_USA_WIND"].le(RW_WIND_CHANGE_LIMIT)
        & all_24h_th30["DIFF_MSWEP_DIST_30"].notna()
    ].copy()
    candidates["DIST30_POSITIVE_FLAG"] = candidates["DIFF_MSWEP_DIST_30"].gt(0)

    outputs = []
    summaries = []
    for overlap_hours, step_hours in OVERLAP_CONFIG.items():
        selected = select_dynamic_overlap(candidates, step_hours)
        selected.insert(0, "OVERLAP_HOURS", overlap_hours)
        selected.insert(1, "MIN_START_STEP_HOURS", step_hours)
        selected.insert(2, "WINDOW_HOURS", 24)
        path = OVERLAP_OUTPUT_DIR / (
            f"RW_TH30_24H_WINDOW_{overlap_hours}H_OVERLAP_DYNAMIC_ANCHOR.csv"
        )
        write_csv_atomic(selected, path)
        outputs.append(selected)
        summaries.append(
            {
                "OVERLAP_HOURS": overlap_hours,
                "MIN_START_STEP_HOURS": step_hours,
                "RW_SEGMENTS": len(selected),
                "DIST30_VALID": int(selected["DIFF_MSWEP_DIST_30"].notna().sum()),
                "DIST30_POSITIVE": int(selected["DIST30_POSITIVE_FLAG"].sum()),
                "PROPORTION_DIST30_POSITIVE": float(
                    selected["DIST30_POSITIVE_FLAG"].mean()
                ) if len(selected) else np.nan,
                "MEDIAN_DIFF_MSWEP_DIST_30": float(
                    selected["DIFF_MSWEP_DIST_30"].median()
                ),
                "OUTPUT_PATH": str(path),
            }
        )
        print(f"{overlap_hours:>2} h overlap: {len(selected):,} RW segments")

    combined = pd.concat(outputs, ignore_index=True)
    summary = pd.DataFrame(summaries)
    write_csv_atomic(
        combined,
        OVERLAP_OUTPUT_DIR / "RW_TH30_24H_OVERLAP_SENSITIVITY_COMBINED.csv",
    )
    write_csv_atomic(
        summary,
        OVERLAP_OUTPUT_DIR / "RW_TH30_24H_OVERLAP_SENSITIVITY_SUMMARY.csv",
    )


# ============================================================
# 7. Main program
# ============================================================

def main() -> None:
    for directory in list(OUTPUT_DIRS.values()) + [
        SPEED_OUTPUT_DIR,
        OVERLAP_OUTPUT_DIR,
        RUN3_LOG_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("RUN3: building all MSWEP analysis windows")
    print("=" * 90)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Window configuration: {WINDOW_THRESHOLDS}")

    threshold_windows: dict[float, list[int]] = {}
    for window_hours, thresholds in WINDOW_THRESHOLDS.items():
        for threshold in thresholds:
            threshold_windows.setdefault(float(threshold), []).append(window_hours)

    summaries = []
    threshold30_results: dict[int, pd.DataFrame] = {}

    for threshold in sorted(threshold_windows):
        source_df, config = read_threshold_table(threshold)
        for window_hours in sorted(threshold_windows[threshold]):
            print("\n" + "-" * 90)
            print(
                f"Building exact {window_hours}-hour pairs for "
                f"threshold {config['label']}"
            )
            all_df = build_window_pairs(source_df, config, window_hours)
            print(f"Exact pairs: {len(all_df):,}")
            summaries.append(
                save_window_outputs(all_df, config, threshold, window_hours)
            )
            if float(threshold) == 30.0:
                threshold30_results[window_hours] = all_df
        del source_df

    summary_table = pd.DataFrame(summaries).sort_values(
        ["WINDOW_HOURS", "THRESHOLD"]
    ).reset_index(drop=True)
    summary_path = RUN3_LOG_DIR / "RUN3_PROCESSING_SUMMARY.csv"
    write_csv_atomic(summary_table, summary_path)

    save_intensity_quantiles(threshold30_results)
    save_overlap_sensitivity(threshold30_results[24])

    print("\n" + "=" * 90)
    print("RUN3 completed successfully")
    print("=" * 90)
    print(
        summary_table[
            [
                "WINDOW_HOURS",
                "THRESHOLD_LABEL",
                "ALL_PAIRS",
                "CLEANED_PAIRS",
                "REMOVED_PAIRS",
                "ET_PAIR_COUNT_ALL",
            ]
        ].to_string(index=False)
    )
    print(f"\nCombined processing summary: {summary_path}")


if __name__ == "__main__":
    main()
