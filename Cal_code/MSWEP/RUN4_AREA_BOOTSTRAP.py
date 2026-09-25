import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)


# ============================================================
# 1. User configuration
# ============================================================

THRESHOLDS = [
    10,
    20,
    30,
    40,
]

BOOTSTRAP_COUNT = 5000
BOOTSTRAP_BATCH_SIZE = 32

RANDOM_SEED = 20260812

CI_LOWER_PERCENTILE = 2.5
CI_UPPER_PERCENTILE = 97.5

COAST_DISTANCE_THRESHOLD_KM = 500.0


# ============================================================
# 2. Project paths
# ============================================================

# The script is expected at TC-RW-V1/Cal_code/MSWEP/.
# TC_RW_PROJECT_ROOT can override automatic project-root detection.
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(
    os.environ.get("TC_RW_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))
).expanduser().resolve()

INPUT_DIR = (
    PROJECT_ROOT / "Data" / "Processed" / "MSWEP" / "Windows" / "24H"
)

OUTPUT_ROOT = (
    PROJECT_ROOT / "Data" / "Processed" / "MSWEP" / "Bootstrap" / "Area"
)

LOG_DIR = (
    PROJECT_ROOT / "Results" / "Quality_control" / "MSWEP" / "RUN4"
)

PROCESSING_SUMMARY_CSV = (
    LOG_DIR
    / "bootstrap_processing_summary.csv"
)


# ============================================================
# 3. Distance-zone configuration
# ============================================================

DISTANCE_ZONES = [
    "0_100",
    "100_200",
    "200_300",
    "300_400",
    "400_500",
    "0_200",
    "200_500",
]

DISTANCE_ZONE_LABELS = {
    "0_100": "0-100 km",
    "100_200": "100-200 km",
    "200_300": "200-300 km",
    "300_400": "300-400 km",
    "400_500": "400-500 km",
    "0_200": "0-200 km",
    "200_500": "200-500 km",
}


# ============================================================
# 4. Spatial categories
# ============================================================

SPATIAL_TYPES = [
    "OPEN_OCEAN",
    "NEARSHORE",
]


# ============================================================
# 5. Metric categories
# ============================================================

METRIC_TYPES = [
    "AREA",
]


# ============================================================
# 6. Intensity-change groups
# ============================================================

INTENSITY_GROUPS = [
    {
        "order": 1,
        "code": "LE_5TH",
        "label": "<=5th",
        "range": "<= -30 kt",
        "mask": lambda values: (
            values <= -30.0
        ),
    },
    {
        "order": 2,
        "code": "P05_P25",
        "label": "(5th,25th]",
        "range": "(-30, -12] kt",
        "mask": lambda values: (
            (values > -30.0)
            & (values <= -12.0)
        ),
    },
    {
        "order": 3,
        "code": "P25_P50",
        "label": "(25th,50th)",
        "range": "(-12, 0) kt",
        "mask": lambda values: (
            (values > -12.0)
            & (values < 0.0)
        ),
    },
    {
        "order": 4,
        "code": "EQ_50TH",
        "label": "=50th",
        "range": "0 kt",
        "mask": lambda values: (
            values == 0.0
        ),
    },
    {
        "order": 5,
        "code": "P50_P75",
        "label": "(50th,75th)",
        "range": "(0, 12) kt",
        "mask": lambda values: (
            (values > 0.0)
            & (values < 12.0)
        ),
    },
    {
        "order": 6,
        "code": "P75_P95",
        "label": "[75th,95th)",
        "range": "[12, 30) kt",
        "mask": lambda values: (
            (values >= 12.0)
            & (values < 30.0)
        ),
    },
    {
        "order": 7,
        "code": "GE_95TH",
        "label": ">=95th",
        "range": ">= 30 kt",
        "mask": lambda values: (
            values >= 30.0
        ),
    },
]


# ============================================================
# 7. General helper functions
# ============================================================

def threshold_label(threshold):
    """
    Convert a numeric threshold to a stable string label.

    Examples
    --------
    10  -> "10"
    """
    numeric_threshold = float(
        threshold
    )

    if numeric_threshold.is_integer():
        return str(
            int(numeric_threshold)
        )

    return format(
        numeric_threshold,
        "g",
    )


def threshold_seed_offset(threshold):
    """
    Convert a threshold to a deterministic integer seed offset.
    """
    return int(
        round(
            float(threshold)
            * 10000
        )
    )


def build_input_path(threshold):
    """
    Build the threshold-specific CLEANED input path.
    """
    label = threshold_label(
        threshold
    )

    return (
        INPUT_DIR
        / (
            "PRE_DATA_IBT_1982_2024_"
            f"MSWEP_TH{label}_"
            "24H_SLIDING_ET_CLEANED.csv"
        )
    )


def build_threshold_output_dir(
    threshold,
):
    """
    Build the output directory for one threshold.
    """
    label = threshold_label(
        threshold
    )

    return (
        OUTPUT_ROOT
        / f"TH{label}"
    )


def build_output_path(
    threshold,
    spatial_type,
    metric_type,
):
    """
    Build one Bootstrap output path.
    """
    label = threshold_label(
        threshold
    )

    threshold_output_dir = (
        build_threshold_output_dir(
            threshold
        )
    )

    return (
        threshold_output_dir
        / (
            "PRE_DATA_IBT_1982_2024_"
            f"MSWEP_TH{label}_"
            f"{spatial_type}_"
            f"{metric_type}_"
            "BOOTSTRAP_TABLE.csv"
        )
    )


def build_metric_columns(
    threshold,
    metric_type,
):
    """
    Build threshold-specific area-difference columns.
    """
    label = threshold_label(
        threshold
    )

    return [
        (
            f"DIFF_MSWEP_{metric_type}_"
            f"{label}_{zone}"
        )
        for zone in DISTANCE_ZONES
    ]


def build_metric_description(
    threshold,
    metric_type,
):
    """
    Build a human-readable description for one metric.
    """
    label = threshold_label(
        threshold
    )

    if metric_type == "AREA":
        return (
            "24-hour change in area occupied by grid cells "
            f"with PRECIP_3H >= {label} mm/3h"
        )

    raise ValueError(
        f"Unsupported metric type: {metric_type}"
    )


def metric_unit(metric_type):
    """
    Return the physical unit for one metric.
    """
    if metric_type == "AREA":
        return "km2"

    raise ValueError(
        f"Unsupported metric type: {metric_type}"
    )


def validate_required_columns(
    df,
    required_columns,
    context,
):
    """
    Raise an error if required columns are missing.
    """
    missing_columns = (
        set(required_columns)
        .difference(df.columns)
    )

    if missing_columns:
        raise KeyError(
            f"{context} is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )


def write_csv_atomic(
    df,
    output_path,
):
    """
    Write a CSV atomically.

    The formal output is replaced only after the temporary file
    has been written successfully.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        output_path.name
        + f".tmp.{os.getpid()}"
    )

    try:
        df.to_csv(
            temporary_path,
            index=False,
        )

        if not temporary_path.exists():
            raise RuntimeError(
                "Temporary output was not created: "
                f"{temporary_path}"
            )

        if temporary_path.stat().st_size == 0:
            raise RuntimeError(
                "Temporary output is empty: "
                f"{temporary_path}"
            )

        os.replace(
            temporary_path,
            output_path,
        )

    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


# ============================================================
# 8. Read and validate one threshold
# ============================================================

def read_threshold_data(
    threshold,
):
    """
    Read one threshold-specific CLEANED 24-hour table.
    """
    label = threshold_label(
        threshold
    )

    input_path = build_input_path(
        threshold
    )

    print("\n" + "=" * 100)
    print(
        f"Reading threshold {label} CLEANED table"
    )
    print("=" * 100)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV does not exist: {input_path}"
        )

    area_columns = build_metric_columns(
        threshold,
        "AREA",
    )

    required_columns = [
        "SID",
        "ROW_ID_S",
        "ROW_ID_E",
        "ISO_TIME_S",
        "ISO_TIME_E",
        "DIFF_USA_WIND",
        "CAL_DIST_REAL_S",
        "CAL_DIST_REAL_E",
        "COUNT_ET_24H",
    ] + area_columns

    df = pd.read_csv(
        input_path,
        usecols=required_columns,
        dtype={
            "SID": "string",
        },
        low_memory=False,
    )

    validate_required_columns(
        df,
        required_columns,
        f"Threshold {label}",
    )

    print(f"Input file: {input_path}")
    print(f"Input CLEANED windows: {len(df)}")
    print(f"Input columns: {len(df.columns)}")

    df["ISO_TIME_S"] = pd.to_datetime(
        df["ISO_TIME_S"],
        errors="coerce",
    )

    df["ISO_TIME_E"] = pd.to_datetime(
        df["ISO_TIME_E"],
        errors="coerce",
    )

    invalid_time_mask = (
        df["ISO_TIME_S"].isna()
        | df["ISO_TIME_E"].isna()
    )

    if invalid_time_mask.any():
        raise ValueError(
            f"Threshold {label} contains invalid pair times. "
            f"Count: {int(invalid_time_mask.sum())}"
        )

    time_difference_hours = (
        (
            df["ISO_TIME_E"]
            - df["ISO_TIME_S"]
        )
        .dt.total_seconds()
        / 3600.0
    )

    invalid_interval_mask = (
        ~np.isclose(
            time_difference_hours,
            24.0,
            rtol=0.0,
            atol=1e-9,
        )
    )

    if invalid_interval_mask.any():
        raise ValueError(
            f"Threshold {label} contains non-24-hour "
            f"windows. Count: "
            f"{int(invalid_interval_mask.sum())}"
        )

    df["ROW_ID_S"] = pd.to_numeric(
        df["ROW_ID_S"],
        errors="coerce",
    )

    df["ROW_ID_E"] = pd.to_numeric(
        df["ROW_ID_E"],
        errors="coerce",
    )

    invalid_row_id_mask = (
        df["ROW_ID_S"].isna()
        | df["ROW_ID_E"].isna()
    )

    if invalid_row_id_mask.any():
        raise ValueError(
            f"Threshold {label} contains invalid pair "
            f"ROW_ID values. Count: "
            f"{int(invalid_row_id_mask.sum())}"
        )

    df["ROW_ID_S"] = df[
        "ROW_ID_S"
    ].astype(np.int64)

    df["ROW_ID_E"] = df[
        "ROW_ID_E"
    ].astype(np.int64)

    duplicate_pair_mask = (
        df.duplicated(
            subset=[
                "SID",
                "ROW_ID_S",
                "ROW_ID_E",
            ],
            keep=False,
        )
    )

    if duplicate_pair_mask.any():
        examples = (
            df.loc[
                duplicate_pair_mask,
                [
                    "SID",
                    "ROW_ID_S",
                    "ROW_ID_E",
                    "ISO_TIME_S",
                    "ISO_TIME_E",
                ],
            ]
            .head(20)
            .to_dict("records")
        )

        raise ValueError(
            f"Threshold {label} contains duplicate pair keys. "
            f"Count: {int(duplicate_pair_mask.sum())}. "
            f"Examples: {examples}"
        )

    numeric_columns = [
        "DIFF_USA_WIND",
        "CAL_DIST_REAL_S",
        "CAL_DIST_REAL_E",
        "COUNT_ET_24H",
    ] + area_columns

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df[numeric_columns] = (
        df[numeric_columns]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    common_filter_columns = [
        "DIFF_USA_WIND",
        "CAL_DIST_REAL_S",
        "CAL_DIST_REAL_E",
        "COUNT_ET_24H",
    ]

    common_valid_mask = (
        df[
            common_filter_columns
        ]
        .notna()
        .all(axis=1)
    )

    common_invalid_count = int(
        (~common_valid_mask).sum()
    )

    if common_invalid_count > 0:
        examples = (
            df.loc[
                ~common_valid_mask,
                [
                    "SID",
                    "ROW_ID_S",
                    "ROW_ID_E",
                ]
                + common_filter_columns,
            ]
            .head(20)
            .to_dict("records")
        )

        raise ValueError(
            f"Threshold {label} contains invalid common "
            f"filter fields. Count: {common_invalid_count}. "
            f"Examples: {examples}"
        )

    invalid_et_mask = (
        (df["COUNT_ET_24H"] < 0)
        | (
            df["COUNT_ET_24H"]
            % 1
            != 0
        )
    )

    if invalid_et_mask.any():
        raise ValueError(
            f"Threshold {label} contains invalid "
            f"COUNT_ET_24H values. Count: "
            f"{int(invalid_et_mask.sum())}"
        )

    print(
        f"[PASS] Threshold {label} input validation completed."
    )

    return (
        df,
        area_columns,
        input_path,
    )


# ============================================================
# 9. Create spatial subsets
# ============================================================

def create_spatial_subsets(
    df,
    threshold,
):
    """
    Create mutually exclusive spatial subsets.

    OPEN_OCEAN:
        Both endpoints are more than 500 km from the coastline.

    NEARSHORE:
        Both endpoints are at most 500 km from the coastline.
        No lower bound is applied.

    MIXED:
        The endpoints belong to different spatial categories.
        Mixed windows are excluded.
    """
    label = threshold_label(
        threshold
    )

    open_ocean_mask = (
        (
            df["CAL_DIST_REAL_S"]
            > COAST_DISTANCE_THRESHOLD_KM
        )
        & (
            df["CAL_DIST_REAL_E"]
            > COAST_DISTANCE_THRESHOLD_KM
        )
    )

    nearshore_mask = (
        (
            df["CAL_DIST_REAL_S"]
            <= COAST_DISTANCE_THRESHOLD_KM
        )
        & (
            df["CAL_DIST_REAL_E"]
            <= COAST_DISTANCE_THRESHOLD_KM
        )
    )

    mixed_mask = (
        ~open_ocean_mask
        & ~nearshore_mask
    )

    spatial_subsets = {
        "OPEN_OCEAN": (
            df.loc[
                open_ocean_mask
            ].copy()
        ),
        "NEARSHORE": (
            df.loc[
                nearshore_mask
            ].copy()
        ),
    }

    mixed_count = int(
        mixed_mask.sum()
    )

    classified_count = (
        len(
            spatial_subsets[
                "OPEN_OCEAN"
            ]
        )
        + len(
            spatial_subsets[
                "NEARSHORE"
            ]
        )
        + mixed_count
    )

    if classified_count != len(df):
        raise RuntimeError(
            f"Threshold {label} spatial categories do not "
            "fully cover the input."
        )

    print(
        f"Threshold {label} spatial classification:"
    )

    print(
        "  OPEN_OCEAN before ET filtering: "
        f"{len(spatial_subsets['OPEN_OCEAN'])}"
    )

    print(
        "  NEARSHORE before ET filtering: "
        f"{len(spatial_subsets['NEARSHORE'])}"
    )

    print(
        f"  MIXED excluded: {mixed_count}"
    )

    return (
        spatial_subsets,
        mixed_count,
    )


# ============================================================
# 10. Apply non-ET filtering
# ============================================================

def apply_non_et_filter(
    spatial_subsets,
    threshold,
):
    """
    Retain only windows with COUNT_ET_24H equal to zero.
    """
    label = threshold_label(
        threshold
    )

    filtered_subsets = {}
    et_summary = {}

    for spatial_type in SPATIAL_TYPES:
        subset_df = spatial_subsets[
            spatial_type
        ]

        non_et_mask = (
            subset_df[
                "COUNT_ET_24H"
            ]
            == 0
        )

        filtered_df = (
            subset_df.loc[
                non_et_mask
            ]
            .copy()
        )

        et_removed_count = (
            len(subset_df)
            - len(filtered_df)
        )

        filtered_subsets[
            spatial_type
        ] = filtered_df

        et_summary[
            spatial_type
        ] = {
            "before_et": len(
                subset_df
            ),
            "after_et": len(
                filtered_df
            ),
            "et_removed": (
                et_removed_count
            ),
        }

        print(
            f"Threshold {label} {spatial_type}: "
            f"before ET={len(subset_df)}, "
            f"after ET={len(filtered_df)}, "
            f"ET removed={et_removed_count}"
        )

    return (
        filtered_subsets,
        et_summary,
    )


# ============================================================
# 11. Validate intensity-group coverage
# ============================================================

def validate_intensity_groups(
    subset_df,
    threshold,
    spatial_type,
):
    """
    Verify that every window belongs to exactly one intensity
    group.
    """
    label = threshold_label(
        threshold
    )

    wind_values = (
        subset_df[
            "DIFF_USA_WIND"
        ]
        .to_numpy(
            dtype=np.float64,
            copy=False,
        )
    )

    membership_count = np.zeros(
        len(subset_df),
        dtype=np.int8,
    )

    group_counts = {}

    for group in INTENSITY_GROUPS:
        group_mask = group[
            "mask"
        ](
            wind_values
        )

        membership_count += (
            group_mask.astype(
                np.int8
            )
        )

        group_counts[
            group["code"]
        ] = int(
            group_mask.sum()
        )

    invalid_membership_mask = (
        membership_count
        != 1
    )

    if invalid_membership_mask.any():
        raise ValueError(
            f"Threshold {label} {spatial_type} contains "
            "windows that do not belong to exactly one "
            "intensity group. "
            f"Count: {int(invalid_membership_mask.sum())}"
        )

    if sum(
        group_counts.values()
    ) != len(subset_df):
        raise RuntimeError(
            f"Threshold {label} {spatial_type} intensity "
            "group counts do not sum to the subset size."
        )

    print(
        f"[PASS] Threshold {label} {spatial_type} "
        "intensity-group coverage completed."
    )

    return group_counts


# ============================================================
# 12. Bootstrap calculation
# ============================================================

def calculate_bootstrap_means(
    values,
    bootstrap_count,
    random_seed,
):
    """
    Calculate window-level Bootstrap means.

    Every resample has the same size as the original group and is
    sampled with replacement. All distance zones share the same
    resampled window indices.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 2:
        raise ValueError(
            "Bootstrap input must be two-dimensional."
        )

    sample_size = values.shape[0]
    variable_count = values.shape[1]

    if sample_size == 0:
        return np.empty(
            (
                0,
                variable_count,
            ),
            dtype=np.float64,
        )

    random_generator = (
        np.random.default_rng(
            random_seed
        )
    )

    bootstrap_means = np.empty(
        (
            bootstrap_count,
            variable_count,
        ),
        dtype=np.float64,
    )

    completed_count = 0

    while (
        completed_count
        < bootstrap_count
    ):
        current_batch_size = min(
            BOOTSTRAP_BATCH_SIZE,
            (
                bootstrap_count
                - completed_count
            ),
        )

        sampled_indices = (
            random_generator.integers(
                low=0,
                high=sample_size,
                size=(
                    current_batch_size,
                    sample_size,
                ),
                endpoint=False,
            )
        )

        sampled_values = values[
            sampled_indices
        ]

        bootstrap_means[
            completed_count:
            completed_count
            + current_batch_size
        ] = sampled_values.mean(
            axis=1,
            dtype=np.float64,
        )

        completed_count += (
            current_batch_size
        )

    return bootstrap_means


# ============================================================
# 13. Build deterministic Bootstrap seed
# ============================================================

def build_group_seed(
    threshold,
    spatial_type,
    metric_type,
    group_order,
):
    """
    Build a deterministic seed for one Bootstrap calculation.
    """
    spatial_offsets = {
        "OPEN_OCEAN": 0,
        "NEARSHORE": 100000,
    }

    metric_offsets = {
        "AREA": 0,
    }

    return (
        RANDOM_SEED
        + threshold_seed_offset(
            threshold
        )
        + spatial_offsets[
            spatial_type
        ]
        + metric_offsets[
            metric_type
        ]
        + int(group_order)
    )


# ============================================================
# 14. Summarise one spatial and metric combination
# ============================================================

def summarise_metric(
    subset_df,
    threshold,
    spatial_type,
    metric_type,
    metric_columns,
    input_cleaned_count,
):
    """
    Calculate original means, Bootstrap confidence intervals, and
    Bootstrap standard errors for all intensity groups and zones.
    """
    label = threshold_label(
        threshold
    )

    unit = metric_unit(
        metric_type
    )

    description = (
        build_metric_description(
            threshold,
            metric_type,
        )
    )

    wind_values = (
        subset_df[
            "DIFF_USA_WIND"
        ]
        .to_numpy(
            dtype=np.float64,
            copy=False,
        )
    )

    output_records = []

    print(
        f"\nCalculating threshold {label}, "
        f"{spatial_type}, {metric_type}"
    )

    for group in INTENSITY_GROUPS:
        group_mask = group[
            "mask"
        ](
            wind_values
        )

        group_df = subset_df.loc[
            group_mask
        ].copy()

        group_window_count = len(
            group_df
        )

        common_valid_mask = (
            group_df[
                metric_columns
            ]
            .notna()
            .all(axis=1)
        )

        valid_group_df = (
            group_df.loc[
                common_valid_mask
            ]
            .copy()
        )

        common_valid_count = len(
            valid_group_df
        )

        invalid_count = (
            group_window_count
            - common_valid_count
        )

        group_seed = build_group_seed(
            threshold,
            spatial_type,
            metric_type,
            group["order"],
        )

        print(
            f"  {group['label']}: "
            f"group={group_window_count}, "
            f"common-valid={common_valid_count}, "
            f"invalid={invalid_count}"
        )

        if common_valid_count > 0:
            values = (
                valid_group_df[
                    metric_columns
                ]
                .to_numpy(
                    dtype=np.float64,
                    copy=True,
                )
            )

            original_means = (
                values.mean(
                    axis=0,
                    dtype=np.float64,
                )
            )

            bootstrap_means = (
                calculate_bootstrap_means(
                    values=values,
                    bootstrap_count=(
                        BOOTSTRAP_COUNT
                    ),
                    random_seed=(
                        group_seed
                    ),
                )
            )

            confidence_lower = (
                np.percentile(
                    bootstrap_means,
                    CI_LOWER_PERCENTILE,
                    axis=0,
                )
            )

            confidence_upper = (
                np.percentile(
                    bootstrap_means,
                    CI_UPPER_PERCENTILE,
                    axis=0,
                )
            )

            bootstrap_standard_error = (
                bootstrap_means.std(
                    axis=0,
                    ddof=1,
                )
            )

        else:
            original_means = np.full(
                len(metric_columns),
                np.nan,
                dtype=np.float64,
            )

            confidence_lower = np.full(
                len(metric_columns),
                np.nan,
                dtype=np.float64,
            )

            confidence_upper = np.full(
                len(metric_columns),
                np.nan,
                dtype=np.float64,
            )

            bootstrap_standard_error = np.full(
                len(metric_columns),
                np.nan,
                dtype=np.float64,
            )

        for zone_order, (
            zone,
            source_column,
        ) in enumerate(
            zip(
                DISTANCE_ZONES,
                metric_columns,
            ),
            start=1,
        ):
            output_records.append(
                {
                    "THRESHOLD": float(
                        threshold
                    ),
                    "THRESHOLD_LABEL": (
                        label
                    ),
                    "THRESHOLD_UNIT": (
                        "mm/3h"
                    ),
                    "INPUT_CLEANED_WINDOW_COUNT": (
                        input_cleaned_count
                    ),
                    "SPATIAL_TYPE": (
                        spatial_type
                    ),
                    "METRIC_TYPE": (
                        metric_type
                    ),
                    "METRIC_DESCRIPTION": (
                        description
                    ),
                    "UNIT": unit,
                    "INTENSITY_GROUP_ORDER": (
                        group["order"]
                    ),
                    "INTENSITY_GROUP_CODE": (
                        group["code"]
                    ),
                    "INTENSITY_GROUP_LABEL": (
                        group["label"]
                    ),
                    "WIND_CHANGE_RANGE": (
                        group["range"]
                    ),
                    "DISTANCE_ZONE_ORDER": (
                        zone_order
                    ),
                    "DISTANCE_ZONE": zone,
                    "DISTANCE_ZONE_LABEL": (
                        DISTANCE_ZONE_LABELS[
                            zone
                        ]
                    ),
                    "SOURCE_COLUMN": (
                        source_column
                    ),
                    "GROUP_WINDOW_COUNT": (
                        group_window_count
                    ),
                    "COMMON_VALID_WINDOW_COUNT": (
                        common_valid_count
                    ),
                    "INVALID_WINDOW_COUNT": (
                        invalid_count
                    ),
                    "MEAN_CHANGE": float(
                        original_means[
                            zone_order - 1
                        ]
                    ),
                    "BOOTSTRAP_CI_LOWER": float(
                        confidence_lower[
                            zone_order - 1
                        ]
                    ),
                    "BOOTSTRAP_CI_UPPER": float(
                        confidence_upper[
                            zone_order - 1
                        ]
                    ),
                    "BOOTSTRAP_STANDARD_ERROR": float(
                        bootstrap_standard_error[
                            zone_order - 1
                        ]
                    ),
                    "BOOTSTRAP_COUNT": (
                        BOOTSTRAP_COUNT
                    ),
                    "BOOTSTRAP_UNIT": (
                        "24-hour window"
                    ),
                    "RANDOM_SEED": (
                        group_seed
                    ),
                }
            )

    result_df = pd.DataFrame(
        output_records
    )

    expected_rows = (
        len(INTENSITY_GROUPS)
        * len(DISTANCE_ZONES)
    )

    if len(result_df) != expected_rows:
        raise RuntimeError(
            f"Threshold {label} {spatial_type} "
            f"{metric_type} produced {len(result_df)} rows; "
            f"expected {expected_rows}."
        )

    result_df = result_df.sort_values(
        [
            "INTENSITY_GROUP_ORDER",
            "DISTANCE_ZONE_ORDER",
        ]
    ).reset_index(
        drop=True
    )

    return result_df


# ============================================================
# 15. Process one threshold
# ============================================================

def process_threshold(
    threshold,
):
    """
    Process one threshold independently.
    """
    label = threshold_label(
        threshold
    )

    (
        df,
        area_columns,
        input_path,
    ) = read_threshold_data(
        threshold
    )

    input_cleaned_count = len(df)

    (
        spatial_subsets,
        mixed_count,
    ) = create_spatial_subsets(
        df,
        threshold,
    )

    (
        filtered_subsets,
        et_summary,
    ) = apply_non_et_filter(
        spatial_subsets,
        threshold,
    )

    output_paths = []
    summary_records = []

    metric_column_map = {
        "AREA": area_columns,
    }

    for spatial_type in SPATIAL_TYPES:
        subset_df = filtered_subsets[
            spatial_type
        ]

        group_counts = (
            validate_intensity_groups(
                subset_df,
                threshold,
                spatial_type,
            )
        )

        for metric_type in METRIC_TYPES:
            result_df = summarise_metric(
                subset_df=subset_df,
                threshold=threshold,
                spatial_type=spatial_type,
                metric_type=metric_type,
                metric_columns=(
                    metric_column_map[
                        metric_type
                    ]
                ),
                input_cleaned_count=(
                    input_cleaned_count
                ),
            )

            output_path = build_output_path(
                threshold,
                spatial_type,
                metric_type,
            )

            write_csv_atomic(
                result_df,
                output_path,
            )

            output_paths.append(
                output_path
            )

            metric_invalid_windows = int(
                result_df[
                    [
                        "INTENSITY_GROUP_CODE",
                        "INVALID_WINDOW_COUNT",
                    ]
                ]
                .drop_duplicates(
                    subset=[
                        "INTENSITY_GROUP_CODE",
                    ]
                )[
                    "INVALID_WINDOW_COUNT"
                ]
                .sum()
            )

            summary_records.append(
                {
                    "THRESHOLD": float(
                        threshold
                    ),
                    "THRESHOLD_LABEL": (
                        label
                    ),
                    "INPUT_PATH": str(
                        input_path
                    ),
                    "INPUT_CLEANED_WINDOWS": (
                        input_cleaned_count
                    ),
                    "OPEN_OCEAN_WINDOWS": len(
                        spatial_subsets[
                            "OPEN_OCEAN"
                        ]
                    ),
                    "NEARSHORE_WINDOWS": len(
                        spatial_subsets[
                            "NEARSHORE"
                        ]
                    ),
                    "MIXED_EXCLUDED_WINDOWS": (
                        mixed_count
                    ),
                    "SPATIAL_TYPE": (
                        spatial_type
                    ),
                    "SPATIAL_WINDOWS_BEFORE_ET": (
                        et_summary[
                            spatial_type
                        ][
                            "before_et"
                        ]
                    ),
                    "NON_ET_WINDOWS": (
                        et_summary[
                            spatial_type
                        ][
                            "after_et"
                        ]
                    ),
                    "ET_REMOVED_WINDOWS": (
                        et_summary[
                            spatial_type
                        ][
                            "et_removed"
                        ]
                    ),
                    "METRIC_TYPE": (
                        metric_type
                    ),
                    "METRIC_INVALID_WINDOWS": (
                        metric_invalid_windows
                    ),
                    "BOOTSTRAP_COUNT": (
                        BOOTSTRAP_COUNT
                    ),
                    "OUTPUT_ROWS": len(
                        result_df
                    ),
                    "OUTPUT_PATH": str(
                        output_path
                    ),
                    "INTENSITY_GROUP_COUNTS": (
                        ";".join(
                            (
                                f"{code}="
                                f"{count}"
                            )
                            for code, count
                            in group_counts.items()
                        )
                    ),
                }
            )

            print(
                f"Saved: {output_path}"
            )

    del df
    del spatial_subsets
    del filtered_subsets

    return (
        summary_records,
        output_paths,
    )


# ============================================================
# 16. Main program
# ============================================================

def main():
    if not THRESHOLDS:
        raise ValueError(
            "THRESHOLDS must contain at least one value."
        )

    labels = [
        threshold_label(
            threshold
        )
        for threshold in THRESHOLDS
    ]

    if len(labels) != len(
        set(labels)
    ):
        raise ValueError(
            "THRESHOLDS contains duplicate values after "
            f"normalization: {labels}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("Starting multi-threshold MSWEP area Bootstrap processing")
    print("=" * 100)

    print(
        "Thresholds (mm/3h): "
        + ", ".join(labels)
    )

    print(
        f"Bootstrap samples per group: "
        f"{BOOTSTRAP_COUNT}"
    )

    print(
        f"Input directory: {INPUT_DIR}"
    )

    print(
        f"Output root: {OUTPUT_ROOT}"
    )

    all_summary_records = []
    all_output_paths = []

    for threshold in THRESHOLDS:
        (
            threshold_summary,
            threshold_outputs,
        ) = process_threshold(
            threshold
        )

        all_summary_records.extend(
            threshold_summary
        )

        all_output_paths.extend(
            threshold_outputs
        )

    summary_df = pd.DataFrame(
        all_summary_records
    )

    summary_df = summary_df.sort_values(
        [
            "THRESHOLD",
            "SPATIAL_TYPE",
            "METRIC_TYPE",
        ]
    ).reset_index(
        drop=True
    )

    write_csv_atomic(
        summary_df,
        PROCESSING_SUMMARY_CSV,
    )

    print("\n" + "=" * 100)
    print("Processing completed successfully")
    print("=" * 100)

    print(
        f"Thresholds processed: "
        f"{len(THRESHOLDS)}"
    )

    print(
        f"Bootstrap tables created: "
        f"{len(all_output_paths)}"
    )

    print(
        f"Processing summary: "
        f"{PROCESSING_SUMMARY_CSV}"
    )

    display_columns = [
        "THRESHOLD_LABEL",
        "INPUT_CLEANED_WINDOWS",
        "SPATIAL_TYPE",
        "SPATIAL_WINDOWS_BEFORE_ET",
        "NON_ET_WINDOWS",
        "ET_REMOVED_WINDOWS",
        "METRIC_TYPE",
        "METRIC_INVALID_WINDOWS",
        "OUTPUT_ROWS",
    ]

    print("\nSummary:")

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\nOutput files:")

    for output_path in all_output_paths:
        print(
            f"  {output_path}"
        )


if __name__ == "__main__":
    main()
