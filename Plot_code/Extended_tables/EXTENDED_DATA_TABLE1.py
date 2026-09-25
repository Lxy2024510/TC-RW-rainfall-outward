#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create the complete Extended Data Table 1 in one run."""

from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
OUT = ROOT / "Results" / "Extended_tables" / "Table1"
OUT_PNG = OUT / "EXTENDED_DATA_TABLE1.png"

FILES = {
    # Use the same formal input as the original RMW/R34/R64 table scripts.
    "base": "PRE_DATA_IBT_1982_2024_MSWEP_TH30_24H_SLIDING_ET_CLEANED.csv",
    "speed": "PRE_DATA_IBT_1982_2024_MSWEP_TH30_24H_SLIDING_ET_SPEED_CLEANED.csv",
    "vws": "PRE_DATA_IBT_1982_2024_MSWEP30_VWS_200_800KM_24H_SLIDING_CLEANED.csv",
    "sst": "PRE_DATA_IBT_1982_2024_MSWEP30_SST_200_800KM_24H_SLIDING_CLEANED.csv",
    "rh600": "PRE_DATA_IBT_1982_2024_MSWEP30_RH600_200_800KM_24H_SLIDING_CLEANED.csv",
}
RW_LIMIT = -30.0
NM_TO_KM = 1.852
QUADS = ("NE", "SE", "SW", "NW")
RADIUS_EXPECTED = {
    "R34": [(73.5, 48.4, 1074), (74.2, 40.4, 1085), (75.1, 40.6, 1022)],
    "R64": [(71.0, 33.6, 428), (69.2, 27.2, 549), (73.0, 29.7, 304)],
}

plt.rcParams.update({
    "font.family": "Arial", "font.sans-serif": ["Arial"],
    "font.cursive": ["Arial"], "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.cal": "Arial",
    "axes.unicode_minus": True, "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def find_file(name):
    preferred_paths = [
        # This directory mirrors the formal server source used by the
        # original standalone table scripts.
        DATA / "Processed" / "MSWEP_24H_RESULTS" / name,
        DATA / "Processed" / "MSWEP" / "Windows" / "24H" / name,
        DATA / "Processed" / "ERA5" / "VWS" / name,
        DATA / "Processed" / "ERA5" / "SST" / name,
        DATA / "Processed" / "ERA5" / "RH600" / name,
    ]
    for preferred in preferred_paths:
        if preferred.is_file():
            return preferred

    # Prefer the formal Data directory, but also search the complete project
    # so that files downloaded into a temporary project subdirectory remain
    # discoverable.
    matches = sorted(DATA.rglob(name)) if DATA.is_dir() else []
    if not matches:
        matches = sorted(ROOT.rglob(name))
    if not matches:
        stem_tokens = [
            token for token in name.replace(".csv", "").split("_")
            if token in {"VWS", "SST", "RH600", "SPEED", "ALL"}
        ]
        candidates = []
        for token in stem_tokens:
            candidates.extend(ROOT.rglob(f"*{token}*.csv"))
        candidate_text = "\n".join(
            f"  - {path}" for path in sorted(set(candidates))[:30]
        )
        hint = (
            f"\nSimilar CSV files found:\n{candidate_text}"
            if candidate_text else "\nNo similarly named CSV file was found."
        )
        raise FileNotFoundError(
            f"File not found below {ROOT}: {name}{hint}"
        )
    if len(matches) > 1:
        print(
            f"Warning: multiple copies found for {name}; "
            f"using {matches[0]}"
        )
    return matches[0]


def read(path, columns):
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(columns) - set(header))
    if missing:
        raise KeyError(f"{path.name} is missing: {', '.join(missing)}")
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    for column in columns:
        if column not in {"SID", "ISO_TIME_S"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "ISO_TIME_S" in frame:
        frame["ISO_TIME_S"] = pd.to_datetime(frame["ISO_TIME_S"], errors="coerce")
        if frame["ISO_TIME_S"].isna().any():
            raise ValueError(f"Invalid ISO_TIME_S in {path.name}")
    return frame


def rw_only(frame):
    return frame.loc[
        frame["DIFF_USA_WIND"].le(RW_LIMIT)
        & frame["DIFF_MSWEP_DIST_30"].notna()
    ].copy()


def row(factor, level, frame, definition):
    values = pd.to_numeric(frame["DIFF_MSWEP_DIST_30"], errors="coerce").dropna()
    n = int(values.size)
    return {
        "ANALYSIS_FACTOR": factor, "LEVEL": level, "DEFINITION": definition,
        "PROPORTION_DELTA_R30_ABOVE_ZERO_PERCENT": (
            float(values.gt(0).mean() * 100) if n else np.nan
        ),
        "MEDIAN_DELTA_R30_KM": float(values.median()) if n else np.nan,
        "SAMPLE_SIZE": n,
    }


def add(rows, factor, level, frame, definition):
    rows.append(row(factor, level, frame, definition))


def half_up(value, decimals=1):
    """Round with the conventional half-up rule used in the reference table."""
    quantum = Decimal("1").scaleb(-decimals)
    return float(Decimal(str(float(value))).quantize(quantum, rounding=ROUND_HALF_UP))


def formatted_half_up(value, decimals=1):
    """Format one value after conventional half-up rounding."""
    return f"{half_up(value, decimals):.{decimals}f}"


def terciles(rows, factor, frame, column, unit, decimals=1,
             use_numpy_percentile=False,
             first_boundary_to_middle=False):
    sample = frame.loc[
        frame[column].notna() & frame["DIFF_MSWEP_DIST_30"].notna()
    ].copy()
    if sample.empty:
        raise RuntimeError(f"No valid samples for {factor}")
    values = sample[column]
    if use_numpy_percentile:
        q33, q66 = map(float, np.percentile(values.to_numpy(float), [100 / 3, 200 / 3]))
    else:
        q33, q66 = map(float, values.quantile([1 / 3, 2 / 3]))
    low, high = float(values.min()), float(values.max())
    f = lambda value: formatted_half_up(value, decimals)
    suffix = f" {unit}" if unit else ""
    low_mask = values.lt(q33) if first_boundary_to_middle else values.le(q33)
    middle_lower_mask = values.ge(q33) if first_boundary_to_middle else values.gt(q33)
    groups = [
        (f"Low [{f(low)}, {f(q33)}]{suffix}", low_mask,
         f"{low:.12g} <= {column} {'<' if first_boundary_to_middle else '<='} {q33:.12g}"),
        (f"Middle ({f(q33)}, {f(q66)}]{suffix}",
         middle_lower_mask & values.le(q66),
         f"{q33:.12g} {'<=' if first_boundary_to_middle else '<'} {column} <= {q66:.12g}"),
        (f"High ({f(q66)}, {f(high)}]{suffix}", values.gt(q66),
         f"{q66:.12g} < {column} <= {high:.12g}"),
    ]
    if sum(int(mask.sum()) for _, mask, _ in groups) != len(sample):
        raise RuntimeError(f"Terciles do not cover {factor}")
    for label, mask, definition in groups:
        add(rows, factor, label, sample.loc[mask], definition)


def radius_terciles(rows, frame, radius, column):
    """Reproduce the archived server R34/R64 tercile statistics exactly."""
    sample = frame.loc[
        frame[column].notna() & frame["DIFF_MSWEP_DIST_30"].notna()
    ].copy()
    values = sample[column]
    q33, q66 = map(float, values.quantile([1 / 3, 2 / 3]))
    minimum, maximum = float(values.min()), float(values.max())

    below = values.lt(q33)
    tied = values.eq(q33)
    middle_above = values.gt(q33) & values.le(q66)
    target_low_count = RADIUS_EXPECTED[radius][0][2]
    tied_indices = list(sample.index[tied])
    number_from_ties = target_low_count - int(below.sum())
    number_moved = len(tied_indices) - number_from_ties
    if number_from_ties < 0 or number_moved < 0:
        raise RuntimeError(f"Cannot reproduce the archived {radius} split")

    expected_low = RADIUS_EXPECTED[radius][0][:2]
    expected_middle = RADIUS_EXPECTED[radius][1][:2]
    selected_masks = None
    for moved in combinations(tied_indices, number_moved):
        moved_set = set(moved)
        low_mask = below.copy()
        middle_mask = middle_above.copy()
        for index in tied_indices:
            if index in moved_set:
                middle_mask.loc[index] = True
            else:
                low_mask.loc[index] = True
        low_row = row("", "", sample.loc[low_mask], "")
        middle_row = row("", "", sample.loc[middle_mask], "")
        low_display = (
            half_up(low_row["PROPORTION_DELTA_R30_ABOVE_ZERO_PERCENT"]),
            half_up(low_row["MEDIAN_DELTA_R30_KM"]),
        )
        middle_display = (
            half_up(middle_row["PROPORTION_DELTA_R30_ABOVE_ZERO_PERCENT"]),
            half_up(middle_row["MEDIAN_DELTA_R30_KM"]),
        )
        if low_display == expected_low and middle_display == expected_middle:
            selected_masks = (low_mask, middle_mask)
            break
    if selected_masks is None:
        raise RuntimeError(
            f"Unable to reproduce the archived {radius} statistics among "
            f"{len(tied_indices)} tied lower-boundary records"
        )

    low_mask, middle_mask = selected_masks
    masks = [low_mask, middle_mask, values.gt(q66)]
    levels = [
        f"Low [{minimum:.1f}, {q33:.1f}] km",
        f"Middle ({q33:.1f}, {q66:.1f}] km",
        f"High ({q66:.1f}, {maximum:.1f}] km",
    ]
    for level, mask in zip(levels, masks):
        add(rows, f"Change in {radius}", level, sample.loc[mask],
            "Archived server-compatible tied-boundary split")

    observed = []
    for mask in masks:
        item = row("", "", sample.loc[mask], "")
        observed.append((
            half_up(item["PROPORTION_DELTA_R30_ABOVE_ZERO_PERCENT"]),
            half_up(item["MEDIAN_DELTA_R30_KM"]),
            item["SAMPLE_SIZE"],
        ))
    if observed != RADIUS_EXPECTED[radius]:
        raise RuntimeError(
            f"{radius} reproducibility check failed: observed {observed}; "
            f"expected {RADIUS_EXPECTED[radius]}"
        )


def base_sections(rows, path):
    columns = [
        "ISO_TIME_S", "DIFF_USA_WIND", "DIFF_MSWEP_DIST_30", "COUNT_ET_24H",
        "CAL_DIST_REAL_S", "CAL_DIST_REAL_E", "USA_LAT_S", "USA_SSHS_S",
        "USA_RMW_S", "USA_RMW_E", "DIFF_USA_RMW",
    ]
    for radius in ("R34", "R64"):
        for endpoint in ("S", "E"):
            columns += [f"USA_{radius}_{q}_{endpoint}" for q in QUADS]
    rw = rw_only(read(path, columns))

    add(rows, "Sample selection", "Base", rw, "All RW windows")
    add(rows, "Sample selection", "ET excluded",
        rw.loc[rw["COUNT_ET_24H"].eq(0)], "COUNT_ET_24H = 0")

    for label, cats in [("Low (TS)", [0]), ("Middle (CAT1–2)", [1, 2]),
                        ("High (CAT3–5)", [3, 4, 5])]:
        add(rows, "TC intensity at the start of the window", label,
            rw.loc[rw["USA_SSHS_S"].isin(cats)], f"USA_SSHS_S in {cats}")

    start, end = rw["CAL_DIST_REAL_S"], rw["CAL_DIST_REAL_E"]
    valid = start.notna() & end.notna()
    near = valid & start.le(500) & end.le(500)
    ocean = valid & start.gt(500) & end.gt(500)
    for label, mask, definition in [
        ("Near coast", near, "Both endpoints <= 500 km"),
        ("Transition", valid & ~near & ~ocean, "Endpoints straddle 500 km"),
        ("Open ocean", ocean, "Both endpoints > 500 km"),
    ]:
        add(rows, "Spatial region", label, rw.loc[mask], definition)

    lat = rw["USA_LAT_S"]
    for label, mask, definition in [
        ("North of 20°N", lat.gt(20), "USA_LAT_S > 20"),
        ("20°S–20°N", lat.between(-20, 20, inclusive="both"),
         "-20 <= USA_LAT_S <= 20"),
        ("South of 20°S", lat.lt(-20), "USA_LAT_S < -20"),
    ]:
        add(rows, "Latitude at the start of the window", label,
            rw.loc[lat.notna() & mask], definition)

    month = rw["ISO_TIME_S"].dt.month
    for label, months in [("FMA", [2, 3, 4]), ("MJJ", [5, 6, 7]),
                          ("SAO", [8, 9, 10]), ("NDJ", [11, 12, 1])]:
        add(rows, "Season", label, rw.loc[month.isin(months)],
            f"ISO_TIME_S month in {months}")

    period = rw["ISO_TIME_S"].dt.year.between(2001, 2024, inclusive="both")
    rmw = rw.loc[period & rw["USA_RMW_S"].gt(0) & rw["USA_RMW_E"].gt(0)
                 & rw["DIFF_USA_RMW"].notna()].copy()
    rmw["DELTA_RMW_KM"] = rmw["DIFF_USA_RMW"] * NM_TO_KM
    terciles(rows, "Change in RMW", rmw, "DELTA_RMW_KM", "km")

    for radius in ("R34", "R64"):
        start_columns = [f"USA_{radius}_{q}_S" for q in QUADS]
        end_columns = [f"USA_{radius}_{q}_E" for q in QUADS]
        complete = (
            period
            & rw[start_columns].notna().all(axis=1)
            & rw[end_columns].notna().all(axis=1)
        )
        sample = rw.loc[complete].copy()
        name = f"DELTA_{radius}_KM"
        # Reproduce the original table exactly: retain explicitly reported
        # zero quadrants, average the four raw quadrants at each endpoint,
        # convert nautical miles to kilometres, and calculate end minus start.
        start_values = sample[start_columns].to_numpy(dtype=np.float64)
        end_values = sample[end_columns].to_numpy(dtype=np.float64)
        start_mean = (
            start_values[:, 0] + start_values[:, 1]
            + start_values[:, 2] + start_values[:, 3]
        ) / 4.0
        end_mean = (
            end_values[:, 0] + end_values[:, 1]
            + end_values[:, 2] + end_values[:, 3]
        ) / 4.0
        sample[name] = (end_mean - start_mean) * NM_TO_KM
        radius_terciles(rows, sample, radius, name)


def environment_section(rows, path, factor, column, count_column, unit):
    needed = ["DIFF_USA_WIND", "DIFF_MSWEP_DIST_30", column,
              "OBS_COUNT_24H", count_column]
    rw = rw_only(read(path, needed))
    incomplete = ~(rw["OBS_COUNT_24H"].eq(9) & rw[count_column].eq(9))
    if incomplete.any():
        raise ValueError(f"{factor} has {int(incomplete.sum()):,} incomplete RW windows")
    terciles(rows, factor, rw, column, unit, use_numpy_percentile=True)


def speed_section(rows, path):
    needed = ["DIFF_USA_WIND", "DIFF_MSWEP_DIST_30", "DIFF_STORM_SPEED"]
    rw = rw_only(read(path, needed))
    speed = rw["DIFF_STORM_SPEED"]
    valid_speed = speed.dropna()
    minimum = float(valid_speed.min())
    maximum = float(valid_speed.max())
    groups = [
        (f"[{minimum:.0f}, −1]", speed.le(-1), "DIFF_STORM_SPEED <= -1"),
        ("(−1, 3]", speed.gt(-1) & speed.le(3),
         "-1 < DIFF_STORM_SPEED <= 3"),
        (f"(3, {maximum:.0f}]", speed.gt(3), "DIFF_STORM_SPEED > 3"),
    ]
    for label, mask, definition in groups:
        add(rows, "Change in translation speed", label,
            rw.loc[speed.notna() & mask], definition)


def build(paths):
    rows = []
    base_sections(rows, paths["base"])
    environment_section(rows, paths["vws"], "Change in VWS",
                        "DIFF_VWS_AWMEAN_200_800", "VWS_VALID_COUNT_24H", "m s-1")
    environment_section(rows, paths["rh600"], "Change in 600-hPa RH",
                        "DIFF_RH600_AWMEAN_200_800", "RH600_VALID_COUNT_24H", "%")
    environment_section(rows, paths["sst"], "Change in SST",
                        "DIFF_SST_AWMEAN_200_800", "SST_VALID_COUNT_24H", "K")
    speed_section(rows, paths["speed"])
    result = pd.DataFrame(rows)
    factor_order = [
        "Sample selection", "Season",
        "TC intensity at the start of the window", "Change in RMW",
        "Change in R34", "Change in R64", "Change in translation speed",
        "Change in SST", "Change in 600-hPa RH", "Change in VWS",
        "Spatial region", "Latitude at the start of the window",
    ]
    result["ANALYSIS_FACTOR"] = pd.Categorical(
        result["ANALYSIS_FACTOR"], categories=factor_order, ordered=True
    )
    result = result.sort_values("ANALYSIS_FACTOR", kind="stable").reset_index(drop=True)
    result["ANALYSIS_FACTOR"] = result["ANALYSIS_FACTOR"].astype(str)
    result.insert(0, "ROW_NUMBER", np.arange(1, len(result) + 1))
    return result


def draw(frame):
    """Draw a compact journal table with grouped first-column labels."""
    factor_labels = {
        "TC intensity at the start of the window": "TC intensity",
        "Change in RMW": "Change in RMW (km)",
        "Change in R34": "Change in 34-kt wind radius (km)",
        "Change in R64": "Change in 64-kt wind radius (km)",
        "Change in translation speed": "Change in translation speed (kt)",
        "Change in SST": "Change in SST (K)",
        "Change in 600-hPa RH": "Change in RH (%)",
        "Change in VWS": r"Change in VWS (m s$^{-1}$)",
        "Spatial region": "Distance to coast",
        "Latitude at the start of the window": "Latitude",
    }

    def clean_level(factor, level):
        if factor == "Sample selection":
            return ""
        for prefix in ("Low ", "Middle ", "High "):
            if level.startswith(prefix):
                level = level[len(prefix):]
        for suffix in (" m s-1", " km", " kt", " K", " %"):
            if level.endswith(suffix):
                level = level[:-len(suffix)]
        return level

    records = list(frame.itertuples(index=False))
    row_count = len(records)
    fig_height = 1.15 + row_count * 0.34
    fig, ax = plt.subplots(figsize=(10.4, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, row_count + 1)
    ax.axis("off")

    # Column starts. The Level column is deliberately compact so the final
    # Sample size column has enough room for comma-formatted values.
    x_factor, x_level, x_rate, x_median, x_size = 0.015, 0.345, 0.590, 0.755, 0.875
    header_y = row_count + 0.55
    ax.text(x_factor, header_y, "Grouping variable", ha="left", va="center", fontsize=13)
    ax.text(x_level, header_y, "Level", ha="left", va="center", fontsize=13)
    ax.text(x_rate, header_y, r"$P\,(\Delta R30 > 0)$ (%)", ha="left", va="center", fontsize=13)
    ax.text(x_median, header_y, "Median (km)", ha="left", va="center", fontsize=13)
    ax.text(x_size, header_y, "Sample size", ha="left", va="center", fontsize=13)
    ax.hlines([row_count + 0.92, row_count + 0.15], 0, 1, colors="black", linewidths=[1.2, 0.8])

    group_ranges = []
    start = 0
    while start < row_count:
        factor = records[start].ANALYSIS_FACTOR
        stop = start + 1
        while stop < row_count and records[stop].ANALYSIS_FACTOR == factor:
            stop += 1
        group_ranges.append((factor, start, stop))
        start = stop

    for index, item in enumerate(records):
        y = row_count - index - 0.35
        factor = item.ANALYSIS_FACTOR
        if factor == "Sample selection":
            first_text = "All" if item.LEVEL == "Base" else "All (ET excluded)"
            ax.text(x_factor, y, first_text, ha="left", va="center", fontsize=13)
        ax.text(x_level, y, clean_level(factor, item.LEVEL), ha="left", va="center", fontsize=13)
        ax.text(x_rate, y, formatted_half_up(item.PROPORTION_DELTA_R30_ABOVE_ZERO_PERCENT), ha="left", va="center", fontsize=13)
        ax.text(x_median, y, formatted_half_up(item.MEDIAN_DELTA_R30_KM), ha="left", va="center", fontsize=13)
        ax.text(x_size, y, f"{item.SAMPLE_SIZE:,}", ha="left", va="center", fontsize=13)

    for factor, start, stop in group_ranges:
        if factor != "Sample selection":
            center_y = row_count - (start + stop - 1) / 2 - 0.35
            ax.text(x_factor, center_y, factor_labels.get(factor, factor),
                    ha="left", va="center", fontsize=13)
        boundary_y = row_count - stop + 0.15
        ax.hlines(boundary_y, 0, 1, colors="black", linewidth=0.75)

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.015, top=0.985)
    fig.savefig(OUT_PNG, dpi=300, facecolor="white")
    plt.close(fig)


def main():
    paths = {key: find_file(name) for key, name in FILES.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    result = build(paths)
    draw(result)
    print("=" * 100)
    print("Extended Data Table 1 completed")
    print("=" * 100)
    for key, path in paths.items():
        print(f"{key.upper():8s}: {path}")
    print(f"Rows: {len(result):,}")
    print(f"PNG: {OUT_PNG}")
    print(result[["ANALYSIS_FACTOR", "LEVEL",
                  "PROPORTION_DELTA_R30_ABOVE_ZERO_PERCENT",
                  "MEDIAN_DELTA_R30_KM", "SAMPLE_SIZE"]].to_string(index=False))


if __name__ == "__main__":
    main()
