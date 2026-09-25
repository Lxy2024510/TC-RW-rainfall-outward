#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXFIG9B: annual nearshore RW 24-h windows contributed per nearshore TC."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd
from scipy.stats import linregress, norm, t

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


WINDOW_CSV = Path(
    str(PROJECT_ROOT / "Data" / "Processed" / "MSWEP_24H_RESULTS") + "/" +
    "PRE_DATA_IBT_1982_2024_MSWEP_TH30_24H_SLIDING_ET_ALL.csv"
)
TRACK_CSV = Path(
    str(PROJECT_ROOT / "Data" / "Processed" / "MSWEP_RESULTS") + "/" +
    "PRE_DATA_IBT_1982_2024_MSWEP_THRESHOLD_30.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "Results" / "Extended_figures" / "EX_FIG9"
OUTPUT_PNG = OUTPUT_DIR / "EX_FIG9B.png"

START_YEAR = 1982
END_YEAR = 2024
RW_THRESHOLD_KT = -30.0
TC_THRESHOLD_KT = 34.0
NEARSHORE_LIMIT_KM = 500.0
EXPECTED_RW_WINDOWS = 5774

FIGSIZE = (9.0, 5.63)
DPI = 300
FONT_SIZE = 20
FRAME_WIDTH = 1.8
SERIES_WIDTH = 2.2
TREND_WIDTH = 2.2
GRID_WIDTH = 1.0
SERIES_COLOR = "#56A1B8"
TREND_COLOR = SERIES_COLOR
GRID_COLOR = "#9E9E9E"


plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.cursive": ["Arial"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.cal": "Arial",
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.linewidth": FRAME_WIDTH,
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "xtick.major.width": FRAME_WIDTH,
    "ytick.major.width": FRAME_WIDTH,
    "xtick.major.size": 6.0,
    "ytick.major.size": 6.0,
})


def clean_sid(series):
    sid = series.astype("string").str.strip()
    return sid.mask(
        sid.isna() | sid.eq("") | sid.str.lower().isin(["nan", "none", "null"])
    )


def mann_kendall_test(values):
    """Classical two-sided Mann–Kendall test with tie correction."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 3:
        raise ValueError("Mann–Kendall检验至少需要3个有效年份。")

    s_value = 0.0
    for index in range(n - 1):
        s_value += np.sign(values[index + 1:] - values[index]).sum()

    _, counts = np.unique(values, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    variance = (
        n * (n - 1) * (2 * n + 5) - tie_term
    ) / 18.0

    if variance <= 0 or s_value == 0:
        z_value = 0.0
    elif s_value > 0:
        z_value = (s_value - 1.0) / np.sqrt(variance)
    else:
        z_value = (s_value + 1.0) / np.sqrt(variance)

    p_value = 2.0 * norm.sf(abs(z_value))
    total_pairs = n * (n - 1) / 2.0
    tied_pairs = np.sum(counts * (counts - 1) / 2.0)
    denominator = np.sqrt(total_pairs * (total_pairs - tied_pairs))
    tau_b = s_value / denominator if denominator > 0 else 0.0

    return {
        "n": int(n),
        "S": float(s_value),
        "variance": float(variance),
        "z": float(z_value),
        "p": float(p_value),
        "tau_b": float(tau_b),
    }


def annual_rw_window_counts():
    columns = [
        "SID", "ISO_TIME_S", "DIFF_USA_WIND",
        "CAL_DIST_REAL_S", "CAL_DIST_REAL_E",
    ]
    if not WINDOW_CSV.exists():
        raise FileNotFoundError(WINDOW_CSV)

    df = pd.read_csv(WINDOW_CSV, usecols=columns, low_memory=False)
    df["SID"] = clean_sid(df["SID"])
    df["ISO_TIME_S"] = pd.to_datetime(df["ISO_TIME_S"], errors="coerce")
    for column in ["DIFF_USA_WIND", "CAL_DIST_REAL_S", "CAL_DIST_REAL_E"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    required = [
        "ISO_TIME_S", "DIFF_USA_WIND",
        "CAL_DIST_REAL_S", "CAL_DIST_REAL_E",
    ]
    valid = df.dropna(subset=required).copy()
    rw = valid[
        (valid["DIFF_USA_WIND"] <= RW_THRESHOLD_KT)
        & (valid["CAL_DIST_REAL_S"] <= NEARSHORE_LIMIT_KM)
        & (valid["CAL_DIST_REAL_E"] <= NEARSHORE_LIMIT_KM)
    ].copy()
    rw["YEAR"] = rw["ISO_TIME_S"].dt.year
    rw = rw[rw["YEAR"].between(START_YEAR, END_YEAR)].copy()

    if len(rw) != EXPECTED_RW_WINDOWS:
        print(
            f"Warning: selected {len(rw):,} RW windows; "
            f"expected {EXPECTED_RW_WINDOWS:,}."
        )

    years = pd.Index(range(START_YEAR, END_YEAR + 1), name="YEAR")
    annual = (
        rw.groupby("YEAR").size()
        .reindex(years, fill_value=0)
        .astype(int)
        .rename("NEARSHORE_RW_24H_WINDOW_COUNT")
        .reset_index()
    )
    return annual, rw


def annual_nearshore_tc_counts():
    columns = ["SID", "ISO_TIME", "USA_WIND", "CAL_DIST_REAL"]
    if not TRACK_CSV.exists():
        raise FileNotFoundError(TRACK_CSV)

    df = pd.read_csv(TRACK_CSV, usecols=columns, low_memory=False)
    df["SID"] = clean_sid(df["SID"])
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    df["USA_WIND"] = pd.to_numeric(df["USA_WIND"], errors="coerce")
    df["CAL_DIST_REAL"] = pd.to_numeric(df["CAL_DIST_REAL"], errors="coerce")
    df = df.dropna(subset=["SID", "ISO_TIME", "USA_WIND", "CAL_DIST_REAL"])

    # A SID is a TC if it reaches at least 34 kt anywhere in its lifetime.
    lifetime_max_wind = df.groupby("SID")["USA_WIND"].max()
    tc_sids = lifetime_max_wind[lifetime_max_wind >= TC_THRESHOLD_KT].index
    tc = df[df["SID"].isin(tc_sids)].copy()

    # One nearshore point is sufficient; assign each TC to its first-entry year.
    nearshore_points = tc[tc["CAL_DIST_REAL"] <= NEARSHORE_LIMIT_KM].copy()
    if nearshore_points.empty:
        raise RuntimeError("没有TC满足CAL_DIST_REAL <= 500 km。")

    first_entry = (
        nearshore_points.sort_values(["SID", "ISO_TIME"])
        .drop_duplicates("SID", keep="first")
        .rename(columns={
            "ISO_TIME": "FIRST_NEARSHORE_TIME",
            "CAL_DIST_REAL": "FIRST_NEARSHORE_DISTANCE_KM",
            "USA_WIND": "WIND_AT_FIRST_NEARSHORE_ENTRY_KT",
        })
    )
    first_entry["YEAR"] = first_entry["FIRST_NEARSHORE_TIME"].dt.year
    first_entry = first_entry[
        first_entry["YEAR"].between(START_YEAR, END_YEAR)
    ].copy()
    first_entry["LIFETIME_MAX_WIND_KT"] = first_entry["SID"].map(lifetime_max_wind)

    if first_entry["SID"].duplicated().any():
        raise RuntimeError("近海TC首次进入记录中仍存在重复SID。")

    years = pd.Index(range(START_YEAR, END_YEAR + 1), name="YEAR")
    annual = (
        first_entry.groupby("YEAR")["SID"].nunique()
        .reindex(years, fill_value=0)
        .astype(int)
        .rename("NEARSHORE_TC_COUNT")
        .reset_index()
    )
    return annual, first_entry


def combine_annual_counts(rw_annual, tc_annual):
    annual = rw_annual.merge(tc_annual, on="YEAR", how="outer", validate="one_to_one")
    annual = annual.sort_values("YEAR").reset_index(drop=True)
    if annual[["NEARSHORE_RW_24H_WINDOW_COUNT", "NEARSHORE_TC_COUNT"]].isna().any().any():
        raise RuntimeError("年度分子或分母存在缺失值。")
    if (annual["NEARSHORE_TC_COUNT"] <= 0).any():
        raise RuntimeError(
            "以下年份没有近海TC，无法计算比值："
            f"{annual.loc[annual['NEARSHORE_TC_COUNT'] <= 0, 'YEAR'].tolist()}"
        )

    annual["NEARSHORE_RW_PER_TC"] = (
        annual["NEARSHORE_RW_24H_WINDOW_COUNT"]
        / annual["NEARSHORE_TC_COUNT"]
    )
    return annual


def plot_trend(annual):
    years = annual["YEAR"].to_numpy(dtype=float)
    ratios = annual["NEARSHORE_RW_PER_TC"].to_numpy(dtype=float)
    mk = mann_kendall_test(ratios)

    # OLS provides the slope, dashed trend line and two-sided 95% CI.
    ols = linregress(years, ratios)
    trend = ols.intercept + ols.slope * years
    slope_decade = ols.slope * 10.0

    n = len(years)
    residuals = ratios - trend
    residual_standard_error = np.sqrt(
        np.sum(residuals ** 2) / (n - 2)
    )
    year_mean = np.mean(years)
    sxx = np.sum((years - year_mean) ** 2)
    mean_fit_standard_error = residual_standard_error * np.sqrt(
        1.0 / n + (years - year_mean) ** 2 / sxx
    )
    t_critical = t.ppf(0.975, df=n - 2)
    trend_ci_low = trend - t_critical * mean_fit_standard_error
    trend_ci_high = trend + t_critical * mean_fit_standard_error

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.18, top=0.94)
    ax.plot(
        years, ratios, color=SERIES_COLOR, linewidth=SERIES_WIDTH,
        zorder=3,
    )
    ax.plot(
        years, trend, color=TREND_COLOR, linewidth=TREND_WIDTH,
        linestyle="--", zorder=4,
    )
    ax.fill_between(
        years,
        trend_ci_low,
        trend_ci_high,
        color=TREND_COLOR,
        alpha=0.20,
        linewidth=0,
        zorder=2,
    )

    ax.set_xlim(START_YEAR - 1, END_YEAR + 1)
    ax.set_xticks(np.arange(1990, 2021, 10))
    ax.set_xlabel("Year", labelpad=10)
    ax.set_ylabel("Number of near-coast RW window\nper TC", labelpad=10)
    ax.set_ylim(0.5, 3.5)
    ax.set_yticks([1, 2, 3])
    ax.grid(
        True, axis="y", color=GRID_COLOR, linestyle="--",
        linewidth=GRID_WIDTH, alpha=0.55, zorder=0,
    )

    p_text = f"$P$ = {mk['p']:.2g}"
    ax.text(
        0.03, 0.96,
        f"Trend: {slope_decade:.1f} decade$^{{-1}}$, {p_text}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=FONT_SIZE, color="black",
    )

    fig.savefig(
        OUTPUT_PNG, dpi=DPI, format="png",
        facecolor="white", edgecolor="white",
    )
    plt.close(fig)

    print("=" * 76)
    print("EXFIG9B: annual nearshore RW contribution per nearshore TC")
    print("=" * 76)
    print(f"Nearshore RW windows: {annual['NEARSHORE_RW_24H_WINDOW_COUNT'].sum():,}")
    print(f"Nearshore TCs: {annual['NEARSHORE_TC_COUNT'].sum():,}")
    print(f"MK tau-b: {mk['tau_b']:.6f}")
    print(f"MK p: {mk['p']:.8f}")
    print(f"OLS slope: {slope_decade:.6f} per decade")
    print(f"OLS R-squared: {ols.rvalue ** 2:.6f}")
    print("Figure P value is from the Mann-Kendall test, not OLS.")
    print(f"Saved figure: {OUTPUT_PNG}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rw_annual, _ = annual_rw_window_counts()
    tc_annual, _ = annual_nearshore_tc_counts()
    annual = combine_annual_counts(rw_annual, tc_annual)
    plot_trend(annual)


if __name__ == "__main__":
    main()
