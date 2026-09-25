#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXFIG9A: annual trend in nearshore RW 24-hour sample counts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd
from scipy.stats import linregress, norm, t

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_CSV = Path(
    str(PROJECT_ROOT / "Data" / "Processed" / "MSWEP_24H_RESULTS") + "/" +
    "PRE_DATA_IBT_1982_2024_MSWEP_TH30_24H_SLIDING_ET_ALL.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "Results" / "Extended_figures" / "EX_FIG9"
OUTPUT_PNG = OUTPUT_DIR / "EX_FIG9A.png"

START_YEAR = 1982
END_YEAR = 2024
EXPECTED_SAMPLE_COUNT = 5774

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
    "legend.fontsize": FONT_SIZE,
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


def mann_kendall_test(values):
    """Classical two-sided Mann–Kendall test with tie correction."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        raise ValueError("Mann–Kendall检验至少需要两个有效年份。")

    s_stat = 0
    for index in range(n - 1):
        s_stat += np.sign(x[index + 1:] - x[index]).sum()

    _, tie_counts = np.unique(x, return_counts=True)
    tie_term = np.sum(
        tie_counts * (tie_counts - 1) * (2 * tie_counts + 5)
    )
    variance = (
        n * (n - 1) * (2 * n + 5) - tie_term
    ) / 18.0

    if variance <= 0:
        z_score = 0.0
    elif s_stat > 0:
        z_score = (s_stat - 1) / np.sqrt(variance)
    elif s_stat < 0:
        z_score = (s_stat + 1) / np.sqrt(variance)
    else:
        z_score = 0.0

    p_value = 2.0 * norm.sf(abs(z_score))
    tau = s_stat / (0.5 * n * (n - 1))
    return float(s_stat), float(z_score), float(p_value), float(tau)


def read_and_count():
    required = [
        "SID",
        "ISO_TIME_S",
        "DIFF_USA_WIND",
        "CAL_DIST_REAL_S",
        "CAL_DIST_REAL_E",
    ]
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV, usecols=required, low_memory=False)
    df["ISO_TIME_S"] = pd.to_datetime(df["ISO_TIME_S"], errors="coerce")
    for column in [
        "DIFF_USA_WIND", "CAL_DIST_REAL_S", "CAL_DIST_REAL_E"
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    selected = df[
        (df["DIFF_USA_WIND"] <= -30)
        & (df["CAL_DIST_REAL_S"] <= 500)
        & (df["CAL_DIST_REAL_E"] <= 500)
    ].copy()

    if len(selected) != EXPECTED_SAMPLE_COUNT:
        print(
            "Warning: selected sample count is "
            f"{len(selected):,}, expected {EXPECTED_SAMPLE_COUNT:,}."
        )

    selected["YEAR"] = selected["ISO_TIME_S"].dt.year
    years = pd.Index(range(START_YEAR, END_YEAR + 1), name="YEAR")
    annual = (
        selected.dropna(subset=["YEAR"])
        .groupby("YEAR")
        .size()
        .reindex(years, fill_value=0)
        .rename("RW_24H_SAMPLE_COUNT")
        .reset_index()
    )
    return selected, annual


def plot_result(annual):
    years = annual["YEAR"].to_numpy(dtype=float)
    counts = annual["RW_24H_SAMPLE_COUNT"].to_numpy(dtype=float)

    s_stat, z_score, p_value, tau = mann_kendall_test(counts)
    # OLS supplies the trend slope, dashed line and two-sided 95% CI.
    ols = linregress(years, counts)
    fitted = ols.intercept + ols.slope * years
    slope_decade = ols.slope * 10.0

    n = len(years)
    residuals = counts - fitted
    residual_standard_error = np.sqrt(
        np.sum(residuals ** 2) / (n - 2)
    )
    year_mean = np.mean(years)
    sxx = np.sum((years - year_mean) ** 2)
    mean_fit_standard_error = residual_standard_error * np.sqrt(
        1.0 / n + (years - year_mean) ** 2 / sxx
    )
    t_critical = t.ppf(0.975, df=n - 2)
    confidence_low = fitted - t_critical * mean_fit_standard_error
    confidence_high = fitted + t_critical * mean_fit_standard_error

    annual["OLS_FITTED_COUNT"] = fitted
    annual["OLS_95CI_LOW"] = confidence_low
    annual["OLS_95CI_HIGH"] = confidence_high
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.18, top=0.94)

    ax.plot(
        years,
        counts,
        color=SERIES_COLOR,
        linewidth=SERIES_WIDTH,
        label="Annual count",
        zorder=3,
    )
    ax.plot(
        years,
        fitted,
        color=TREND_COLOR,
        linewidth=TREND_WIDTH,
        linestyle="--",
        zorder=4,
    )
    ax.fill_between(
        years,
        confidence_low,
        confidence_high,
        color=TREND_COLOR,
        alpha=0.20,
        linewidth=0,
        zorder=2,
    )

    ax.set_xlim(START_YEAR - 1, END_YEAR + 1)
    ax.set_xticks(np.arange(1990, 2021, 10))
    ax.set_xlabel("Year", labelpad=10)
    ax.set_ylabel("Number of near-coast RW window", labelpad=10)
    ax.set_ylim(0, 250)
    ax.set_yticks(np.arange(0, 251, 50))

    ax.grid(
        True,
        axis="y",
        color=GRID_COLOR,
        linestyle="--",
        linewidth=GRID_WIDTH,
        alpha=0.55,
        zorder=0,
    )

    p_text = "$P$ < 0.01"
    annotation = (
        f"Trend: {slope_decade:.1f} decade$^{{-1}}$, "
        f"{p_text}"
    )
    ax.text(
        0.03,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_SIZE,
        color="black",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_PNG,
        dpi=DPI,
        format="png",
        facecolor="white",
        edgecolor="white",
    )
    plt.close(fig)

    print("=" * 72)
    print("EXFIG9A: annual nearshore RW trend")
    print("=" * 72)
    print(f"Selected 24-hour samples: {int(annual['RW_24H_SAMPLE_COUNT'].sum()):,}")
    print(f"Years: {START_YEAR}-{END_YEAR}")
    print(f"Mann-Kendall S: {s_stat:.0f}")
    print(f"Mann-Kendall Z: {z_score:.4f}")
    print(f"Mann-Kendall tau: {tau:.4f}")
    print(f"Mann-Kendall p: {p_value:.6g}")
    print(f"OLS slope: {slope_decade:.6f} RW/decade")
    print(f"OLS R-squared: {ols.rvalue ** 2:.6f}")
    print(f"OLS p (not used in figure): {ols.pvalue:.8f}")
    print("Figure P value is from the Mann-Kendall test, not OLS.")
    print(f"Saved figure: {OUTPUT_PNG}")


def main():
    _, annual = read_and_count()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_result(annual)


if __name__ == "__main__":
    main()
