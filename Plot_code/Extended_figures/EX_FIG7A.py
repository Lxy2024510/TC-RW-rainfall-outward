#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize, LinearSegmentedColormap

DATA_DIR = Path(PROJECT_ROOT / "Data" / "Figure_data" / "Extended_figures")
FIG_DIR = PROJECT_ROOT / "Results" / "Extended_figures" / "EX_FIG7"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_DIR / "CTRL_EXP_intensity_timeseries.csv"

OUT_VMAX_FIG = FIG_DIR / "EX_FIG7A.png"

plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.cursive": ["Arial"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.cal": "Arial",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.titlesize": 20,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.linewidth": 1.8,
    "grid.linewidth": 1.0,
    "lines.linewidth": 2.2,
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "xtick.major.width": 1.8,
    "ytick.major.width": 1.8,
    "xtick.minor.width": 1.4,
    "ytick.minor.width": 1.4,
    "xtick.major.size": 6.0,
    "ytick.major.size": 6.0,
})

CTRL_REFERENCE_START = datetime(2007, 9, 2, 0, 0, 0)

INITIAL_CASES = {
    42: datetime(2007, 9, 4, 19, 0, 0),
    43: datetime(2007, 9, 4, 23, 0, 0),
    44: datetime(2007, 9, 5, 15, 0, 0),
    45: datetime(2007, 9, 5, 19, 0, 0),
    46: datetime(2007, 9, 5, 1, 0, 0),
    47: datetime(2007, 9, 5, 2, 0, 0),
    48: datetime(2007, 9, 6, 3, 0, 0),
    49: datetime(2007, 9, 6, 0, 0, 0),
    50: datetime(2007, 9, 6, 13, 0, 0),
    51: datetime(2007, 9, 7, 3, 0, 0),
}

CTRL_START_HOUR = {
    strength: int(
        (start_time - CTRL_REFERENCE_START).total_seconds() / 3600.0
    )
    for strength, start_time in INITIAL_CASES.items()
}

PLOT_START_HOUR = 0
PLOT_END_HOUR = 30

SMOOTH_WINDOW = 5
SMOOTH_CENTER = True
SMOOTH_MIN_PERIODS = 1

DPI = 300
FIGSIZE = (9.0, 5.63)

AXES_LEFT = 0.14
AXES_RIGHT = 0.98
AXES_BOTTOM = 0.17
AXES_TOP = 0.95

PRESSURE_BLUE_COLORS = ["#56A1B8", "#6FBFD7", "#A3DCEC"]

CTRL_LINESTYLE = "--"
EXP_LINESTYLE = "-"
CTRL_LINEWIDTH = 1.8
EXP_LINEWIDTH = 2.2
CTRL_ALPHA = 0.9
EXP_ALPHA = 0.9

GRID_ALPHA = 0.55
SHOW_GRID = True

VMAX_YMIN = 15.0
VMAX_YMAX = 55.0
VMAX_YTICK = 10.0

REQUIRED_COLUMNS = [
    "experiment",
    "case_id",
    "initial_strength",
    "forecast_hour",
    "vmax10_ms",
]


def read_data():
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"找不到CSV：{RAW_CSV}")

    df = pd.read_csv(
        RAW_CSV,
        dtype={
            "experiment": "string",
            "case_id": "string",
        },
    )

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"CSV缺少列：{missing}；现有列：{df.columns.tolist()}"
        )

    df["experiment"] = (
        df["experiment"].astype("string").str.strip()
    )

    for c in [
        "initial_strength",
        "forecast_hour",
        "vmax10_ms",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def build_ctrl_branches(df):
    ctrl = df[df["experiment"] == "CTRL"].copy()

    if ctrl.empty:
        raise RuntimeError("CSV中没有CTRL数据。")

    rows = []

    for strength in sorted(INITIAL_CASES):
        start_hour = CTRL_START_HOUR[strength]
        end_hour = start_hour + PLOT_END_HOUR

        branch = ctrl[
            (ctrl["forecast_hour"] >= start_hour)
            & (ctrl["forecast_hour"] <= end_hour)
        ].copy()

        if branch.empty:
            raise RuntimeError(
                f"无法构造CTRL strength={strength} 分支；"
                f"需要CTRL forecast_hour={start_hour}–{end_hour}。"
            )

        branch["experiment"] = "CTRL"
        branch["case_id"] = f"CTRL_{strength}"
        branch["initial_strength"] = float(strength)
        branch["forecast_hour"] = (
            branch["forecast_hour"] - start_hour
        )

        rows.append(branch)

    return pd.concat(rows, ignore_index=True)


def build_exp_branches(df):
    exp = df[df["experiment"] == "EXP"].copy()

    if exp.empty:
        raise RuntimeError("CSV中没有EXP数据。")

    exp = exp[
        exp["initial_strength"].isin(sorted(INITIAL_CASES))
        & (exp["forecast_hour"] >= PLOT_START_HOUR)
        & (exp["forecast_hour"] <= PLOT_END_HOUR)
    ].copy()

    exp["case_id"] = exp["initial_strength"].apply(
        lambda x: f"EXP_{int(x)}" if pd.notna(x) else "EXP"
    )

    return exp


def build_branches(df):
    combined = pd.concat(
        [
            build_ctrl_branches(df),
            build_exp_branches(df),
        ],
        ignore_index=True,
    )

    return combined.sort_values(
        [
            "experiment",
            "initial_strength",
            "forecast_hour",
        ]
    ).reset_index(drop=True)


def smooth_one_case(group):
    group = group.sort_values("forecast_hour").copy()

    full_hours = np.arange(
        PLOT_START_HOUR,
        PLOT_END_HOUR + 1,
        1,
        dtype=float,
    )

    indexed = (
        group
        .set_index("forecast_hour")
        .reindex(full_hours)
    )
    indexed.index.name = "forecast_hour"

    for c in ["experiment", "case_id", "initial_strength"]:
        valid = group[c].dropna()
        indexed[c] = (
            valid.iloc[0]
            if not valid.empty
            else np.nan
        )

    indexed["vmax10_ms_5point"] = (
        indexed["vmax10_ms"]
        .rolling(
            window=SMOOTH_WINDOW,
            center=SMOOTH_CENTER,
            min_periods=SMOOTH_MIN_PERIODS,
        )
        .mean()
    )

    return indexed.reset_index()


def smooth_all(branches):
    parts = []

    for _, group in branches.groupby(
        "case_id",
        sort=False,
    ):
        parts.append(
            smooth_one_case(group)
        )

    return pd.concat(
        parts,
        ignore_index=True,
    ).sort_values(
        [
            "experiment",
            "initial_strength",
            "forecast_hour",
        ]
    )


def build_strength_colors():
    cmap = LinearSegmentedColormap.from_list(
        "strength_gradient_custom",
        PRESSURE_BLUE_COLORS,
        N=256,
    )

    strengths = sorted(INITIAL_CASES)
    norm = Normalize(
        vmin=min(strengths),
        vmax=max(strengths),
    )

    return {
        strength: cmap(norm(strength))
        for strength in strengths
    }


def plot_timeseries(
    df,
    variable,
    ylabel,
    output_path,
    y_min,
    y_max,
    y_tick,
):
    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    fig.subplots_adjust(
        left=AXES_LEFT,
        right=AXES_RIGHT,
        bottom=AXES_BOTTOM,
        top=AXES_TOP,
    )

    strength_colors = (
        build_strength_colors()
    )

    for strength in sorted(INITIAL_CASES):
        color = strength_colors[strength]

        ctrl = df[
            (df["experiment"] == "CTRL")
            & (df["initial_strength"] == strength)
        ].sort_values("forecast_hour")

        exp = df[
            (df["experiment"] == "EXP")
            & (df["initial_strength"] == strength)
        ].sort_values("forecast_hour")

        if not ctrl.empty:
            ax.plot(
                ctrl["forecast_hour"],
                ctrl[variable],
                color=color,
                linestyle=CTRL_LINESTYLE,
                linewidth=CTRL_LINEWIDTH,
                alpha=CTRL_ALPHA,
                zorder=3,
            )

        if not exp.empty:
            ax.plot(
                exp["forecast_hour"],
                exp[variable],
                color=color,
                linestyle=EXP_LINESTYLE,
                linewidth=EXP_LINEWIDTH,
                alpha=EXP_ALPHA,
                zorder=4,
            )

    ax.set_xlim(
        PLOT_START_HOUR,
        PLOT_END_HOUR,
    )
    ax.set_xticks(
        np.arange(0, 31, 5)
    )
    ax.set_xlabel("Time (h)")
    ax.set_ylabel(ylabel)

    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(20.0, 51.0, y_tick))

    if SHOW_GRID:
        ax.grid(
            True,
            linestyle="--",
            linewidth=1.0,
            alpha=GRID_ALPHA,
        )

    fig.savefig(
        output_path,
        format="png",
        dpi=DPI,
        facecolor="white",
        edgecolor="white",
    )
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    raw = read_data()
    branches = build_branches(raw)
    smoothed = smooth_all(branches)

    plot_timeseries(
        df=smoothed,
        variable="vmax10_ms_5point",
        ylabel=r"Maximum 10-m wind speed (m s$^{-1}$)",
        output_path=OUT_VMAX_FIG,
        y_min=VMAX_YMIN,
        y_max=VMAX_YMAX,
        y_tick=VMAX_YTICK,
    )


if __name__ == "__main__":
    main()
