#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FIG7B: CTRL and EXP minimum-central-pressure time series."""

from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

DATA_DIR = Path(PROJECT_ROOT / "Data" / "Figure_data" / "Extended_figures")
FIG_DIR = PROJECT_ROOT / "Results" / "Extended_figures" / "EX_FIG7"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RAW_CSV = DATA_DIR / "CTRL_EXP_intensity_timeseries.csv"
OUT_FIG = FIG_DIR / "EX_FIG7B.png"

DPI = 300
FIGSIZE = (9.0, 5.63)
PLOT_END_HOUR = 30
SMOOTH_WINDOW = 5
YMIN, YMAX = 950.0, 1010.0
YTICKS = [960.0, 980.0, 1000.0]
COLORS = ["#56A1B8", "#6FBFD7", "#A3DCEC"]

CTRL_REFERENCE_START = datetime(2007, 9, 2, 0)
INITIAL_CASES = {
    42: datetime(2007, 9, 4, 19), 43: datetime(2007, 9, 4, 23),
    44: datetime(2007, 9, 5, 15), 45: datetime(2007, 9, 5, 19),
    46: datetime(2007, 9, 5, 1), 47: datetime(2007, 9, 5, 2),
    48: datetime(2007, 9, 6, 3), 49: datetime(2007, 9, 6, 0),
    50: datetime(2007, 9, 6, 13), 51: datetime(2007, 9, 7, 3),
}
CTRL_START_HOUR = {
    key: int((value - CTRL_REFERENCE_START).total_seconds() / 3600)
    for key, value in INITIAL_CASES.items()
}

plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.cursive": ["Arial"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.cal": "Arial",
    "font.size": 20, "axes.labelsize": 20,
    "xtick.labelsize": 20, "ytick.labelsize": 20,
    "axes.edgecolor": "black", "axes.labelcolor": "black",
    "axes.linewidth": 1.8, "xtick.color": "black",
    "ytick.color": "black", "text.color": "black",
    "xtick.major.width": 1.8, "ytick.major.width": 1.8,
    "xtick.major.size": 6.0, "ytick.major.size": 6.0,
})


def read_data():
    if not RAW_CSV.exists():
        raise FileNotFoundError(RAW_CSV)
    df = pd.read_csv(RAW_CSV, dtype={"experiment": "string", "case_id": "string"})
    required = ["experiment", "case_id", "initial_strength", "forecast_hour", "pmin_hpa"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise KeyError(f"缺少列：{missing}；现有列：{df.columns.tolist()}")
    df["experiment"] = df["experiment"].astype("string").str.strip()
    for name in ["initial_strength", "forecast_hour", "pmin_hpa"]:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    return df


def build_branches(df):
    ctrl = df[df["experiment"] == "CTRL"].copy()
    exp = df[df["experiment"] == "EXP"].copy()
    if ctrl.empty or exp.empty:
        raise RuntimeError("CSV必须同时包含CTRL和EXP数据。")

    ctrl_parts = []
    for strength in sorted(INITIAL_CASES):
        start = CTRL_START_HOUR[strength]
        part = ctrl[ctrl["forecast_hour"].between(start, start + PLOT_END_HOUR)].copy()
        if part.empty:
            raise RuntimeError(f"无法构造CTRL strength={strength}分支。")
        part["experiment"] = "CTRL"
        part["case_id"] = f"CTRL_{strength}"
        part["initial_strength"] = float(strength)
        part["forecast_hour"] -= start
        ctrl_parts.append(part)

    exp = exp[
        exp["initial_strength"].isin(INITIAL_CASES)
        & exp["forecast_hour"].between(0, PLOT_END_HOUR)
    ].copy()
    exp["case_id"] = exp["initial_strength"].map(lambda x: f"EXP_{int(x)}")
    return pd.concat(ctrl_parts + [exp], ignore_index=True)


def smooth_case(group):
    hours = np.arange(0, PLOT_END_HOUR + 1, dtype=float)
    group = group.sort_values("forecast_hour")
    out = group.set_index("forecast_hour").reindex(hours)
    out.index.name = "forecast_hour"
    for name in ["experiment", "case_id", "initial_strength"]:
        valid = group[name].dropna()
        out[name] = valid.iloc[0] if len(valid) else np.nan
    out["pmin_hpa_5point"] = out["pmin_hpa"].rolling(
        SMOOTH_WINDOW, center=True, min_periods=1
    ).mean()
    return out.reset_index()


def prepare_data(df):
    branches = build_branches(df)
    return pd.concat(
        [smooth_case(group) for _, group in branches.groupby("case_id", sort=False)],
        ignore_index=True,
    )


def strength_colors():
    cmap = LinearSegmentedColormap.from_list("strength_blue", COLORS, N=256)
    norm = Normalize(42, 51)
    return {strength: cmap(norm(strength)) for strength in INITIAL_CASES}


def plot_figure(df):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.17, top=0.95)
    colors = strength_colors()
    for strength in sorted(INITIAL_CASES):
        for experiment, style, width, zorder in [
            ("CTRL", "--", 1.8, 3), ("EXP", "-", 2.2, 4)
        ]:
            part = df[
                (df["experiment"] == experiment)
                & (df["initial_strength"] == strength)
            ].sort_values("forecast_hour")
            if not part.empty:
                ax.plot(
                    part["forecast_hour"], part["pmin_hpa_5point"],
                    color=colors[strength], linestyle=style,
                    linewidth=width, alpha=0.9, zorder=zorder,
                )

    legend_handles = [
        Line2D(
            [0], [0], color="black", linestyle="--", linewidth=1.8,
            label="CTRL",
        ),
        Line2D(
            [0], [0], color="black", linestyle="-", linewidth=2.2,
            label="EXP",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(1.02, 1.025),
        ncol=1,
        frameon=False,
        fontsize=20,
        handlelength=1.6,
        handletextpad=0.55,
        labelspacing=0.35,
        borderaxespad=0.0,
    )

    ax.set_xlim(0, 30)
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Minimum central pressure (hPa)")
    ax.set_ylim(YMIN, YMAX)
    ax.set_yticks(YTICKS)
    ax.grid(True, color="#9E9E9E", linestyle="--", linewidth=1.0, alpha=0.55)
    fig.savefig(
        OUT_FIG,
        format="png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
    )
    plt.close(fig)
    print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    plot_figure(prepare_data(read_data()))
