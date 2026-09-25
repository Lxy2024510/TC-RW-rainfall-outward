#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot closed-ring 500-700-hPa near-coast and open-ocean RW mass flux."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

INPUT_CSV = Path(PROJECT_ROOT / "Data" / "Processed" / "ERA5" / "MASS_FLUX_BOOTSTRAP_RESULTS_CLOSED_RINGS" / "RADIAL_MASS_FLUX_RW_STEADY_RI_1982_2024_MSWEP30_BOOTSTRAP_LONG.csv")
OUTPUT_PNG = PROJECT_ROOT / "Results" / "Extended_figures" / "EX_FIG6" / "EX_FIG6C.png"
PRODUCT = "500_700"
COMPONENTS = ["INNER_BOUNDARY", "NET_INFLUX", "OUTER_BOUNDARY"]
X_LABELS = ["300 km", "300–500 km\nnet flux", "500 km"]
SERIES = [("NEAR_COAST", "Near coast", "#D5A0B6"), ("OPEN_OCEAN", "Open ocean", "#4F9DB8")]

plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial"], "font.cursive": ["Arial"], "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold", "mathtext.cal": "Arial", "font.size": 32, "axes.labelsize": 32, "xtick.labelsize": 32, "ytick.labelsize": 32})

def tick_label(value, _position):
    if np.isclose(value, 0.0, atol=1.0): return "0"
    value /= 1.0e9
    return f"−{abs(value):g}" if value < 0 else f"{value:g}"

def get_data(frame, spatial_type):
    data = frame.loc[frame["SPATIAL_TYPE"].eq(spatial_type) & frame["INTENSITY_GROUP"].eq("RW") & frame["PRODUCT"].eq(PRODUCT) & frame["COMPONENT"].isin(COMPONENTS)].copy()
    data = data.set_index("COMPONENT").reindex(COMPONENTS)
    cols = ["OBSERVED_MEAN", "BOOTSTRAP_CI_LOWER", "BOOTSTRAP_CI_UPPER"]
    if len(data) != 3 or data[cols].isna().any(axis=None): raise ValueError(f"Incomplete plotting data for {spatial_type}")
    return data

def main():
    if not INPUT_CSV.is_file(): raise FileNotFoundError(f"Input CSV does not exist: {INPUT_CSV}")
    frame = pd.read_csv(INPUT_CSV, low_memory=False)
    required = {"SPATIAL_TYPE", "INTENSITY_GROUP", "PRODUCT", "COMPONENT", "OBSERVED_MEAN", "BOOTSTRAP_CI_LOWER", "BOOTSTRAP_CI_UPPER"}
    missing = sorted(required.difference(frame.columns))
    if missing: raise KeyError(f"Input CSV is missing columns: {missing}")
    x = np.arange(3, dtype=float)
    figure, axis = plt.subplots(figsize=(6.4, 12.8))
    for spatial_type, label, color in SERIES:
        data = get_data(frame, spatial_type)
        mean = data["OBSERVED_MEAN"].to_numpy(float); lower = data["BOOTSTRAP_CI_LOWER"].to_numpy(float); upper = data["BOOTSTRAP_CI_UPPER"].to_numpy(float)
        axis.fill_between(x, lower, upper, color=color, alpha=0.18, linewidth=0)
        axis.plot(x, mean, color=color, linewidth=4.0, marker="o", markersize=10.0, label=label)
    axis.legend(loc="lower right", bbox_to_anchor=(1.04, 1.025), frameon=False, fontsize=30, handlelength=1.6, handletextpad=0.55, labelspacing=0.18, borderaxespad=0.0)
    axis.axhline(0.0, color="#333333", linestyle="--", linewidth=1.4)
    axis.set_xticks(x); axis.set_xticklabels(X_LABELS, rotation=45, ha="right", rotation_mode="anchor", multialignment="center")
    axis.set_xlim(-0.1, 2.1); axis.set_ylim(-1.9e9, 1.5e9); axis.set_yticks(np.array([-1.0, 0.0, 1.0]) * 1.0e9)
    axis.yaxis.set_major_formatter(FuncFormatter(tick_label)); axis.text(0.0, 1.005, "1e9", transform=axis.transAxes, ha="left", va="bottom", fontsize=32)
    axis.set_axisbelow(True); axis.yaxis.grid(True, color="#C4C4C4", linestyle="--", linewidth=1.2, alpha=0.85)
    axis.set_ylabel(r"Change in mass flux (kg s$^{-1}$)", labelpad=16)
    axis.tick_params(axis="x", pad=8, direction="out", length=10, width=2)
    axis.tick_params(axis="y", direction="out", length=10, width=2)
    for spine in axis.spines.values(): spine.set_linewidth(2)
    figure.subplots_adjust(left=0.13, right=0.98, bottom=0.25, top=0.96)
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True); figure.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight", facecolor="white"); plt.close(figure)
    print(f"Saved: {OUTPUT_PNG}")

if __name__ == "__main__": main()
