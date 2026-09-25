#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot open-ocean versus nearshore RW 700-hPa-and-below convergence changes."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


INPUT_CSV = Path(
    str(PROJECT_ROOT / "Data" / "Processed" / "MSWEP_CONVERGENCE_LAYERS_BOOTSTRAP_RESULTS") + "/" +
    "CONVERGENCE_LAYERS_1982_2024_MSWEP30_BOOTSTRAP_LONG.csv"
)
VERTICAL_PRODUCT = "700_PB"
OUTPUT_DIR = PROJECT_ROOT / "Results" / "Extended_figures" / "EX_FIG6"
OUTPUT_PNG = OUTPUT_DIR / "EX_FIG6E.png"

RW_GROUP_CODE = "LE_5TH"
METRIC_TYPE = "CONV_AW"
DISTANCE_ZONES = [
    "R000_100",
    "R100_200",
    "R200_300",
    "R300_400",
    "R400_500",
]
DISTANCE_LABELS = [
    "0\u2013100 km",
    "100\u2013200 km",
    "200\u2013300 km",
    "300\u2013400 km",
    "400\u2013500 km",
]
SPATIAL_CONFIG = [
    ("OPEN_OCEAN", "#4F9DB8", "o"),
    ("NEARSHORE", "#D5A0B6", "o"),
]
SPATIAL_LABELS = {
    "OPEN_OCEAN": "Open ocean",
    "NEARSHORE": "Near coast",
}

FIGURE_SIZE = (6.4, 12.8)
FIGURE_DPI = 600
FONT_SIZE = 32
TABLE_FONT_SIZE = 12
TABLE_SIZE = (12.0, 6.0)
Y_MIN = -11.0e-6
Y_MAX = 4.0e-6
Y_TICKS = np.array([-8.0, -4.0, 0.0, 4.0]) * 1.0e-6


plt.rcParams.update(
    {
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
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
    }
)


def read_one_table(spatial_type: str) -> pd.DataFrame:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Input table does not exist: {INPUT_CSV}")

    frame = pd.read_csv(INPUT_CSV, low_memory=False)
    required = {
        "INTENSITY_GROUP_CODE",
        "SPATIAL_TYPE",
        "VERTICAL_PRODUCT",
        "DISTANCE_ZONE",
        "MEAN_CHANGE",
        "BOOTSTRAP_CI_LOWER",
        "BOOTSTRAP_CI_UPPER",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"{spatial_type} table is missing columns: {missing}")

    frame = frame.loc[
        frame["INTENSITY_GROUP_CODE"].eq(RW_GROUP_CODE)
        & frame["SPATIAL_TYPE"].eq(spatial_type)
        & frame["VERTICAL_PRODUCT"].eq(VERTICAL_PRODUCT)
        & frame["DISTANCE_ZONE"].isin(DISTANCE_ZONES)
    ].copy()
    numeric = ["MEAN_CHANGE", "BOOTSTRAP_CI_LOWER", "BOOTSTRAP_CI_UPPER"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)

    if len(frame) != len(DISTANCE_ZONES):
        raise ValueError(
            f"Expected five {spatial_type} RW rows, found {len(frame)}"
        )
    if frame["DISTANCE_ZONE"].duplicated().any():
        raise ValueError(f"Duplicate {spatial_type} distance-zone rows found.")

    frame = frame.set_index("DISTANCE_ZONE").reindex(DISTANCE_ZONES)
    if frame[numeric].isna().any(axis=None):
        raise ValueError(f"Incomplete or invalid data for {spatial_type}")
    if (frame["BOOTSTRAP_CI_LOWER"] > frame["BOOTSTRAP_CI_UPPER"]).any():
        raise ValueError(f"Invalid confidence interval for {spatial_type}")
    return frame


def format_scaled_tick(value: float, _position: int) -> str:
    scaled = value / 1.0e-6
    if np.isclose(scaled, 0.0, atol=1.0e-12):
        return "0"
    if scaled < 0.0:
        return f"\N{MINUS SIGN}{abs(scaled):g}"
    return f"{scaled:g}"


def plot_figure(data_by_type: dict[str, pd.DataFrame]) -> None:
    x = np.arange(len(DISTANCE_ZONES), dtype=float)
    fig, axis = plt.subplots(figsize=FIGURE_SIZE)

    for spatial_type, color, marker in SPATIAL_CONFIG:
        frame = data_by_type[spatial_type]
        mean = frame["MEAN_CHANGE"].to_numpy(dtype=float)
        lower = frame["BOOTSTRAP_CI_LOWER"].to_numpy(dtype=float)
        upper = frame["BOOTSTRAP_CI_UPPER"].to_numpy(dtype=float)

        axis.fill_between(x, lower, upper, color=color, alpha=0.18, linewidth=0)
        axis.plot(
            x,
            mean,
            color=color,
            linewidth=4.0,
            marker=marker,
            markersize=10.0,
        )

    axis.set_xlim(-0.2, len(x) - 0.8)
    axis.set_ylim(Y_MIN, Y_MAX)
    axis.set_yticks(Y_TICKS)
    axis.yaxis.set_major_formatter(FuncFormatter(format_scaled_tick))

    axis.set_axisbelow(True)
    axis.yaxis.grid(
        True,
        color="#C4C4C4",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
    )
    axis.xaxis.grid(False)
    axis.axhline(0.0, color="#333333", linestyle="--", linewidth=1.4)

    axis.set_xticks(x)
    axis.set_xticklabels(
        DISTANCE_LABELS,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    axis.set_xlabel("")
    axis.set_ylabel(r"Change in convergence (s$^{-1}$)", labelpad=16)

    axis.tick_params(
        axis="x",
        pad=8,
        length=10,
        width=2.0,
        direction="out",
        colors="#000000",
    )
    axis.tick_params(
        axis="y",
        length=10,
        width=2.0,
        direction="out",
        colors="#000000",
    )
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(2.0)

    axis.text(
        0.0,
        1.010,
        "1e\N{MINUS SIGN}6",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_SIZE,
        color="#000000",
        clip_on=False,
    )

    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.25, top=0.96)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved figure: {OUTPUT_PNG}")
    print("Y-axis: -11 to 4 × 10^-6 s^-1; ticks = -8, -4, 0, 4")


def main() -> None:
    data = {code: read_one_table(code) for code, _, _ in SPATIAL_CONFIG}
    plot_figure(data)


if __name__ == "__main__":
    main()
