#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot the EXP-minus-CTRL convergence and radial/vertical wind vectors."""

from pathlib import Path

import matplotlib
import numpy as np
from netCDF4 import Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import distance_transform_edt


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data" / "Figure_data" / "Main_figures" / "FIG2"
FIG_DIR = PROJECT_ROOT / "Results" / "Main_figures" / "FIG2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONV_NC_FILE = DATA_DIR / "convergence_radius_pressure_CTRL_EXP_clean.nc"
VECTOR_NC_FILE = (
    DATA_DIR / "radial_vertical_velocity_radius_pressure_CTRL_EXP_clean.nc"
)

OUT_FIG_DIFF = FIG_DIR / "FIG2E.png"


# ============================================================
# 2. Global plotting style
# ============================================================

DPI = 300

plt.rcParams.update(
    {
        "font.family": "Arial",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 25,
        "ytick.labelsize": 24,
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "axes.edgecolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "axes.linewidth": 2.5,
        "xtick.major.width": 2.5,
        "ytick.major.width": 2.5,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
    }
)


# ============================================================
# 3. Background and vector configuration
# ============================================================

SOFT_PINK_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "soft_blue_white_pink",
    [
        "#56A1B8",
        "#6FBFD7",
        "#A3DCEC",
        "#FFFFFF",
        "#FBC8DB",
        "#ECAFC6",
        "#DEAEC3",
    ],
    N=256,
)

SHARED_VMIN = -1.0e-6
SHARED_VMAX = 1.0e-6
SMOOTH_INTERPOLATION = "bilinear"

RING_STRIDE = 2
PRESSURE_STRIDE = 2
MIN_RADIUS_KM_TO_PLOT = 5.0

INNER_BOUNDARY_KM = 100.0
OUTER_BOUNDARY_KM = 200.0

# W only affects the vertical display component.
# The horizontal component U/Vr is NOT multiplied by these factors.
INNER_W_VISUAL_FACTOR = 10.0
MIDDLE_W_VISUAL_FACTOR = 100.0
OUTER_W_VISUAL_FACTOR = 1000.0

MASK_BOUNDARY_COLUMNS = True

QUIVER_SCALE = 0.28
QUIVER_WIDTH = 0.004
QUIVER_HEADWIDTH = 4.5
QUIVER_HEADLENGTH = 5.5
QUIVER_HEADAXISLENGTH = 5.0

XTICKS = np.arange(0, 501, 100)
YTICKS = np.arange(200, 1000, 200)

REFERENCE_RADII_KM = (INNER_BOUNDARY_KM, OUTER_BOUNDARY_KM)
REFERENCE_LINE_COLOR = "#FFFFFF"
REFERENCE_LINE_ALPHA = 1.0
REFERENCE_LINE_WIDTH = 2.0

SHOW_GRID = False

SINGLE_CBAR_GAP = 0.035
SINGLE_CBAR_WIDTH = 0.025
SINGLE_CBAR_HEIGHT_FRACTION = 0.88


# ============================================================
# 4. Quiver-key configuration
# ============================================================

# The quiver key refers ONLY to the horizontal wind component U/Vr.
QUIVER_KEY_U = 10.0

# Place the key OUTSIDE the main panel at the upper-right side.
# These are offsets in FIGURE coordinates relative to the main-axis box.
QUIVER_KEY_FIG_DX = -0.1   # larger -> farther right
QUIVER_KEY_FIG_DY = 0.030   # larger -> farther above

QUIVER_KEY_LABEL = r"10 m s$^{-1}$"
QUIVER_KEY_LABELPOS = "E"
QUIVER_KEY_FONTSIZE = 24


# ============================================================
# 5. Helpers
# ============================================================

def decode_text(value):
    """Decode a NetCDF string value."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def fill_nan_with_nearest(field):
    """Fill missing cells with their nearest finite neighbor for display."""
    field = np.asarray(field, dtype=np.float64)
    valid = np.isfinite(field)

    if np.all(valid):
        return field.copy()

    if not np.any(valid):
        return np.zeros_like(field)

    _, nearest_indices = distance_transform_edt(
        ~valid,
        return_indices=True,
    )

    return field[tuple(nearest_indices)]


def configure_axis(axis):
    """Configure the radius-pressure axes."""
    axis.set_xlim(0, 500)
    axis.set_ylim(1000, 200)

    axis.set_xticks(XTICKS)
    axis.set_yticks(YTICKS)

    axis.set_xlabel(
        "Radial distance from TC center (km)",
        fontsize=25,
        labelpad=12,
    )
    axis.set_ylabel(
        "Vertical pressure (hPa)",
        labelpad=20,
    )

    axis.yaxis.tick_left()
    axis.yaxis.set_label_position("left")

    axis.tick_params(
        axis="both",
        colors="#000000",
        width=2.5,
        labelcolor="#000000",
    )

    axis.tick_params(
        axis="y",
        left=True,
        right=False,
        labelleft=True,
        labelright=False,
        pad=7,
    )

    axis.tick_params(
        axis="x",
        pad=11,
    )

    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(2.5)

    for radius in REFERENCE_RADII_KM:
        axis.axvline(
            radius,
            color=REFERENCE_LINE_COLOR,
            linewidth=REFERENCE_LINE_WIDTH,
            linestyle="--",
            alpha=REFERENCE_LINE_ALPHA,
            zorder=30,
        )

    if SHOW_GRID:
        axis.grid(
            True,
            linestyle="--",
            linewidth=0.5,
            alpha=0.35,
        )


def draw_smoothed_background(axis, field, norm, image_extent):
    """Draw the convergence field and restore missing cells as white."""
    field = np.asarray(field, dtype=np.float64)

    invalid_mask = ~np.isfinite(field)
    display_field = fill_nan_with_nearest(field)

    artist = axis.imshow(
        display_field,
        origin="upper",
        aspect="auto",
        extent=image_extent,
        cmap=SOFT_PINK_BLUE_CMAP,
        norm=norm,
        interpolation=SMOOTH_INTERPOLATION,
        zorder=1,
    )

    mask_cmap = ListedColormap(
        [
            (1.0, 1.0, 1.0, 0.0),
            (1.0, 1.0, 1.0, 1.0),
        ]
    )

    axis.imshow(
        invalid_mask.astype(np.float64),
        origin="upper",
        aspect="auto",
        extent=image_extent,
        cmap=mask_cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        zorder=2,
    )

    return artist


def draw_vector_panel(axis, background, u_field, v_field, norm, plot_data):
    """
    Draw the convergence background and wind-vector overlay.

    Returns
    -------
    background_artist
        Image artist used for the colorbar.
    quiver_handle
        One of the main-panel Quiver objects.  All three radial zones use the
        same quiver scale/width/head settings, so this object can be used by
        ax.quiverkey() to make an exact horizontal U/Vr reference key.
    """
    background_artist = draw_smoothed_background(
        axis,
        background,
        norm,
        plot_data["image_extent"],
    )

    configure_axis(axis)

    radius = plot_data["R"][0, :]

    zones = [
        (0.0, INNER_BOUNDARY_KM),
        (INNER_BOUNDARY_KM, OUTER_BOUNDARY_KM),
        (OUTER_BOUNDARY_KM, 505.0),
    ]

    quiver_handle = None

    for radius_min, radius_max in zones:
        mask = (radius >= radius_min) & (radius < radius_max)

        if not np.any(mask):
            continue

        q = axis.quiver(
            plot_data["R"][:, mask],
            plot_data["P"][:, mask],
            u_field[:, mask],
            v_field[:, mask],
            angles="xy",
            scale_units="xy",
            scale=QUIVER_SCALE,
            units="width",
            width=QUIVER_WIDTH,
            headwidth=QUIVER_HEADWIDTH,
            headlength=QUIVER_HEADLENGTH,
            headaxislength=QUIVER_HEADAXISLENGTH,
            pivot="mid",
            color="#000000",
            zorder=10,
        )

        # Save one Quiver object for quiverkey().
        # All zones have identical U scaling, so any of them can be used.
        if quiver_handle is None:
            quiver_handle = q

    return background_artist, quiver_handle


def add_horizontal_wind_quiverkey(axis, quiver_handle, axis_position):
    """
    Add a U/Vr = 10 m/s horizontal reference arrow OUTSIDE the main panel,
    at the upper-right side, using figure coordinates.

    Important:
    This key is a scale for the HORIZONTAL component only.
    It should not be used to infer W or the total length of a slanted vector.
    """
    if quiver_handle is None:
        raise RuntimeError("No quiver vectors were drawn; cannot create quiver key.")

    # axis_position is the main-axis bounding box in figure coordinates.
    key_x = axis_position.x1 + QUIVER_KEY_FIG_DX
    key_y = axis_position.y1 + QUIVER_KEY_FIG_DY

    axis.quiverkey(
        quiver_handle,
        X=key_x,
        Y=key_y,
        U=QUIVER_KEY_U,
        label=QUIVER_KEY_LABEL,
        labelpos=QUIVER_KEY_LABELPOS,
        coordinates="figure",
        color="#000000",
        labelcolor="#000000",
        fontproperties={"size": QUIVER_KEY_FONTSIZE},
    )


# ============================================================
# 6. Read data
# ============================================================

def read_data():
    """Read convergence and vector NetCDF inputs."""
    if not CONV_NC_FILE.exists():
        raise FileNotFoundError(
            f"Convergence NetCDF does not exist: {CONV_NC_FILE}"
        )

    if not VECTOR_NC_FILE.exists():
        raise FileNotFoundError(
            f"Vector NetCDF does not exist: {VECTOR_NC_FILE}"
        )

    with Dataset(CONV_NC_FILE, "r") as nc_file:
        convergence_names = [
            decode_text(value)
            for value in nc_file.variables["experiment_name"][:]
        ]

        pressure_levels = np.asarray(
            nc_file.variables["pressure_level_hPa"][:],
            dtype=np.float64,
        )

        ring_inner = np.asarray(
            nc_file.variables["ring_inner_km"][:],
            dtype=np.float64,
        )

        ring_outer = np.asarray(
            nc_file.variables["ring_outer_km"][:],
            dtype=np.float64,
        )

        convergence = np.ma.filled(
            nc_file.variables["composite_mean_convergence"][:],
            np.nan,
        ).astype(np.float64)

    with Dataset(VECTOR_NC_FILE, "r") as nc_file:
        vector_names = [
            decode_text(value)
            for value in nc_file.variables["experiment_name"][:]
        ]

        vector_pressure = np.asarray(
            nc_file.variables["pressure_level_hPa"][:],
            dtype=np.float64,
        )

        ring_mid = np.asarray(
            nc_file.variables["ring_mid_km"][:],
            dtype=np.float64,
        )

        radial = np.ma.filled(
            nc_file.variables["composite_mean_radial_velocity"][:],
            np.nan,
        ).astype(np.float64)

        vertical = np.ma.filled(
            nc_file.variables["composite_mean_vertical_velocity"][:],
            np.nan,
        ).astype(np.float64)

    if "CTRL" not in convergence_names or "EXP" not in convergence_names:
        raise KeyError(
            f"Convergence data must contain CTRL and EXP: {convergence_names}"
        )

    if "CTRL" not in vector_names or "EXP" not in vector_names:
        raise KeyError(
            f"Vector data must contain CTRL and EXP: {vector_names}"
        )

    if not np.allclose(
        pressure_levels,
        vector_pressure,
        equal_nan=True,
    ):
        raise ValueError(
            "The pressure levels differ between the two inputs"
        )

    convergence_difference = (
        convergence[convergence_names.index("EXP"), :, :]
        - convergence[convergence_names.index("CTRL"), :, :]
    )

    radial_difference = (
        radial[vector_names.index("EXP"), :, :]
        - radial[vector_names.index("CTRL"), :, :]
    )

    vertical_difference = (
        vertical[vector_names.index("EXP"), :, :]
        - vertical[vector_names.index("CTRL"), :, :]
    )

    return {
        "pressure": pressure_levels,
        "ring_inner": ring_inner,
        "ring_outer": ring_outer,
        "ring_mid": ring_mid,
        "convergence_difference": convergence_difference,
        "radial_difference": radial_difference,
        "vertical_difference": vertical_difference,
    }


# ============================================================
# 7. Prepare vectors
# ============================================================

def prepare_plot_data(data):
    """Subsample the fields and apply zone-specific W display factors."""
    pressure = data["pressure"]
    ring_mid = data["ring_mid"]

    pressure_step = float(
        np.median(np.diff(pressure))
    )

    image_extent = [
        float(data["ring_inner"][0]),
        float(data["ring_outer"][-1]),
        float(pressure[-1] + pressure_step / 2.0),
        float(pressure[0] - pressure_step / 2.0),
    ]

    ring_mask = ring_mid >= MIN_RADIUS_KM_TO_PLOT

    ring_plot = ring_mid[ring_mask][::RING_STRIDE]
    pressure_plot = pressure[::PRESSURE_STRIDE]

    radius_grid, pressure_grid = np.meshgrid(
        ring_plot,
        pressure_plot,
    )

    def subset(field):
        return field[
            ::PRESSURE_STRIDE,
            :
        ][:, ring_mask][
            :,
            ::RING_STRIDE
        ]

    radial_difference = subset(
        data["radial_difference"]
    )

    vertical_difference = subset(
        data["vertical_difference"]
    )

    vertical_factor_1d = np.select(
        [
            ring_plot < INNER_BOUNDARY_KM,
            ring_plot < OUTER_BOUNDARY_KM,
        ],
        [
            INNER_W_VISUAL_FACTOR,
            MIDDLE_W_VISUAL_FACTOR,
        ],
        default=OUTER_W_VISUAL_FACTOR,
    ).astype(np.float64)

    vertical_factor_2d = np.broadcast_to(
        vertical_factor_1d[np.newaxis, :],
        radius_grid.shape,
    )

    boundary_columns = (
        np.isclose(
            ring_plot,
            INNER_BOUNDARY_KM,
        )
        | np.isclose(
            ring_plot,
            OUTER_BOUNDARY_KM,
        )
    )

    # Horizontal component: unchanged.
    radial_vector = np.asarray(
        radial_difference,
        dtype=np.float64,
    )

    # Vertical component: sign reversed for pressure coordinates
    # and visually amplified by radial zone.
    vertical_vector = (
        -np.asarray(
            vertical_difference,
            dtype=np.float64,
        )
        * vertical_factor_2d
    )

    invalid = (
        ~np.isfinite(radial_vector)
        | ~np.isfinite(vertical_vector)
    )

    if MASK_BOUNDARY_COLUMNS and np.any(boundary_columns):
        invalid |= np.broadcast_to(
            boundary_columns[np.newaxis, :],
            radius_grid.shape,
        )

    radial_vector = np.ma.masked_where(
        invalid,
        radial_vector,
    )

    vertical_vector = np.ma.masked_where(
        invalid,
        vertical_vector,
    )

    return {
        "image_extent": image_extent,
        "R": radius_grid,
        "P": pressure_grid,
        "U": radial_vector,
        "V": vertical_vector,
    }


# ============================================================
# 8. Main figure
# ============================================================

def plot_main_figure(background, plot_data, norm):
    """Draw the final figure with a U=10 m/s quiver key."""

    # Restore the canvas size used by the previous version.
    figure = plt.figure(
        figsize=(12.0, 9.6),
        facecolor="white",
    )

    # No separate legend axis is needed anymore.
    grid = figure.add_gridspec(
        nrows=1,
        ncols=1,
        left=0.150,
        right=0.800,
        bottom=0.105,
        top=0.900,
    )

    main_axis = figure.add_subplot(
        grid[0, 0]
    )

    # Restore the previous plotting-frame aspect ratio:
    # width : height = 1.2 : 1
    main_axis.set_box_aspect(1.0 / 1.2)

    background_artist, quiver_handle = draw_vector_panel(
        main_axis,
        background,
        plot_data["U"],
        plot_data["V"],
        norm,
        plot_data,
    )

    figure.canvas.draw()
    axis_position = main_axis.get_position()

    # Horizontal wind reference key OUTSIDE the main panel, upper-right.
    add_horizontal_wind_quiverkey(
        main_axis,
        quiver_handle,
        axis_position,
    )

    colorbar_axis = figure.add_axes(
        [
            axis_position.x1 + SINGLE_CBAR_GAP,
            axis_position.y0
            + axis_position.height
            * (1.0 - SINGLE_CBAR_HEIGHT_FRACTION)
            / 2.0,
            SINGLE_CBAR_WIDTH,
            axis_position.height
            * SINGLE_CBAR_HEIGHT_FRACTION,
        ]
    )

    colorbar = figure.colorbar(
        background_artist,
        cax=colorbar_axis,
        extend="both",
        extendfrac=0.035,
    )

    colorbar.set_ticks(
        np.array(
            [-1.0, 0.0, 1.0]
        )
        * 1.0e-6
    )

    colorbar.set_label(
        r"Convergence of wind ($10^{-6}$ s$^{-1}$)",
        labelpad=10,
        color="#000000",
    )

    colorbar.ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: (
                f"\N{MINUS SIGN}{abs(value * 1.0e6):g}"
                if value < 0.0
                else f"{value * 1.0e6:g}"
            )
        )
    )

    colorbar.ax.yaxis.offsetText.set_visible(False)
    colorbar.ax.yaxis.set_ticks_position("right")
    colorbar.ax.yaxis.set_label_position("right")
    colorbar.ax.yaxis.labelpad = 10

    colorbar.ax.tick_params(
        pad=6,
        colors="#000000",
        width=2.0,
        labelcolor="#000000",
    )

    colorbar.outline.set_edgecolor("#000000")
    colorbar.outline.set_linewidth(2.0)

    figure.savefig(
        OUT_FIG_DIFF,
        format="png",
        dpi=DPI,
        facecolor="white",
        edgecolor="white",
    )

    plt.close(figure)

    print(f"Saved: {OUT_FIG_DIFF}")


# ============================================================
# 9. Main program
# ============================================================

def main():
    data = read_data()

    plot_data = prepare_plot_data(
        data
    )

    norm = TwoSlopeNorm(
        vmin=SHARED_VMIN,
        vcenter=0.0,
        vmax=SHARED_VMAX,
    )

    plot_main_figure(
        background=data["convergence_difference"],
        plot_data=plot_data,
        norm=norm,
    )


if __name__ == "__main__":
    main()
