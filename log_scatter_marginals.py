#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib",
#   "numpy",
# ]
# ///

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# --- Figure layout -----------------------------------------------------------
FIGURE_SIZE = (10, 10)
GRID_SPACING = 0.08
MARGINAL_SIZE_FRACTION = 0.5
DIAGONAL_PANEL_SHIFT = 3.9

# --- Data colors / markers ---------------------------------------------------
SCATTER_MARKER_STYLE = {"s": 16, "alpha": 0.55, "edgecolors": "none"}
HIST_ALPHA = 0.75
HIST_X_COLOR = "tab:blue"
HIST_Y_COLOR = "tab:orange"
DIAGONAL_HIST_COLOR = "tab:green"

# --- Shared line styles ------------------------------------------------------
REFERENCE_LINE_STYLE = {
    "color": "0.25",
    "linestyle": "--",
    "linewidth": 0.9,
    "alpha": 0.45,
}
SCATTER_GRID_STYLE = {
    "which": "both",
    "color": "0.9",
    "linewidth": 0.8,
}

# --- Diagonal (ratio) panel geometry -----------------------------------------
# The diagonal axis runs from the upper-left to the lower-right of the panel.
# Bars and labels are offset along the perpendicular "normal" toward the
# upper-right corner, which is the direction the distribution grows.
DIAGONAL_AXIS_START = np.array([0.02, 0.98])
DIAGONAL_AXIS_END = np.array([0.98, 0.02])
DIAGONAL_DIRECTION = np.array([1.0, -1.0]) / math.sqrt(2)
DIAGONAL_NORMAL = np.array([1.0, 1.0]) / math.sqrt(2)

DIAGONAL_ROTATION_DEGREES = -45
DIAGONAL_LABEL_POSITION = (0.82, 0.81)
DIAGONAL_AXIS_LINE_STYLE = {"color": "0.35", "linewidth": 1.0}
DIAGONAL_TICK_STYLE = {"color": "0.25", "linewidth": 0.8}

# Offsets are expressed in axes-fraction units along DIAGONAL_NORMAL.
REFERENCE_LINE_BACK = 0.08
REFERENCE_LINE_FORWARD = 0.38
TICK_HALF_LENGTH = 0.025
LABEL_OFFSET = 0.085
BAR_WIDTH_FRACTION = 0.95
BAR_MAX_HEIGHT = 0.26

MAX_REFERENCE_LINES = 7


def _reference_offsets(log_x: np.ndarray, log_y: np.ndarray) -> list[float]:
    """Choose one-decade reference offsets (log10(y / x)) that cover the data."""
    offsets = log_y - log_x
    low = math.floor(offsets.min())
    high = math.ceil(offsets.max())
    refs = list(range(low, high + 1))

    if len(refs) > MAX_REFERENCE_LINES:
        refs = [round(float(value)) for value in np.linspace(low, high, MAX_REFERENCE_LINES)]

    refs.append(0)
    return sorted(set(float(ref) for ref in refs))


def _shared_log_limits(log_x: np.ndarray, log_y: np.ndarray) -> tuple[float, float]:
    """Return padded, shared linear-axis limits from log10 data."""
    combined = np.concatenate([log_x, log_y])
    low = combined.min()
    high = combined.max()
    padding = max((high - low) * 0.05, 0.1)

    return 10 ** (low - padding), 10 ** (high + padding)


def _reference_label(offset: float) -> str:
    if offset == 0:
        return "y = x"
    return f"y = {10**offset:g}x"


def _hide_spines(ax: plt.Axes, spines: tuple[str, ...]) -> None:
    for spine in spines:
        ax.spines[spine].set_visible(False)


@dataclass(frozen=True)
class _DiagonalProjection:
    """Maps a log10(x / y) value onto a point along the diagonal panel axis."""

    axis_min: float
    axis_max: float

    def point(self, value: float) -> np.ndarray:
        fraction = (value - self.axis_min) / (self.axis_max - self.axis_min)
        return DIAGONAL_AXIS_START + fraction * (DIAGONAL_AXIS_END - DIAGONAL_AXIS_START)


def _plot_scatter_reference_lines(ax: plt.Axes, reference_offsets: list[float]) -> None:
    """Draw constant-ratio (y = c * x) guide lines on the scatter panel."""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    for offset in reference_offsets:
        multiplier = 10**offset
        start = max(x_min, y_min / multiplier)
        end = min(x_max, y_max / multiplier)

        if start >= end:
            continue

        ax.plot(
            [start, end],
            [start * multiplier, end * multiplier],
            **REFERENCE_LINE_STYLE,
            zorder=0,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def _draw_diagonal_reference_lines(
    ax: plt.Axes,
    projection: _DiagonalProjection,
    reference_offsets: list[float],
) -> None:
    """Draw the dashed guide lines, ticks, and y = c * x labels for each offset."""
    for offset in reference_offsets:
        point = projection.point(-offset)
        back = point - DIAGONAL_NORMAL * REFERENCE_LINE_BACK
        forward = point + DIAGONAL_NORMAL * REFERENCE_LINE_FORWARD
        ax.plot([back[0], forward[0]], [back[1], forward[1]], **REFERENCE_LINE_STYLE, clip_on=True)

        tick_start = point - DIAGONAL_NORMAL * TICK_HALF_LENGTH
        tick_end = point + DIAGONAL_NORMAL * TICK_HALF_LENGTH
        ax.plot([tick_start[0], tick_end[0]], [tick_start[1], tick_end[1]], **DIAGONAL_TICK_STYLE)

        label_point = point - DIAGONAL_NORMAL * LABEL_OFFSET
        ax.text(
            label_point[0],
            label_point[1],
            _reference_label(offset),
            ha="right",
            va="center",
            rotation=DIAGONAL_ROTATION_DEGREES,
            fontsize=8,
        )


def _draw_diagonal_bars(
    ax: plt.Axes,
    projection: _DiagonalProjection,
    counts: np.ndarray,
    edges: np.ndarray,
    color: str,
) -> None:
    """Draw histogram bars rising perpendicular to the diagonal axis."""
    max_count = counts.max(initial=0)
    if max_count == 0:
        return

    half_width = 0.5 * (BAR_WIDTH_FRACTION / len(counts)) * DIAGONAL_DIRECTION

    for count, low_edge, high_edge in zip(counts, edges[:-1], edges[1:]):
        center = projection.point((low_edge + high_edge) / 2)
        height = BAR_MAX_HEIGHT * (count / max_count) * DIAGONAL_NORMAL

        ax.add_patch(
            Polygon(
                [
                    center - half_width,
                    center + half_width,
                    center + half_width + height,
                    center - half_width + height,
                ],
                closed=True,
                facecolor=color,
                edgecolor="white",
                linewidth=0.4,
                alpha=HIST_ALPHA,
            )
        )


def _draw_diagonal_axis_and_label(ax: plt.Axes) -> None:
    """Draw the diagonal axis line and its rotated log10(x/y) title."""
    ax.plot(
        [DIAGONAL_AXIS_START[0], DIAGONAL_AXIS_END[0]],
        [DIAGONAL_AXIS_START[1], DIAGONAL_AXIS_END[1]],
        **DIAGONAL_AXIS_LINE_STYLE,
    )
    ax.text(
        *DIAGONAL_LABEL_POSITION,
        "log10(x/y)",
        transform=ax.transAxes,
        ha="center",
        va="center",
        rotation=DIAGONAL_ROTATION_DEGREES,
        fontsize=10,
    )


def _plot_diagonal_histogram(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    bins: int,
    color: str,
    reference_offsets: list[float],
    log_axis_min: float,
    log_axis_max: float,
) -> None:
    """Draw the log10(x / y) distribution along the panel's diagonal axis."""
    log_axis_span = log_axis_max - log_axis_min
    projection = _DiagonalProjection(axis_min=-log_axis_span, axis_max=log_axis_span)
    counts, edges = np.histogram(values, bins=bins, range=(projection.axis_min, projection.axis_max))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    if counts.max(initial=0) == 0:
        return

    _draw_diagonal_reference_lines(ax, projection, reference_offsets)
    _draw_diagonal_bars(ax, projection, counts, edges, color)
    _draw_diagonal_axis_and_label(ax)


def _align_marginal_axes(
    ax_scatter: plt.Axes,
    ax_hist_x: plt.Axes,
    ax_hist_y: plt.Axes,
) -> None:
    """Shrink marginal plots inside their grid cells while keeping scatter alignment."""
    scatter_position = ax_scatter.get_position()
    top_cell = ax_hist_x.get_position()
    right_cell = ax_hist_y.get_position()

    ax_hist_x.set_position(
        [
            scatter_position.x0,
            top_cell.y0,
            scatter_position.width,
            top_cell.height * MARGINAL_SIZE_FRACTION,
        ]
    )
    ax_hist_y.set_position(
        [
            right_cell.x0,
            scatter_position.y0,
            right_cell.width * MARGINAL_SIZE_FRACTION,
            scatter_position.height,
        ]
    )


def _shift_diagonal_axis_toward_scatter(
    ax_scatter: plt.Axes,
    ax_ratio: plt.Axes,
    *,
    gap_fraction: float = DIAGONAL_PANEL_SHIFT,
) -> None:
    """Move the diagonal plot closer without changing its size."""
    scatter_position = ax_scatter.get_position()
    ratio_position = ax_ratio.get_position()
    horizontal_gap = max(ratio_position.x0 - scatter_position.x1, 0)
    vertical_gap = max(ratio_position.y0 - scatter_position.y1, 0)

    ax_ratio.set_position(
        [
            ratio_position.x0 - horizontal_gap * gap_fraction,
            ratio_position.y0 - vertical_gap * gap_fraction,
            ratio_position.width,
            ratio_position.height,
        ]
    )


def _style_scatter_axis(ax: plt.Axes, axis_min: float, axis_max: float) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, **SCATTER_GRID_STYLE)


def _style_top_marginal_axis(ax: plt.Axes, axis_min: float, axis_max: float) -> None:
    ax.set_xscale("log")
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    _hide_spines(ax, ("left", "right", "top"))


def _style_right_marginal_axis(ax: plt.Axes, axis_min: float, axis_max: float) -> None:
    ax.set_yscale("log")
    ax.set_ylim(axis_min, axis_max)
    ax.set_xlabel("")
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", labelleft=False)
    _hide_spines(ax, ("bottom", "right", "top"))


def plot_log_scatter_with_distributions(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int = 40,
    output_path: str | Path | None = "log_scatter_marginals.png",
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Plot a log-log scatter chart with marginal and ratio distributions."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("x and y must contain only positive values for log scales")

    log_x = np.log10(x)
    log_y = np.log10(y)
    reference_offsets = _reference_offsets(log_x, log_y)
    axis_min, axis_max = _shared_log_limits(log_x, log_y)
    shared_bins = np.geomspace(axis_min, axis_max, bins + 1)

    fig = plt.figure(figsize=FIGURE_SIZE)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1, 1),
        height_ratios=(1, 1),
        hspace=GRID_SPACING,
        wspace=GRID_SPACING,
    )

    ax_hist_x = fig.add_subplot(grid[0, 0])
    ax_scatter = fig.add_subplot(grid[1, 0], sharex=ax_hist_x)
    ax_hist_y = fig.add_subplot(grid[1, 1], sharey=ax_scatter)
    ax_ratio = fig.add_subplot(grid[0, 1])

    ax_scatter.set_box_aspect(1)
    ax_ratio.set_box_aspect(1)

    ax_scatter.scatter(x, y, **SCATTER_MARKER_STYLE)
    _style_scatter_axis(ax_scatter, axis_min, axis_max)
    _plot_scatter_reference_lines(ax_scatter, reference_offsets)

    ax_hist_x.hist(x, bins=shared_bins, color=HIST_X_COLOR, alpha=HIST_ALPHA)
    _style_top_marginal_axis(ax_hist_x, axis_min, axis_max)

    ax_hist_y.hist(y, bins=shared_bins, orientation="horizontal", color=HIST_Y_COLOR, alpha=HIST_ALPHA)
    _style_right_marginal_axis(ax_hist_y, axis_min, axis_max)

    _plot_diagonal_histogram(
        ax_ratio,
        np.log10(x / y),
        bins=bins,
        color=DIAGONAL_HIST_COLOR,
        reference_offsets=reference_offsets,
        log_axis_min=np.log10(axis_min),
        log_axis_max=np.log10(axis_max),
    )

    fig.suptitle("Log-Scale Scatter Plot with Marginal Distributions", fontsize=14)
    fig.canvas.draw()
    _align_marginal_axes(ax_scatter, ax_hist_x, ax_hist_y)
    _shift_diagonal_axis_toward_scatter(ax_scatter, ax_ratio)

    if output_path is not None:
        fig.savefig(output_path, dpi=180)

    return fig, {
        "scatter": ax_scatter,
        "hist_x": ax_hist_x,
        "hist_y": ax_hist_y,
        "ratio": ax_ratio,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(seed=7)
    sample_size = 1_200

    x_data = rng.lognormal(mean=1.0, sigma=0.75, size=sample_size)
    y_data = rng.lognormal(mean=0.8, sigma=0.9, size=sample_size)

    plot_log_scatter_with_distributions(x_data, y_data)
    plt.show()
