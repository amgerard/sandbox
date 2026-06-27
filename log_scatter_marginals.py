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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def _reference_offsets(log_x: np.ndarray, log_y: np.ndarray) -> list[float]:
    """Choose one-decade reference offsets that cover the data."""
    offsets = log_y - log_x
    low = math.floor(offsets.min())
    high = math.ceil(offsets.max())
    refs = list(range(low, high + 1))

    if 0 not in refs:
        refs.append(0)

    if len(refs) > 7:
        refs = [round(float(value)) for value in np.linspace(low, high, 7)]
        if 0 not in refs:
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
    if offset > 0:
        multiplier = f"{10**offset:g}"
        return f"y = {multiplier}x"

    multiplier = f"{10**offset:g}"
    return f"y = {multiplier}x"


def _plot_scatter_reference_lines(
    ax: plt.Axes,
    reference_offsets: list[float],
) -> None:
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
            color="0.25",
            linestyle="--",
            linewidth=0.9,
            alpha=0.45,
            zorder=0,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


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
    """Draw a compact histogram along the projection axis for log10(x / y)."""
    log_axis_span = log_axis_max - log_axis_min
    axis_min = -log_axis_span
    axis_max = log_axis_span

    counts, edges = np.histogram(values, bins=bins, range=(axis_min, axis_max))
    max_count = counts.max(initial=0)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    if max_count == 0:
        return

    diagonal = np.array([1 / math.sqrt(2), -1 / math.sqrt(2)])
    normal = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
    start = np.array([0.02, 0.98])
    end = np.array([0.98, 0.02])
    span = end - start
    bar_width = 0.95 / bins

    def point_for_value(value: float) -> np.ndarray:
        t = (value - axis_min) / (axis_max - axis_min)
        return start + t * span

    for offset in reference_offsets:
        point = point_for_value(-offset)
        ax.plot(
            [point[0] - normal[0] * 0.08, point[0] + normal[0] * 0.38],
            [point[1] - normal[1] * 0.08, point[1] + normal[1] * 0.38],
            color="0.25",
            linestyle="--",
            linewidth=0.9,
            alpha=0.45,
            clip_on=True,
        )
        tick_start = point - normal * 0.025
        tick_end = point + normal * 0.025
        ax.plot(
            [tick_start[0], tick_end[0]],
            [tick_start[1], tick_end[1]],
            color="0.25",
            linewidth=0.8,
        )
        label_point = point - normal * 0.085
        ax.text(
            label_point[0],
            label_point[1],
            _reference_label(offset),
            ha="right",
            va="center",
            rotation=-45,
            fontsize=8,
        )

    for idx, count in enumerate(counts):
        center = point_for_value((edges[idx] + edges[idx + 1]) / 2)
        half_width = 0.5 * bar_width * diagonal
        height = 0.26 * (count / max_count) * normal

        polygon = Polygon(
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
            alpha=0.75,
        )
        ax.add_patch(polygon)

    ax.plot([start[0], end[0]], [start[1], end[1]], color="0.35", linewidth=1.0)
    ax.text(
        0.48,
        0.02,
        "log10(x / y)",
        ha="center",
        va="center",
        rotation=-45,
        fontsize=10,
    )


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
            top_cell.height * 0.5,
        ]
    )
    ax_hist_y.set_position(
        [
            right_cell.x0,
            scatter_position.y0,
            right_cell.width * 0.5,
            scatter_position.height,
        ]
    )


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

    fig = plt.figure(figsize=(10, 10))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1, 1),
        height_ratios=(1, 1),
        hspace=0.08,
        wspace=0.08,
    )

    ax_hist_x = fig.add_subplot(grid[0, 0])
    ax_scatter = fig.add_subplot(grid[1, 0], sharex=ax_hist_x)
    ax_hist_y = fig.add_subplot(grid[1, 1], sharey=ax_scatter)
    ax_ratio = fig.add_subplot(grid[0, 1])

    ax_scatter.set_box_aspect(1)
    ax_ratio.set_box_aspect(1)

    ax_scatter.scatter(x, y, s=16, alpha=0.55, edgecolors="none")
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlim(axis_min, axis_max)
    ax_scatter.set_ylim(axis_min, axis_max)
    ax_scatter.set_xlabel("x")
    ax_scatter.set_ylabel("y")
    ax_scatter.grid(True, which="both", color="0.9", linewidth=0.8)
    _plot_scatter_reference_lines(ax_scatter, reference_offsets)

    shared_bins = np.geomspace(axis_min, axis_max, bins + 1)

    ax_hist_x.hist(x, bins=shared_bins, color="tab:blue", alpha=0.75)
    ax_hist_x.set_xscale("log")
    ax_hist_x.set_xlim(axis_min, axis_max)
    ax_hist_x.set_ylabel("count")
    ax_hist_x.tick_params(axis="x", labelbottom=False)
    ax_hist_x.grid(True, axis="x", which="both", color="0.9", linewidth=0.8)

    ax_hist_y.hist(y, bins=shared_bins, orientation="horizontal", color="tab:orange", alpha=0.75)
    ax_hist_y.set_yscale("log")
    ax_hist_y.set_ylim(axis_min, axis_max)
    ax_hist_y.set_xlabel("count")
    ax_hist_y.tick_params(axis="y", labelleft=False)
    ax_hist_y.grid(True, axis="y", which="both", color="0.9", linewidth=0.8)

    _plot_diagonal_histogram(
        ax_ratio,
        np.log10(x / y),
        bins=bins,
        color="tab:green",
        reference_offsets=reference_offsets,
        log_axis_min=np.log10(axis_min),
        log_axis_max=np.log10(axis_max),
    )

    fig.suptitle("Log-Scale Scatter Plot with Marginal Distributions", fontsize=14)
    fig.canvas.draw()
    _align_marginal_axes(ax_scatter, ax_hist_x, ax_hist_y)

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
