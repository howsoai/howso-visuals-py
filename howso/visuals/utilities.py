import math
from typing import Literal

import plotly.graph_objects as go


def nice_range(minimum: float, maximum: float) -> tuple[float, float]:
    """
    Expand a value interval to rounded bounds.

    Parameters
    ----------
    minimum : float
        The lower bound of the data range.
    maximum : float
        The upper bound of the data range.

    Returns
    -------
    tuple of float
        A (nice_min, nice_max) pair suitable for use as an axis range.
    """
    if maximum < minimum:
        minimum, maximum = maximum, minimum

    span = maximum - minimum
    if span == 0:
        return minimum, maximum

    # Find power of 10 of span
    power = math.floor(math.log10(span))
    step = 10**power

    # Adjust step to 1, 2, or 5 multiple
    error = span / step
    if error < 2:
        step /= 5
    elif error < 5:
        step /= 2

    nice_min = math.floor(minimum / step) * step
    nice_max = math.ceil(maximum / step) * step

    return nice_min, nice_max


def normalize_axis_range(
    *figures: go.Figure,
    axis: Literal["y", "x"] = "y",
    bounds: tuple[float, float] | None = None,
) -> None:
    """
    Normalize the y or x axis range of all Figures.

    Parameters
    ----------
    axis : {"x", "y"}, default "y"
        The axis to adjust.
    bounds : tuple of float, optional
        The axis range to normalize all figures to. If unset, calculates a range
        given the axis values across all figures.
    """
    if bounds is None:
        bounds = (0, 0)
        for fig in figures:
            for trace in fig.data:
                ax = getattr(trace, axis, None)
                if ax is None:
                    ax = []
                for val in ax:
                    if val is not None and math.isfinite(float(val)):
                        bounds = min(bounds[0], float(val)), max(bounds[1], float(val))
        bounds = nice_range(*bounds)

    for fig in figures:
        if axis == "x":
            fig.update_xaxes(range=bounds)
        elif axis == "y":
            fig.update_yaxes(range=bounds)
