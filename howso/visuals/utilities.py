import math
from typing import Literal, SupportsFloat, SupportsInt

import plotly.graph_objects as go

SI_PREFIXES = [
    (1e30, "Q"),  # quetta
    (1e27, "R"),  # ronna
    (1e24, "Y"),  # yotta
    (1e21, "Z"),  # zetta
    (1e18, "E"),  # exa
    (1e15, "P"),  # peta
    (1e12, "T"),  # tera
    (1e9, "B"),  # giga (use B for billion over G)
    (1e6, "M"),  # mega
    (1e3, "k"),  # kilo
    (1e0, ""),  # base
    (1e-3, "m"),  # milli
    (1e-6, "µ"),  # micro
    (1e-9, "n"),  # nano
    (1e-12, "p"),  # pico
    (1e-15, "f"),  # femto
    (1e-18, "a"),  # atto
    (1e-21, "z"),  # zepto
    (1e-24, "y"),  # yocto
    (1e-27, "r"),  # ronto
    (1e-30, "q"),  # quecto
]


def compact_number(value: SupportsFloat, digits: SupportsInt = 3) -> str:
    """
    Format a number to specified digits with SI prefix.

    Parameters
    ----------
    value : float
        The value to format.
    digits : int, default 3
        The number of digits to format to.

    Returns
    -------
    str
        The formatted number.
    """
    value = float(value)
    digits = int(digits)
    abs_value = abs(value)

    if value == 0:
        return "0"

    exp = math.floor(math.log10(abs_value))

    # Fallback to sci notation if we run out of SI prefix
    if abs(exp) > 30:
        return f"{value:.{digits}g}"

    # Don't use SI prefix if decimal places can fit it
    if -digits <= exp < 0:
        rounded = round(value, digits)
        if rounded != 0:
            formatted = f"{rounded:.{digits}f}"
            return formatted.rstrip("0")

    # Use SI prefix
    exp_si = (exp // 3) * 3  # Snap down to nearest SI prefix boundary
    scaled = value / 10**exp_si

    if math.floor(math.log10(abs(round(scaled)))) >= 3:
        # Move up to next prefix if rounding pushes scaled to 1000
        exp_si += 3
        scaled = value / 10**exp_si

    index = (30 - exp_si) // 3
    prefix = SI_PREFIXES[index][1]
    formatted = f"{scaled:.{digits}g}"
    if "e" in formatted:
        # fallback if cant fit in digits and g produces sci notation
        formatted = f"{scaled:.0f}"
    return f"{formatted}{prefix}"


def nice_range(lower: float, upper: float) -> tuple[float, float]:
    """
    Expand a value interval to rounded bounds.

    Parameters
    ----------
    lower : float
        The lower bound of the data range.
    upper : float
        The upper bound of the data range.

    Returns
    -------
    tuple of float
        A (nice_min, nice_max) pair suitable for use as an axis range.
    """
    if upper < lower:
        lower, upper = upper, lower

    span = upper - lower
    if span == 0:
        return lower, upper

    # Find power of 10 of span
    power = math.floor(math.log10(span))
    step = 10**power

    # Adjust step to 1, 2, or 5 multiple
    error = span / step
    if error < 2:
        step /= 5
    elif error < 5:
        step /= 2

    nice_min = math.floor(lower / step) * step
    nice_max = math.ceil(upper / step) * step

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
