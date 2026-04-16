import plotly.graph_objects as go
import pytest

from howso.visuals.utilities import compact_number, nice_range, normalize_axis_range


@pytest.mark.parametrize(
    ("value", "digits", "expected"),
    [
        # Zero
        (0, 3, "0"),
        # Small decimals (milli range, handled without SI suffix)
        (0.001, 3, "0.001"),
        (0.001, 1, "1m"),  # too few digits, uses m
        (0.0099, 3, "0.01"),
        (0.0001, 4, "0.0001"),
        (0.0001, 3, "100µ"),
        (0.00001, 8, "0.00001"),  # no tailing 0s
        (-0.00001, 6, "-0.00001"),
        (0.00000009999, 4, "99.99n"),
        (-0.00000009999, 4, "-99.99n"),
        (-0.000000099999, 4, "-100n"),  # round up
        (1e-6, 3, "1µ"),
        (2.5e-6, 3, "2.5µ"),
        (1e-9, 3, "1n"),
        # Base range
        (1, 3, "1"),
        (42, 3, "42"),
        (999, 3, "999"),
        (999, 1, "999"),
        # Kilo
        (1000, 3, "1k"),
        (1500, 3, "1.5k"),
        (1234, 4, "1.234k"),
        (1234, 2, "1.2k"),
        (1234, 1, "1k"),
        # Mega
        (999_999, 3, "1M"),  # round up
        (1_000_000, 3, "1M"),
        (2_500_000, 3, "2.5M"),
        # Giga (B)
        (1_000_000_000, 3, "1B"),
        (1_234_000_000, 3, "1.23B"),
        # Tera
        (1e12, 3, "1T"),
        # Peta
        (1e15, 3, "1P"),
        # Negative values
        (-1000, 3, "-1k"),
        (-0.001, 3, "-0.001"),
        (-1_500_000, 3, "-1.5M"),
    ],
)
def test_compact_number(value, digits, expected):
    """Test compact number creates expected formatted number."""
    assert compact_number(value, digits) == expected


@pytest.mark.parametrize(
    ("lo", "hi"),
    [
        (1.2, 8.7),
        (-50, -10),
        (-3.5, 6.5),
        (0, 100),
        (0.0001, 0.0009),
        (-0.0005, 0.0005),
        (0, 1_000_000),
    ],
)
def test_nice_range(lo: float, hi: float):
    result = nice_range(lo, hi)
    assert isinstance(result, tuple)
    assert len(result) == 2

    nice_lo, nice_hi = result
    assert nice_lo <= lo
    assert nice_hi >= hi
    assert nice_lo <= nice_hi


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ((1.2, 8.7), (1, 9)),
        ((0.0035, 0.0095), (0.003, 0.01)),
        ((10, 1), (1, 10)),
        ((5, 5), (5, 5)),
    ],
)
def test_nice_range_matches_expected(bounds: tuple[float, float], expected: tuple[float, float]):
    nice_bounds = nice_range(*bounds)
    assert nice_bounds == expected


@pytest.mark.parametrize(
    ("axis", "xs", "ys", "lo", "hi"),
    [
        ("y", [1, 2, 3], [10, 20, 30], 10, 30),
        ("x", [5, 10, 15], [1, 2, 3], 5, 15),
        ("x", [None, 20, None], [1, 2, 3], 0, 20),
        ("y", [1, 2, 3], [float("inf"), 20, float("-inf")], 0, 20),
    ],
)
def test_normalize_axis_range(axis: str, xs: list[float], ys: list[float], lo: float, hi: float):
    fig = go.Figure(go.Scatter(x=xs, y=ys))
    normalize_axis_range(fig, axis=axis)
    ax_layout = fig.layout.yaxis if axis == "y" else fig.layout.xaxis
    assert ax_layout.range[0] <= lo
    assert ax_layout.range[1] >= hi


@pytest.mark.parametrize("bounds", [(-5, 100), (0, 50), (10, 10)])
def test_normalize_axis_range_explicit_bounds(bounds: tuple[float, float]):
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[0, 500, 1000]))
    normalize_axis_range(fig, bounds=bounds)
    assert tuple(fig.layout.yaxis.range) == bounds


def test_normalize_axis_range_figures_share_range():
    fig1 = go.Figure(go.Scatter(x=[1, 2], y=[0, 50]))
    fig2 = go.Figure(go.Scatter(x=[1, 2], y=[10, 90]))
    normalize_axis_range(fig1, fig2)
    assert fig1.layout.yaxis.range == fig2.layout.yaxis.range


def test_normalize_axis_range_empty_figure_does_not_raise():
    normalize_axis_range(go.Figure())
