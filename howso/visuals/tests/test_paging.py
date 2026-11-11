import plotly.graph_objects as go
import pytest

from howso.visuals import RowPagedFigure

DASH = "\N{EN DASH}"


def _heatmap_figure(num_rows: int) -> RowPagedFigure:
    """Build a heatmap with one categorical row per item, top to bottom."""
    fig = RowPagedFigure(go.Heatmap(z=[[r] for r in range(num_rows)], y=[f"case {r}" for r in range(num_rows)]))
    fig.update_layout(yaxis={"autorange": "reversed"})
    return fig


def test_row_paged_figure_is_a_figure():
    """A RowPagedFigure can be used anywhere a plotly Figure is expected."""
    assert isinstance(_heatmap_figure(3), go.Figure)


@pytest.mark.parametrize(
    ("num_rows", "max_visible_rows", "expected"),
    [(3, None, 3), (3, 5, 3), (3, 3, 3), (30, 20, 20)],
)
def test_visible_rows(num_rows, max_visible_rows, expected):
    """visible_rows caps the number of rows at max_visible_rows."""
    assert RowPagedFigure.visible_rows(num_rows, max_visible_rows) == expected


@pytest.mark.parametrize("max_visible_rows", [0, -1])
def test_visible_rows_rejects_non_positive(max_visible_rows):
    """visible_rows rejects a non-positive max_visible_rows."""
    with pytest.raises(ValueError, match="max_visible_rows"):
        RowPagedFigure.visible_rows(3, max_visible_rows)


@pytest.mark.parametrize("max_visible_rows", [3, None])
def test_page_rows_does_nothing_when_rows_fit(max_visible_rows):
    """page_rows leaves the figure unchanged when every row fits."""
    fig = _heatmap_figure(3)

    assert fig.page_rows(3, max_visible_rows, menu_height_px=500) is False
    assert fig.layout.yaxis.autorange == "reversed"
    assert fig.layout.yaxis.range is None
    assert fig.layout.dragmode is None
    assert fig.layout.updatemenus == ()
    assert fig.layout.annotations == ()


def test_page_rows_stacks_buttons():
    """page_rows shows the first page and adds one stacked button per page, the last page full."""
    fig = _heatmap_figure(95)

    assert fig.page_rows(95, 20, menu_height_px=500, label="Cases") is True

    assert fig.layout.yaxis.autorange is False
    assert tuple(fig.layout.yaxis.range) == (19.5, -0.5)
    assert fig.layout.dragmode == "pan"
    assert fig.layout.xaxis.fixedrange is True

    (menu,) = fig.layout.updatemenus
    assert menu.type == "buttons"
    assert menu.direction == "down"
    pages = [(1, 20), (21, 40), (41, 60), (61, 80), (76, 95)]
    assert [b.label for b in menu.buttons] == [f"{first}{DASH}{last}" for first, last in pages]
    assert menu.buttons[-1].args[0]["yaxis.range"] == [94.5, 74.5]

    (label,) = fig.layout.annotations
    assert label.text == "Cases"
    assert label.yshift == 5 * RowPagedFigure.page_button_height_px + 2


def test_page_rows_collapses_to_dropdown():
    """page_rows uses an upward dropdown when the buttons do not fit in menu_height_px."""
    fig = _heatmap_figure(1000)

    fig.page_rows(1000, 20, menu_height_px=300)

    (menu,) = fig.layout.updatemenus
    assert menu.type == "dropdown"
    assert menu.direction == "up"
    assert len(menu.buttons) == 50
    assert menu.buttons[-1].label == f"981{DASH}1000"
    assert fig.layout.annotations[0].yshift == RowPagedFigure.page_dropdown_height_px + 2


def test_page_rows_bottom_up_on_another_axis():
    """page_rows pages the named y axis, first row at the bottom when top_down is False."""
    fig = RowPagedFigure()
    fig.add_bar(x=list(range(30)), y=[str(r) for r in range(30)], orientation="h", yaxis="y2")
    fig.update_layout(updatemenus=[{"buttons": []}])

    fig.page_rows(30, 10, menu_height_px=500, yaxis="yaxis2", top_down=False)

    assert tuple(fig.layout.yaxis2.range) == (-0.5, 9.5)
    assert len(fig.layout.updatemenus) == 2
    assert fig.layout.updatemenus[1].buttons[1].args[0]["yaxis2.range"] == [9.5, 19.5]


def test_page_label_can_be_overridden():
    """Subclasses can change the page button text."""

    class OfFigure(RowPagedFigure):
        def page_label(self, first_row, last_row):
            return f"{first_row} to {last_row}"

    fig = OfFigure(go.Heatmap(z=[[r] for r in range(4)]))
    fig.page_rows(4, 2, menu_height_px=500)

    assert [b.label for b in fig.layout.updatemenus[0].buttons] == ["1 to 2", "3 to 4"]
