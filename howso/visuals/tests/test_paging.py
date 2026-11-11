import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
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
    # Dragging is left off for renderers that cannot run the page sync script, which turns it on.
    assert fig.layout.dragmode is False
    assert fig.layout.yaxis.fixedrange is True
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
    assert fig.layout.yaxis2.fixedrange is True
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


def _sync_script_count(html: str) -> int:
    """Count how many times the page sync script is attached in rendered HTML."""
    return html.count('gd.on("plotly_relayout"')


def test_is_paged():
    """is_paged reports whether page_rows added page controls."""
    fig = _heatmap_figure(30)
    assert fig.is_paged is False
    fig.page_rows(30, 10, menu_height_px=500)
    assert fig.is_paged is True


def test_html_includes_page_sync_script_only_when_paged():
    """to_html attaches the script that follows dragging only to paged figures."""
    unpaged = _heatmap_figure(3)
    unpaged.page_rows(3, 5, menu_height_px=500)
    assert _sync_script_count(unpaged.to_html(include_plotlyjs=False)) == 0

    paged = _heatmap_figure(30)
    paged.page_rows(30, 10, menu_height_px=500)
    assert _sync_script_count(paged.to_html(include_plotlyjs=False)) == 1


def test_html_keeps_caller_post_script():
    """The page sync script is added before, not instead of, the caller's post_script."""
    fig = _heatmap_figure(30)
    fig.page_rows(30, 10, menu_height_px=500)

    html = fig.to_html(include_plotlyjs=False, post_script="console.log('caller');")

    assert _sync_script_count(html) == 1
    assert html.index('gd.on("plotly_relayout"') < html.index("console.log('caller');")


def test_write_html_includes_page_sync_script(tmp_path):
    """write_html attaches the script that follows dragging."""
    fig = _heatmap_figure(30)
    fig.page_rows(30, 10, menu_height_px=500)
    path = tmp_path / "figure.html"

    fig.write_html(path, include_plotlyjs=False)

    assert _sync_script_count(path.read_text(encoding="utf-8")) == 1


@pytest.mark.parametrize(("renderer", "expected_count"), [("notebook", 1), ("plotly_mimetype", 0)])
def test_notebook_display_includes_page_sync_script(monkeypatch, renderer, expected_count):
    """HTML notebook renderers get the sync script; JSON-only renderers still display the figure."""
    monkeypatch.setattr(pio.renderers, "default", renderer)
    fig = _heatmap_figure(30)
    fig.page_rows(30, 10, menu_height_px=500)

    bundle = fig._repr_mimebundle_()

    assert bundle
    assert _sync_script_count(bundle.get("text/html", "")) == expected_count


def _paged_figure() -> RowPagedFigure:
    """Build a heatmap whose 30 rows are paged 10 at a time."""
    fig = _heatmap_figure(30)
    fig.page_rows(30, 10, menu_height_px=500)
    return fig


def test_to_html_keeps_positional_post_script():
    """to_html accepts post_script by position, as plotly does, and keeps it alongside the sync script."""
    html = _paged_figure().to_html(None, True, False, False, "console.log('caller');")

    assert _sync_script_count(html) == 1
    assert "console.log('caller');" in html


def test_write_html_keeps_positional_post_script(tmp_path):
    """write_html accepts post_script by position, as plotly does, and keeps it alongside the sync script."""
    path = tmp_path / "figure.html"

    _paged_figure().write_html(path, None, True, False, False, "console.log('caller');")

    html = path.read_text(encoding="utf-8")
    assert _sync_script_count(html) == 1
    assert "console.log('caller');" in html


def test_notebook_display_keeps_renderer_post_script(monkeypatch):
    """Displaying a paged figure keeps the renderer's configured post_script, and leaves the renderer as it was."""
    monkeypatch.setattr(pio.renderers, "default", "notebook")
    monkeypatch.setattr(pio.renderers["notebook"], "post_script", "console.log('renderer');")

    html = _paged_figure()._repr_mimebundle_()["text/html"]

    assert _sync_script_count(html) == 1
    assert html.index('gd.on("plotly_relayout"') < html.index("console.log('renderer');")
    assert pio.renderers["notebook"].post_script == "console.log('renderer');"


def test_notebook_display_keeps_call_post_script(monkeypatch):
    """A post_script given to the display call is kept alongside the sync script."""
    monkeypatch.setattr(pio.renderers, "default", "notebook")

    html = _paged_figure()._repr_mimebundle_(post_script="console.log('caller');")["text/html"]

    assert _sync_script_count(html) == 1
    assert "console.log('caller');" in html


@pytest.mark.parametrize(
    ("args", "kwargs", "expected_renderer"),
    [((), {}, "notebook"), (("browser",), {}, "browser"), ((), {"renderer": "browser"}, "browser")],
)
def test_show_keeps_renderer_post_script(monkeypatch, args, expected_renderer, kwargs):
    """`show` adds the sync script to the chosen renderer's own post_script only for the call."""
    monkeypatch.setattr(pio.renderers, "default", "notebook")
    for name in ("notebook", "browser"):
        monkeypatch.setattr(pio.renderers[name], "post_script", f"console.log('{name}');")
    seen = {}

    def fake_show(fig, renderer=None, validate=True, **show_kwargs: object) -> None:  # noqa: ARG001
        seen["renderer"] = renderer or pio.renderers.default
        seen["post_script"] = pio.renderers[seen["renderer"]].post_script
        seen["kwargs"] = show_kwargs

    monkeypatch.setattr(pio, "show", fake_show)

    _paged_figure().show(*args, **kwargs)

    assert seen["renderer"] == expected_renderer
    assert seen["post_script"][1:] == [f"console.log('{expected_renderer}');"]
    assert _sync_script_count(seen["post_script"][0]) == 1
    assert "post_script" not in seen["kwargs"]
    assert pio.renderers[expected_renderer].post_script == f"console.log('{expected_renderer}');"


def test_show_keeps_call_post_script(monkeypatch):
    """A post_script given to show is kept alongside the sync script."""
    seen = {}

    def fake_show(fig, renderer=None, validate=True, **show_kwargs: object) -> None:  # noqa: ARG001
        seen["post_script"] = show_kwargs["post_script"]

    monkeypatch.setattr(pio, "show", fake_show)

    _paged_figure().show(post_script="console.log('caller');")

    assert seen["post_script"][1:] == ["console.log('caller');"]
    assert _sync_script_count(seen["post_script"][0]) == 1


def test_wraps_make_subplots_figure():
    """Wrapping a make_subplots figure keeps its grid, so traces can still be added by row and column."""
    fig = RowPagedFigure(make_subplots(rows=2, cols=1))

    fig.add_scatter(y=[1], row=2, col=1)

    assert fig.data[0].yaxis == "y2"
