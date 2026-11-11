"""Figures whose rows can be paged through when there are too many to show at once."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import inspect
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

__all__ = ["RowPagedFigure"]

_PAGE_MENU_NAME = "howso_row_pager"
"""The ``name`` of the page menus that :class:`RowPagedFigure` adds, used to find them in the browser."""

# Plotly.js can only tie a button's active state to an axis range that matches the button's range exactly,
# which a drag never does. This script, run after the plot is drawn, keeps the page buttons in step with
# dragging: it marks the page closest to the rows in view as active whenever a paged axis moves, then turns
# on vertical dragging, which the figure leaves off so that it is only possible where this script runs. It
# reads everything it needs from the page menus, and ignores the relayout it makes to update a menu because
# that changes only the menu. Plotly replaces ``{plot_id}``.
_PAGE_SYNC_SCRIPT = """
(function () {
    var gd = document.getElementById("{plot_id}");
    if (!gd || typeof gd.on !== "function") {
        return;
    }
    function forEachPager(callback) {
        (gd.layout.updatemenus || []).forEach(function (menu, index) {
            if (menu.name === "__MENU_NAME__" && menu.buttons && menu.buttons.length) {
                var key = Object.keys(menu.buttons[0].args[0])[0];
                callback(menu, index, key, key.slice(0, -".range".length));
            }
        });
    }
    gd.on("plotly_relayout", function (eventData) {
        var update = {};
        forEachPager(function (menu, index, key, axis) {
            var moved = Object.keys(eventData || {}).some(function (k) {
                return k === axis || k.indexOf(axis + ".") === 0;
            });
            var range = gd.layout[axis] && gd.layout[axis].range;
            if (!moved || !range) {
                return;
            }
            var firstRow = Math.min(range[0], range[1]);
            var closest = 0;
            var closestDistance = Infinity;
            menu.buttons.forEach(function (button, i) {
                var pageRange = button.args[0][key];
                var distance = Math.abs(Math.min(pageRange[0], pageRange[1]) - firstRow);
                if (distance < closestDistance) {
                    closest = i;
                    closestDistance = distance;
                }
            });
            if (menu.active !== closest) {
                update["updatemenus[" + index + "].active"] = closest;
            }
        });
        if (Object.keys(update).length) {
            Plotly.relayout(gd, update);
        }
    });
    var enableDragging = {};
    forEachPager(function (menu, index, key, axis) {
        enableDragging.dragmode = "pan";
        enableDragging[axis + ".fixedrange"] = false;
    });
    if (Object.keys(enableDragging).length) {
        Plotly.relayout(gd, enableDragging);
    }
})();
""".replace("__MENU_NAME__", _PAGE_MENU_NAME)


def _with_page_sync_script(post_script: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Add the page sync script to the start of ``post_script``, keeping any scripts already there."""
    if post_script is None:
        scripts = []
    elif isinstance(post_script, str):
        scripts = [post_script]
    else:
        scripts = list(post_script)
    # Run first so the buttons follow any relayout that the other scripts make.
    return [_PAGE_SYNC_SCRIPT, *scripts]


class RowPagedFigure(go.Figure):
    """
    A Plotly figure that shows a window of its rows and pages vertically through the rest.

    Use this as the base figure for plots with one row per item along a categorical y axis, such as a
    heatmap of cases by features or a horizontal bar chart, when there may be more rows than fit on screen.
    Build the figure as usual, size it for :meth:`visible_rows` rows, then call :meth:`page_rows`.

    Plotly has no vertical scrollbar, so the figure pages through its rows with a vertical stack of
    buttons to the right of the plot, one per page, under a small label. When there are too many pages to
    stack in the available space, the buttons collapse into a dropdown that opens upward and scrolls.

    The rows can also be dragged vertically, and dragging moves the active button to the page closest to
    the rows in view. That needs a small script, which the figure adds whenever it is rendered as HTML: with
    :meth:`show`, :meth:`to_html` or :meth:`write_html`, or when displayed in a notebook with an HTML
    renderer such as "notebook" or "browser". Renderers that pass Plotly only the figure's JSON, such as
    "plotly_mimetype" (the default in JupyterLab and VS Code), cannot run the script, so the figure leaves
    dragging off and the script turns it on. Where the script cannot run, the page buttons are the only way
    to move through the rows and always show the page in view.

    A ``RowPagedFigure`` is a ``plotly.graph_objects.Figure``, so it can be returned anywhere a figure is
    expected. To page an existing figure, including one from ``make_subplots``, wrap it with
    ``RowPagedFigure(fig)``.

    Subclasses can change the size estimates used to lay out the menu with :attr:`page_button_height_px`
    and :attr:`page_dropdown_height_px`, and the button text with :meth:`page_label`.

    Examples
    --------
    .. code-block:: python

        fig = RowPagedFigure(go.Heatmap(z=z, x=features, y=cases))
        num_visible_rows = RowPagedFigure.visible_rows(len(cases), max_visible_rows=20)
        fig.update_layout(height=margin_px + num_visible_rows * row_height_px)
        fig.page_rows(len(cases), max_visible_rows=20, menu_height_px=num_visible_rows * row_height_px)
    """

    page_button_height_px: float = 35
    """The approximate rendered height of each stacked page button, including the gap between buttons."""

    page_dropdown_height_px: float = 37
    """The approximate rendered height of the collapsed page dropdown."""

    @staticmethod
    def visible_rows(num_rows: int, max_visible_rows: int | None) -> int:
        """
        Get the number of rows that are visible at once.

        Use this to size the figure before calling :meth:`page_rows`.

        Parameters
        ----------
        num_rows : int
            The total number of rows in the figure.
        max_visible_rows : int, optional
            The maximum number of rows visible at once. When None, all rows are visible.

        Returns
        -------
        int
            The number of rows that are visible at once.

        Raises
        ------
        ValueError
            If ``max_visible_rows`` is not greater than 0.
        """
        if max_visible_rows is None:
            return num_rows
        if max_visible_rows <= 0:
            raise ValueError("`max_visible_rows` must be greater than 0.")
        return min(num_rows, max_visible_rows)

    def page_label(self, first_row: int, last_row: int) -> str:
        """
        Get the text for the page button that shows ``first_row`` through ``last_row``.

        Parameters
        ----------
        first_row : int
            The 1-based position of the first row on the page.
        last_row : int
            The 1-based position of the last row on the page.

        Returns
        -------
        str
            The button text.
        """
        return f"{first_row}\N{EN DASH}{last_row}"

    def page_rows(
        self,
        num_rows: int,
        max_visible_rows: int | None,
        *,
        menu_height_px: float,
        label: str = "Rows",
        yaxis: str = "yaxis",
        top_down: bool = True,
    ) -> bool:
        """
        Show a window of the figure's rows and add controls to page through the rest.

        Does nothing when every row fits within ``max_visible_rows``. Otherwise the y axis shows the first
        page of rows and page buttons are added to the right of the plot. Zooming and dragging are turned
        off; when the figure is rendered as HTML, a script turns vertical dragging back on and keeps the
        page buttons in step with it. Double clicking a draggable plot resets it to the first page.

        The rows must sit at integer positions 0 through ``num_rows - 1`` along the y axis, as they do for a
        categorical axis.

        Parameters
        ----------
        num_rows : int
            The total number of rows in the figure.
        max_visible_rows : int, optional
            The maximum number of rows visible at once. When None, all rows are visible.
        menu_height_px : float
            The vertical space, in pixels, available for the stacked page buttons, measured up from the
            bottom of the plot. Leave out any space taken by a colorbar or legend above them. When the
            buttons do not fit, they collapse into a dropdown.
        label : str, default "Rows"
            The text shown above the page buttons, naming what the rows are.
        yaxis : str, default "yaxis"
            The layout name of the y axis to page, such as "yaxis" or "yaxis2".
        top_down : bool, default True
            Whether the first row is at the top of the plot, as on a reversed axis. When False, the first
            row is at the bottom.

        Returns
        -------
        bool
            Whether the rows are paged.

        Raises
        ------
        ValueError
            If ``max_visible_rows`` is not greater than 0.
        """
        page_size = self.visible_rows(num_rows, max_visible_rows)
        if page_size >= num_rows:
            return False

        # Each page shows ``page_size`` rows. The last page shows the final rows so it is always full, even
        # though it overlaps the page before it.
        last_start = num_rows - page_size
        starts = [*range(0, last_start, page_size), last_start]

        # Rows sit at integer positions, so each spans +/- 0.5 around its position.
        def page_range(start: int) -> list[float]:
            first, last = start - 0.5, start + page_size - 0.5
            return [last, first] if top_down else [first, last]

        buttons = [
            {
                "method": "relayout",
                "args": [{f"{yaxis}.range": page_range(start)}],
                "label": self.page_label(start + 1, start + page_size),
            }
            for start in starts
        ]
        stacked = len(buttons) * self.page_button_height_px <= menu_height_px
        menu_height = len(buttons) * self.page_button_height_px if stacked else self.page_dropdown_height_px

        # Dragging stays off unless the page sync script runs, so the active button always shows the page in
        # view. When the script turns dragging on, the pan is intentionally unbounded: Plotly's
        # ``minallowed``/``maxallowed`` clamp each end of the range independently, which shrinks or stalls
        # the window at the edges.
        self.update_layout({
            "dragmode": False,
            yaxis: {"autorange": False, "range": page_range(0), "fixedrange": True},
        })
        self.update_xaxes(fixedrange=True)
        page_menu = {
            "name": _PAGE_MENU_NAME,
            "type": "buttons" if stacked else "dropdown",
            "direction": "down" if stacked else "up",
            "active": 0,
            "buttons": buttons,
            "showactive": True,
            "x": 1,
            "xanchor": "left",
            "y": 0,
            "yanchor": "bottom",
            "pad": {"l": 10},
            "font": {"size": 11},
        }
        self.update_layout(updatemenus=[*self.layout.updatemenus, page_menu])
        self.add_annotation(self._page_menu_label(label, menu_height))
        return True

    @property
    def is_paged(self) -> bool:
        """Whether :meth:`page_rows` has added page controls to the figure."""
        return any(menu.name == _PAGE_MENU_NAME for menu in self.layout.updatemenus)

    def show(self, *args: Any, **kwargs: Any) -> None:  # noqa: D102 - Documented by plotly.
        if not self.is_paged:
            return super().show(*args, **kwargs)
        signature = inspect.signature(pio.show)
        bound = signature.bind(self, *args, **kwargs)
        # pio.show takes renderer options such as post_script through its ``**kwargs``.
        var_keyword = next(
            (name for name, param in signature.parameters.items() if param.kind is param.VAR_KEYWORD), None
        )
        call_kwargs = bound.arguments.get(var_keyword, {}) if var_keyword else {}
        if "post_script" in call_kwargs:
            # A post_script given in the call replaces the renderers' own, so add the page sync script to it.
            call_kwargs["post_script"] = _with_page_sync_script(call_kwargs["post_script"])
            return pio.show(*bound.args, **bound.kwargs)
        renderer = bound.arguments.get("renderer") or pio.renderers.default
        with self._page_sync_in_renderers(renderer):
            return pio.show(*bound.args, **bound.kwargs)

    def to_html(self, *args: Any, **kwargs: Any) -> str:  # noqa: D102 - Documented by plotly.
        return self._call_with_page_sync(pio.to_html, args, kwargs)

    def write_html(self, *args: Any, **kwargs: Any) -> None:  # noqa: D102 - Documented by plotly.
        return self._call_with_page_sync(pio.write_html, args, kwargs)

    def _repr_mimebundle_(
        self, include: Any = None, exclude: Any = None, validate: bool = True, **kwargs: Any
    ) -> dict[str, Any]:
        """Return the notebook display of the figure, keeping the page buttons in step with dragging."""
        if not self.is_paged:
            return super()._repr_mimebundle_(include, exclude, validate, **kwargs)
        if "post_script" in kwargs:
            kwargs["post_script"] = _with_page_sync_script(kwargs["post_script"])
            return super()._repr_mimebundle_(include, exclude, validate, **kwargs)
        with self._page_sync_in_renderers(pio.renderers.default):
            return super()._repr_mimebundle_(include, exclude, validate, **kwargs)

    def _ipython_display_(self) -> None:
        """Display the figure in IPython, keeping the page buttons in step with dragging."""
        if pio.renderers.render_on_display and pio.renderers.default:
            self.show()
        else:
            print(repr(self))

    def _call_with_page_sync(self, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """
        Call a ``plotly.io`` HTML function on the figure with the page sync script added to ``post_script``.

        ``post_script`` may be given by position or keyword, so the call is bound to ``func``'s signature to find
        it rather than adding it as another keyword.
        """
        if not self.is_paged:
            return func(self, *args, **kwargs)
        bound = inspect.signature(func).bind(self, *args, **kwargs)
        bound.arguments["post_script"] = _with_page_sync_script(bound.arguments.get("post_script"))
        return func(*bound.args, **bound.kwargs)

    @contextmanager
    def _page_sync_in_renderers(self, renderers: str | None) -> Iterator[None]:
        """
        Add the page sync script to each named renderer's own ``post_script`` while the context is open.

        Used when the call gives no ``post_script``, so each renderer keeps any ``post_script`` configured on
        it, such as ``pio.renderers["notebook"].post_script``. ``renderers`` is a renderer string such as
        "plotly_mimetype+notebook". Names that are not registered are left for Plotly to report.
        """
        patched = []
        try:
            for name in dict.fromkeys((renderers or "").split("+")):
                try:
                    renderer = pio.renderers[name]
                except (KeyError, ValueError):
                    continue
                if hasattr(renderer, "post_script"):
                    patched.append((renderer, renderer.post_script))
                    renderer.post_script = _with_page_sync_script(renderer.post_script)
            yield
        finally:
            for renderer, post_script in patched:
                renderer.post_script = post_script

    @staticmethod
    def _page_menu_label(text: str, menu_height_px: float) -> dict[str, Any]:
        """Build the annotation that labels the page menu, sitting just above it."""
        return {
            "text": text,
            "showarrow": False,
            "xref": "paper",
            "yref": "paper",
            "x": 1,
            "xanchor": "left",
            "xshift": 10,
            "y": 0,
            "yanchor": "bottom",
            "yshift": menu_height_px + 2,
            "font": {"size": 11, "color": "gray"},
        }
