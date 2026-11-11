"""Figures whose rows can be paged through when there are too many to show at once."""

from typing import Any

import plotly.graph_objects as go

__all__ = ["RowPagedFigure"]


class RowPagedFigure(go.Figure):
    """
    A Plotly figure that shows a window of its rows and pages vertically through the rest.

    Use this as the base figure for plots with one row per item along a categorical y axis, such as a
    heatmap of cases by features or a horizontal bar chart, when there may be more rows than fit on screen.
    Build the figure as usual, size it for :meth:`visible_rows` rows, then call :meth:`page_rows`.

    Plotly has no vertical scrollbar, so the figure pages through its rows with a vertical stack of
    buttons to the right of the plot, one per page, under a small label. The rows can also be dragged
    vertically. When there are too many pages to stack in the available space, the buttons collapse
    into a dropdown that opens upward and scrolls. Dragging the plot does not change the active button.

    A ``RowPagedFigure`` is a ``plotly.graph_objects.Figure``, so it can be returned anywhere a figure is
    expected. To page an existing figure, wrap it with ``RowPagedFigure(fig)``. Subplot grid information
    from ``make_subplots`` is not kept when wrapping, so add traces by axis reference rather than by
    ``row``/``col`` afterward.

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
        page of rows, the plot can be dragged vertically but not horizontally, and page buttons are added to
        the right of the plot. Double clicking the plot resets it to the first page.

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

        # The pan is intentionally unbounded: Plotly's ``minallowed``/``maxallowed`` clamp each end of the
        # range independently, which shrinks or stalls the window at the edges.
        self.update_layout({
            "dragmode": "pan",
            yaxis: {"autorange": False, "range": page_range(0)},
        })
        self.update_xaxes(fixedrange=True)
        page_menu = {
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
