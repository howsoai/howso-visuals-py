from collections.abc import Callable, Collection, Mapping, Sequence
import math
from typing import Any, SupportsInt, TypeAlias

import networkx as nx
import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import minmax_scale

LayoutMapping: TypeAlias = Mapping[Any, tuple[float, float]]


def _create_edge_annotations(
    G: nx.Graph,  # noqa: N803
    pos: LayoutMapping,
    edge_attr: str | None = None,
    edge_attr_sigfigs: SupportsInt | None = 4,
    label_edges: bool = True,
    uncertain_edges: Collection[tuple[str, str]] | None = None,
    uncertain_edge_opacity: float = 0.3,
) -> tuple[list[go.layout.Annotation], list[dict[str, Any]]]:
    # Annotations are created to show the edges between nodes,
    # while invisible shapes with labels are created to label them with the edge weight.
    annotations = []
    shapes = []
    directed = nx.is_directed(G)

    widths = None
    unscaled_widths = None
    if edge_attr is not None:
        unscaled_widths = [d[edge_attr] for _, _, d in G.edges(data=True)]
        widths = minmax_scale(np.array(unscaled_widths).reshape(-1, 1), (2, 5))
        widths = widths.reshape(-1)

    edge_blacklist = set()

    for i, (s, d) in enumerate(G.edges()):
        if (s, d) in edge_blacklist:
            continue

        x0, y0 = pos[s]
        x1, y1 = pos[d]
        width = widths[i] if widths is not None else 2

        if directed and G.has_edge(d, s):
            edge_blacklist.add((d, s))
            arrowside = "end+start"
        elif not directed:
            arrowside = "none"
        else:
            arrowside = "end"

        if uncertain_edges and ((s, d) in uncertain_edges or (d, s) in uncertain_edges):
            opacity = uncertain_edge_opacity
            arrowside = "none"
        else:
            opacity = 0.8

        annotations.append(
            go.layout.Annotation(
                ax=x0,
                ay=y0,
                axref="x",
                ayref="y",
                x=x1,
                y=y1,
                xref="x",
                yref="y",
                showarrow=True,
                arrowhead=4,
                standoff=40.5,
                startstandoff=37.5,
                arrowside=arrowside,
                arrowwidth=width,
                opacity=opacity,
                captureevents=True,
            )
        )

        if label_edges:
            if edge_attr_sigfigs is not None and unscaled_widths is not None:
                shape_label = f"{round(unscaled_widths[i], edge_attr_sigfigs)}"
            elif unscaled_widths is not None:
                shape_label = f"{unscaled_widths[i]}"
            else:
                shape_label = ""
        else:
            shape_label = ""

        shape_label = (
            '<span style="text-shadow: -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff;">'
            f"{shape_label}</span>"
        )

        shapes.append(
            dict(
                type="line",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                xref="x",
                yref="y",
                label=dict(text=shape_label),
                opacity=0,
            )
        )

    return annotations, shapes


def plot_graph(
    G: nx.Graph,  # noqa: N803
    *,
    colorscale: str | Sequence[tuple[float, str]] = "Bluered",
    cscale_tuple: tuple[float, float, float] = None,
    edge_attr_sigfigs: SupportsInt | None = 4,
    edge_attr: str | None = None,
    label_edges: bool = True,
    layout: Callable[[nx.Graph], LayoutMapping] = nx.shell_layout,
    node_color: list[float] | None = None,
    subtitle: str | None = None,
    title: str = "Causal Graph",
    uncertain_edges: Collection[tuple[str, str]] | None = None,
    uncertain_edge_opacity: float = 0.3,
) -> go.Figure:
    """
    Plot a ``networkx`` graph using `Plotly`.

    Parameters
    ----------
    G : nx.Graph
        The graph to plot.
    colorscale : str | Sequence[tuple[float, str]], default "Bluered"
        The colorscale to use when plotting nodes using ``node_color``. Defaults to `Plotly`'s reversed "Bluered"
        colorscale.
    cscale_tuple : tuple[float, float, float], optional
        The tuple of values (``cmin``, ``cmid``, ``cmax``) to use for the colorscale. If None, ``(3, 15, 30)`` will be used.
    edge_attr : str, optional
        The name of the edge attribute to use when scaling the size of the edges. This should
        be an attribute that is contained within ``G``.
    edge_attr_sigfigs : SupportsInt | None, default 4
        The number of significant figures to round to when labelling each edge. If None, no rounding
        will be performed.
    label_edges : bool, default True
        Whether to label plotted edges.
    layout : Callable[nx.Graph, Mapping[Any, tuple[float, float]]], default nx.shell_layout
        A callable which generates a mapping of nodes to ``(x, y)`` coordinates.
    node_color : list[float], optional
        The data to use when determining the color for each node.
    subtitle : str, optional
        The subtitle of the plot.
    title : str, default "Causal Graph"
        The title of the plot.
    uncertain_edges : Collection[tuple[str, str]], optional
        Edges that are deemed uncertain by the caller. These will be plotted with an opacity equal to
        ``uncertain_edge_opacity`` and will not have directional arrows.
    uncertain_edge_opacity : float, default 0.3
        The opacity use when plotting edges contained in ``uncertain_edges``.

    Returns
    -------
    go.Figure
        The resultant `Plotly` figure.
    """
    pos = layout(G, center=(1, 1))

    text = []
    node_x = []
    node_y = []
    for node in G.nodes():
        text.append(node)
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    # This places a 1px black border around the node labels.
    text = [
        f'<span style="text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;">{t}</span>'
        for t in text
    ]
    hovertemplate = "<b>%{text}</b>"
    if node_color is not None:
        hovertemplate += "<br>Destination MIR: %{customdata[0]:.4f}</br>"

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        text=text,
        textposition="middle center",
        mode="markers+text",
        marker=dict(
            color=node_color,
            coloraxis="coloraxis",
            size=75,
        ),
        zorder=999,
        textfont=dict(color="white"),
        name="Nodes",
        customdata=[[x] for x in node_color] if node_color is not None else None,
        hovertemplate=hovertemplate,
    )

    annotations, shapes = _create_edge_annotations(
        G,
        pos,
        edge_attr=edge_attr,
        edge_attr_sigfigs=edge_attr_sigfigs,
        label_edges=label_edges,
        uncertain_edges=uncertain_edges,
        uncertain_edge_opacity=uncertain_edge_opacity,
    )
    fig = go.Figure(
        layout=go.Layout(
            title=dict(text="<br>Network graph made with Python", font=dict(size=16)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=8, l=8, r=8, t=48),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, constrain="domain"),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, constrain="domain"),
            annotations=annotations,
        )
    )

    if cscale_tuple is None:
        cbot = 0
        cmin = 1
        cmid = 3
        cmax = 10
    else:
        cbot = 0
        cmin = cscale_tuple[0]
        cmid = cscale_tuple[1]
        cmax = cscale_tuple[2]

    fig.update_layout(
        coloraxis=dict(
            colorscale=colorscale,
            cmin=cbot,
            cmid=cmid,
            cmax=cmax,
            colorbar=dict(
                title="Missing Information",
                tickvals=[cbot, cmin, cmid, cmax],
                ticktext=[f"{cbot}", f"{cmin}", f"{cmid}", f"≥{cmax}"],
            ),
        ),
    )

    for s in shapes:
        fig.add_shape(**s)
    fig.add_trace(node_trace)

    fig.update_layout(
        title=dict(text=title, subtitle=dict(text=subtitle), xref="paper", xanchor="left", x=0),
        width=1000,
        height=750,
    )
    return fig


def _remap_axis_ref(ref: str | None, ax_idx: int) -> str | None:
    """Remap an axis ref given new index."""
    if ref in (None, "pixel", "paper"):
        return ref
    if ref.startswith("x"):
        return "x" if ax_idx == 1 else f"x{ax_idx}"
    if ref.startswith("y"):
        return "y" if ax_idx == 1 else f"y{ax_idx}"
    return ref


def compare_network_figures(  # noqa: PLR0912, PLR0915
    figures: Sequence[go.Figure],
    *,
    columns: int | None = None,
    per_row_colorbar: bool = True,
    subplot_titles: list[str | None] | None = None,
    title: str | None = None,
    width: int = 800,
    height: int = 650,
) -> go.Figure:
    """
    Combine multiple network graphs for comparison.

    Parameters
    ----------
    figures : Sequence[go.Figure]
        Network figures to compare. All must share the same colorbar scale.
    columns : int, optional
        The number of columns of figures. If unspecified, all figures will be rendered side by side on the same row.
    per_row_colorbar : bool, default True
        Show the colorbar on each row.
    subplot_titles : list[str | None], optional
        Set the title of each individual figure. Using None will inherit the original figure title.
    title : str, optional
        Set an overall figure title.
    width : int, default 800
        The width of each individual figure.
    height : int, default 650
        The height of each individual figure.

    Returns
    -------
    go.Figure
        The resulting `Plotly` figure.
    """
    n_figs = len(figures)
    n_cols = min(n_figs, columns) if columns else n_figs
    n_rows = math.ceil(n_figs / n_cols)
    height = max(height, 400)
    width = max(width, 400)

    if columns is not None and columns < 1:
        raise ValueError("When specified, `columns` must be greater than 0.")

    uses_coloraxis = False
    coloraxis = None
    color_ranges = []
    resolved_titles: list[str] = []
    for i, fig in enumerate(figures):
        # Capture titles provided or from figures (each title must be truthy)
        if subplot_titles is not None and i < len(subplot_titles):
            if subplot_titles[i] is None:
                resolved_titles.append(fig.layout.title.text or " ")
            else:
                resolved_titles.append(subplot_titles[i] or " ")
        else:
            resolved_titles.append(fig.layout.title.text or " ")

        if fig.data == tuple():
            continue  # Skip empty figures

        # Validate trace and coloraxis
        for trace in fig.data:
            if not isinstance(trace, go.Scatter):
                raise TypeError("All figures must be network graph figures.")
        ca = fig.layout.coloraxis
        if ca is not None:
            color_ranges.append((ca.cmin, ca.cmid, ca.cmax))
            if coloraxis is None:
                # Capture first non empty network plot's color axis
                coloraxis = ca.to_plotly_json()
    if len(set(color_ranges)) > 1:
        raise ValueError("All figures must share the same colorbar scale.")

    horizontal_spacing = 0.02 / n_cols
    vertical_gap = 0.01
    if any(t != " " for t in resolved_titles):
        vertical_gap = 0.05
    vertical_spacing = (vertical_gap / n_rows) * (800 / height)
    sub = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=resolved_titles,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
    )

    # Left align titles
    for i in range(len(sub.layout.annotations)):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        # Get the actual domain Plotly computed for this subplot
        ax_key = "xaxis" if (row == 1 and col == 1) else f"xaxis{(row - 1) * n_cols + col}"
        left_edge = sub.layout[ax_key].domain[0]
        sub.layout.annotations[i].x = left_edge
        sub.layout.annotations[i].xanchor = "left"

    # Add all the plots
    for index, fig in enumerate(figures):
        row = (index // n_cols) + 1
        col = (index % n_cols) + 1

        if fig.data == tuple():
            # Render empty plot
            sub.update_xaxes(range=[0, 1], row=row, col=col)  # Prepare axes for centering label
            sub.update_yaxes(range=[0, 1], row=row, col=col)
            sub.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1, line=dict(width=0), row=row, col=col)
            sub.add_annotation(
                x=0.5,
                y=0.5,
                text="No Results",
                showarrow=False,
                font=dict(size=14),
                xref=_remap_axis_ref("x", index + 1),
                yref=_remap_axis_ref("y", index + 1),
            )
        else:
            # Traces
            for trace in fig.data:
                if hasattr(trace.marker, "coloraxis"):
                    # Apply coloraxis or marker color to each trace
                    if trace.marker.color is not None:
                        uses_coloraxis = coloraxis is not None  # If at least one plot uses MIR
                        if per_row_colorbar:
                            # coloraxis per row
                            trace.marker.coloraxis = f"coloraxis{row}" if row > 1 else "coloraxis"
                        else:
                            # single shared coloraxis
                            trace.marker.coloraxis = "coloraxis"
                    else:
                        # Figure doesn't use MIR, use static color for all nodes
                        trace.marker.color = pc.qualitative.Safe_r[0]
                        trace.hoverlabel.font.color = "#ffffff"
                sub.add_trace(trace, row=row, col=col)

            # Shapes (edges)
            for shape in fig.layout.shapes:
                s = shape.to_plotly_json()
                s.pop("xref", None)
                s.pop("yref", None)
                sub.add_shape(**s, row=row, col=col)

            # Annotations (edge labels)
            for ann in fig.layout.annotations:
                a = ann.to_plotly_json()
                a["xref"] = _remap_axis_ref(a.get("xref"), index + 1)
                a["yref"] = _remap_axis_ref(a.get("yref"), index + 1)
                a["axref"] = _remap_axis_ref(a.get("axref"), index + 1)
                a["ayref"] = _remap_axis_ref(a.get("ayref"), index + 1)
                sub.add_annotation(**a)

        sub.update_xaxes(
            showgrid=False,
            zeroline=False,
            fixedrange=True,
            showticklabels=False,
            constrain="domain",
            row=row,
            col=col,
        )
        sub.update_yaxes(
            showgrid=False,
            zeroline=False,
            fixedrange=True,
            showticklabels=False,
            constrain="domain",
            row=row,
            col=col,
        )

    # Overall figure layout
    colorbar_width = 100 if uses_coloraxis else 0
    sub.update_layout(
        title=dict(text=title, xref="paper", xanchor="left", x=0),
        height=height * n_rows,
        width=(width * n_cols) + colorbar_width,
        showlegend=False,
        dragmode=False,
        hovermode="closest",
        margin=dict(b=8, l=8, r=8, t=60 if title else 40),
    )

    # Update layout of each coloraxis
    if uses_coloraxis and coloraxis is not None:
        for row in range(1, n_rows + 1):
            coloraxis_id = "coloraxis" if row == 1 else f"coloraxis{row}"
            row_height = (1 - vertical_spacing * (n_rows - 1)) / n_rows
            y_top = 1 - (row - 1) * (row_height + vertical_spacing)
            sub.update_layout(
                {
                    coloraxis_id: {
                        **coloraxis,
                        "colorbar": {
                            **coloraxis.get("colorbar", {}),
                            "len": height - 60,
                            "lenmode": "pixels",
                            "y": y_top,
                            "yanchor": "top",
                        },
                    }
                }
            )

    return sub
