from collections.abc import Callable, Mapping
from typing import Any, SupportsInt, TypeAlias

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import minmax_scale

LayoutMapping: TypeAlias = Mapping[Any, tuple[float, float]]


def _create_edge_annotations(
    G: nx.Graph,  # noqa: N803
    pos: LayoutMapping,
    edge_attr: str | None = None,
    edge_attr_sigfigs: SupportsInt | None = 4,
    label_edges: bool = True,
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
                opacity=0.8,
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
    cscale_tuple: tuple[float, float, float] = None,
    edge_attr_sigfigs: SupportsInt | None = 4,
    edge_attr: str | None = None,
    label_edges: bool = True,
    layout: Callable[[nx.Graph], LayoutMapping] = nx.shell_layout,
    node_color: list[float] | None = None,
    subtitle: str | None = None,
    title: str = "Causal Graph",
) -> go.Figure:
    """
    Plot a ``networkx`` graph using `Plotly`.

    Parameters
    ----------
    G : nx.Graph
        The graph to plot.
    cscale_tuple : tuple[float, float, float], optional
        The tuple of values to use for the colorscale. If None, (3, 15, 30) will be used.
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
    title : str, default "Causal Graph"
        The title of the plot.
    subtitle : str, optional
        The subtitle of the plot.

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
        )
    fig = go.Figure(
        layout=go.Layout(
            title=dict(text="<br>Network graph made with Python", font=dict(size=16)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, constrain="domain"),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, constrain="domain"),
            annotations=annotations,
        )
    )
    
    if cscale_tuple is None:
        cbot = 1
        cmin = 3
        cmid = 15
        cmax = 30
    else:
        cbot = 1
        cmin = cscale_tuple[0]
        cmid = cscale_tuple[1]
        cmax = cscale_tuple[2]

    fig.update_layout(
        coloraxis=dict(
            colorscale="Bluered_r",
            cmin=cbot,
            cmid=cmid,
            cmax=cmax,
            colorbar=dict(
                title="Missing Information Ratio",
                tickvals=[cbot, cmin, cmid, cmax],
                ticktext=[f"{cbot}", f"{cmin}", f"{cmid}", f"≥{cmax}"],
            ),
            reversescale=True,
        ),
    )

    for s in shapes:
        fig.add_shape(**s)
    fig.add_trace(node_trace)

    fig.update_layout(
        title=dict(text=title, subtitle=dict(text=subtitle)),
        width=1000,
        height=750,
    )
    return fig
