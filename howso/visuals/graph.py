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

        if edge_attr_sigfigs is not None and unscaled_widths is not None:
            shape_label = f"{round(unscaled_widths[i], edge_attr_sigfigs)}"
        elif unscaled_widths is not None:
            shape_label = f"{unscaled_widths[i]}"
        else:
            shape_label = ""

        shapes.append(
            dict(
                type="line",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                xref="x",
                yref="y",
                label=dict(text=shape_label, xanchor="left"),
                opacity=0,
            )
        )

    return annotations, shapes


def plot_graph(
    G: nx.Graph,  # noqa: N803
    *,
    layout: Callable[[nx.Graph], LayoutMapping] = nx.spring_layout,
    edge_attr: str | None = None,
    edge_attr_sigfigs: SupportsInt | None = 4,
    node_color: list[float],
    title: str = "Causal Graph",
    subtitle: str | None = None,
) -> go.Figure:
    fig = go.Figure(
        layout=go.Layout(
            title=dict(text="<br>Network graph made with Python", font=dict(size=16)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, constrain="domain"),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, constrain="domain"),
        )
    )
    pos = layout(G, center=(1, 1), weight=edge_attr)

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
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        text=text,
        textposition="middle center",
        mode="markers+text",
        marker=dict(
            showscale=True,
            reversescale=True,
            color=node_color,
            colorscale="Bluered_r",
            cmin=3,
            cmax=30,
            cmid=15,
            size=75,
        ),
        zorder=999,
        textfont=dict(color="white"),
        name="Nodes",
        customdata=[[x] for x in node_color],
        hovertemplate=("<b>%{text}</b><br>Destination MIR: %{customdata[0]:.4f}</br>"),
    )

    annotations, shapes = _create_edge_annotations(G, pos, edge_attr=edge_attr, edge_attr_sigfigs=edge_attr_sigfigs)
    for a in annotations:
        fig.add_annotation(a)
    for s in shapes:
        fig.add_shape(**s)
    fig.add_trace(node_trace)

    fig.update_layout(title=dict(text=title, subtitle=dict(text=subtitle)))
    return fig
