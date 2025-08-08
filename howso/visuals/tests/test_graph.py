import networkx as nx
import pytest

from howso.visuals.graph import _create_edge_annotations, plot_graph


@pytest.fixture
def G():  # noqa: N802
    return nx.random_geometric_graph(10, 0.5)


@pytest.fixture
def pos(G):  # noqa: N803
    return nx.shell_layout(G)


def test_create_edge_annotations(G, pos):  # noqa: N803
    annotations, shapes = _create_edge_annotations(G, pos)

    n_edges = len(G.edges())
    assert len(annotations) == n_edges
    assert len(shapes) == n_edges


@pytest.mark.parametrize("layout", [nx.shell_layout, nx.spring_layout])
@pytest.mark.parametrize("title", ["Causal Graph", "My Causal Graph"])
@pytest.mark.parametrize("subtitle", ["A subtitle", None])
def test_plot_graph(G, layout, title, subtitle):  # noqa: N803
    fig = plot_graph(G, layout=layout, title=title, subtitle=subtitle)

    assert fig.layout.title["text"] == title
    if subtitle is not None:
        assert fig.layout.title["subtitle"]["text"] == subtitle

    assert len(fig.data[-1].x) == len(G.nodes())
