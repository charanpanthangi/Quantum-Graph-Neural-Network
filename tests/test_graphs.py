"""Tests for graph generation utilities."""
import networkx as nx
from app.graphs import generate_graph_dataset, graph_to_adjacency_and_features


def test_generate_graph_dataset():
    dataset = generate_graph_dataset()
    assert len(dataset) >= 3
    graph, label = dataset[0]
    assert isinstance(graph, nx.Graph)
    assert label in [0, 1]


def test_graph_to_adjacency_and_features():
    graph = nx.path_graph(3)
    adjacency, features = graph_to_adjacency_and_features(graph)
    assert adjacency.shape == (3, 3)
    assert features.shape == (3, 1)
