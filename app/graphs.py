"""Graph generation utilities for the QGNN demo.

This module builds tiny NetworkX graphs and provides simple labels so that
beginners can run a full classification workflow without heavy compute. All
examples are intentionally small to keep the quantum simulation fast.
"""
from typing import List, Tuple
import networkx as nx
import numpy as np


def generate_graph_dataset(task: str = "graph_classification") -> List[Tuple[nx.Graph, int]]:
    """Create a small list of graphs with labels.

    Parameters
    ----------
    task: str
        Either "graph_classification" (default) or "node_classification".
        Graph classification labels are 1 if the graph contains a triangle,
        else 0. Node classification labels mark star centers as 1 and leaves as
        0.

    Returns
    -------
    list of tuple
        Each tuple is (graph, label). For node classification the label is a
        dictionary keyed by node with small integers.
    """
    dataset: List[Tuple[nx.Graph, int]] = []

    # Path graph: simple chain, no triangle.
    path_graph = nx.path_graph(4)
    dataset.append((path_graph, 0))

    # Star graph: one center with leaves.
    star_graph = nx.star_graph(3)
    dataset.append((star_graph, 0))

    # Triangle cycle: contains a 3-cycle.
    triangle_graph = nx.cycle_graph(3)
    dataset.append((triangle_graph, 1))

    # Square cycle: no triangle, but still cyclic.
    square_graph = nx.cycle_graph(4)
    dataset.append((square_graph, 0))

    if task == "node_classification":
        # For node classification we attach node-wise labels to the star.
        labels = {node: (1 if node == 0 else 0) for node in star_graph.nodes}
        return [(star_graph, labels)]

    return dataset


def graph_to_adjacency_and_features(graph: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a NetworkX graph to adjacency matrix and degree features.

    The node features are simple degree counts normalized to [0, 1]. This keeps
    the example focused on the message-passing logic rather than complex
    feature engineering.

    Parameters
    ----------
    graph: nx.Graph
        Input graph.

    Returns
    -------
    tuple
        (adjacency_matrix, node_features) where adjacency_matrix is shape
        (n_nodes, n_nodes) and node_features is shape (n_nodes, 1).
    """
    adjacency = nx.to_numpy_array(graph)
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    max_degree = max(degrees.max(), 1.0)
    features = (degrees / max_degree).reshape(-1, 1)
    return adjacency, features
