"""Tests for a single QGNN layer."""
import numpy as np
from app.qgnn_layer import run_qgnn_layer


def test_run_qgnn_layer_output_shape():
    adjacency = np.array([[0, 1], [1, 0]])
    features = np.array([[0.0], [1.0]])
    params = {"edge_weights": np.ones((2, 2)) * 0.1, "node_weights": np.zeros(2)}
    embeddings = run_qgnn_layer(params, (adjacency, features))
    assert embeddings.shape == (2, 1)
