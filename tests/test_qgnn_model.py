"""Tests for the stacked QGNN model."""
import numpy as np
from app.qgnn_model import QGNNModel


def test_qgnn_model_forward():
    adjacency = np.array([[0, 1], [1, 0]])
    features = np.array([[0.0], [1.0]])
    model = QGNNModel(num_layers=1, num_nodes=2, seed=0)
    prob, embedding = model.forward((adjacency, features))
    assert 0.0 <= prob <= 1.0
    assert embedding.shape == (1,)
