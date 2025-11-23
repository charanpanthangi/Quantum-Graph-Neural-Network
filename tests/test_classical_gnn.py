"""Tests for the classical GNN baseline."""
import numpy as np
from app.classical_gnn import ClassicalGNN


def test_classical_gnn_forward():
    adjacency = np.array([[0, 1], [1, 0]])
    features = np.array([[1.0], [0.0]])
    model = ClassicalGNN(input_dim=1, hidden_dim=2, seed=0)
    prob = model.forward(adjacency, features)
    assert 0.0 <= prob <= 1.0
