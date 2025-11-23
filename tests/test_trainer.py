"""Tests for training loops."""
import numpy as np
from app.graphs import generate_graph_dataset, graph_to_adjacency_and_features
from app.trainer import split_and_prepare, train_qgnn, train_classical_gnn


def build_dataset():
    raw = []
    for graph, label in generate_graph_dataset():
        adjacency, features = graph_to_adjacency_and_features(graph)
        raw.append(((adjacency, features), label))
    return raw


def test_split_and_train_runs():
    dataset_raw = build_dataset()
    train_set, test_set = split_and_prepare(dataset_raw, test_ratio=0.5)
    qgnn_model, history_qgnn = train_qgnn(train_set, num_layers=1, epochs=1, lr=0.1)
    classical_model, history_classical = train_classical_gnn(train_set, hidden_dim=2, epochs=1, lr=0.1)
    assert len(history_qgnn["loss"]) == 1
    assert len(history_classical["loss"]) == 1
