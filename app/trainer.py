"""Training loops for the QGNN and classical GNN."""
from typing import List, Tuple
import numpy as np
from .qgnn_model import QGNNModel
from .classical_gnn import ClassicalGNN
from .utils import train_test_split, compute_accuracy


def train_qgnn(dataset: List[Tuple[np.ndarray, np.ndarray, int]], num_layers: int = 1, epochs: int = 10, lr: float = 0.1) -> Tuple[QGNNModel, dict]:
    """Train the QGNN with a basic parameter-shift style gradient."""
    num_nodes = dataset[0][0].shape[0]
    model = QGNNModel(num_layers=num_layers, num_nodes=num_nodes, seed=0)
    history = {"loss": [], "acc": []}

    for _ in range(epochs):
        losses = []
        preds = []
        targets = []
        for adjacency, features, label in dataset:
            prob, _ = model.forward((adjacency, features))
            loss = -label * np.log(prob + 1e-8) - (1 - label) * np.log(1 - prob + 1e-8)
            grads = model.parameter_shift_gradient((adjacency, features), label)
            model.apply_gradients(grads, lr)
            losses.append(loss)
            preds.append(1 if prob >= 0.5 else 0)
            targets.append(label)
        history["loss"].append(float(np.mean(losses)))
        history["acc"].append(compute_accuracy(preds, targets))
    return model, history


def train_classical_gnn(dataset: List[Tuple[np.ndarray, np.ndarray, int]], hidden_dim: int = 4, epochs: int = 10, lr: float = 0.1) -> Tuple[ClassicalGNN, dict]:
    """Train the classical GNN with analytic gradients."""
    input_dim = dataset[0][1].shape[1]
    model = ClassicalGNN(input_dim=input_dim, hidden_dim=hidden_dim, seed=0)
    history = {"loss": [], "acc": []}

    for _ in range(epochs):
        losses = []
        preds = []
        targets = []
        for adjacency, features, label in dataset:
            prob = model.forward(adjacency, features)
            loss = -label * np.log(prob + 1e-8) - (1 - label) * np.log(1 - prob + 1e-8)
            grads = model.gradients(adjacency, features, label)
            model.apply_gradients(grads, lr)
            losses.append(loss)
            preds.append(1 if prob >= 0.5 else 0)
            targets.append(label)
        history["loss"].append(float(np.mean(losses)))
        history["acc"].append(compute_accuracy(preds, targets))
    return model, history


def prepare_dataset(raw_dataset: List[Tuple]) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Convert raw graphs into adjacency, features, and labels."""
    processed = []
    for graph, label in raw_dataset:
        adjacency, features = graph
        processed.append((adjacency, features, label))
    return processed


def split_and_prepare(dataset_raw: List[Tuple], test_ratio: float = 0.25) -> Tuple[List, List]:
    """Split dataset and convert graphs to numpy arrays."""
    train_raw, test_raw = train_test_split(dataset_raw, test_ratio=test_ratio)
    train = prepare_dataset(train_raw)
    test = prepare_dataset(test_raw)
    return train, test
