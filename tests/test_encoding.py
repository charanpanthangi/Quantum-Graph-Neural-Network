"""Tests for encoding node features to qubits."""
import numpy as np
from qiskit import QuantumCircuit
from app.encoding import encode_node_features


def test_encode_node_features_runs():
    circuit = QuantumCircuit(2)
    features = np.array([[0.0], [1.0]])
    mapping = {0: 0, 1: 1}
    encode_node_features(circuit, features, mapping)
    assert circuit.num_qubits == 2
