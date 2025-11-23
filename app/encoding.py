"""Node feature encoding for the QGNN demo.

Each node feature value is mapped to a simple Y-rotation on its dedicated
qubit. This keeps the encoding easy to understand while still producing
non-trivial quantum states.
"""
from typing import Dict
import numpy as np
from qiskit import QuantumCircuit


def encode_node_features(circuit: QuantumCircuit, node_features: np.ndarray, qubit_mapping: Dict[int, int]) -> None:
    """Encode node features as rotation angles on qubits.

    Parameters
    ----------
    circuit: QuantumCircuit
        Circuit where the rotations will be added.
    node_features: np.ndarray
        Array of shape (n_nodes, feature_dim). Only the first column is used in
        this tiny demo.
    qubit_mapping: dict
        Mapping from node index to qubit index so that features are placed on
        the correct qubits.
    """
    for node_idx, qubit_idx in qubit_mapping.items():
        # Normalize the feature to [0, 1] and map to an angle.
        feature_value = float(node_features[node_idx, 0])
        angle = np.pi * feature_value
        circuit.ry(angle, qubit_idx)
