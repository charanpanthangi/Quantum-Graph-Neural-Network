"""Single QGNN message-passing layer.

A layer encodes node features onto qubits and applies parameterized entangling
gates for every edge. The output is a set of node embeddings computed from the
expectation value of the Pauli-Z operator on each qubit.
"""
from typing import Dict, Tuple
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from .encoding import encode_node_features


def build_qgnn_layer_circuit(adj_matrix: np.ndarray, node_features: np.ndarray, params_layer: Dict[str, np.ndarray]) -> QuantumCircuit:
    """Create a QuantumCircuit representing one QGNN layer.

    Parameters
    ----------
    adj_matrix: np.ndarray
        Adjacency matrix describing where entangling gates should be placed.
    node_features: np.ndarray
        Node features for this layer; in this demo we use degree-based features.
    params_layer: dict
        Dictionary containing edge weights and optional node rotation weights.

    Returns
    -------
    QuantumCircuit
        Circuit with feature encoding and entangling gates.
    """
    num_nodes = adj_matrix.shape[0]
    circuit = QuantumCircuit(num_nodes)

    # Map each node to a qubit with the same index for clarity.
    qubit_mapping = {node: node for node in range(num_nodes)}

    # Encode classical features into qubit rotations.
    encode_node_features(circuit, node_features, qubit_mapping)

    # Entangling gates on edges simulate quantum message passing.
    edge_weights = params_layer.get("edge_weights", np.zeros_like(adj_matrix))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i, j] > 0:
                theta = edge_weights[i, j]
                # Parameterized ZZ rotation is a simple entangling gate.
                circuit.rzz(theta, i, j)

    # Optional node-wise trainable rotations for extra flexibility.
    node_weights = params_layer.get("node_weights", np.zeros(num_nodes))
    for qubit_idx in range(num_nodes):
        circuit.rz(node_weights[qubit_idx], qubit_idx)

    return circuit


def run_qgnn_layer(params_layer: Dict[str, np.ndarray], graph_data: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Run one QGNN layer and produce node embeddings.

    Parameters
    ----------
    params_layer: dict
        Contains edge_weights (square matrix) and node_weights (vector).
    graph_data: tuple
        (adjacency_matrix, node_features) representing the graph.

    Returns
    -------
    np.ndarray
        Array of shape (n_nodes, 1) containing expectation values of Z on each
        qubit. These act as node embeddings.
    """
    adjacency, features = graph_data
    circuit = build_qgnn_layer_circuit(adjacency, features, params_layer)

    # Statevector simulation keeps the example lightweight and deterministic.
    state = Statevector.from_instruction(circuit)

    embeddings = []
    for qubit_idx in range(adjacency.shape[0]):
        # Build Z operator for the specific qubit.
        z_ops = ["I"] * adjacency.shape[0]
        z_ops[qubit_idx] = "Z"
        op = SparsePauliOp.from_list([( "".join(reversed(z_ops)), 1.0)])
        expectation = np.real(state.expectation_value(op))
        embeddings.append([expectation])

    return np.array(embeddings)
