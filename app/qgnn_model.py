"""Stacked QGNN model with a simple classical readout."""
from typing import List, Tuple
import numpy as np
from .qgnn_layer import run_qgnn_layer


class QGNNModel:
    """Tiny QGNN model with multiple layers and a logistic readout."""

    def __init__(self, num_layers: int, num_nodes: int, seed: int = 0):
        self.num_layers = num_layers
        rng = np.random.default_rng(seed)
        # Each layer stores edge weights and node weights.
        self.layers: List[dict] = []
        for _ in range(num_layers):
            self.layers.append(
                {
                    "edge_weights": rng.normal(0, 0.2, size=(num_nodes, num_nodes)),
                    "node_weights": rng.normal(0, 0.2, size=(num_nodes,)),
                }
            )
        # Simple linear readout from pooled embedding to single logit.
        self.readout_weights = rng.normal(0, 0.5, size=(num_nodes, 1))
        self.readout_bias = 0.0

    def forward(self, graph_data: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, np.ndarray]:
        """Run the QGNN forward pass and return probability and embedding."""
        adjacency, features = graph_data
        node_embeddings = features
        # Sequentially apply layers, using outputs as new features.
        for layer_params in self.layers:
            node_embeddings = run_qgnn_layer(layer_params, (adjacency, node_embeddings))
        # Global mean pooling to obtain graph-level vector.
        graph_embedding = node_embeddings.mean(axis=0)
        logit = float(graph_embedding @ self.readout_weights.mean(axis=0) + self.readout_bias)
        prob = 1 / (1 + np.exp(-logit))
        return prob, graph_embedding

    def parameter_shift_gradient(self, graph_data: Tuple[np.ndarray, np.ndarray], target: int, shift: float = 0.2) -> List[dict]:
        """Estimate gradients with a finite difference style shift rule."""
        base_prob, _ = self.forward(graph_data)
        grad_layers: List[dict] = []

        for layer_params in self.layers:
            grad_edge = np.zeros_like(layer_params["edge_weights"])
            grad_node = np.zeros_like(layer_params["node_weights"])
            # Edge weight gradients.
            for i in range(grad_edge.shape[0]):
                for j in range(grad_edge.shape[1]):
                    if i >= j:
                        continue
                    layer_params["edge_weights"][i, j] += shift
                    plus_prob, _ = self.forward(graph_data)
                    layer_params["edge_weights"][i, j] -= 2 * shift
                    minus_prob, _ = self.forward(graph_data)
                    layer_params["edge_weights"][i, j] += shift
                    grad_edge[i, j] = (plus_prob - minus_prob) / (2 * shift)
            # Node weight gradients.
            for idx in range(grad_node.shape[0]):
                layer_params["node_weights"][idx] += shift
                plus_prob, _ = self.forward(graph_data)
                layer_params["node_weights"][idx] -= 2 * shift
                minus_prob, _ = self.forward(graph_data)
                layer_params["node_weights"][idx] += shift
                grad_node[idx] = (plus_prob - minus_prob) / (2 * shift)
            grad_layers.append({"edge_weights": grad_edge, "node_weights": grad_node})

        # Readout gradient is analytic because it is classical.
        _, final_emb = self.forward(graph_data)
        logit = float(final_emb @ self.readout_weights.mean(axis=0) + self.readout_bias)
        prob = 1 / (1 + np.exp(-logit))
        grad_logit = prob - target
        grad_readout = final_emb * grad_logit
        grad_bias = grad_logit
        grad_layers.append({"readout": grad_readout, "bias": grad_bias})
        return grad_layers

    def apply_gradients(self, grads: List[dict], lr: float) -> None:
        """Update parameters with the provided gradients."""
        for layer, grad in zip(self.layers, grads[:-1]):
            layer["edge_weights"] -= lr * grad["edge_weights"]
            layer["node_weights"] -= lr * grad["node_weights"]
        readout_grad = grads[-1]
        self.readout_weights -= lr * readout_grad["readout"].reshape(self.readout_weights.shape)
        self.readout_bias -= lr * readout_grad["bias"]
