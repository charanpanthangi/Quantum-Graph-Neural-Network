"""Simple classical GNN baseline for comparison with the QGNN."""
import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation implemented with numpy."""
    return np.maximum(0, x)


class ClassicalGNN:
    """Very small GNN using matrix multiplications and ReLU activations."""

    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.5, size=(input_dim, hidden_dim))
        self.w2 = rng.normal(0, 0.5, size=(hidden_dim, 1))

    def forward(self, adjacency: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Run a single pass of the classical GNN."""
        # Add self-loops to stabilize the aggregation.
        adj = adjacency + np.eye(adjacency.shape[0])
        hidden = relu(adj @ features @ self.w1)
        pooled = hidden.mean(axis=0, keepdims=True)
        logits = pooled @ self.w2
        probs = 1 / (1 + np.exp(-logits))
        return probs.squeeze()

    def gradients(self, adjacency: np.ndarray, features: np.ndarray, target: int) -> dict:
        """Analytic gradients for the tiny network using basic calculus."""
        adj = adjacency + np.eye(adjacency.shape[0])
        hidden_linear = adj @ features @ self.w1
        hidden = relu(hidden_linear)
        pooled = hidden.mean(axis=0, keepdims=True)
        logit = pooled @ self.w2
        prob = 1 / (1 + np.exp(-logit))
        grad_logit = prob - target
        grad_w2 = pooled.T * grad_logit
        grad_hidden = grad_logit * self.w2.T / hidden.shape[0]
        grad_hidden[hidden_linear <= 0] = 0
        grad_w1 = features.T @ adj.T @ grad_hidden
        return {"w1": grad_w1, "w2": grad_w2}

    def apply_gradients(self, grads: dict, lr: float) -> None:
        """Update weights with gradient descent."""
        self.w1 -= lr * grads["w1"]
        self.w2 -= lr * grads["w2"]
