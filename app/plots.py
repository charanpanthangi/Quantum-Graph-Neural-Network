"""SVG plotting utilities for the QGNN project."""
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx

sns.set_style("whitegrid")


def plot_graph_example(graph: nx.Graph, output_path: str) -> None:
    """Draw a NetworkX graph and save as SVG."""
    plt.figure(figsize=(4, 3))
    pos = nx.spring_layout(graph, seed=0)
    nx.draw(graph, pos, with_labels=True, node_color="#99c1f1", edge_color="#4c566a")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_embedding_projection(embeddings: np.ndarray, labels: List[int], output_path: str) -> None:
    """Project embeddings to 2D (PCA) and save a scatter plot as SVG."""
    # Simple 2D projection using mean-centering; dataset is tiny.
    mean = embeddings.mean(axis=0, keepdims=True)
    centered = embeddings - mean
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    top_vecs = eigvecs[:, -2:]
    projected = centered @ top_vecs
    plt.figure(figsize=(4, 3))
    scatter = plt.scatter(projected[:, 0], projected[:, 1], c=labels, cmap="coolwarm", s=60)
    plt.colorbar(scatter, label="Class label")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_training_loss(history_qgnn: dict, history_classical: dict, output_path: str) -> None:
    """Plot training losses for both models as SVG."""
    plt.figure(figsize=(4, 3))
    plt.plot(history_qgnn.get("loss", []), label="QGNN loss", marker="o")
    plt.plot(history_classical.get("loss", []), label="Classical GNN loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_performance_bar(acc_qgnn: float, acc_classical: float, output_path: str) -> None:
    """Bar plot comparing accuracies as SVG."""
    plt.figure(figsize=(4, 3))
    models = ["QGNN", "Classical GNN"]
    accuracies = [acc_qgnn, acc_classical]
    sns.barplot(x=models, y=accuracies, palette="pastel")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()
