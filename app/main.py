"""Command-line entrypoint for the QGNN demo."""
import argparse
import os
import numpy as np
from .graphs import generate_graph_dataset, graph_to_adjacency_and_features
from .trainer import train_qgnn, train_classical_gnn, split_and_prepare
from .plots import (
    plot_graph_example,
    plot_embedding_projection,
    plot_training_loss,
    plot_performance_bar,
)


def run_demo(epochs: int, lr: float, task: str) -> None:
    """Run the full pipeline: dataset, training, and SVG plots."""
    raw_graphs = generate_graph_dataset(task=task)
    # Convert graphs to numpy arrays for training.
    processed = []
    for graph, label in raw_graphs:
        adjacency, features = graph_to_adjacency_and_features(graph)
        processed.append(((adjacency, features), label))

    train_set, test_set = split_and_prepare(processed, test_ratio=0.5)

    # Train QGNN
    qgnn_model, history_qgnn = train_qgnn(train_set, num_layers=1, epochs=epochs, lr=lr)

    # Train classical baseline
    classical_model, history_classical = train_classical_gnn(train_set, hidden_dim=4, epochs=epochs, lr=lr)

    # Evaluate on test split
    qgnn_preds = []
    classical_preds = []
    targets = []
    embeddings = []
    for adjacency, features, label in test_set:
        prob_qgnn, emb = qgnn_model.forward((adjacency, features))
        prob_classical = classical_model.forward(adjacency, features)
        qgnn_preds.append(1 if prob_qgnn >= 0.5 else 0)
        classical_preds.append(1 if prob_classical >= 0.5 else 0)
        targets.append(label)
        embeddings.append(emb.flatten())

    acc_qgnn = sum(int(p == t) for p, t in zip(qgnn_preds, targets)) / len(targets)
    acc_classical = sum(int(p == t) for p, t in zip(classical_preds, targets)) / len(targets)

    os.makedirs("examples", exist_ok=True)
    # Plot example graph
    plot_graph_example(raw_graphs[0][0], "examples/qgnn_graph_example.svg")
    # Plot embedding projection
    if len(embeddings) >= 2:
        plot_embedding_projection(np.vstack(embeddings), targets, "examples/qgnn_embedding_projection.svg")
    else:
        plot_embedding_projection(np.vstack([embeddings[0], embeddings[0]]), [targets[0], targets[0]], "examples/qgnn_embedding_projection.svg")
    # Plot training loss curves
    plot_training_loss(history_qgnn, history_classical, "examples/qgnn_training_loss.svg")
    # Plot performance comparison
    plot_performance_bar(acc_qgnn, acc_classical, "examples/qgnn_vs_classical_gnn_performance.svg")

    print(f"QGNN test accuracy: {acc_qgnn:.2f}")
    print(f"Classical GNN test accuracy: {acc_classical:.2f}")
    print("SVG visualizations saved to examples/ directory")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the demo."""
    parser = argparse.ArgumentParser(description="Quantum Graph Neural Network demo")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs for both models")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--task", type=str, default="graph_classification", choices=["graph_classification", "node_classification"], help="Task type")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(epochs=args.epochs, lr=args.lr, task=args.task)
