"""Utility helpers for dataset handling and metrics."""
from typing import List, Tuple
import numpy as np


def train_test_split(dataset: List, test_ratio: float = 0.25, seed: int = 0) -> Tuple[List, List]:
    """Split a list into train and test parts."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    split = int(len(dataset) * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]
    train = [dataset[i] for i in train_idx]
    test = [dataset[i] for i in test_idx]
    return train, test


def compute_accuracy(preds: List[int], targets: List[int]) -> float:
    """Compute simple classification accuracy."""
    if len(preds) == 0:
        return 0.0
    correct = sum(int(p == t) for p, t in zip(preds, targets))
    return correct / len(preds)
