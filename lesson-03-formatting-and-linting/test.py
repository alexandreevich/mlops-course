import numpy as np


def calculate_precision(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    precision = tp / (tp + fp)
    return precision


y_true = np.array([1, 1, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 1, 0])

precision: float = calculate_precision(y_pred, y_true)
print(f"Precision: {precision:.2f}")
