import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def find_best_threshold(y_true, y_proba, metric: str = "f1"):
    """
    Find threshold in [0.05, 0.95] that maximizes the chosen metric.
    metric: 'f1' (default), 'precision', or 'recall'
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_val = 0.5, -1e18

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        if metric == "f1":
            v = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            v = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            v = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError("metric must be one of: 'f1', 'precision', 'recall'")

        if v > best_val:
            best_val = v
            best_t = t

    return float(best_t)