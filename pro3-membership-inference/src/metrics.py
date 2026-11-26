"""The metric of the task: TPR at 5% FPR, calculated as the server does it."""
import numpy as np
from sklearn.metrics import roc_auc_score


def tpr_at_5fpr(scores, labels, max_fpr: float = 0.05) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(-scores, kind="stable")
    scores, labels = scores[order], labels[order]
    positives = labels.sum()
    negatives = len(labels) - positives
    # Records with the same score go above the threshold together, as on the server.
    ends = np.r_[np.flatnonzero(np.diff(scores) != 0), len(scores) - 1]
    tp = np.cumsum(labels)[ends]
    fp = np.cumsum(1 - labels)[ends]
    allowed = fp / negatives <= max_fpr
    return float(tp[allowed].max() / positives) if allowed.any() else 0.0


def auc(scores, labels) -> float:
    return float(roc_auc_score(labels, scores))
