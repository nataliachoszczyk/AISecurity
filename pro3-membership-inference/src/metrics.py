"""The metric of the task: TPR at 5% FPR."""
from sklearn.metrics import roc_auc_score


def tpr_at_5fpr(scores, labels) -> float:
    """A copy of the server function from the task text, with English names."""
    pairs = sorted(zip(scores, labels), key=lambda p: -p[0])
    P = sum(labels); N = len(labels) - P
    tp = fp = 0; best = 0.0; i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1  # records with the same score are counted together
        tp += sum(e for _, e in pairs[i:j]); fp += sum(1 - e for _, e in pairs[i:j])
        if fp / N > 0.05:
            break
        best = tp / P
        i = j
    return best


def auc(scores, labels) -> float:
    return float(roc_auc_score(labels, scores))
