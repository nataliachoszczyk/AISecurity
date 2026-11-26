"""Local validation on known.jsonl: folds that copy the relation between known and test, and the score of each fold."""
import numpy as np
import pandas as pd

from .metrics import auc, tpr_at_5fpr


def make_folds(known: pd.DataFrame, n_folds: int = 2, seed: int = 0) -> np.ndarray:
    """Return the fold of each known record.

    Each set owner has records in known and in test. Thus the split is inside each owner, so that each fold sees each owner.
    Single records (forum posts, web photos) are split inside their source type. Both splits keep the member rate.
    """
    rng = np.random.default_rng(seed)
    owner_size = known.groupby("source")["source"].transform("size")
    group = known["source"].where(owner_size > 1, known["source_type"])
    fold = np.empty(len(known), dtype=int)
    for idx in known.groupby([group, known["member"]]).indices.values():
        idx = rng.permutation(idx)
        # A random start for each group keeps the folds equal in size.
        fold[idx] = (rng.integers(n_folds) + np.arange(len(idx))) % n_folds
    return fold


def cross_validate(known: pd.DataFrame, score_fn, n_folds: int = 2, seeds=range(5), by: str | None = "source_type") -> pd.DataFrame:
    """Score each fold with score_fn(fit, evaluate) -> scores of the evaluate records.

    The metric uses all records of a fold together, as the server does. With `by`, each group also gets its own row.
    """
    rows = []
    for seed in seeds:
        fold = make_folds(known, n_folds, seed)
        for f in range(n_folds):
            fit, evaluate = known[fold != f], known[fold == f]
            scores = pd.Series(np.asarray(score_fn(fit, evaluate), dtype=float), index=evaluate.index)
            parts = [("all", evaluate)] + (list(evaluate.groupby(by)) if by else [])
            for group, part in parts:
                labels = part["member"].to_numpy()
                values = scores.loc[part.index].to_numpy()
                rows.append({"seed": seed, "fold": f, "records": group, "n": len(part),
                             "TPR@5%FPR": tpr_at_5fpr(values.tolist(), labels.tolist()),
                             "AUC": auc(values, labels) if 0 < labels.sum() < len(labels) else np.nan})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over all seeds and folds."""
    table = results.groupby("records", sort=False)[["TPR@5%FPR", "AUC"]].agg(["mean", "std"])
    table.columns = [f"{metric} {stat}" for metric, stat in table.columns]
    return table.reset_index()
