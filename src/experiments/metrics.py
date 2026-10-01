"""Shared probability metrics for offline experiment reports."""

import numpy as np
from sklearn.metrics import roc_auc_score


def correlation(a, b):
    return (
        None if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0 else float(np.corrcoef(a, b)[0, 1])
    )


def metrics(y, probabilities):
    return dict(
        auc=float(roc_auc_score(y, probabilities[:, 1])) if len(np.unique(y)) == 2 else None,
        accuracy=float(np.mean(probabilities.argmax(1) == y)),
    )
