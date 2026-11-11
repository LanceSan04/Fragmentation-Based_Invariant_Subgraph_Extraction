"""Reported-only within-assay ranking with fixed class-count eligibility."""

from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score


def within_assay_report(labels, probabilities, assays, min_per_class=5):
    if min_per_class < 1:
        raise ValueError("min_per_class must be positive")
    if not (len(labels) == len(probabilities) == len(assays)):
        raise ValueError("Predictions and assay identities must align")
    groups = defaultdict(list)
    for i, assay in enumerate(assays):
        if assay is not None:
            groups[str(assay)].append(i)
    rows = []
    y, p = np.asarray(labels), np.asarray(probabilities)
    for assay, indices in groups.items():
        target = y[indices]
        positive = int((target == 1).sum())
        negative = int((target == 0).sum())
        if min(positive, negative) >= min_per_class:
            rows.append(dict(assay=assay, n=len(indices), pairs=positive * negative,
                             auc=float(roc_auc_score(target, p[indices]))))
    pairs = sum(r["pairs"] for r in rows)
    total_pairs = int((y == 1).sum()) * int((y == 0).sum())
    return dict(macro_auc=float(np.mean([r["auc"] for r in rows])) if rows else None,
                pair_weighted_auc=sum(r["auc"] * r["pairs"] for r in rows) / pairs if pairs else None,
                eligible_assays=len(rows), total_assays=len(groups),
                molecule_coverage=sum(r["n"] for r in rows) / len(y) if len(y) else 0,
                pair_coverage=pairs / total_pairs if total_pairs else 0,
                min_per_class=min_per_class, reported_only=True, per_assay=rows)
