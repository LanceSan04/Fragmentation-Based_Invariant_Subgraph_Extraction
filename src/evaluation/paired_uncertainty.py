"""Paired assay-cluster bootstrap, conditional on the fitted seed models."""
import numpy as np
from sklearn.metrics import roc_auc_score


def cluster_difference(first_paths, second_paths, repeats=1000, seed=41010):
    if not first_paths or len(first_paths) != len(second_paths):
        raise ValueError("Equal nonempty paired seed lists required")
    first, second = [], []
    reference = None
    for a, b in zip(first_paths, second_paths):
        for path, values in ((a, first), (b, second)):
            with np.load(path, allow_pickle=False) as data:
                membership = tuple(data[k].copy() for k in ("labels", "assays", "row_index"))
                if reference is None:
                    reference = membership
                elif any(not np.array_equal(x, y) for x, y in zip(reference, membership)):
                    raise ValueError("Paired prediction membership differs")
                values.append(data["probabilities"].copy())
    labels, assays, _ = reference
    if any(str(a) in ("", "None") for a in assays):
        return dict(available=False, reason="Missing assay identifiers")
    groups, inverse = np.unique(assays, return_inverse=True)
    rng = np.random.RandomState(seed)
    differences = []
    for _ in range(repeats):
        weights = np.bincount(rng.randint(len(groups), size=len(groups)),
                              minlength=len(groups))[inverse]
        if len(np.unique(labels[weights > 0])) != 2:
            continue
        differences.append(np.mean([
            roc_auc_score(labels, a, sample_weight=weights) -
            roc_auc_score(labels, b, sample_weight=weights)
            for a, b in zip(first, second)]))
    return dict(available=bool(differences), requested_samples=repeats,
                valid_samples=len(differences), assay_count=len(groups), seed=seed,
                percentile_95=np.quantile(differences, [.025, .975]).tolist() if differences else None,
                interpretation="Conditional assay-cluster uncertainty for the mean paired seed difference; does not include retraining variation or remove adaptive benchmark reuse.")
