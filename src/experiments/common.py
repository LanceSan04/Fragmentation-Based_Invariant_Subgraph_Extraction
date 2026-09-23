"""Shared experiment RNG setup and numeric summaries."""
import random
import math
import statistics
import numpy as np
import torch

def set_all_seeds(seed):
    """Seed every RNG the training path draws from, model init included.

    run_training seeds these again from config.seed; both are kept so a run is
    reproducible whether it is launched from here or from train_sugar.py.
    Note that CUDA scatter reductions stay nondeterministic, so runs are
    repeatable in distribution but not bit-identical.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _summarize(values):
    """mean/std/n over a list that may contain None (e.g. single-class AUC)."""
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.mean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "n": len(clean),
    }


