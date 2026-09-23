"""Shared finite, complete-window metric policy for training and search."""
import math


def trailing_mean(values, window):
    """Return a complete trailing mean, or None during warm-up/invalid windows."""
    if window < 1:
        raise ValueError("selection window must be a positive integer")
    if len(values) < window:
        return None
    chunk = values[-window:]
    if any(v is None or not math.isfinite(float(v)) for v in chunk):
        return None
    return sum(float(v) for v in chunk) / window


def best_moving_average(values, window):
    """Highest eligible trailing mean; no partial-window fallback."""
    if window < 1:
        raise ValueError("selection window must be a positive integer")
    means = (trailing_mean(values[max(0, end - window):end], window)
             for end in range(window, len(values) + 1))
    return max((value for value in means if value is not None), default=None)
