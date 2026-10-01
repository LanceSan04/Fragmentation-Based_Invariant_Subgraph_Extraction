"""One-based epoch schedule. Delay D means epochs 1..D have zero weight."""


def diversity_weight(target, epoch, delay=0, ramp=0):
    if target < 0 or epoch < 1 or delay < 0 or ramp < 0:
        raise ValueError("Nonnegative weights/durations and a positive epoch are required")
    if epoch <= delay:
        return 0.0
    return float(target) * (min(1.0, (epoch - delay) / ramp) if ramp else 1.0)
