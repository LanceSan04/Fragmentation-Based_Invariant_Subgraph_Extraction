"""Resolve the single documented cache relocation without editing saved configs."""

from pathlib import Path
import warnings

ROOT = Path(__file__).resolve().parents[2]
OLD_ASSAY_CACHE = "outputs_overnight_losses_20260913_102501/processed"
NEW_ASSAY_CACHE = "data/processed/ec50_followup_v1/assay"


def resolve_processed_path(value):
    if not value:
        return value
    path = Path(value)
    if path.exists():
        return value
    normalized = str(value).replace("\\", "/").rstrip("/")
    target = ROOT / NEW_ASSAY_CACHE
    if (
        normalized == OLD_ASSAY_CACHE or normalized.endswith("/" + OLD_ASSAY_CACHE)
    ) and target.is_dir():
        warnings.warn(f"Resolving relocated Assay cache: {value} -> {target}", UserWarning)
        return str(target)
    return value
