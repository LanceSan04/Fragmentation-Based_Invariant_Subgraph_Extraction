"""Fragmentation adapters and registry."""

from src.fragmentation.brics import BRICSFragmenter
from src.fragmentation.group_graph import GroupGraphFragmenter
from src.fragmentation.recap import RECAPFragmenter


FRAGMENTERS = {
    "group_graph": GroupGraphFragmenter,
    "brics": BRICSFragmenter,
    "recap": RECAPFragmenter,
}


def build_fragmenter(name="group_graph"):
    """Create a fragmenter by config name."""
    key = (name or "group_graph").lower()
    if key not in FRAGMENTERS:
        available = ", ".join(sorted(FRAGMENTERS))
        raise ValueError(f"Unknown fragmentation method {name!r}. Available: {available}")
    return FRAGMENTERS[key]()
