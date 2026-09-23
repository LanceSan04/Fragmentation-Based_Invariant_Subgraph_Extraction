"""Compatibility entry point. Prefer python main.py experiment; implementation is shared."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments import parameter_search as _implementation
if __name__ == "__main__":
    _implementation.main()
else:
    # Preserve imports and monkeypatch targets used by existing callers/tests.
    sys.modules[__name__] = _implementation
