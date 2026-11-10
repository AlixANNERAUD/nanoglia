"""What hardware/tools shares with emulation/tools: the comparison of golden.rs (golden.py) and
the reading of a firmware ELF (run.py). Imported from there, not copied."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# `python -I` leaves the directory of the script out of the module path.
sys.path.insert(0, str(REPO / "emulation" / "tools"))

import golden  # noqa: E402

__all__ = ["REPO", "golden"]
