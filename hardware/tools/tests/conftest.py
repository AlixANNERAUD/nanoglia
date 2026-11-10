"""Puts hardware/tools on the module path, as hardware.py does."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
