"""Ensure the tests directory is importable so ``from _exact import ...`` works."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
