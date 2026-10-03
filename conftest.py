"""
Pytest configuration.

Adds the repository root to sys.path so tests can import from `src.*`
without requiring an editable install.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))