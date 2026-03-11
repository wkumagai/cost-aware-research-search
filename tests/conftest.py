"""
Pytest configuration.

Adds the repository root to sys.path so that `from src.loop import ...` works
in test files without needing an installed package.
"""

import sys
from pathlib import Path

# Add repo root to path so `from src.loop import ...` works
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
