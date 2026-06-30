"""Local runtime defaults for cleaner execution in constrained environments."""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".runtime_cache"
CACHE_DIR.mkdir(exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 2))
