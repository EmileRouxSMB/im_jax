from __future__ import annotations

import sys
from pathlib import Path

import jax
import pytest


@pytest.fixture(autouse=True, scope="session")
def _enable_x64():
    jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True, scope="session")
def _add_repo_root():
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
