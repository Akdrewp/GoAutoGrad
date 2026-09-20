"""Global test configuration and dynamic backend selector.

This is so I don't have to switch it out after I've
implemented it in C++
"""

from __future__ import annotations

import os

import pytest

# Determine active backend:
# Set GOTORCH_BACKEND="cpp" to use backend implementation
BACKEND_CHOICE = os.getenv("GOTORCH_BACKEND", "auto").lower()

def _resolve_backend():
    if BACKEND_CHOICE in ("cpp", "c", "native"):
        from GoTorch.backend import native_backend
        return native_backend

    if BACKEND_CHOICE == "python":
        from GoTorch.backend import ndarray
        return ndarray

    # Try importing from c++ by default
    try:
        from GoTorch.backend import native_backend
        return native_backend
    except ImportError:
        from GoTorch.backend import ndarray
        return ndarray


ACTIVE_BACKEND = _resolve_backend()


@pytest.fixture(scope="session")
def backend():
    """Provides the active backend module (Python or C/C++).
    """
    return ACTIVE_BACKEND


@pytest.fixture(scope="session")
def backend_name():
    """Identifies which implementation is currently running."""
    return getattr(ACTIVE_BACKEND, "__backend_type__", "python")
