"""Global test configuration and native backend fixture."""

from __future__ import annotations

import pytest

from GoTorch.backend import native_backend

ACTIVE_BACKEND = native_backend


@pytest.fixture(scope="session")
def backend():
    """Provides the active native C++ backend module."""
    return ACTIVE_BACKEND


@pytest.fixture(scope="session")
def backend_name():
    """Identifies which implementation is currently running."""
    return getattr(ACTIVE_BACKEND, "__backend_type__", "cpp")
