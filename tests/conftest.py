"""Shared test fixtures.

Registries are process-global, so a test that registers a plugin (or calls
``discover_plugins(force=True)``) would otherwise leak into every test that
runs after it. The autouse fixture snapshots and restores them through the
public registry API, so tests never reach into ``_entries``.
"""
from __future__ import annotations

import pytest

from maatml.registry import discover_plugins, restore_registries, snapshot_registries


@pytest.fixture(autouse=True)
def _isolate_registries():
    # Discover first so the baseline snapshot holds the built-ins: snapshotting
    # an empty registry and restoring it later would wipe whatever the test
    # (or an earlier import) registered.
    discover_plugins()
    snapshot = snapshot_registries()
    try:
        yield
    finally:
        restore_registries(snapshot)
