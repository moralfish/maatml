"""Tests for the decorator-based plugin registry."""
from __future__ import annotations

from pathlib import Path

import pytest

from flow_ml.registry import (
    TRAINERS,
    VALIDATORS,
    discover_plugins,
    load_model_plugins,
    register_trainer,
)


@pytest.fixture(autouse=True)
def _isolate_registries():
    """Snapshot + restore all registries so tests don't leak registrations."""
    import flow_ml.registry as reg

    snaps = {kind: dict(r._entries) for kind, r in reg._ALL_REGISTRIES.items()}
    discovered_snap = reg._discovered
    yield
    for kind, entries in snaps.items():
        r = reg._ALL_REGISTRIES[kind]
        r._entries.clear()
        r._entries.update(entries)
    reg._discovered = discovered_snap


def test_register_get_require() -> None:
    @register_trainer("test_toy_trainer")
    def toy_train(x: int) -> int:
        return x + 1

    assert TRAINERS.get("test_toy_trainer") is toy_train
    assert TRAINERS.require("test_toy_trainer")(3) == 4
    assert "test_toy_trainer" in TRAINERS.names()
    with pytest.raises(KeyError):
        TRAINERS.require("does-not-exist")


def test_folder_local_plugin_load(tmp_path: Path) -> None:
    plugin = tmp_path / "local_plugin.py"
    plugin.write_text(
        "from flow_ml.registry import register_validator\n"
        "@register_validator('local_test_validator')\n"
        "def _v(*a, **k):\n"
        "    return {'ok': True}\n",
        encoding="utf-8",
    )
    loaded = load_model_plugins(tmp_path, ["local_plugin.py"])
    assert loaded
    assert VALIDATORS.get("local_test_validator") is not None
    assert VALIDATORS.require("local_test_validator")() == {"ok": True}


def test_discover_plugins_registers_contrib() -> None:
    discover_plugins(force=True)
    assert VALIDATORS.get("jcl") is not None
    assert VALIDATORS.get("spool") is not None
    assert TRAINERS.get("causal_sft") is not None
    assert TRAINERS.get("seq2seq") is not None
    assert TRAINERS.get("multi_head_classifier") is not None
    assert TRAINERS.get("classifier") is not None
