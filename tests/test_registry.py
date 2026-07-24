"""Tests for the decorator-based plugin registry."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from maatml.config import load_model_def
from maatml.registry import (
    EXPORTERS,
    FORMATS,
    PREDICTORS,
    TRAINERS,
    VALIDATORS,
    discover_plugins,
    load_model_plugins,
    register_trainer,
    reset_registries,
)


# Registry isolation is the autouse fixture in tests/conftest.py.


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
        "from maatml.registry import register_validator\n"
        "@register_validator('local_test_validator')\n"
        "def _v(*a, **k):\n"
        "    return {'ok': True}\n",
        encoding="utf-8",
    )
    loaded = load_model_plugins(tmp_path, ["local_plugin.py"])
    assert loaded
    assert VALIDATORS.get("local_test_validator") is not None
    assert VALIDATORS.require("local_test_validator")() == {"ok": True}


def test_folder_local_package_plugin_load(tmp_path: Path) -> None:
    pkg = tmp_path / "toy_plugin"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "from maatml.registry import register_validator\n"
        "from .helper import VALUE\n"
        "@register_validator('pkg_test_validator')\n"
        "def _v(*a, **k):\n"
        "    return {'value': VALUE}\n",
        encoding="utf-8",
    )
    (pkg / "helper.py").write_text("VALUE = 42\n", encoding="utf-8")
    loaded = load_model_plugins(tmp_path, ["./toy_plugin"])
    assert loaded
    assert VALIDATORS.require("pkg_test_validator")() == {"value": 42}


def test_discover_plugins_registers_core() -> None:
    # reset_registries is the blank slate; discover_plugins only adds to the
    # registry (it must never wipe what a model folder registered).
    reset_registries()
    discover_plugins()
    assert TRAINERS.get("causal_sft") is not None
    assert TRAINERS.get("seq2seq") is not None
    assert TRAINERS.get("multi_head_classifier") is not None
    assert TRAINERS.get("classifier") is not None
    assert FORMATS.get("jsonl_seed") is not None
    assert FORMATS.get("alpaca") is not None
    assert FORMATS.get("sharegpt") is not None
    assert PREDICTORS.get("seq2seq") is not None
    assert PREDICTORS.get("classifier") is not None
    assert EXPORTERS.get("safetensors") is not None
    assert EXPORTERS.get("gguf") is not None
    assert EXPORTERS.get("mlx") is not None
    # Task validators live in example plugins, not core discovery.
    assert VALIDATORS.get("jcl") is None
    assert VALIDATORS.get("spool") is None


def test_load_model_def_registers_jcl_plugin() -> None:
    discover_plugins(force=True)
    repo = Path(__file__).resolve().parents[1]
    md = load_model_def(repo / "examples" / "jcl-validator")
    assert "jcl" in (md.evaluation or {}).get("validator", "")
    assert VALIDATORS.get("jcl") is not None
    assert PREDICTORS.get("jcl_classifier") is not None
    from maatml.registry import GENERATORS

    assert GENERATORS.get("jcl") is not None


def test_model_plugins_execute_once_per_process(tmp_path: Path) -> None:
    """load_model_def and the CLI both ask for plugins; the code runs once."""
    plugin = tmp_path / "counting_plugin.py"
    plugin.write_text(
        "import os\n"
        "from maatml.registry import register_validator\n"
        "os.environ['MAATML_TEST_PLUGIN_RUNS'] = str(\n"
        "    int(os.environ.get('MAATML_TEST_PLUGIN_RUNS', '0')) + 1\n"
        ")\n"
        "@register_validator('counting_test_validator')\n"
        "def _v(*a, **k):\n"
        "    return {'ok': True}\n",
        encoding="utf-8",
    )
    os.environ.pop("MAATML_TEST_PLUGIN_RUNS", None)
    try:
        load_model_plugins(tmp_path, ["counting_plugin.py"])
        assert os.environ["MAATML_TEST_PLUGIN_RUNS"] == "1"

        loaded = load_model_plugins(tmp_path, ["counting_plugin.py"])
        assert os.environ["MAATML_TEST_PLUGIN_RUNS"] == "1", "plugin re-executed"
        assert loaded, "the module name is still reported to the caller"

        load_model_plugins(tmp_path, ["counting_plugin.py"], force=True)
        assert os.environ["MAATML_TEST_PLUGIN_RUNS"] == "2"
    finally:
        os.environ.pop("MAATML_TEST_PLUGIN_RUNS", None)


def test_discover_plugins_does_not_wipe_model_plugin_registrations(tmp_path: Path) -> None:
    plugin = tmp_path / "keepme_plugin.py"
    plugin.write_text(
        "from maatml.registry import register_validator\n"
        "@register_validator('keepme_validator')\n"
        "def _v(*a, **k):\n"
        "    return {'ok': True}\n",
        encoding="utf-8",
    )
    load_model_plugins(tmp_path, ["keepme_plugin.py"])
    assert VALIDATORS.get("keepme_validator") is not None

    discover_plugins(force=True)
    assert VALIDATORS.get("keepme_validator") is not None, (
        "rediscovery must not drop what a model folder registered"
    )
    assert TRAINERS.get("causal_sft") is not None


def test_entry_point_failures_are_surfaced(monkeypatch) -> None:
    """A broken third-party plugin is reported, not silently skipped."""
    import maatml.registry as reg

    class _BrokenEntryPoint:
        name = "broken_plugin"

        def load(self):
            raise ImportError("no module named 'nope'")

    class _Eps:
        def select(self, group):
            del group
            return [_BrokenEntryPoint()]

    monkeypatch.setattr("importlib.metadata.entry_points", lambda: _Eps())
    reg.discover_plugins(force=True)

    errors = dict(reg.load_errors())
    assert "entry_point:broken_plugin" in errors
    assert "no module named" in errors["entry_point:broken_plugin"]

    # The failure is named in "Unknown … plugin" errors, where it explains why.
    with pytest.raises(KeyError, match="broken_plugin"):
        TRAINERS.require("some_missing_trainer")


def test_windows_style_plugin_paths_are_treated_as_paths() -> None:
    """A backslash path must not be sent down the import_module branch."""
    from maatml.registry import looks_like_plugin_path

    for entry in (
        r"C:\models\my-model\my_plugin",
        r"plugins\hook.py",
        r".\vision_plugin",
        "./vision_plugin",
        "plugins/hook.py",
        "sub/dir",
    ):
        assert looks_like_plugin_path(entry), entry

    # Dotted module paths stay module paths.
    for entry in ("my_pkg.plugins.foo", "maatml_vision"):
        assert not looks_like_plugin_path(entry), entry


def test_bare_directory_name_next_to_the_model_is_a_path(tmp_path: Path) -> None:
    from maatml.registry import looks_like_plugin_path

    (tmp_path / "sibling_plugin").mkdir()
    assert looks_like_plugin_path("sibling_plugin", tmp_path)
    assert not looks_like_plugin_path("sibling_plugin", tmp_path / "elsewhere")
