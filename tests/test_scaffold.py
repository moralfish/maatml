"""Tests for model scaffolding and validate_model_dir."""
from __future__ import annotations

from pathlib import Path

from maatml.scaffold import scaffold_model, validate_model_dir


def test_scaffold_causal_sft_and_validate(tmp_path: Path) -> None:
    target = tmp_path / "my-sft"
    scaffold_model(target, architecture="causal_sft", name="my-sft")
    assert (target / "model.yml").is_file()
    assert (target / "README.md").is_file()
    assert (target / ".gitignore").is_file()
    assert (target / "datasets" / "schema.json").is_file()
    assert (target / "datasets" / "prompt_spec.json").is_file()
    assert (target / "datasets" / "samples" / "seed_samples.jsonl").is_file()
    errors = validate_model_dir(target)
    assert errors == [], errors


def test_scaffold_classifier(tmp_path: Path) -> None:
    target = tmp_path / "toy-classifier"
    scaffold_model(target, architecture="classifier")
    body = (target / "model.yml").read_text(encoding="utf-8")
    assert "architecture: classifier" in body
    assert "expected_output" in (
        target / "datasets" / "samples" / "seed_samples.jsonl"
    ).read_text(encoding="utf-8")
    errors = validate_model_dir(target)
    assert errors == [], errors


def test_scaffold_dpo(tmp_path: Path) -> None:
    target = tmp_path / "toy-dpo"
    scaffold_model(target, architecture="dpo", name="toy-dpo")
    body = (target / "model.yml").read_text(encoding="utf-8")
    assert "architecture: dpo" in body
    assert "preference_jsonl" in body
    seed = (target / "datasets" / "samples" / "seed_samples.jsonl").read_text(
        encoding="utf-8"
    )
    assert "chosen" in seed and "rejected" in seed
    errors = validate_model_dir(target)
    assert errors == [], errors


def test_scaffold_refuses_overwrite_without_force(tmp_path: Path) -> None:
    """D1: re-scaffolding an existing model folder must not clobber it."""
    import json

    import pytest

    target = tmp_path / "keep"
    scaffold_model(target, architecture="causal_sft", name="keep")
    seed = target / "datasets" / "samples" / "seed_samples.jsonl"
    seed.write_text(json.dumps({"my": "curated"}) + "\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        scaffold_model(target, architecture="causal_sft", name="keep")
    # seed corpus untouched
    assert json.loads(seed.read_text().splitlines()[0]) == {"my": "curated"}

    # --force regenerates
    scaffold_model(target, architecture="causal_sft", name="keep", force=True)
    assert (target / "model.yml").is_file()


def test_validate_no_plugins_skips_plugin_import(tmp_path: Path, monkeypatch) -> None:
    """S2: validate --no-plugins must not import trainer or model plugin code."""
    import maatml.registry as registry_mod
    import maatml.scaffold as scaffold_mod

    target = tmp_path / "plug"
    scaffold_model(target, architecture="causal_sft", name="plug")
    # Declare a (nonexistent-but-unloaded) plugin so load would try to import it.
    yml = (target / "model.yml").read_text(encoding="utf-8")
    (target / "model.yml").write_text(yml + "plugins:\n  - not_a_real_module\n", encoding="utf-8")

    calls = {"discover": 0, "load": 0}
    monkeypatch.setattr(scaffold_mod, "discover_plugins", lambda *a, **k: calls.__setitem__("discover", calls["discover"] + 1))
    monkeypatch.setattr(registry_mod, "load_model_plugins", lambda *a, **k: calls.__setitem__("load", calls["load"] + 1))

    errors = validate_model_dir(target, load_plugins=False)
    assert calls == {"discover": 0, "load": 0}
    # schema + paths still validated; no 'not a registered trainer' error
    assert all("registered trainer" not in e for e in errors)


def test_config_key_warnings_clean_and_typos(tmp_path: Path) -> None:
    """C2: unrecognized dataset:/evaluation: keys warn (never fail)."""
    from maatml.config import config_key_warnings, load_model_def

    target = tmp_path / "warn"
    scaffold_model(target, architecture="causal_sft", name="warn")
    md = load_model_def(target)
    assert config_key_warnings(md) == []

    md.dataset["requst_field"] = "x"  # typo
    md.evaluation["metricz"] = 1  # typo
    warns = config_key_warnings(md)
    assert any("dataset.requst_field" in w for w in warns)
    assert any("evaluation.metricz" in w for w in warns)


def test_scaffold_seq2seq(tmp_path: Path) -> None:
    target = tmp_path / "toy-seq2seq"
    scaffold_model(target, architecture="seq2seq", name="toy-seq2seq")
    body = (target / "model.yml").read_text(encoding="utf-8")
    assert "architecture: seq2seq" in body
    assert "source_prefix" in body
    assert "target_field: expected_output" in body
    # seq2seq renders a prompt spec; the trainer and predictor both read it.
    assert (target / "datasets" / "prompt_spec.json").is_file()
    seed = (target / "datasets" / "samples" / "seed_samples.jsonl").read_text(
        encoding="utf-8"
    )
    assert "expected_output" in seed
    assert validate_model_dir(target) == []


def test_scaffold_multi_head_classifier(tmp_path: Path) -> None:
    target = tmp_path / "toy-heads"
    scaffold_model(target, architecture="multi_head_classifier", name="toy-heads")
    body = (target / "model.yml").read_text(encoding="utf-8")
    assert "architecture: multi_head_classifier" in body
    assert "predictor: classifier" in body
    # A classifier has no prompt spec to render.
    assert not (target / "datasets" / "prompt_spec.json").exists()
    assert validate_model_dir(target) == []


# --- plugin-owned architectures --------------------------------------------


def _write_plugin(tmp_path: Path) -> Path:
    """A plugin that owns an architecture and its scaffold defaults."""
    plugin = tmp_path / "toy_plugin"
    plugin.mkdir()
    (plugin / "__init__.py").write_text(
        "from maatml.registry import register_scaffold_hook, register_trainer\n"
        "\n"
        "@register_trainer('toy_arch')\n"
        "def _train(model_def, **kwargs):\n"
        "    return None\n"
        "\n"
        "@register_scaffold_hook('toy_arch')\n"
        "def _scaffold(target_dir, *, architecture, name):\n"
        "    return {\n"
        "        'model_yml': {\n"
        "            'dataset': {\n"
        "                'format': 'jsonl_seed',\n"
        "                'seed_samples': 'datasets/samples/seed_samples.jsonl',\n"
        "                'request_field': 'image',\n"
        "                'target_field': 'expected',\n"
        "            },\n"
        "            'training': {'backbone': 'toy-net', 'epochs': 3},\n"
        "        },\n"
        "        'seed_rows': [],\n"
        "        'files': {'GENERATE.md': 'run maatml datagen first\\n'},\n"
        "    }\n",
        encoding="utf-8",
    )
    return plugin


def test_plugin_owned_architecture_needs_its_plugin(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="need --plugin"):
        scaffold_model(tmp_path / "nope", architecture="toy_arch")


def test_scaffold_loads_plugin_and_records_it(tmp_path: Path) -> None:
    plugin = _write_plugin(tmp_path)
    target = tmp_path / "toy-model"
    scaffold_model(target, architecture="toy_arch", plugins=[str(plugin)])

    body = (target / "model.yml").read_text(encoding="utf-8")
    assert "architecture: toy_arch" in body
    # The folder keeps working after scaffolding because it records the plugin.
    assert str(plugin) in body
    # The hook's sections replace core's guesses rather than merging with them.
    assert "backbone: toy-net" in body
    assert "CHANGE_ME" not in body
    assert "prompt_spec" not in body
    assert not (target / "datasets" / "prompt_spec.json").exists()
    # An explicitly empty seed_rows means "generated corpus", not "use the
    # generic sample row".
    assert (target / "datasets" / "samples" / "seed_samples.jsonl").read_text() == ""
    assert (target / "GENERATE.md").is_file()
    assert validate_model_dir(target) == []


def test_scaffold_records_a_plugin_inside_the_model_folder_as_relative(
    tmp_path: Path,
) -> None:
    target = tmp_path / "vendored"
    target.mkdir()
    plugin = _write_plugin(target)
    scaffold_model(target, architecture="toy_arch", plugins=[str(plugin)])
    body = (target / "model.yml").read_text(encoding="utf-8")
    assert "./toy_plugin" in body
    assert str(tmp_path) not in body
