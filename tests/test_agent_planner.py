from __future__ import annotations

import json
import shutil
from pathlib import Path

from flow_ml.config import load_model_def
from flow_ml.data.pipeline import prepare_agent
from flow_ml.data.schemas import AgentPlan
from flow_ml.utils.io import iter_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models" / "agent-planner"
DATASET_DIR = MODEL_DIR / "datasets"
PROMPT_SPEC = DATASET_DIR / "prompt_spec.json"
SEED_SAMPLES = DATASET_DIR / "samples" / "seed_samples.jsonl"
EVAL_SAMPLES = DATASET_DIR / "samples" / "eval_samples.jsonl"


def _make_agent_model_folder(tmp_path: Path) -> Path:
    mdir = tmp_path / "agent-model"
    (mdir / "datasets" / "samples").mkdir(parents=True)
    shutil.copy2(SEED_SAMPLES, mdir / "datasets" / "samples" / "seed_samples.jsonl")
    shutil.copy2(EVAL_SAMPLES, mdir / "datasets" / "samples" / "eval_samples.jsonl")
    (mdir / "datasets" / "prompt_spec.json").write_text(
        PROMPT_SPEC.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (mdir / "model.yml").write_text(
        "\n".join(
            [
                "name: agent-test",
                "model_id: agent-test:v1",
                "task: agent_planning",
                "data:",
                "  seed: 7331",
                "  prompt_spec: datasets/prompt_spec.json",
                "  seed_samples: datasets/samples/seed_samples.jsonl",
                "  benchmark_samples: datasets/samples/eval_samples.jsonl",
                "  split_ratios: [0.7, 0.2, 0.1]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return mdir


def test_agent_prompt_spec_has_required_fields() -> None:
    spec = json.loads(PROMPT_SPEC.read_text(encoding="utf-8"))
    assert spec["schema_version"] == "1"
    assert "<<AGENT_INPUT>>" in spec["user_template"]
    schema = spec["response_schema"]
    assert schema["type"] == "object"
    assert schema["required"] == [
        "intent_summary",
        "plan_steps",
        "tool_calls",
        "dsl_patch",
        "dsl",
        "confidence",
        "refusal_reason",
    ]
    assert spec["candidate_models"]["primary"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert spec["candidate_models"]["fallback"] == "HuggingFaceTB/SmolLM3-3B"
    assert spec["temperature"] == 0.0


def test_agent_samples_validate_against_runtime_schema() -> None:
    rows = list(iter_jsonl(SEED_SAMPLES)) + list(iter_jsonl(EVAL_SAMPLES))
    assert rows
    seen_ids: set[str] = set()
    for row in rows:
        for key in ("sample_id", "source", "request", "expected_intent", "agent_plan"):
            assert key in row, f"missing {key} in {row.get('sample_id')}"
        assert row["sample_id"] not in seen_ids
        seen_ids.add(row["sample_id"])
        plan = AgentPlan.model_validate(row["agent_plan"])
        assert plan.intent_summary
        assert plan.plan_steps
        assert 0.0 <= plan.confidence <= 1.0


def test_prepare_agent_writes_fixed_benchmark_to_test(tmp_path: Path) -> None:
    mdir = _make_agent_model_folder(tmp_path)
    md = load_model_def(mdir)
    summary = prepare_agent(md)

    seed_total = sum(1 for _ in iter_jsonl(SEED_SAMPLES))
    eval_total = sum(1 for _ in iter_jsonl(EVAL_SAMPLES))
    assert sum(summary["split_counts"].values()) == seed_total + eval_total
    assert summary["split_counts"]["test"] >= eval_total

    test_rows = list(iter_jsonl(md.prepared_dir / "test.jsonl"))
    test_ids = {row["sample_id"] for row in test_rows}
    eval_ids = {row["sample_id"] for row in iter_jsonl(EVAL_SAMPLES)}
    assert eval_ids <= test_ids
    for split in ("train", "val", "test"):
        path = md.prepared_dir / f"{split}.jsonl"
        assert path.exists()
        for row in iter_jsonl(path):
            assert row["split"] == split
            AgentPlan.model_validate(row["agent_plan"])


def test_package_agent_emits_agent_planning_task(tmp_path: Path) -> None:
    from flow_ml.packaging.package_model import package_agent

    fake_ckpt = tmp_path / "fake-ckpt"
    fake_ckpt.mkdir()
    (fake_ckpt / "model.safetensors").write_bytes(b"\x00\x00")
    (fake_ckpt / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
    (fake_ckpt / "tokenizer.json").write_text("{}", encoding="utf-8")

    result = package_agent(
        fake_ckpt,
        tmp_path / "dist" / "agent-planner-smoke",
        prompt_spec_path=PROMPT_SPEC,
        model_id="agent-planner:smoke",
        version="smoke",
    )

    manifest = json.loads((result.pkg_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["task"] == "agent_planning"
    assert manifest["model_id"] == "agent-planner:smoke"
    assert manifest["prompt_spec_file"] == "prompt_spec.json"
    spec = json.loads((result.pkg_dir / "prompt_spec.json").read_text(encoding="utf-8"))
    assert spec["response_schema"]["required"][0] == "intent_summary"
    assert result.fm_path is not None and result.fm_path.exists()
