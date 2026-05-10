"""Pipeline tests for FlowGraphGenerator: prepare_flow_graph + schema +
prompt_spec invariants."""
from __future__ import annotations

import json
from pathlib import Path

from flow_ml.config import load_model_def
from flow_ml.data.pipeline import prepare_flow_graph
from flow_ml.data.schemas import FlowGraphSample, FlowGraphProposal
from flow_ml.utils.io import iter_jsonl

REPO = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO / "models" / "flow-graph-generator"
DATASET_DIR = MODEL_DIR / "datasets"
PROMPT_SPEC = DATASET_DIR / "prompt_spec.json"
SCHEMA = DATASET_DIR / "flow_graph_schema.json"
CONTRACTS = DATASET_DIR / "node_contracts.json"
SEED_SAMPLES = DATASET_DIR / "samples" / "seed_samples.jsonl"


def test_prompt_spec_has_required_fields() -> None:
    spec = json.loads(PROMPT_SPEC.read_text(encoding="utf-8"))
    assert spec["schema_version"] == "1"
    assert spec["base_model"] == "Qwen/Qwen3-1.7B"
    assert "<<USER_REQUEST>>" in spec["user_template"]
    assert "FlowGraphGenerator" in spec["system"]
    assert spec["stop"] == ["<|im_end|>"]
    assert spec["max_new_tokens"] >= 256
    assert spec["temperature"] == 0.0
    assert spec["json_keys_order"] == ["id", "name", "version", "nodes", "edges", "warnings"]


def test_node_contracts_enumerates_real_handlers() -> None:
    contracts = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    assert set(contracts["node_kinds"]) == {"action", "ai", "cloud_ai", "utility"}
    triples = {(t["adapter"], t["actionId"]) for t in contracts["action_triples"]}
    assert ("shell", "run-command") in triples
    assert ("shell", "pnpm") in triples
    assert ("zowe", "cli-raw") in triples
    assert ("mri-toolkit", "prepare") in triples
    assert set(contracts["forbidden_adapters"]) == {"ssh", "zosmf", "mock"}
    assert {c["name"] for c in contracts["refusal_categories"]} == {
        "credential.read_secret",
        "shell.exec_unrestricted",
        "external.http_post",
        "network.upload_file",
    }


def test_seed_samples_consistent_shape() -> None:
    rows = list(iter_jsonl(SEED_SAMPLES))
    assert rows, "seed_samples.jsonl must not be empty"
    seen_ids: set[str] = set()
    for row in rows:
        for key in ("sample_id", "category", "source", "request", "expected_graph"):
            assert key in row, f"missing {key} in {row.get('sample_id')}"
        # round-trip through Pydantic
        FlowGraphProposal.model_validate(row["expected_graph"])
        assert row["sample_id"] not in seen_ids, f"duplicate id {row['sample_id']}"
        seen_ids.add(row["sample_id"])


def test_seed_samples_cover_all_thirteen_categories() -> None:
    """Every §7 category must have at least one seed; coverage gap = bug."""
    expected = {
        "simple", "conditional", "parallel", "jcl-validation", "job-submission",
        "spool-inspection", "db2", "notification", "report-generation",
        "ambiguous", "unsafe", "unsupported", "repair",
    }
    rows = list(iter_jsonl(SEED_SAMPLES))
    seen = {row["category"] for row in rows}
    missing = expected - seen
    assert not missing, f"seed corpus missing categories: {missing}"


def test_prepare_flow_graph_writes_splits(tmp_path: Path) -> None:
    md = load_model_def(MODEL_DIR)
    summary = prepare_flow_graph(md, out_dir=tmp_path)
    total = sum(summary["split_counts"].values())
    seed_total = sum(1 for _ in iter_jsonl(SEED_SAMPLES))
    assert total == seed_total
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.jsonl"
        assert path.exists()
        rows = list(iter_jsonl(path))
        assert len(rows) == summary["split_counts"][split]
        for row in rows:
            assert row["request"]
            assert row["expected_graph"]
            assert row["category"]
            assert row["split"] == split
    # All 13 categories appear in the prepared corpus
    assert len(summary["category_counts"]) == 13


def test_flow_graph_proposal_roundtrip() -> None:
    """Pydantic model round-trips a refusal sample (empty nodes + warnings)."""
    raw = {
        "id": "refused",
        "name": "Refused",
        "version": "0.1.0",
        "nodes": [],
        "edges": [],
        "warnings": ["Refused: credential.read_secret is forbidden."],
    }
    proposal = FlowGraphProposal.model_validate(raw)
    dumped = proposal.model_dump(mode="json")
    assert dumped["nodes"] == []
    assert dumped["edges"] == []
    assert dumped["warnings"] == ["Refused: credential.read_secret is forbidden."]


def test_flow_graph_sample_validates_against_seed_row() -> None:
    """A FlowGraphSample built from a seed row + split must validate cleanly."""
    rows = list(iter_jsonl(SEED_SAMPLES))
    row = rows[0]
    sample = FlowGraphSample.model_validate({**row, "split": "train"})
    assert sample.expected_graph.id
    assert sample.category == row["category"]
