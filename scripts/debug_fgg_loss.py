"""Probe each FGG training sample to find the one producing absurd loss.

Loads Qwen3-1.7B (no LoRA, no training), tokenizes each train.jsonl row
through `build_chat_example`, runs a single forward pass, prints per-sample
loss. Anything > 50 is a corrupt label-mask candidate worth inspecting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from flow_ml.training.sft_base import build_chat_example  # noqa: E402
from flow_ml.utils.io import iter_jsonl, read_json  # noqa: E402


MODEL_DIR = REPO / "models" / "flow-graph-generator"
TRAIN = MODEL_DIR / "output" / "prepared" / "train.jsonl"
SPEC = read_json(MODEL_DIR / "datasets" / "prompt_spec.json")
TARGET_FIELD = "expected_graph"


def main() -> int:
    rows = list(iter_jsonl(TRAIN))
    print(f"loaded {len(rows)} train rows")

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("loading Qwen3-1.7B (fp32, cpu) ...")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
    model.config.pad_token_id = tok.pad_token_id
    model.eval()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)

    bad = []
    seq_lens = []
    label_counts = []
    for i, row in enumerate(rows):
        ex = build_chat_example(
            row, SPEC, tok, max_length=4096,
            target_field=TARGET_FIELD, request_field="request",
            user_placeholder="<<USER_REQUEST>>",
        )
        ids = torch.tensor([ex["input_ids"]], dtype=torch.long, device=device)
        labels = torch.tensor([ex["labels"]], dtype=torch.long, device=device)
        n_unmasked = int((labels != -100).sum().item())
        seq_lens.append(ids.shape[1])
        label_counts.append(n_unmasked)
        with torch.inference_mode():
            try:
                out = model(input_ids=ids, labels=labels)
                loss = float(out.loss.item())
            except Exception as exc:
                print(f"[error] sample {i} ({row.get('sample_id')}): {exc}")
                continue
        finite = bool(torch.isfinite(torch.tensor(loss)))
        if not finite or loss > 50 or n_unmasked == 0:
            bad.append((i, row.get("sample_id"), row.get("category"), loss, n_unmasked, ids.shape[1]))
            print(f"[BAD] {i} {row.get('sample_id')} ({row.get('category')}): loss={loss:.2f} unmasked={n_unmasked} seq={ids.shape[1]}")
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(rows)} processed (bad so far: {len(bad)})")

    print()
    print(f"total bad samples: {len(bad)} / {len(rows)}")
    print(f"seq_len p50={sorted(seq_lens)[len(seq_lens)//2]} p95={sorted(seq_lens)[int(len(seq_lens)*0.95)]} max={max(seq_lens)}")
    print(f"unmasked label tokens: min={min(label_counts)} median={sorted(label_counts)[len(label_counts)//2]} max={max(label_counts)}")
    if bad:
        print("\nfirst 5 bad rows:")
        for entry in bad[:5]:
            print(f"  idx={entry[0]} id={entry[1]} cat={entry[2]} loss={entry[3]:.2f} unmasked={entry[4]} seq={entry[5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
