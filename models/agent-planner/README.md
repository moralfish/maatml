# Agent Planner

Local Flow Studio workflow-planning agent. The model turns a user request plus optional Flow context into strict JSON containing an intent summary, ordered plan steps, tool calls, Flow DSL output or patches, confidence, and a refusal/fallback reason when needed.

- **Task string (manifest):** `agent_planning`
- **Model id:** `agent-planner:v1`
- **Base model:** `Qwen/Qwen3-4B-Instruct-2507` (LoRA fine-tune via PEFT)
- **Runtime backend:** `CandleGenerativeBackend`
- **Response schema:**
  ```json
  {
    "intent": "string",
    "plan": [{ "step": 1, "action": "string", "tool": "string" }],
    "dsl": "string | null",
    "dsl_patch": "string | null",
    "confidence": 0.0,
    "refusal_reason": "string | null"
  }
  ```

## Model Choice

- **Primary (shipped):** `Qwen/Qwen3-4B-Instruct-2507`
  - Apache-2.0, 4B-class, strong local small-model adoption, and a good quality/latency tradeoff for Apple Silicon with 16 GB+ unified memory.
- **Deployment fallback:** `HuggingFaceTB/SmolLM3-3B`
  - Apache-2.0, fully open training story, smaller bundle, and a sensible option if dense Qwen3 Candle support becomes expensive. Not the smoke model.
- **Smoke/CI model:** `Qwen/Qwen3-0.6B`
  - Used only for fast pipeline validation (`flow_ml train --smoke`). Produces intentionally undertrained weights that are never shipped.
- **Baselines to benchmark, not ship first:** `microsoft/Phi-4-mini-instruct`, `meta-llama/Llama-3.2-3B-Instruct`, `google/gemma-3-1b-it`, `HuggingFaceTB/SmolLM2-360M-Instruct`.

## Dataset

- `datasets/schema.json` — sample shape (`request`, `context`, `agent_plan`).
- `datasets/prompt_spec.json` — system prompt + `<<AGENT_INPUT>>` ChatML template + decoding params.
- `datasets/samples/seed_samples.jsonl` — hand-authored seed samples covering the five request categories listed below.
- `datasets/samples/eval_samples.jsonl` — held-out evaluation samples; always routed to the test split by `prepare_agent` to keep the benchmark fixed across retrains and base-model comparisons.

### Data Strategy

The seed corpus covers:

- natural-language workflow requests to Flow DSL
- multi-step planning with tool calls
- existing-flow edits expressed as `dsl_patch`
- ambiguous requests that should ask for clarification
- unsupported or unsafe requests that should refuse with a clear reason

## End-to-end commands

All commands below take this model folder as their primary argument. Outputs land under `output/` (gitignored).

```bash
# 1. Prepare splits (hash-stable 70/20/10; eval_samples.jsonl always → test)
python -m flow_ml.cli prepare models/agent-planner/

# 2. Smoke training (Qwen3-0.6B, 6 steps)
python -m flow_ml.cli train models/agent-planner/ --smoke

# 3. Full training (Qwen3-4B-Instruct-2507, 5 epochs)
python -m flow_ml.cli train models/agent-planner/

# 4. Evaluate the most recent checkpoint
python -m flow_ml.cli evaluate models/agent-planner/

# 5. Package -> f16 .fm archive
python -m flow_ml.cli package models/agent-planner/ --version v1

# 6. Verify the .fm
python -m flow_ml.cli verify models/agent-planner/output/dist/agent-planner-v1.fm
```

## Evaluation criteria

`evaluate_agent` computes:

- **`json_validity`** — fraction of generations that parse as JSON conforming to the response schema.
- **`schema_validity`** — fraction of JSON-valid generations that also pass Pydantic schema validation.
- **`intent_match`** — semantic match between predicted and gold `intent` strings.
- **`dsl_presence_accuracy`** — whether `dsl` / `dsl_patch` is present or absent as expected by the gold label.
- **`refusal_accuracy`** — exact match on whether the model refused (non-null `refusal_reason`) vs. produced a plan.
- **`action_jaccard`** — Jaccard similarity of the predicted vs. gold action sets across plan steps.
- **latency statistics** — p50 / p95 / mean inference latency measured during eval.

Reports land in `output/eval/<run-name>.{json,md}`.

For v1, the quality gate is structural reliability first. The model should emit valid, schema-conforming JSON before optimizing broader agent reasoning quality.

## Packaging

`package_agent` merges the LoRA adapter back into the base weights, converts to **f16** (halves the on-disk size vs. f32), and writes:

```
output/dist/agent-planner-v1/
  manifest.json          # ModelManifest (task, runtime, weights_dtype: f16, ...)
  model.safetensors      # merged base+LoRA weights at f16
  config.json
  tokenizer.json
  tokenizer_config.json
  prompt_spec.json
output/dist/agent-planner-v1.fm   # deflated zip of the above
```

Key packaging parameters from `model.yml`:

- `max_input_tokens: 2048`
- `expected_latency_ms: 2500`
- `weights_dtype: f16`

## Runtime contract

Flow Studio's `CandleGenerativeBackend` loads the merged f16 safetensors, renders the user message through `prompt_spec.json`'s `<<AGENT_INPUT>>` template, and decodes greedily to the configured stop sequence. The response payload is JSON-extracted and validated against the response schema before being returned to the canvas.
