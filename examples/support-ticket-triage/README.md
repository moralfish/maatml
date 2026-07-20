# support-ticket-triage

Causal LoRA SFT example: given a customer support ticket, emit triage JSON.

```json
{
  "priority": "p1|p2|p3|p4",
  "category": "billing|access|bug|how_to|other",
  "team": "payments|identity|platform|docs|general",
  "summary": "one-line agent summary"
}
```

## Lifecycle

```bash
flow_ml prepare examples/support-ticket-triage/
flow_ml train   examples/support-ticket-triage/ --smoke
flow_ml train   examples/support-ticket-triage/
flow_ml evaluate examples/support-ticket-triage/
```

Add more tickets under `datasets/samples/seed_samples.jsonl`, then re-prepare.
This folder is intentionally small — it demonstrates the `causal_sft` path and
standalone model-folder layout, not production triage quality.
