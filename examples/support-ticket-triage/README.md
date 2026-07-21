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
maatml prepare examples/support-ticket-triage/
maatml train   examples/support-ticket-triage/ --smoke
maatml train   examples/support-ticket-triage/
maatml evaluate examples/support-ticket-triage/
```

Add more tickets under `datasets/samples/seed_samples.jsonl`, then re-prepare.
This folder is intentionally small — it demonstrates the `causal_sft` path and
standalone model-folder layout, not production triage quality.
