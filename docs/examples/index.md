# Examples

Six reference models share the identical folder layout and CLI, from a
one-command support-ticket triage to a vLLM-servable vision-language model.
Each is a standalone folder under `examples/` with its own `model.yml`: install `maatml` from PyPI, clone the repo for the seed data and plugins, and
point the CLI at the folder.

| Model | Task | Architecture | Base |
|-------|------|--------------|------|
| [Support Ticket Triage](support-ticket-triage.md) | triage → JSON | `causal_sft` (LoRA) | Qwen3-0.6B |
| [Vision VLM](vision-vlm.md) | describe a scene image | `vlm_sft` (vLLM-servable) | SmolVLM-256M-Instruct |
| [Vision](vision.md) | scene + detect + pose | `vision_multitask` | MobileNetV3-Large |
| [Vision Describer](vision-describer.md) | caption from vision JSON | `seq2seq` | flan-t5-small |
| [JCL Validator](jcl-validator.md) | `jcl_validation` | `classifier` (4-head) | ModernBERT-base |
| [Spool Interpreter](spool-interpreter.md) | `spool_interpretation` | `seq2seq` | flan-t5-base |

New to MaatML? [Support Ticket Triage](support-ticket-triage.md) is the
shortest path end to end, see the [5-minute quickstart](../getting-started.md).
For the multimodal path, start with [Vision](vision.md) (trains a checkpoint)
and [Vision Describer](vision-describer.md) (reads its output and captions it).
[Vision VLM](vision-vlm.md) folds both into one vision-language model, servable
directly by vLLM.

Any directory with a valid `model.yml` works the same way. Scaffold a new one
with:

```bash
maatml scaffold ~/models/my-task --architecture causal_sft --name my-task
```

See the [plugin author guide](../plugins.md) for wiring up a validator,
metrics, and a predictor for your own task.
