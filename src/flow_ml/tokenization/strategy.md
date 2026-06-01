# Tokenization Strategy

## Decisions made

The generative and seq2seq trainers use the **base model's tokenizer as-is** via `AutoTokenizer.from_pretrained(model_id)`. No custom vocabulary is added; no extra tokens are injected. This was evaluated against the alternative of adding task-specific special tokens and rejected because:

- JCL and spool text fragments are handled well by the BERT / Qwen2 / SmolLM2 sub-word tokenizers with no measurable fragmentation penalty at the sequence lengths in use.
- Adding tokens to a frozen (or LoRA-frozen) base would require embedding resize + random initialisation of the new rows, which introduces training instability without a clear quality benefit at these corpus sizes.

## Pad token

Models that do not define a `pad_token` in their tokenizer config (Qwen2, SmolLM2) have their pad token set to the eos token before collation:

```python
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
```

This is done in the trainer modules (`jcl_classifier.py`, `spool_seq2seq.py`, `flow_graph_generator.py`) and is safe because attention masks correctly exclude the padded positions from loss computation.

## Generative models: prompt masking

For causal-LM trainers the loss is masked over the prompt tokens (labels set to `-100`); only the target / completion tokens contribute to the loss. Each trainer's data collator applies `apply_chat_template` to build the prompt token sequence, then appends the target token sequence with labels unmasked.

## Sanitization ordering

`sanitizer.py` runs at the **raw-text level** before any tokenization, during `flow_ml prepare`. By the time text reaches the tokenizer at training or inference time, PII and secrets have already been redacted to placeholder strings. The tokenizer therefore sees sanitized text only.

## Tokenizer assets in the checkpoint

Training writes the tokenizer files verbatim into the checkpoint directory:

```
tokenizer.json
tokenizer_config.json
special_tokens_map.json    (when present)
tokenizer.model            (SentencePiece; when present)
```

Versioning the tokenizer with the weights ensures inference always uses the exact tokenizer the model was trained with.
