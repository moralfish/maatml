# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
for the Python package and per-model versions under `models/`.

## [0.1.0] - Unreleased

### Added

- Public contribution surface: Apache-2.0 LICENSE, CONTRIBUTING, CODE_OF_CONDUCT,
  SECURITY, CHANGELOG, CODEOWNERS, issue/PR templates, Dependabot, pre-commit
- Plugin registry (`flow_ml.registry`) with trainers, validators, metrics,
  predictors, formats, and scaffold hooks; entry-point group `flow_ml.plugins`
- Registry-driven CLI: `prepare`, `train`, `evaluate`, `scaffold`, `validate`,
  `plugins`, `plan`
- Standalone model folders: all paths resolve relative to the model dir; no
  repo-relative fallbacks; wheel package-data for sanitization rules and fixtures
- Device profiles (`mps` / `cuda` / `cpu`) and training guards (NaN abort,
  tokenizer↔embedding contract, `run_metadata.json` provenance)
- Shared validation base + generic evaluation harness
- Reference contrib plugins: `flow_ml.contrib.jcl`, `flow_ml.contrib.spool`
- Example model: `examples/support-ticket-triage` (`causal_sft` — ticket → triage JSON)
- Reference models: JCL Validator (ModernBERT multi-head classifier) and Spool
  Interpreter (flan-t5 seq2seq), versioned at `0.1.0`
- Group-aware (`family`) dataset splits and family-stamped seed corpora
- CI: lint (ruff/mypy), Python 3.10–3.12 test matrix, wheel standalone install job
