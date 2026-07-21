# MaatML

MaatML takes task-specific language models from **experimentation to
production**: prepare → train → evaluate → export → deploy, with a
validator-gated data flywheel.

This docs site is optional (`pip install maatml[docs]` + `mkdocs serve`).
The canonical references in the repo are:

- [README.md](../README.md) — install, CLI overview, end-to-end examples
- [AGENTS.md](../AGENTS.md) — command cheat-sheet for contributors / agents
- [ROADMAP.md](../ROADMAP.md) — tranche status (v0.4 product surface)
- [CHANGELOG.md](../CHANGELOG.md) — release notes

## Quick CLI map (v0.4)

| Command | Purpose |
|---------|---------|
| `maatml export` | Bundle checkpoint (+ optional GGUF/MLX) with `manifest.json` |
| `maatml verify` | Recompute sha256 of manifest files |
| `maatml datagen` | Validator-gated seed generation (`dataset.generator` or `--teacher`) |
| `maatml ingest` | Map / sanitize / validate external JSONL into seed corpus |

See [Plugin author guide](plugins.md) for registering generators, exporters,
validators, and metrics.
