# MaatML

MaatML takes task-specific language models from **experimentation to
production**: prepare → train → evaluate → export → deploy, with a
validator-gated data flywheel.

**Install from PyPI** (not from source):

```bash
pip install maatml
pip install "maatml[ml]"       # training stack
pip install "maatml[docs]"     # this site: mkdocs serve
```

- Site: [maatml.org](https://maatml.org) · [maatml.com](https://maatml.com)
- PyPI: [pypi.org/project/maatml](https://pypi.org/project/maatml/)
- Source: [github.com/moralfish/maatml](https://github.com/moralfish/maatml)

Canonical references in the repo:

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
