# Contributing to flow-ml

Thanks for contributing. This guide is for human contributors. AI coding agents
should also read [AGENTS.md](AGENTS.md).

## Code of conduct

Please read and follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Do not open public issues for security reports. See [SECURITY.md](SECURITY.md).

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate

# CPU-free contributions (tests run without torch)
pip install -e ".[dev]"

# Training / evaluation work
pip install -e ".[dev,ml]"
```

## Lint and tests

```bash
ruff check src tests scripts
pytest tests/
```

## Pull requests

- Keep PRs focused — one concern per PR when practical.
- Add or update tests for behavior changes.
- Sign off every commit with the [Developer Certificate of Origin](https://developercertificate.org/)
  (`git commit -s` adds a `Signed-off-by:` trailer).

## Versioning

### Python package

The package stays **0.x** while the plugin API is still churning. Breaking
changes to public surfaces are expected before 1.0.

### Per-model versions (`models/<name>/model.yml`)

| Bump | When |
| ---- | ---- |
| **major** | Breaking change to the model's output schema or public contract |
| **minor** | Retrain, data change, or training/config change that affects quality |
| **patch** | Metadata-only (docs, packaging notes, non-behavioral edits) |

## Models

Each task model lives under `models/` as a standalone folder with its own
`model.yml`. Any folder with a valid `model.yml` can be prepared, trained, and
evaluated via the `flow_ml` CLI. Scaffolding new models with `flow_ml scaffold`
is on the roadmap (or available when that command lands) — open a plugin request
issue if you want to propose a new task.

## AI agent guidance

Operational architecture notes for coding agents live in [AGENTS.md](AGENTS.md).
