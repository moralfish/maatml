# Going public checklist

Manual maintainer steps before flipping this repository to public. Do **not**
automate a git history rewrite from CI — cut fresh history (if needed) is a
local, intentional decision.

## Pre-visibility

- [ ] Decide whether to cut a **fresh history** (orphan branch / new repo) vs.
      publish existing history. If cutting, do it locally before the visibility
      flip; never force-rewrite a shared public default branch after the fact.
- [ ] Run a full **secret scan** on the tree and history (`gitleaks`,
      `trufflehog`, or equivalent). Redact or rotate anything found before
      publish.
- [ ] Confirm no private vendor paths, internal hostnames, or sibling-repo
      defaults remain in docs, scripts, or comments.

## GitHub settings

- [ ] Enable **branch protection** on `main` with required CI checks
      (`lint`, `test`, `standalone` from `.github/workflows/ci.yml`).
- [ ] Enable **Dependabot** alerts (and review weekly Dependabot PRs from
      `.github/dependabot.yml`).
- [ ] Enable **private vulnerability reporting** (Security → Reporting).
- [ ] Confirm [CODEOWNERS](../.github/CODEOWNERS), issue/PR templates, and
      [SECURITY.md](../SECURITY.md) look correct for a public audience.

## Packaging

- [ ] Verify the `flow-ml` name is available (or owned) on **PyPI**.
- [ ] Smoke-test a clean install: `pip install dist/*.whl && flow_ml --help && flow_ml plugins`.

## Flip

- [ ] Flip repository visibility to **public**.
- [ ] Announce / tag `v0.1.0` (or next agreed release) when ready.
