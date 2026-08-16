# Contributing

Thank you for improving Applied Unsupervised Learning. This repository is notebook-first:
the notebooks tell the modelling story, and `src/unsup_lab` keeps reusable logic
tested and reproducible.

## Development setup

Install dependencies with Poetry:

```bash
poetry install --with api
poetry run python -m ipykernel install --user --name applied-unsupervised-learning
```

Install the optional deep-learning group only when working on notebook 15 or
`unsup_lab.deep`:

```bash
poetry install --with deep
```

## Quality gate

Run the same checks expected by CI before opening a pull request:

```bash
make check
```

For a faster source-only pass:

```bash
make lint
make typecheck
make test
```

Notebook execution can take longer. Use the smoke target during normal
development and the full target before larger notebook changes:

```bash
make notebook-smoke
make notebooks
```

## Contribution rules

- Keep code, comments, docs, variables, functions, and file names in English.
- Add type hints and docstrings to public functions.
- Validate inputs where wrong shapes or types could silently produce bad results.
- Add tests for reusable functions under `src/unsup_lab`.
- Keep notebooks readable; do not hide the modelling narrative inside too many
  helpers.
- Do not present unsupervised clusters as ground truth. Treat them as hypotheses
  that need stability checks, interpretation, and limitations.
- Avoid new dependencies unless they materially improve the project.
- Do not add API-key requirements or examples that depend on private data.

## Pull request checklist

- The change has a clear modelling or maintainability purpose.
- New reusable source code has focused tests.
- Affected notebooks still run, or the PR explains why they were not executed.
- Public functions include type hints and docstrings.
- The README, docs, or notebook text were updated when behaviour changed.

## Release checklist

1. Update `CHANGELOG.md`.
2. Update the version in `pyproject.toml`.
3. Run `make check`.
4. Build the package with `make build`.
5. Create a GitHub release so Zenodo can archive the snapshot.
