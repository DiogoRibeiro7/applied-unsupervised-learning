# Applied Unsupervised Learning

Reproducible workflows for label-free modelling: seventeen executed notebooks, and the typed, tested package behind them.

The argument running through the project is that unsupervised methods deserve the same discipline as supervised ones — deliberate assumptions, evaluation without labels, and honesty about what a label-free result can support. That last part is why the [findings](findings.md) page reports the results that *did not* flatter the method alongside the ones that did.

## Where to start

<div class="grid cards" markdown>

- :material-chart-scatter-plot: **[What the notebooks found](findings.md)**

    Results with numbers, including the awkward ones — three algorithms returning an identical partition, and an "ensemble" that was one detector in disguise.

- :material-notebook-outline: **[The notebooks](notebooks/00_project_overview.ipynb)**

    The product. Every one is stored with its executed outputs and regenerates from fixed seeds without private data.

- :material-file-document-outline: **[Notebook briefs](notebook_briefs.md)**

    One page per notebook: the question, the approach, the result, and the judgement call that mattered.

- :material-api: **[API reference](reference.md)**

    `unsup_lab`, generated from the docstrings — data generators, evaluation, stability, thresholds, explanations and more.

</div>

## Running it yourself

```bash
poetry install
poetry run jupyter lab
```

The quality gate that guards every change:

```bash
make check    # poetry check, ruff format, ruff, mypy, pytest, notebook smoke run
```

A second CI job exercises the PyTorch deep-clustering suite against the exact version pinned in `poetry.lock`, using the CPU build so it costs 190 MB rather than 2.5 GB of CUDA packages.

## Citing

Every release is archived on Zenodo. Cite the concept DOI, which always resolves to the newest version:

> Ribeiro, D. (2026). *Applied Unsupervised Learning: reproducible workflows for label-free modelling*. Zenodo. [https://doi.org/10.5281/zenodo.21963335](https://doi.org/10.5281/zenodo.21963335)

## Licence

Code under `src/`, `scripts/` and `tests/` is MIT. The notebooks and prose are CC BY 4.0. The built distribution contains only `src/unsup_lab`, so anything installed from a wheel is MIT in its entirety.
