# Applied Unsupervised Learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21963335.svg)](https://doi.org/10.5281/zenodo.21963335)
[![CI](https://github.com/DiogoRibeiro7/applied-unsupervised-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/applied-unsupervised-learning/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](pyproject.toml)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-46a2f1.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/code%20license-MIT-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/docs%20license-CC%20BY%204.0-lightgrey.svg)](LICENSE-CC-BY-4.0.txt)

A notebook-first applied machine learning project focused on **unsupervised learning**, designed around high-quality Jupyter notebooks with supporting production-style Python code.

The goal is not to show isolated algorithms. The goal is to show that unsupervised learning can be treated as a serious modelling workflow: data generation, representation learning, clustering, anomaly detection, topic discovery, recommender embeddings, model selection, stability analysis, interpretation, and deployment-minded reporting.

## What this repo showcases

- Clustering with KMeans, Gaussian Mixtures, DBSCAN, Agglomerative Clustering, consensus ensembles, and spectral community detection on graphs.
- Dimensionality reduction with PCA, Kernel PCA, Isomap, Locally Linear Embedding, explained-variance diagnostics, and a note on when 2D separation misleads.
- Anomaly detection with Isolation Forest, Local Outlier Factor, robust covariance, One-Class SVM, PCA reconstruction error, and precision-at-k ranking evaluation.
- Unsupervised NLP with TF-IDF, NMF topic models, latent semantic analysis, coherence/diversity diagnostics, and four reproduced failure modes.
- Recommender-style embeddings with matrix factorization and customer/product latent spaces.
- Cluster validation using silhouette score, Davies-Bouldin, Calinski-Harabasz, bootstrap and seed stability, scaling/outlier sensitivity, and business-facing diagnostics.
- Beyond the core: Dirichlet-process mixtures, DTW clustering, matrix profile, unsupervised shapelets, graph node embeddings, streaming drift monitoring, and deep clustering (optional PyTorch extra).
- Notebook-first storytelling with reusable source code.

## Repository layout

```text
applied-unsupervised-learning/
├── notebooks/
│   ├── 00_project_overview.ipynb
│   ├── 01_customer_segmentation_clustering.ipynb
│   ├── 02_dimensionality_reduction_manifold_learning.ipynb
│   ├── 03_anomaly_detection_sensor_events.ipynb
│   ├── 04_unsupervised_nlp_topic_modeling.ipynb
│   ├── 05_recommender_embeddings_matrix_factorization.ipynb
│   ├── 06_model_selection_stability_explainability.ipynb
│   ├── 07_consensus_clustering.ipynb
│   ├── 08_streaming_clustering_drift.ipynb
│   ├── 09_bayesian_nonparametric_mixtures.ipynb
│   ├── 10_time_series_dtw_clustering.ipynb
│   ├── 11_graph_community_detection.ipynb
│   ├── 12_matrix_profile_motifs_discords.ipynb
│   ├── 13_graph_node_embeddings_anomaly.ipynb
│   ├── 14_unsupervised_shapelets.ipynb
│   ├── 15_deep_clustering_autoencoder_dec.ipynb
│   └── 99_lessons_learned.ipynb
├── src/unsup_lab/
│   ├── data.py            # synthetic data generators
│   ├── evaluation.py      # internal clustering metrics
│   ├── stability.py       # model selection & stability analysis
│   ├── consensus.py       # ensemble / consensus clustering
│   ├── streaming.py       # streaming clustering & drift monitoring
│   ├── bayesian.py        # Dirichlet-process nonparametric mixtures
│   ├── timeseries.py      # DTW clustering, matrix profile, u-shapelets
│   ├── graphs.py          # graph communities, node embeddings, node anomalies
│   ├── config.py          # typed YAML run configuration
│   ├── tracking.py        # dependency-free JSONL experiment tracker
│   ├── deep.py            # autoencoder, DEC & contrastive clustering (optional torch)
│   ├── recommenders.py    # matrix factorization helpers
│   ├── nlp.py             # topic-model cleaning, labels & diagnostics
│   ├── service.py         # reusable clustering/anomaly/topic pipelines
│   ├── artifacts.py       # model persistence with metadata
│   ├── cli.py             # `applied-unsupervised-learning` command line interface
│   ├── api.py             # FastAPI scoring service
│   ├── plotting.py
│   ├── preprocessing.py
│   └── reporting.py
├── tests/
├── docs/
├── data/
├── outputs/
├── CONTRIBUTING.md
├── SECURITY.md
├── CHANGELOG.md
├── ROADMAP.md
└── pyproject.toml
```

## Suggested notebook narrative

1. **Start with the modelling problem**, not the algorithm.
2. Build a synthetic-but-realistic dataset that has known latent structure.
3. Apply several methods and explain their assumptions.
4. Evaluate without labels using internal metrics and stability tests.
5. Translate the discovered structure into decisions.
6. Document limitations and failure modes.

## Getting started

This repo uses Poetry.

```bash
poetry install
poetry run python -m ipykernel install --user --name applied-unsupervised-learning
poetry run jupyter lab
```

Run the quality gate (mirrors CI):

```bash
make check
```

Individual checks are also available:

```bash
make lint
make typecheck
make test
make notebook-smoke
```

Execute notebooks as a reproducibility smoke test:

```bash
poetry run python scripts/run_all_notebooks.py                       # all notebooks
poetry run python scripts/run_all_notebooks.py --only 00_project_overview.ipynb
```

Continuous integration runs ruff, mypy, pytest, and a single-notebook smoke
execution on every push and pull request (see `.github/workflows/ci.yml`). A
second job covers `unsup_lab.deep`, installing the CPU build of the PyTorch
version the lock pins - 190 MB rather than the ~2.5 GB of CUDA packages the
default wheel brings - so the deep-clustering tests run for real instead of
silently skipping.

## Project standards

The repository includes the maintenance files expected for a professional open
source or applied machine learning project:

- [`CONTRIBUTING.md`](CONTRIBUTING.md) - setup, quality gate, contribution rules,
  and release checklist.
- [`CHANGELOG.md`](CHANGELOG.md) - release history and unreleased changes.
- [`SECURITY.md`](SECURITY.md) - private vulnerability reporting guidance.
- [`SUPPORT.md`](SUPPORT.md) - issue expectations for reproducible modelling
  questions.
- [`.github/ISSUE_TEMPLATE`](.github/ISSUE_TEMPLATE) and
  [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md) -
  structured GitHub collaboration.
- [`.github/dependabot.yml`](.github/dependabot.yml) - weekly dependency and
  GitHub Actions update checks.

## Production layer

The notebooks stay the presentation layer; the same modelling code is also
exposed through a thin CLI and API so the work can be operationalised.

Command line interface (installed as `applied-unsupervised-learning`):

```bash
poetry run applied-unsupervised-learning generate-data --dataset customers --output data/customers.csv
poetry run applied-unsupervised-learning train-clustering --k 5      # saves model + JSON report under outputs/
poetry run applied-unsupervised-learning detect-anomalies            # precision@k against hidden labels
poetry run applied-unsupervised-learning build-topic-model --n-topics 4
poetry run applied-unsupervised-learning report --model outputs/models/clustering.joblib
poetry run applied-unsupervised-learning batch-score \
    --model outputs/models/clustering.joblib \
    --input data/customers.csv --output outputs/scored.csv
```

Each training command saves a joblib artifact plus a JSON metadata sidecar
(`unsup_lab.artifacts`) and writes a JSON report under `outputs/reports/`.

**Configuration, tracking and scheduling.**

- `unsup_lab.config` loads typed, validated run settings from YAML (see
  [`configs/example_clustering.yaml`](configs/example_clustering.yaml)).
- `unsup_lab.tracking` is a dependency-free experiment log: training commands
  append a run to a JSONL file when given `--track-path`, and `load_runs` /
  `best_run` read it back.
- [`scripts/scheduled_report.py`](scripts/scheduled_report.py) retrains from a
  config and writes timestamped reports; point cron or Task Scheduler at it.

```bash
poetry run applied-unsupervised-learning train-clustering --k 5 --track-path outputs/experiments/runs.jsonl
poetry run python scripts/scheduled_report.py --config configs/example_clustering.yaml
```

Scoring API (FastAPI), serving the saved artifacts:

```bash
poetry install --with api
poetry run uvicorn unsup_lab.api:app --reload
# GET  /health
# POST /cluster/assign   {"records": [{...features...}]}
# POST /anomaly/score    {"records": [{...features...}]}
```

Deep clustering (optional PyTorch extra, used only by notebook 15 and
`unsup_lab.deep`):

```bash
poetry install --with deep
```

Container build (trains default models, then serves the API on port 8000):

```bash
docker build -t applied-unsupervised-learning .
docker run -p 8000:8000 applied-unsupervised-learning
```

## Professional positioning

This project is useful for Senior Data Scientist, Machine Learning Scientist, Applied Scientist, and AI Engineer roles where unsupervised learning appears in customer segmentation, anomaly detection, product analytics, document mining, recommender systems, sensor analytics, fraud discovery, healthcare operations, or exploratory representation learning.

For reviewers and hiring managers:

- [`docs/project_summary.md`](docs/project_summary.md) - one-page project overview with figures.
- [`docs/notebook_briefs.md`](docs/notebook_briefs.md) - a brief per notebook.
- [`docs/interview_talking_points.md`](docs/interview_talking_points.md) - discussion prompts.
- [`docs/cv_bullets.md`](docs/cv_bullets.md) and [`docs/linkedin_post.md`](docs/linkedin_post.md).
- [`docs/workflows.md`](docs/workflows.md) - workflow diagrams; [`docs/decision_notes.md`](docs/decision_notes.md) - modelling tradeoffs.

Key figures live in [`outputs/figures/`](outputs/figures/) (regenerate with `python scripts/export_figures.py`).

## Recommended extensions

See [`ROADMAP.md`](ROADMAP.md) for the full ambitious roadmap and [`PROMPTS.md`](PROMPTS.md) for development prompts.

## License

Two licenses, split by what the content is:

| Content | License | File |
| --- | --- | --- |
| Code - `src/`, `scripts/`, `tests/`, configuration | MIT | [`LICENSE`](LICENSE) |
| Notebooks and prose - `notebooks/`, `docs/`, `README.md`, `ROADMAP.md` | CC BY 4.0 | [`LICENSE-CC-BY-4.0.txt`](LICENSE-CC-BY-4.0.txt) |

The reasoning: MIT is what makes the reusable machinery genuinely reusable - drop
`unsup_lab` into your own project and go. CC BY 4.0 fits the notebooks and
written analysis, which are closer to an article than to a library, and asks
only for attribution when they are reproduced or adapted.

The packaged distribution (`poetry build`) contains only `src/unsup_lab`, so
anything installed from a wheel is MIT in its entirety.

## Citation

Every release is archived on Zenodo. Cite the **concept DOI**, which always
resolves to the most recent version:

> Ribeiro, D. (2026). *Applied Unsupervised Learning: reproducible workflows
> for label-free modelling*. Zenodo. https://doi.org/10.5281/zenodo.21963335

```bibtex
@software{ribeiro_applied_unsupervised_learning,
  author    = {Ribeiro, Diogo},
  title     = {Applied Unsupervised Learning: reproducible workflows
               for label-free modelling},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.21963335},
  url       = {https://doi.org/10.5281/zenodo.21963335}
}
```

To cite one exact snapshot instead, use that release's own version DOI - for
`v0.1.0` it is [10.5281/zenodo.21963336](https://doi.org/10.5281/zenodo.21963336).
[`CITATION.cff`](CITATION.cff) carries the same metadata and drives GitHub's
"Cite this repository" button; [`.zenodo.json`](.zenodo.json) describes each
deposit as it is archived.
