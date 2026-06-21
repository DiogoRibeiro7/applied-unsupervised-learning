# Unsupervised Learning Lab

A portfolio repository focused on **unsupervised learning**, designed around high-quality Jupyter notebooks with supporting production-style Python code.

The goal is not to show isolated algorithms. The goal is to show that unsupervised learning can be treated as a serious modelling workflow: data generation, representation learning, clustering, anomaly detection, topic discovery, recommender embeddings, model selection, stability analysis, interpretation, and deployment-minded reporting.

## What this repo showcases

- Clustering with KMeans, Gaussian Mixtures, DBSCAN, Agglomerative Clustering, and spectral-style workflows.
- Dimensionality reduction with PCA, Kernel PCA, t-SNE-style discussion, Isomap, and practical embedding diagnostics.
- Anomaly detection with Isolation Forest, Local Outlier Factor, robust covariance, reconstruction-style scores, and threshold calibration.
- Unsupervised NLP with TF-IDF, NMF topic models, latent semantic analysis, and topic interpretation.
- Recommender-style embeddings with matrix factorization and customer/product latent spaces.
- Cluster validation using silhouette score, Davies-Bouldin, Calinski-Harabasz, bootstrap stability, and business-facing diagnostics.
- Notebook-first storytelling with reusable source code.

## Repository layout

```text
unsupervised-learning-lab/
├── notebooks/
│   ├── 00_project_overview.ipynb
│   ├── 01_customer_segmentation_clustering.ipynb
│   ├── 02_dimensionality_reduction_manifold_learning.ipynb
│   ├── 03_anomaly_detection_sensor_events.ipynb
│   ├── 04_unsupervised_nlp_topic_modeling.ipynb
│   ├── 05_recommender_embeddings_matrix_factorization.ipynb
│   ├── 06_model_selection_stability_explainability.ipynb
│   ├── 07_consensus_clustering.ipynb
│   └── 08_streaming_clustering_drift.ipynb
├── src/unsup_lab/
│   ├── data.py            # synthetic data generators
│   ├── evaluation.py      # internal clustering metrics
│   ├── stability.py       # model selection & stability analysis
│   ├── consensus.py       # ensemble / consensus clustering
│   ├── streaming.py       # streaming clustering & drift monitoring
│   ├── recommenders.py    # matrix factorization helpers
│   ├── nlp.py             # topic-model cleaning, labels & diagnostics
│   ├── service.py         # reusable clustering/anomaly/topic pipelines
│   ├── artifacts.py       # model persistence with metadata
│   ├── cli.py             # `unsup-lab` command line interface
│   ├── api.py             # FastAPI scoring service
│   ├── plotting.py
│   ├── preprocessing.py
│   └── reporting.py
├── tests/
├── docs/
├── data/
├── outputs/
├── ROADMAP.md
├── PROMPTS.md
├── AGENTS.md
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
poetry run python -m ipykernel install --user --name unsup-lab
poetry run jupyter lab
```

Run the quality gate (mirrors CI):

```bash
poetry run ruff check .      # lint
poetry run mypy src          # type check
poetry run pytest            # unit tests
```

Execute notebooks as a reproducibility smoke test:

```bash
poetry run python scripts/run_all_notebooks.py                       # all notebooks
poetry run python scripts/run_all_notebooks.py --only 00_project_overview.ipynb
```

Continuous integration runs ruff, mypy, pytest, and a single-notebook smoke
execution on every push and pull request (see `.github/workflows/ci.yml`).

## Production layer

The notebooks stay the presentation layer; the same modelling code is also
exposed through a thin CLI and API so the work can be operationalised.

Command line interface (installed as `unsup-lab`):

```bash
poetry run unsup-lab generate-data --dataset customers --output data/customers.csv
poetry run unsup-lab train-clustering --k 5      # saves model + JSON report under outputs/
poetry run unsup-lab detect-anomalies            # precision@k against hidden labels
poetry run unsup-lab build-topic-model --n-topics 4
poetry run unsup-lab report --model outputs/models/clustering.joblib
```

Each training command saves a joblib artifact plus a JSON metadata sidecar
(`unsup_lab.artifacts`) and writes a JSON report under `outputs/reports/`.

Scoring API (FastAPI), serving the saved artifacts:

```bash
poetry install --with api
poetry run uvicorn unsup_lab.api:app --reload
# GET  /health
# POST /cluster/assign   {"records": [{...features...}]}
# POST /anomaly/score    {"records": [{...features...}]}
```

Container build (trains default models, then serves the API on port 8000):

```bash
docker build -t unsup-lab .
docker run -p 8000:8000 unsup-lab
```

## Portfolio positioning

This project is useful for Senior Data Scientist, Machine Learning Scientist, Applied Scientist, and AI Engineer roles where unsupervised learning appears in customer segmentation, anomaly detection, product analytics, document mining, recommender systems, sensor analytics, fraud discovery, healthcare operations, or exploratory representation learning.

For reviewers and hiring managers:

- [`docs/portfolio_summary.md`](docs/portfolio_summary.md) - one-page overview with figures.
- [`docs/notebook_briefs.md`](docs/notebook_briefs.md) - a brief per notebook.
- [`docs/interview_talking_points.md`](docs/interview_talking_points.md) - discussion prompts.
- [`docs/cv_bullets.md`](docs/cv_bullets.md) and [`docs/linkedin_post.md`](docs/linkedin_post.md).

Key figures live in [`outputs/figures/`](outputs/figures/) (regenerate with `python scripts/export_figures.py`).

## Recommended extensions

See [`ROADMAP.md`](ROADMAP.md) for the full ambitious roadmap and [`PROMPTS.md`](PROMPTS.md) for development prompts.
