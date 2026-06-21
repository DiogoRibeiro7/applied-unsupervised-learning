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
│   └── 06_model_selection_stability_explainability.ipynb
├── src/unsup_lab/
│   ├── data.py
│   ├── evaluation.py
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

Run tests:

```bash
poetry run pytest
```

Execute notebooks:

```bash
poetry run python scripts/run_all_notebooks.py
```

## Portfolio positioning

This project is useful for Senior Data Scientist, Machine Learning Scientist, Applied Scientist, and AI Engineer roles where unsupervised learning appears in customer segmentation, anomaly detection, product analytics, document mining, recommender systems, sensor analytics, fraud discovery, healthcare operations, or exploratory representation learning.

## Recommended extensions

See [`ROADMAP.md`](ROADMAP.md) for the full ambitious roadmap and [`PROMPTS.md`](PROMPTS.md) for development prompts.
