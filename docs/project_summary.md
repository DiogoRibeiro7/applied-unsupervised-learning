# Project Summary

A notebook-first project for **applied unsupervised learning**. The notebooks
tell the modelling story; a typed, tested Python package (`src/unsup_lab`) keeps
that story reproducible, and a thin CLI/API layer shows the same models can be
operationalised.

The emphasis throughout is on **modelling judgement** rather than algorithm
trivia: framing a problem without labels, choosing methods deliberately,
validating structure with internal metrics and stability analysis, and being
explicit about what unsupervised results can and cannot support.

## What it covers

| Area | Methods | Reusable module |
| --- | --- | --- |
| Customer segmentation | KMeans, GMM, Agglomerative, DBSCAN | `data`, `evaluation`, `reporting` |
| Dimensionality reduction | PCA, Kernel PCA, Isomap, LLE | `preprocessing`, `plotting` |
| Anomaly detection | Isolation Forest, LOF, robust covariance, PCA error | `service`, `data` |
| Topic modelling | TF-IDF, NMF, LSA, coherence & diversity | `nlp` |
| Recommender embeddings | TruncatedSVD / NMF matrix factorisation | `recommenders` |
| Model selection & stability | bootstrap stability, AMI, scaling/outlier sensitivity | `stability` |
| Consensus clustering | co-association ensemble | `consensus` |
| Streaming & drift | mini-batch KMeans, PSI drift monitoring | `streaming` |
| Nonparametric mixtures | Dirichlet-process Gaussian mixtures, soft-assignment entropy | `bayesian` |
| Time series | DTW clustering, matrix profile, unsupervised shapelets | `timeseries` |
| Graphs | community detection, spectral node embeddings, node anomalies | `graphs` |
| Deep clustering | autoencoder, DEC, contrastive (optional PyTorch extra) | `deep` |

## Engineering practices

- Typed, documented, input-validated functions with **175+ unit tests**.
- Quality gate of **ruff + mypy + pytest** plus a notebook smoke-execution,
  wired into GitHub Actions CI.
- Deterministic synthetic data generators (fixed seeds) so every notebook is
  reproducible without private data.
- A production layer: `applied-unsupervised-learning` CLI, model artifacts with metadata, JSON
  reports, a FastAPI scoring service, and a Dockerfile.

## What it deliberately avoids

- Treating unsupervised clusters or anomaly scores as ground truth.
- Heavy dependencies in the core install: everything except notebook 15 runs on
  NumPy, pandas, scipy, scikit-learn and matplotlib. PyTorch is an opt-in extra
  (`poetry install --with deep`) used solely by the deep-clustering notebook.
- Dependence on external APIs or private datasets.

## Selected figures

Regenerate with `python scripts/export_figures.py`.

![Customer segments](https://raw.githubusercontent.com/DiogoRibeiro7/applied-unsupervised-learning/main/outputs/figures/customer_segments_pca.png)

*Customer segments (KMeans) in a PCA projection.*

![Consensus co-association](https://raw.githubusercontent.com/DiogoRibeiro7/applied-unsupervised-learning/main/outputs/figures/consensus_coassociation.png)

*Co-association matrix from consensus clustering; block structure is the
evidence that the segments survive resampling.*

![Streaming drift](https://raw.githubusercontent.com/DiogoRibeiro7/applied-unsupervised-learning/main/outputs/figures/streaming_drift.png)

*Streaming drift monitoring: PSI and centroid shift spike when an injected
regime change arrives.*

## Further reading

- [`notebook_briefs.md`](notebook_briefs.md) - one-page brief per notebook.
- [`workflows.md`](workflows.md) - diagrams of each modelling workflow.
- [`decision_notes.md`](decision_notes.md) - modelling tradeoffs and why.
- [`interview_talking_points.md`](interview_talking_points.md) - discussion prompts.
- [`cv_bullets.md`](cv_bullets.md) - CV bullet points.
- [`notebooks/99_lessons_learned.ipynb`](https://github.com/DiogoRibeiro7/applied-unsupervised-learning/blob/main/notebooks/99_lessons_learned.ipynb) - capstone synthesis.
- [`ROADMAP.md`](https://github.com/DiogoRibeiro7/applied-unsupervised-learning/blob/main/ROADMAP.md) - the wider roadmap and progress.
