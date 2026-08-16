# CV Bullet Points

Drawn from this project. Pick the few that match the role; keep the numbers
honest (they describe the repository, not a production deployment).

## Concise (one-liner)

- Built a notebook-first applied unsupervised learning project covering clustering, anomaly
  detection, topic modelling, recommender embeddings and stability analysis,
  backed by a typed, tested Python package and CI.

## Standard (3-4 bullets)

- Designed and implemented end-to-end unsupervised workflows (segmentation,
  anomaly detection, topic modelling, recommender embeddings) using
  scikit-learn, NumPy and pandas, structured as reproducible case-study
  notebooks.
- Built reusable, type-annotated modules with 175+ unit tests and a
  ruff/mypy/pytest CI gate, including bootstrap stability, consensus clustering
  and streaming drift monitoring.
- Emphasised model validation without labels - internal metrics, bootstrap
  stability and adjusted mutual information - to distinguish robust structure
  from initialisation artefacts.
- Added a lightweight production layer (CLI, model artifacts with metadata,
  JSON reports, FastAPI scoring service, Dockerfile) to demonstrate the path
  from research notebook to deployable model.

## Emphasis variants

**For an applied/ML-scientist role:**

- Implemented and compared KMeans, Gaussian Mixtures, Agglomerative, DBSCAN and
  consensus clustering, selecting methods by their cluster-shape assumptions and
  validating results with stability and sensitivity analysis.

**For an ML-engineer role:**

- Packaged unsupervised pipelines behind a CLI and FastAPI service with
  versioned artifacts and metadata, deterministic synthetic data generators, and
  a CI pipeline running lint, type checks, tests and notebook smoke execution.

**For a data-science generalist role:**

- Translated unsupervised model output into business-facing segment profiles,
  named personas and recommended actions, while documenting the limits of what
  label-free methods can support.
