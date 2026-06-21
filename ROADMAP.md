# Ambitious Roadmap

This roadmap is designed to turn the repository from a notebook portfolio into a mature applied unsupervised learning lab.

## Phase 0 — Foundation

**Goal:** Make the repository clean, reproducible, and credible.

- [ ] Add deterministic synthetic data generators with configurable random seeds.
- [ ] Add typed reusable functions for preprocessing, evaluation, plotting, and reporting.
- [ ] Add notebook execution checks using `nbclient` or `papermill`.
- [ ] Add unit tests for all non-notebook code.
- [ ] Add CI with linting, type checks, tests, and notebook smoke tests.
- [ ] Add a clear README with problem framing, business use cases, and modelling decisions.

**Success criteria**

- A recruiter or hiring manager can understand the repo in under 3 minutes.
- A technical reviewer can run the notebooks without manual fixes.
- Every notebook has a clear modelling question, conclusion, and limitations section.

---

## Phase 1 — Core Unsupervised Learning Workflows

**Goal:** Demonstrate strong practical knowledge of unsupervised learning methods.

### Clustering

- [ ] Implement comparative clustering workflows:
  - KMeans
  - Gaussian Mixture Models
  - Agglomerative Clustering
  - DBSCAN
  - HDBSCAN, optional dependency
  - Spectral Clustering
- [ ] Add cluster profiling tables.
- [ ] Add automatic cluster naming based on feature summaries.
- [ ] Compare compact, density-based, hierarchical, and probabilistic cluster assumptions.
- [ ] Add failure-case examples where each method performs poorly.

### Dimensionality reduction

- [ ] Add PCA from both scikit-learn and a small NumPy implementation.
- [ ] Add reconstruction error analysis.
- [ ] Add explained variance diagnostics.
- [ ] Add manifold learning comparisons:
  - Isomap
  - Locally Linear Embedding
  - Kernel PCA
  - UMAP as optional extension
- [ ] Add a section on when 2D visual separation is misleading.

### Anomaly detection

- [ ] Implement anomaly detection notebooks for:
  - Isolation Forest
  - Local Outlier Factor
  - Robust covariance
  - One-Class SVM
  - PCA reconstruction error
  - Autoencoder, optional deep learning extension
- [ ] Add threshold selection strategies.
- [ ] Add analyst review tables.
- [ ] Add drift-aware anomaly monitoring.
- [ ] Add precision-at-k evaluation when synthetic labels are known but hidden during modelling.

---

## Phase 2 — Realistic Case Studies

**Goal:** Move from algorithm demos to credible applied projects.

### Case Study 1 — Customer Segmentation

- [ ] Simulate customer behaviour with latent personas.
- [ ] Add RFM features, product affinity, price sensitivity, churn-risk proxies, and campaign response features.
- [ ] Build segmentation pipelines.
- [ ] Produce persona cards.
- [ ] Add recommended business actions per segment.

### Case Study 2 — Sensor Anomaly Discovery

- [ ] Simulate IoT sensor streams with missingness, drift, unusual usage, and device faults.
- [ ] Compare point anomalies, contextual anomalies, and collective anomalies.
- [ ] Add time-aware aggregation.
- [ ] Add analyst-facing anomaly explanations.

### Case Study 3 — Document Topic Discovery

- [ ] Add unsupervised topic modelling with TF-IDF + NMF.
- [ ] Add Latent Semantic Analysis.
- [ ] Add topic coherence approximation.
- [ ] Add automatic topic labels using top terms.
- [ ] Add optional BERTopic-style extension with embeddings.

### Case Study 4 — Recommender Embeddings

- [ ] Build a synthetic user-item interaction matrix.
- [ ] Learn latent factors with TruncatedSVD or NMF.
- [ ] Visualise user and item embeddings.
- [ ] Identify product groups and user taste communities.
- [ ] Add cold-start discussion.

---

## Phase 3 — Model Selection, Robustness, and Stability

**Goal:** Show senior-level judgement.

- [ ] Add bootstrap cluster stability.
- [ ] Add adjusted mutual information between repeated clusterings.
- [ ] Add sensitivity analysis for:
  - random seeds
  - number of clusters
  - feature scaling
  - outlier contamination
  - dimensionality reduction method
- [ ] Add consensus clustering.
- [ ] Add uncertainty estimates for soft clustering.
- [ ] Add cluster transition analysis under data drift.
- [ ] Add explainability using feature distributions, centroids, SHAP-style surrogate models, and local examples.

**Success criteria**

- The repo does not merely report one clustering result.
- It shows how fragile or stable the discovered structure is.
- It explains what can and cannot be concluded from unsupervised methods.

---

## Phase 4 — Productionisation Layer

**Goal:** Show that the notebook work can become a maintainable product.

- [ ] Package reusable pipelines under `src/unsup_lab`.
- [ ] Add a CLI:
  - `unsup-lab generate-data`
  - `unsup-lab train-clustering`
  - `unsup-lab detect-anomalies`
  - `unsup-lab build-topic-model`
  - `unsup-lab report`
- [ ] Add model artifact saving with metadata.
- [ ] Add experiment tracking with MLflow or a lightweight local JSON tracker.
- [ ] Add configuration with YAML.
- [ ] Add Dockerfile.
- [ ] Add FastAPI service for cluster assignment and anomaly scoring.
- [ ] Add batch inference script.
- [ ] Add scheduled report generation.

---

## Phase 5 — Advanced Research Extensions

**Goal:** Make the repo stand out for senior and research-oriented roles.

- [ ] Implement deep clustering:
  - autoencoder embeddings
  - DEC-style clustering
  - contrastive representation learning
- [ ] Add graph-based unsupervised learning:
  - community detection
  - node embeddings
  - graph anomaly detection
- [ ] Add streaming clustering:
  - mini-batch KMeans
  - online centroid updates
  - drift-aware cluster monitoring
- [ ] Add probabilistic modelling:
  - Bayesian Gaussian mixtures
  - Dirichlet Process Mixtures
  - Hidden Markov Models for latent states
- [ ] Add time-series representation learning:
  - shapelets
  - matrix profiles
  - dynamic time warping clustering
  - seasonal anomaly detection
- [ ] Add privacy-preserving or federated unsupervised learning discussion.

---

## Phase 6 — Communication and Portfolio Polish

**Goal:** Make the repository readable and convincing.

- [ ] Add a short portfolio landing page in `docs/`.
- [ ] Add one-page executive summaries for each case study.
- [ ] Add diagrams for each modelling workflow.
- [ ] Add “decision notes” explaining modelling tradeoffs.
- [ ] Add a final notebook called `99_lessons_learned.ipynb`.
- [ ] Add LinkedIn post draft and CV bullet points based on the repo.
- [ ] Add screenshots of plots and reports.

## Final target

The completed project should feel like a serious internal data science lab for unsupervised learning, not a collection of toy notebooks. It should show:

- mathematical understanding,
- modelling judgement,
- practical feature engineering,
- reproducibility,
- communication,
- robustness analysis,
- and a path from research notebook to production.
