# Ambitious Roadmap

This roadmap is designed to turn the repository from a notebook portfolio into a mature applied unsupervised learning lab.

## Phase 0 — Foundation

**Goal:** Make the repository clean, reproducible, and credible.

- [x] Add deterministic synthetic data generators with configurable random seeds.
- [x] Add typed reusable functions for preprocessing, evaluation, plotting, and reporting.
- [x] Add notebook execution checks using `nbclient` or `papermill`.
- [x] Add unit tests for all non-notebook code.
- [x] Add CI with linting, type checks, tests, and notebook smoke tests.
- [x] Add a clear README with problem framing, business use cases, and modelling decisions.

**Success criteria**

- A recruiter or hiring manager can understand the repo in under 3 minutes.
- A technical reviewer can run the notebooks without manual fixes.
- Every notebook has a clear modelling question, conclusion, and limitations section.

---

## Phase 1 — Core Unsupervised Learning Workflows

**Goal:** Demonstrate strong practical knowledge of unsupervised learning methods.

### Clustering

- Implement comparative clustering workflows:
  - [x] KMeans
  - [x] Gaussian Mixture Models
  - [x] Agglomerative Clustering
  - [x] DBSCAN
  - [ ] HDBSCAN, optional dependency
  - [x] Spectral Clustering (on graphs, `unsup_lab.graphs`; not yet in the tabular comparison)
- [x] Add cluster profiling tables.
- [ ] Add automatic cluster naming based on feature summaries.
- [x] Compare compact, density-based, hierarchical, and probabilistic cluster assumptions.
- [x] Add failure-case examples where each method performs poorly. (Notebook 01: four diagnostic
      geometries scored against five candidates, each of which wins somewhere and fails somewhere.)

### Dimensionality reduction

- [ ] Add PCA from both scikit-learn and a small NumPy implementation.
- [ ] Add reconstruction error analysis. (Used for anomaly scoring in notebook 03, not yet as a
      dimensionality-reduction diagnostic in notebook 02.)
- [x] Add explained variance diagnostics.
- Add manifold learning comparisons:
  - [x] Isomap
  - [x] Locally Linear Embedding
  - [x] Kernel PCA
  - [ ] UMAP as optional extension
- [x] Add a section on when 2D visual separation is misleading.

### Anomaly detection

- Implement anomaly detection notebooks for:
  - [x] Isolation Forest
  - [x] Local Outlier Factor
  - [x] Robust covariance
  - [x] One-Class SVM
  - [x] PCA reconstruction error
  - [ ] Autoencoder, optional deep learning extension (the autoencoder in `unsup_lab.deep` is used
        for clustering, not yet for anomaly scoring)
- [ ] Add threshold selection strategies.
- [x] Add analyst review tables.
- [x] Add drift-aware anomaly monitoring. (Cluster/feature drift in notebook 08; not yet tied back
      to the anomaly detectors.)
- [x] Add precision-at-k evaluation when synthetic labels are known but hidden during modelling.

---

## Phase 2 — Realistic Case Studies

**Goal:** Move from algorithm demos to credible applied projects.

### Case Study 1 — Customer Segmentation

- [x] Simulate customer behaviour with latent personas.
- [x] Add RFM features, product affinity, price sensitivity, churn-risk proxies, and campaign response features.
- [x] Build segmentation pipelines.
- [ ] Produce persona cards.
- [ ] Add recommended business actions per segment.

### Case Study 2 — Sensor Anomaly Discovery

- [ ] Simulate IoT sensor streams with missingness, drift, unusual usage, and device faults.
      (The generator has a daily cycle and injected point anomalies; missingness is a feature
      column rather than actual gaps, and there is no drift or device-fault mode yet.)
- [ ] Compare point anomalies, contextual anomalies, and collective anomalies. (Notebook 12
      contrasts a shape anomaly against a point-wise z-score; notebook 03 is point-only.)
- [ ] Add time-aware aggregation.
- [ ] Add analyst-facing anomaly explanations. (Notebook 03 ranks and reviews; it does not yet
      attribute a score to the features that drove it.)

### Case Study 3 — Document Topic Discovery

- [x] Add unsupervised topic modelling with TF-IDF + NMF.
- [x] Add Latent Semantic Analysis.
- [x] Add topic coherence approximation.
- [x] Add automatic topic labels using top terms.
- [ ] Add optional BERTopic-style extension with embeddings.

### Case Study 4 — Recommender Embeddings

- [x] Build a synthetic user-item interaction matrix.
- [x] Learn latent factors with TruncatedSVD or NMF.
- [x] Visualise user and item embeddings.
- [x] Identify product groups and user taste communities.
- [x] Add cold-start discussion.

---

## Phase 3 — Model Selection, Robustness, and Stability

**Goal:** Show senior-level judgement.

- [x] Add bootstrap cluster stability.
- [x] Add adjusted mutual information between repeated clusterings.
- Add sensitivity analysis for:
  - [x] random seeds
  - [x] number of clusters
  - [x] feature scaling
  - [x] outlier contamination
  - [ ] dimensionality reduction method
- [x] Add consensus clustering.
- [x] Add uncertainty estimates for soft clustering.
- [ ] Add cluster transition analysis under data drift.
- [ ] Add explainability using feature distributions, centroids, SHAP-style surrogate models, and local examples.
      (Cluster profiling tables exist; surrogate models and local examples do not.)

**Success criteria**

- The repo does not merely report one clustering result.
- It shows how fragile or stable the discovered structure is.
- It explains what can and cannot be concluded from unsupervised methods.

---

## Phase 4 — Productionisation Layer

**Goal:** Show that the notebook work can become a maintainable product.

- [x] Package reusable pipelines under `src/unsup_lab`.
- [x] Add a CLI:
  - `unsup-lab generate-data`
  - `unsup-lab train-clustering`
  - `unsup-lab detect-anomalies`
  - `unsup-lab build-topic-model`
  - `unsup-lab report`
  - `unsup-lab batch-score`
- [x] Add model artifact saving with metadata.
- [x] Add experiment tracking with MLflow or a lightweight local JSON tracker.
- [x] Add configuration with YAML.
- [x] Add Dockerfile.
- [x] Add FastAPI service for cluster assignment and anomaly scoring.
- [x] Add batch inference script.
- [x] Add scheduled report generation.

---

## Phase 5 — Advanced Research Extensions

**Goal:** Make the repo stand out for senior and research-oriented roles.

- [x] Implement deep clustering:
  - autoencoder embeddings
  - DEC-style clustering
  - contrastive representation learning
- [x] Add graph-based unsupervised learning:
  - community detection
  - node embeddings
  - graph anomaly detection
- [x] Add streaming clustering:
  - mini-batch KMeans
  - online centroid updates
  - drift-aware cluster monitoring
- Add probabilistic modelling:
  - [x] Bayesian Gaussian mixtures
  - [x] Dirichlet Process Mixtures
  - [ ] Hidden Markov Models for latent states
- Add time-series representation learning:
  - [x] shapelets
  - [x] matrix profiles
  - [x] dynamic time warping clustering
  - [ ] seasonal anomaly detection
- [ ] Add privacy-preserving or federated unsupervised learning discussion.

---

## Phase 6 — Communication and Portfolio Polish

**Goal:** Make the repository readable and convincing.

- [x] Add a short portfolio landing page in `docs/`.
- [x] Add one-page executive summaries for each case study.
- [x] Add diagrams for each modelling workflow.
- [x] Add “decision notes” explaining modelling tradeoffs.
- [x] Add a final notebook called `99_lessons_learned.ipynb`.
- [x] Add LinkedIn post draft and CV bullet points based on the repo.
- [x] Add screenshots of plots and reports.

## Final target

The completed project should feel like a serious internal data science lab for unsupervised learning, not a collection of toy notebooks. It should show:

- mathematical understanding,
- modelling judgement,
- practical feature engineering,
- reproducibility,
- communication,
- robustness analysis,
- and a path from research notebook to production.
