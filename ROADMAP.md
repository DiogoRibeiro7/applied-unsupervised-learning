# Ambitious Roadmap

This roadmap is designed to turn the repository from a notebook project into a mature applied unsupervised learning project.

## Delivery order

The phases below are a *catalogue*: they list what could exist, in no particular
order, and several entries that amount to a single piece of work sit in
different phases. This section is the *plan* — which of those entries ship
together, in what sequence, and why.

Nothing here is a new item. Every bullet points back at checkboxes already
listed below, so there stays exactly one place to tick and no second list to
drift out of date.

### 1. Explain a cluster, then name it — no new dependencies

Closes the widest gap between what the project claims and what it does: the
README promises business-facing diagnostics and per-segment actions, while
notebook 01 profiles clusters with feature means and stops there.

Delivers, together: Phase 3 *explainability … surrogate models and local
examples*, Phase 1 *automatic cluster naming*, Phase 2 *persona cards* and
*recommended business actions per segment*.

Shape: a `unsup_lab.explain` module. A shallow decision tree fitted to the
cluster labels turns a partition into readable rules (`recency < 30 and
frequency > 20 → segment 2`), alongside representative and boundary members per
cluster, feeding persona cards in notebook 01. The surrogate's own fidelity to
the partition is reported, because a rule set that only half-describes the
clusters is a misleading explanation rather than a simple one.

### 2. Anomaly detection as a system — no new dependencies

Continues directly from threshold selection, and retires the overstatement
currently annotated in Phase 2: the generator claims missingness, drift and
device faults, and has none of them.

Delivers, together: Phase 2 *simulate IoT streams with missingness, drift …
and device faults*, *point / contextual / collective anomalies*, *time-aware
aggregation* and *analyst-facing anomaly explanations*, plus Phase 5 *seasonal
anomaly detection*.

Shape: a richer sensor generator with genuine gaps, a slow drift and a fault
mode; per-anomaly feature attribution so a flagged row comes with the reason it
was flagged; and a contrast between the three anomaly kinds, which the current
notebook cannot express because it only produces point anomalies.

### 3. HDBSCAN — no new dependencies

Small, and it lands somewhere useful: notebook 01's diagnostic table has a
*mixed density* row where DBSCAN scores 0.06, and varying density is precisely
the failure HDBSCAN exists to fix. It belongs as a sixth column there, not as a
standalone demonstration.

### 4. Depth for notebook 02

Notebook 02 is the thinnest analysis in the project. Three entries land in it
together: Phase 1 *PCA from scikit-learn and a small NumPy implementation* and
*reconstruction error analysis*, plus Phase 3 *sensitivity to the
dimensionality reduction method*.

### 5. Hidden Markov models — no new dependencies

The largest single build left, and the most research-facing: a Gaussian HMM with
Baum-Welch is a few hundred lines of NumPy. It is the only method under
consideration that models a latent *sequence of states* rather than latent
groups, which nothing else in the project does.

### Deferred, with the reason

- **UMAP** and **BERTopic** need substantial new dependencies (`umap-learn`;
  sentence-transformers and its transformer stack) against a standing preference
  for NumPy, pandas, scipy, scikit-learn and matplotlib. Worth revisiting only
  as a deliberate decision to cover the modern embedding stack.
- **Privacy-preserving or federated discussion** is prose, and reads as
  speculation until there is an implementation to discuss.
- **Cluster transition analysis under drift** waits for item 2, which is where
  drift becomes a real property of the data rather than an injected step.

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
  - [ ] HDBSCAN (native in scikit-learn since 1.3, so no optional dependency is needed after all)
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
- [x] Add threshold selection strategies. (`unsup_lab.thresholds`: quantile, robust MAD, knee and
      Otsu cuts, compared in notebook 03 and scored offline only after the cut is chosen.)
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
  - `applied-unsupervised-learning generate-data`
  - `applied-unsupervised-learning train-clustering`
  - `applied-unsupervised-learning detect-anomalies`
  - `applied-unsupervised-learning build-topic-model`
  - `applied-unsupervised-learning report`
  - `applied-unsupervised-learning batch-score`
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

## Phase 6 — Communication and Professional Polish

**Goal:** Make the repository readable and convincing.

- [x] Add a short project summary page in `docs/`.
- [x] Add one-page executive summaries for each case study.
- [x] Add diagrams for each modelling workflow.
- [x] Add “decision notes” explaining modelling tradeoffs.
- [x] Add a final notebook called `99_lessons_learned.ipynb`.
- [x] Add LinkedIn post draft and CV bullet points based on the repo.
- [x] Add screenshots of plots and reports.

## Final target

The completed project should feel like serious internal data science work for unsupervised learning, not a collection of toy notebooks. It should show:

- mathematical understanding,
- modelling judgement,
- practical feature engineering,
- reproducibility,
- communication,
- robustness analysis,
- and a path from research notebook to production.
