# Notebook Briefs

One-page briefs for each notebook. Every brief follows the same shape: the
business question, the modelling approach, the key result, and the judgement
call that matters most.

---

## 00 - Project Overview

**Question.** What is this lab and how should a reviewer read it?

**Approach.** A short orientation notebook: the philosophy (notebook-first,
label-free, reproducible), the repository map, and how the pieces connect.

**Takeaway.** Sets expectations - the notebooks are the product; the package
exists to keep them honest and reusable.

---

## 01 - Customer Segmentation Clustering

**Question.** Do customers fall into actionable behavioural segments, and how
confident can we be in them?

**Approach.** Engineer RFM-style features (recency, frequency, monetary value,
discount sensitivity, engagement, diversity), then compare KMeans, Gaussian
Mixtures, Agglomerative Clustering and DBSCAN. Evaluate with silhouette,
Davies-Bouldin and Calinski-Harabasz; profile and name each segment; recommend
concrete actions.

**Result.** Compact, interpretable segments with business-facing names and
recommended actions, validated by internal metrics rather than labels.

**Judgement.** Different algorithms encode different cluster shapes; the choice
is a modelling assumption, and the segments are decision aids, not facts.

---

## 02 - Dimensionality Reduction & Manifold Learning

**Question.** What low-dimensional structure underlies the features, and when is
a 2D picture misleading?

**Approach.** PCA with explained-variance and reconstruction-error diagnostics,
plus manifold methods (Kernel PCA, Isomap, LLE) for non-linear structure.

**Result.** A side-by-side view of linear vs non-linear embeddings with explicit
caveats about distortion.

**Judgement.** Visual separation in 2D is not proof of cluster structure;
neighbour-preservation and reconstruction error matter more than a pretty plot.

---

## 03 - Anomaly Detection on Sensor Events

**Question.** Which operational events are abnormal, given no labels at scoring
time?

**Approach.** Synthetic IoT streams with noise, missingness, drift and injected
anomalies. Compare Isolation Forest, LOF, robust covariance and PCA
reconstruction error; calibrate thresholds; evaluate precision-at-k using hidden
labels *only after* scoring.

**Result.** Ranked anomalies with analyst-facing explanations and a precision-at-k
read on quality.

**Judgement.** Anomaly detection is ranking under uncertainty, not binary
classification; labels are for offline evaluation, never for fitting.

---

## 04 - Unsupervised NLP Topic Modelling

**Question.** What themes run through a document corpus, and are the discovered
topics trustworthy?

**Approach.** Text cleaning, TF-IDF, NMF and LSA. Auto-generate topic labels
from top terms, assign documents with a confidence, and score topics with UMass
coherence and topic diversity. Demonstrate four failure modes: over-factorising,
short documents, vocabulary drift, and aggressive frequency filtering.

**Result.** Coherent, labelled topics that recover the latent themes, with
diagnostics that quantify quality and named failure modes.

**Judgement.** Topic models fail quietly; coherence, diversity and an awareness
of failure modes are what separate a usable model from a plausible-looking one.

---

## 05 - Recommender Embeddings via Matrix Factorisation

**Question.** Can we learn useful user and item embeddings from implicit
interactions alone?

**Approach.** Build a sparse user-item matrix, factorise with TruncatedSVD/NMF,
visualise the latent spaces, discover product groups, and run similar-item and
per-user recommendation.

**Result.** Latent factors that group co-consumed items and surface sensible
neighbours and recommendations.

**Judgement.** Implicit feedback is not preference; scores are relevance
rankings, and cold-start items have no signal to factorise.

---

## 06 - Model Selection, Stability & Explainability

**Question.** Is the discovered clustering real, or an artefact of one fit?

**Approach.** Internal metrics across k, seed-agreement via adjusted mutual
information, bootstrap stability, and sensitivity to scaling and outliers,
finished with an automated Markdown stability report.

**Result.** A quantified verdict on how stable the segmentation is, not just a
single partition.

**Judgement.** Reporting one clustering result is the most common unsupervised
mistake; stability analysis is the senior-level differentiator.

---

## 07 - Consensus Clustering

**Question.** How do we get a partition that survives random initialisation and
resampling?

**Approach.** Run many base clusterings on subsamples, accumulate a
co-association matrix, and derive a consensus partition agglomeratively. Compare
against a single KMeans fit.

**Result.** A more stable partition with a co-association heatmap as visual
evidence and an intra-cluster stability score.

**Judgement.** Consensus stabilises whatever bias the base method has; it
reduces variance, not bias.

---

## 08 - Streaming Clustering with Drift Monitoring

**Question.** How do we cluster a never-ending stream and notice when the world
changes?

**Approach.** Incremental mini-batch KMeans with per-batch drift signals -
centroid shift and Population Stability Index - and automatic drift-point
detection. Compare against full-batch KMeans.

**Result.** Single-pass clustering that tracks a known injected drift and stays
close to the full-batch solution.

**Judgement.** A fixed cluster count cannot represent a genuinely new regime;
the monitor signals when to re-fit rather than silently degrading.

---

## 09 - Bayesian Nonparametric Mixtures

**Question.** Can the model infer the number of clusters instead of us choosing
it?

**Approach.** A Dirichlet-process Gaussian mixture with a generous component
upper bound; the variational prior prunes unused components toward zero weight.
Report effective components, soft-assignment uncertainty, and prior sensitivity.
Compare against a manual silhouette sweep for k.

**Result.** Recovers the five latent segments from a cap of fifteen without a
manual search, and flags boundary customers via assignment entropy.

**Judgement.** It trades the choice of k for the choice of an upper bound, a
prior and a Gaussian assumption - fewer knobs, not none, and non-elliptical
clusters inflate the count.

---

## 10 - Time-Series Clustering with Dynamic Time Warping

**Question.** How do we group time series by shape when they are slightly out of
phase?

**Approach.** A banded DTW distance aligns series before measuring distance;
the pairwise distance matrix feeds agglomerative clustering. Compare against
point-by-point Euclidean distance on three phase-shifted shape families.

**Result.** DTW recovers the shape families almost perfectly (ARI ~1.0) where
Euclidean distance is misled by the shifts (ARI ~0.6).

**Judgement.** The alignment is the point - but DTW is O(L^2) per pair, not a
true metric, and amplitude-sensitive, so z-normalisation and banding are part of
using it well.
