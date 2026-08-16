# Decision Notes

The modelling tradeoffs behind the project, and why each call was made. These are
the "why", not the "how" - the code and notebooks cover the how.

## Synthetic data instead of a real dataset

**Decision.** Use deterministic synthetic generators with planted structure.
**Why.** It keeps the repo self-contained (no private data, no API keys),
reproducible (fixed seeds), and lets each notebook validate against known hidden
labels *offline*. **Tradeoff.** Synthetic data is cleaner than reality, so
results read as upper bounds; the failure-mode sections exist to counter that
optimism.

## Compare clustering algorithms rather than pick one

**Decision.** Run KMeans, GMM, Agglomerative and DBSCAN side by side.
**Why.** Each encodes a different assumption (compact, elliptical/soft,
hierarchical, density-based). Comparing them makes the assumption explicit.
**Tradeoff.** More to maintain and explain, but it avoids smuggling in a hidden
modelling choice.

## Standardise features before distance-based clustering

**Decision.** Default to `StandardScaler`; offer `RobustScaler`.
**Why.** KMeans and DTW are scale-sensitive; an unscaled large-magnitude feature
would dominate the distance. **Tradeoff.** Standardising assumes roughly
symmetric features; heavy tails or outliers argue for the robust scaler, which
notebook 06's scaling-sensitivity analysis quantifies.

## Stability analysis over a single clustering result

**Decision.** Report bootstrap stability and seed-agreement (AMI), not one fit.
**Why.** The most common unsupervised mistake is presenting one partition as the
answer. Stability tells you whether structure is real or an initialisation
artefact. **Tradeoff.** More computation; mitigated by subsampling and fixed
seeds.

## Consensus clustering to reduce variance

**Decision.** Aggregate many base clusterings via a co-association matrix.
**Why.** It stabilises assignments and yields a stability score a single fit
cannot. **Tradeoff.** The co-association matrix is O(n^2) memory, so it does not
scale without sampling - stated in notebook 07's limitations.

## Dirichlet-process mixtures to infer the cluster count

**Decision.** Offer a nonparametric model that prunes unused components.
**Why.** It trades a hard choice of `k` for a softer choice of an upper bound and
a prior. **Tradeoff.** It assumes Gaussian components, so non-elliptical clusters
inflate the count; "effective components" depends on a weight threshold that we
report rather than hide.

## Anomaly detection as ranking, not classification

**Decision.** Score first, use labels only afterwards for precision@k.
**Why.** In production there are no labels at scoring time; treating it as binary
classification leaks information and overstates performance. **Tradeoff.** No
single accuracy number - but that honestly reflects the unsupervised setting.

## DTW band for time-series clustering

**Decision.** Constrain alignment with a Sakoe-Chiba band.
**Why.** It bounds cost and blocks pathological alignments (peak matched to
trough). **Tradeoff.** The band encodes an assumption about how much warping is
plausible; too wide defeats the purpose, too narrow approaches Euclidean.

## Modularity to choose the number of communities

**Decision.** Select the community count by maximising Newman modularity.
**Why.** It needs no labels and rewards within-community density above chance.
**Tradeoff.** Modularity has a resolution limit (it can merge small communities)
and can look high on random graphs, so it deserves a null-model sanity check.

## A lightweight JSON tracker instead of MLflow

**Decision.** Append runs to a JSONL file; no tracking server.
**Why.** MLflow is heavy for this project and adds infrastructure. A JSONL log
gives parameter/metric comparison and an audit trail with zero dependencies.
**Tradeoff.** No UI, no artifact store, no distributed runs - fine at this scale,
and an obvious upgrade point if the work grew.

## Deep learning behind an optional extra

**Decision.** Keep the core install to NumPy, pandas, scipy, scikit-learn and
matplotlib, and put PyTorch behind an optional group (`poetry install --with
deep`) used only by `unsup_lab.deep` and notebook 15. **Why.** The deep
clustering methods (autoencoder embeddings, DEC, contrastive) earn their place
in the argument about *objectives*, but a reader who only wants the classical
notebooks should not have to download a deep-learning framework to run them.
**Tradeoff.** One notebook and one module cannot run on a bare install, so the
dependency is stated wherever they are referenced, and no other module imports
them.

## Rank-normalise before combining anomaly detectors

**Decision.** In notebook 03, average detector *ranks*, not raw scores.
**Why.** The five detectors output incomparable quantities - a Mahalanobis-style
distance in the hundreds next to an isolation-forest score in tenths - so a
plain mean is a weighted vote in which one detector holds all the weight (the
correlation between the raw mean and robust covariance alone is 1.00).
**Tradeoff.** On this dataset the honest rank ensemble scores *worse* than the
best single detector, because local outlier factor is near chance here. That is
the real lesson and it is left in the notebook rather than tuned away:
ensembling helps when members are individually decent and fail differently.
