# Interview Talking Points

Discussion prompts for walking a technical reviewer through this project. Each
point pairs a likely question with the substance of a strong answer. The
through-line is modelling judgement, not algorithm recall.

## Framing and evaluation

**"How do you evaluate clustering without labels?"**
Internal metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) measure
cohesion and separation, but no single number is decisive. The more important
question is stability: does the structure survive a different seed, a bootstrap
resample, a different scaling, or a few injected outliers? Notebook 06 and the
`stability` module answer exactly that.

**"Your synthetic data has hidden labels - isn't that cheating?"**
The labels exist only for *offline* validation and are never passed to a model
during fitting. In anomaly detection, scoring happens first and labels are used
afterwards purely to compute precision-at-k. This mirrors reality, where you
might back-test against a later-confirmed outcome.

## Method choices

**"Why compare four clustering algorithms instead of picking one?"**
Each encodes a different assumption about cluster shape - KMeans assumes
compact spherical clusters, GMM allows elliptical soft assignments,
Agglomerative builds a hierarchy, DBSCAN finds density-connected regions and
flags noise. Comparing them makes the assumption explicit rather than hidden.

**"When does a 2D embedding mislead you?"**
t-SNE/UMAP-style plots optimise local neighbourhoods and can manufacture
apparent clusters or distort global distances. Notebook 02 pairs embeddings
with reconstruction error and explained variance so the picture is not trusted
on its own.

## Robustness and stability

**"What's the most common unsupervised mistake?"**
Reporting a single clustering result as if it were the answer. The fix is to
quantify fragility: bootstrap stability and pairwise adjusted mutual information
across seeds tell you whether the segments are real or an initialisation
artefact. Consensus clustering (notebook 07) then reduces that variance.

**"How would you detect that a deployed clustering model has gone stale?"**
Monitor drift. Notebook 08 tracks centroid movement and the Population
Stability Index per batch and flags batches above a threshold. A fixed cluster
count cannot represent a new regime, so the right response to detected drift is
to re-fit or revisit k - the monitor signals that rather than hiding it.

## Topic modelling

**"How do you know your topics are any good?"**
Two diagnostics: UMass coherence (do the top terms actually co-occur in
documents?) and topic diversity (are topics distinct or repeating vocabulary?).
Notebook 04 also reproduces four concrete failure modes - over-factorising,
short documents, vocabulary drift, domain-term filtering - so the symptoms are
recognisable.

## Engineering and productionisation

**"How is this more than a pile of notebooks?"**
The modelling logic lives in a typed, tested package (175+ unit tests) behind a
ruff + mypy + pytest + notebook-smoke CI gate. The same code backs a CLI, model
artifacts with metadata, JSON reports, a FastAPI scoring service and a
Dockerfile, so the path from research to a running endpoint is concrete.

**"What would you do differently at production scale?"**
The co-association matrix in consensus clustering is O(n^2) memory and would
need sampling or an approximate method; PSI here is univariate and should be
extended to multivariate or model-based drift; and the synthetic generators
would be replaced by real feature pipelines with proper data contracts.

## Honesty

**"Tell me about something that didn't work."**
The first draft of the short-document failure mode claimed assignment confidence
collapses. It didn't - the synthetic corpus tokens are so topic-specific that
short documents stayed confident. I reframed the demonstration around the metric
that genuinely degrades (signal terms per document) so the notebook's narrative
matches its own output instead of asserting a convenient but false result.
