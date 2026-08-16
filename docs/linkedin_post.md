# LinkedIn Post Draft

A short announcement post. Professional, specific, not over-claimed. Trim the
hashtags to taste.

---

I just published an **Applied Unsupervised Learning** - a notebook-first applied machine learning project
that treats unsupervised learning as a serious modelling workflow rather than a
gallery of algorithms.

What's inside:

- Customer segmentation comparing KMeans, Gaussian Mixtures, Agglomerative and
  DBSCAN, with named segments and recommended actions
- Anomaly detection on synthetic sensor streams, scored before any label is
  touched
- Topic modelling with TF-IDF, NMF and LSA, plus coherence/diversity diagnostics
  and four reproduced failure modes
- Recommender embeddings from implicit feedback via matrix factorisation
- Model selection and **stability analysis** - bootstrap stability, consensus
  clustering, and streaming drift monitoring

The part I care most about isn't the algorithms - it's the judgement around
them: how do you trust a clustering with no labels? My answer is to stop
reporting a single result and start measuring how fragile it is under different
seeds, resampling, scaling and outliers.

Everything is reproducible: typed, tested Python modules (90+ unit tests), a
ruff/mypy/pytest CI gate, and a small production layer (CLI + FastAPI + Docker)
showing the path from notebook to endpoint. No external APIs, no private data,
no heavy deep-learning stack - just NumPy, pandas, scikit-learn and matplotlib.

Feedback from fellow practitioners very welcome.

#MachineLearning #DataScience #UnsupervisedLearning #MLOps #Python
