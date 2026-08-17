# What the notebooks found

Results, not features. Every number here comes from the executed notebook stored in the repository, and the list deliberately includes the results that did **not** flatter the method — those are the ones that took work to find and are worth more than the wins.

## Clustering

**Three algorithms, one partition.** On the customer data, KMeans, the Gaussian mixture and Ward agglomerative clustering return *the identical partition* — ARI 1.000, down to identical cluster sizes. That is a property of the data, not evidence about the methods: the generator plants compact, well-separated, roughly spherical segments, exactly the regime where a centroid, a full-covariance Gaussian and Ward linkage describe the same thing. A comparison run only on data that convenient cannot justify the choice of algorithm.

**So the assumptions were tested where they bite.** Four diagnostic geometries, five candidates, one configuration everywhere:

| | KMeans | GMM (full) | Agglom (ward) | Agglom (single) | DBSCAN |
| --- | --- | --- | --- | --- | --- |
| spherical blobs | **0.98** | 0.98 | 0.92 | 0.00 | 0.93 |
| elongated blobs | 0.71 | **1.00** | 0.77 | 0.57 | 0.90 |
| non-convex moons | 0.49 | 0.51 | 0.65 | **1.00** | 0.96 |
| mixed density | 0.65 | **0.97** | 0.90 | 0.00 | 0.06 |

Every method wins somewhere and fails somewhere. Single linkage is the sharpest case: the only perfect score on the moons, and the worst method on the other three, because the permissiveness that lets it follow a crescent is the same one that chains two clusters through a single bridging point.

→ [Notebook 01](notebooks/01_customer_segmentation_clustering.ipynb)

## Anomaly detection

**The ensemble was one detector in disguise.** Averaging five detectors' raw scores correlates **1.00 with robust covariance alone**, whose Mahalanobis-scale scores swamp the other four votes. Rank-normalised so they vote on equal terms, the honest ensemble scores **0.24** against **1.00** for the best single detector — because local outlier factor is near chance on globally-outlying anomalies and drags four decent votes down.

**Where you cut matters as much as what you score with.** Four label-free threshold rules demand between **116 and 322 alerts** against the same 120 real anomalies. Otsu is simultaneously the best rule (precision 1.000 on robust covariance) and among the worst (0.585 on the isolation forest) — the rule did not get worse, its two-population assumption stopped holding.

→ [Notebook 03](notebooks/03_anomaly_detection_sensor_events.ipynb)

## Explaining a segmentation

**Six rules reproduce 99.1% of the partition.** A depth-3 surrogate tree turns the clustering into thresholds on the original features, and reports that fidelity — at 0.6 the honest reading would be that the clusters are not rule-shaped and the tidy rules are a lie about them.

**Names derived without labels recover every planted persona at purity 1.000.** Generated purely from feature deviations, then checked against the hidden labels afterwards. It works here for the same reason the three algorithms agreed: planted, well-separated personas.

**The narrowest margin is 0.40 standardised units.** Some customers sit between two segments and would be filed differently by a slightly different fit — which a profile of means structurally cannot show.

→ [Notebook 01](notebooks/01_customer_segmentation_clustering.ipynb)

## Time series and graphs

- **DTW recovers phase-shifted shape families at ARI 1.000**, where point-by-point Euclidean distance manages 0.589. → [Notebook 10](notebooks/10_time_series_dtw_clustering.ipynb)
- **The matrix profile flags a shape anomaly at 8.31 against a median of 0.53**, while the largest absolute z-score anywhere inside that anomaly is **0.17** — a point-wise detector sees nothing at all. → [Notebook 12](notebooks/12_matrix_profile_motifs_discords.ipynb)
- **An unsupervised shapelet separates hidden classes at ARI 1.000**; whole-series KMeans gets 0.225, because the discriminating pattern moves between series. → [Notebook 14](notebooks/14_unsupervised_shapelets.ipynb)
- **Spectral community detection reaches ARI 1.000** on a clearly-structured graph, and still beats KMeans-on-adjacency 0.607 to 0.214 on a weakly-structured one. → [Notebook 11](notebooks/11_graph_community_detection.ipynb)

## Representation learning

**DEC 0.928, contrastive 0.893, plain autoencoder 0.869, raw KMeans 0.865, PCA 0.858.** On non-linearly tangled, noisy data, a reconstruction autoencoder is no free lunch — it barely beats clustering the raw features, because reconstruction is the wrong objective for separation. Methods that optimise for what is actually wanted (separation, invariance) beat the one that optimises for reproducing the input, noise included.

→ [Notebook 15](notebooks/15_deep_clustering_autoencoder_dec.ipynb)

## Model selection

- **A Dirichlet-process mixture infers 5 components from a ceiling of 15** — the same answer a manual silhouette sweep gives, without the sweep. → [Notebook 09](notebooks/09_bayesian_nonparametric_mixtures.ipynb)
- **Consensus clustering reaches 0.995 mean intra-cluster agreement**, and comes with a stability score a single fit cannot provide. → [Notebook 07](notebooks/07_consensus_clustering.ipynb)
- **Single-pass streaming holds AMI 0.742** against a full-batch fit while flagging every drifted batch. → [Notebook 08](notebooks/08_streaming_clustering_drift.ipynb)
