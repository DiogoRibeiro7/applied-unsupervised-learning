"""Reusable utilities for the unsupervised learning lab."""

from unsup_lab.bayesian import (
    BayesianMixtureResult,
    assignment_uncertainty,
    component_weight_table,
    dirichlet_process_clustering,
    effective_components,
    fit_dirichlet_process_mixture,
)
from unsup_lab.consensus import (
    ConsensusResult,
    build_coassociation_matrix,
    consensus_clustering,
)
from unsup_lab.data import (
    make_customer_segmentation_data,
    make_document_corpus,
    make_sensor_anomaly_data,
    make_user_item_interactions,
)
from unsup_lab.evaluation import (
    ClusteringMetrics,
    evaluate_clustering,
    evaluate_k_range,
)
from unsup_lab.nlp import (
    assign_documents,
    clean_text,
    label_topics,
    topic_diversity,
    topic_term_table,
    umass_topic_coherence,
)
from unsup_lab.recommenders import (
    MatrixFactorization,
    build_sparse_interactions,
    discover_item_groups,
    factorize_interactions,
    recommend_for_user,
    similar_items,
)
from unsup_lab.stability import (
    StabilitySummary,
    bootstrap_cluster_stability,
    outlier_sensitivity,
    pairwise_adjusted_mutual_information,
    repeated_run_labels,
    scaling_sensitivity,
    stability_report,
)
from unsup_lab.streaming import (
    detect_drift_points,
    iter_batches,
    monitor_streaming_clusters,
    population_stability_index,
)
from unsup_lab.timeseries import (
    cluster_time_series,
    dtw_distance,
    dtw_distance_matrix,
)

__all__ = [
    "BayesianMixtureResult",
    "ClusteringMetrics",
    "ConsensusResult",
    "MatrixFactorization",
    "StabilitySummary",
    "assign_documents",
    "assignment_uncertainty",
    "bootstrap_cluster_stability",
    "build_coassociation_matrix",
    "component_weight_table",
    "dirichlet_process_clustering",
    "effective_components",
    "fit_dirichlet_process_mixture",
    "build_sparse_interactions",
    "cluster_time_series",
    "consensus_clustering",
    "detect_drift_points",
    "discover_item_groups",
    "dtw_distance",
    "dtw_distance_matrix",
    "evaluate_clustering",
    "evaluate_k_range",
    "factorize_interactions",
    "iter_batches",
    "label_topics",
    "make_customer_segmentation_data",
    "make_document_corpus",
    "make_sensor_anomaly_data",
    "make_user_item_interactions",
    "monitor_streaming_clusters",
    "outlier_sensitivity",
    "pairwise_adjusted_mutual_information",
    "population_stability_index",
    "recommend_for_user",
    "repeated_run_labels",
    "scaling_sensitivity",
    "similar_items",
    "stability_report",
    "topic_diversity",
    "topic_term_table",
    "umass_topic_coherence",
    "clean_text",
]
