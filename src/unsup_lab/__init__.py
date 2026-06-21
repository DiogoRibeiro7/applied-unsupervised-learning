"""Reusable utilities for the unsupervised learning lab."""

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

__all__ = [
    "ClusteringMetrics",
    "MatrixFactorization",
    "StabilitySummary",
    "bootstrap_cluster_stability",
    "build_sparse_interactions",
    "discover_item_groups",
    "evaluate_clustering",
    "evaluate_k_range",
    "factorize_interactions",
    "make_customer_segmentation_data",
    "make_document_corpus",
    "make_sensor_anomaly_data",
    "make_user_item_interactions",
    "outlier_sensitivity",
    "pairwise_adjusted_mutual_information",
    "recommend_for_user",
    "repeated_run_labels",
    "scaling_sensitivity",
    "similar_items",
    "stability_report",
]
