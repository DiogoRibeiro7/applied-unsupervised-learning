"""Command line interface for the unsupervised learning lab.

The CLI operationalises the notebook workflows: it generates synthetic data,
trains clustering / anomaly / topic models, saves versioned artifacts with
metadata, and writes JSON reports. It is intentionally thin -- all modelling
logic lives in :mod:`unsup_lab.service` so the notebooks, CLI and API agree.

Examples
--------
::

    unsup-lab generate-data --dataset customers --output data/customers.csv
    unsup-lab train-clustering --k 5 --model-out outputs/models/clustering.joblib
    unsup-lab detect-anomalies --model-out outputs/models/anomaly.joblib
    unsup-lab build-topic-model --n-topics 4
    unsup-lab report --model outputs/models/clustering.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from unsup_lab.artifacts import load_artifact, save_artifact
from unsup_lab.data import (
    make_customer_segmentation_data,
    make_document_corpus,
    make_sensor_anomaly_data,
)
from unsup_lab.evaluation import evaluate_clustering
from unsup_lab.service import (
    anomaly_scores,
    assign_clusters,
    train_anomaly_model,
    train_clustering,
    train_topic_model,
)
from unsup_lab.tracking import log_run

_DEFAULT_MODEL_DIR = Path("outputs/models")
_DEFAULT_REPORT_DIR = Path("outputs/reports")


def _write_json(payload: dict[str, Any], path: Path) -> None:
    """Write a JSON report, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote report to {path}")


def _maybe_track(
    args: argparse.Namespace,
    name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Log a run to the experiment tracker when ``--track-path`` is supplied."""
    track_path = getattr(args, "track_path", None)
    if track_path:
        log_run(name, params=params, metrics=metrics, path=track_path)
        print(f"Logged run to {track_path}")


def cmd_generate_data(args: argparse.Namespace) -> None:
    """Generate a synthetic dataset and write it to CSV."""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset == "customers":
        frame = make_customer_segmentation_data(args.n, random_state=args.random_state).features
    elif args.dataset == "sensors":
        frame = make_sensor_anomaly_data(args.n, random_state=args.random_state).features
    elif args.dataset == "documents":
        frame = make_document_corpus(random_state=args.random_state)
    else:  # pragma: no cover - argparse choices guard this
        raise ValueError(f"Unknown dataset {args.dataset!r}.")

    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} rows to {output}")


def cmd_train_clustering(args: argparse.Namespace) -> None:
    """Train a clustering pipeline on synthetic customer data."""
    dataset = make_customer_segmentation_data(args.n, random_state=args.random_state)
    pipeline, labels = train_clustering(dataset.features, args.k, random_state=args.random_state)
    # Score in the space the model actually clusters in (after the pipeline's
    # scaler). Scoring the raw features would let the largest-magnitude column
    # dominate a distance the model never saw.
    transformed = pipeline[:-1].transform(dataset.features.to_numpy())
    metrics = evaluate_clustering(transformed, labels)

    save_artifact(
        pipeline,
        args.model_out,
        model_kind="clustering",
        parameters={"k": args.k, "random_state": args.random_state, "n_rows": args.n},
        metrics={"silhouette": metrics.silhouette, "n_clusters": metrics.n_clusters},
    )
    print(f"Saved clustering model to {args.model_out}")

    unique, counts = np.unique(labels, return_counts=True)
    report = {
        "model_kind": "clustering",
        "k": args.k,
        "silhouette": metrics.silhouette,
        "davies_bouldin": metrics.davies_bouldin,
        "calinski_harabasz": metrics.calinski_harabasz,
        "cluster_sizes": {
            int(label): int(count) for label, count in zip(unique, counts, strict=True)
        },
    }
    _write_json(report, Path(args.report_out))
    _maybe_track(
        args,
        "clustering",
        params={"k": args.k, "n": args.n},
        metrics={"silhouette": metrics.silhouette, "davies_bouldin": metrics.davies_bouldin},
    )


def cmd_detect_anomalies(args: argparse.Namespace) -> None:
    """Train an anomaly model on sensor data and report the top anomalies."""
    dataset = make_sensor_anomaly_data(
        args.n, contamination=args.contamination, random_state=args.random_state
    )
    numeric = dataset.features.drop(columns=["timestamp"])
    pipeline = train_anomaly_model(
        numeric, contamination=args.contamination, random_state=args.random_state
    )
    scores = anomaly_scores(pipeline, numeric)

    order = np.argsort(scores)[::-1]
    top_k = order[: args.top_k]
    hidden = dataset.hidden_labels.to_numpy()
    precision_at_k = float((hidden[top_k] == "anomaly").mean())

    save_artifact(
        pipeline,
        args.model_out,
        model_kind="anomaly",
        parameters={"contamination": args.contamination, "random_state": args.random_state},
        metrics={"precision_at_k": precision_at_k, "k": args.top_k},
    )
    print(f"Saved anomaly model to {args.model_out}")

    report = {
        "model_kind": "anomaly",
        "contamination": args.contamination,
        "top_k": args.top_k,
        "precision_at_k": precision_at_k,
        "top_anomaly_indices": [int(i) for i in top_k],
    }
    _write_json(report, Path(args.report_out))
    _maybe_track(
        args,
        "anomaly",
        params={"contamination": args.contamination, "top_k": args.top_k},
        metrics={"precision_at_k": precision_at_k},
    )


def cmd_build_topic_model(args: argparse.Namespace) -> None:
    """Fit a topic model on the synthetic corpus and report top terms."""
    corpus = make_document_corpus(random_state=args.random_state)
    pipeline, topics = train_topic_model(
        corpus["text"].tolist(),
        n_topics=args.n_topics,
        random_state=args.random_state,
    )

    save_artifact(
        pipeline,
        args.model_out,
        model_kind="topic",
        parameters={"n_topics": args.n_topics, "random_state": args.random_state},
        extra={"topics": topics},
    )
    print(f"Saved topic model to {args.model_out}")

    report = {"model_kind": "topic", "n_topics": args.n_topics, "topics": topics}
    _write_json(report, Path(args.report_out))
    _maybe_track(args, "topic", params={"n_topics": args.n_topics}, metrics={})


def cmd_report(args: argparse.Namespace) -> None:
    """Print the metadata sidecar of a saved artifact as JSON."""
    _, metadata = load_artifact(args.model)
    print(json.dumps(metadata.__dict__, indent=2))


def cmd_batch_score(args: argparse.Namespace) -> None:
    """Score a CSV of feature rows with a saved clustering or anomaly model."""
    model, metadata = load_artifact(args.model)
    frame = pd.read_csv(args.input)

    if metadata.model_kind == "clustering":
        frame["cluster"] = assign_clusters(model, frame)
    elif metadata.model_kind == "anomaly":
        frame["anomaly_score"] = anomaly_scores(model, frame)
    else:
        raise ValueError(
            f"batch-score supports 'clustering' and 'anomaly' models, not {metadata.model_kind!r}."
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Scored {len(frame)} rows with the {metadata.model_kind} model -> {output}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the CLI."""
    parser = argparse.ArgumentParser(prog="unsup-lab", description=__doc__)
    parser.add_argument("--random-state", type=int, default=42, help="Global random seed.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-data", help="Generate a synthetic dataset as CSV.")
    gen.add_argument(
        "--dataset", choices=["customers", "sensors", "documents"], default="customers"
    )
    gen.add_argument("--n", type=int, default=2000, help="Number of rows (where applicable).")
    gen.add_argument("--output", default="data/dataset.csv")
    gen.set_defaults(func=cmd_generate_data)

    clust = sub.add_parser("train-clustering", help="Train a clustering model.")
    clust.add_argument("--n", type=int, default=2000)
    clust.add_argument("--k", type=int, default=5)
    clust.add_argument("--model-out", default=str(_DEFAULT_MODEL_DIR / "clustering.joblib"))
    clust.add_argument("--report-out", default=str(_DEFAULT_REPORT_DIR / "clustering.json"))
    clust.add_argument("--track-path", default=None, help="JSONL experiment log to append to.")
    clust.set_defaults(func=cmd_train_clustering)

    anom = sub.add_parser("detect-anomalies", help="Train an anomaly model and score data.")
    anom.add_argument("--n", type=int, default=3000)
    anom.add_argument("--contamination", type=float, default=0.04)
    anom.add_argument("--top-k", type=int, default=50)
    anom.add_argument("--model-out", default=str(_DEFAULT_MODEL_DIR / "anomaly.joblib"))
    anom.add_argument("--report-out", default=str(_DEFAULT_REPORT_DIR / "anomaly.json"))
    anom.add_argument("--track-path", default=None, help="JSONL experiment log to append to.")
    anom.set_defaults(func=cmd_detect_anomalies)

    topic = sub.add_parser("build-topic-model", help="Fit a topic model on the corpus.")
    topic.add_argument("--n-topics", type=int, default=4)
    topic.add_argument("--model-out", default=str(_DEFAULT_MODEL_DIR / "topic.joblib"))
    topic.add_argument("--report-out", default=str(_DEFAULT_REPORT_DIR / "topic.json"))
    topic.add_argument("--track-path", default=None, help="JSONL experiment log to append to.")
    topic.set_defaults(func=cmd_build_topic_model)

    report = sub.add_parser("report", help="Print the metadata of a saved artifact.")
    report.add_argument("--model", required=True)
    report.set_defaults(func=cmd_report)

    batch = sub.add_parser("batch-score", help="Score a CSV with a saved model.")
    batch.add_argument("--model", required=True, help="Path to the saved model artifact.")
    batch.add_argument("--input", required=True, help="Input CSV of feature rows.")
    batch.add_argument("--output", required=True, help="Destination CSV for scored rows.")
    batch.set_defaults(func=cmd_batch_score)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``unsup-lab`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
