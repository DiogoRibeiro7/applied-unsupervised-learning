"""Scheduled report generation.

Retrains the standard models from a YAML config and writes timestamped JSON
reports plus an experiment-tracker entry. Designed to be driven on a schedule
(cron on Linux/macOS, Task Scheduler on Windows), for example daily::

    # crontab: run at 02:00 every day
    0 2 * * * cd /path/to/repo && poetry run python scripts/scheduled_report.py \
        --config configs/example_clustering.yaml

The timestamp is read from the system clock at run time, which is appropriate
for a scheduled job (unlike the deterministic notebooks).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from unsup_lab.cli import main as cli_main  # noqa: E402
from unsup_lab.config import load_config  # noqa: E402


def run_scheduled_report(config_path: str, output_dir: Path) -> Path:
    """Run the configured task and write a timestamped report.

    Returns
    -------
    pathlib.Path
        The report file that was written.
    """
    config = load_config(config_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{config.task}_{stamp}.joblib"
    # The artifact writes its metadata sidecar next to the model as
    # `<stem>.json`, so the run report must not claim that same name or it
    # overwrites the sidecar and the saved model can no longer be loaded.
    report_path = output_dir / f"{config.task}_{stamp}_report.json"
    track_path = output_dir / "runs.jsonl"

    # `--random-state` is a top-level option, so it has to precede the
    # subcommand; otherwise the seed in the config file is silently ignored.
    seed_argv = ["--random-state", str(config.random_state)]

    if config.task == "clustering":
        argv = [
            "train-clustering",
            "--n",
            str(config.n_samples),
            "--k",
            str(config.n_clusters),
            "--model-out",
            str(model_path),
            "--report-out",
            str(report_path),
            "--track-path",
            str(track_path),
        ]
    elif config.task == "anomaly":
        argv = [
            "detect-anomalies",
            "--n",
            str(config.n_samples),
            "--contamination",
            str(config.contamination),
            "--model-out",
            str(model_path),
            "--report-out",
            str(report_path),
            "--track-path",
            str(track_path),
        ]
    else:  # topic
        argv = [
            "build-topic-model",
            "--n-topics",
            str(config.n_topics),
            "--model-out",
            str(model_path),
            "--report-out",
            str(report_path),
            "--track-path",
            str(track_path),
        ]

    cli_main(seed_argv + argv)
    print(f"Scheduled {config.task} report written to {report_path}")
    return report_path


def main() -> None:
    """Parse arguments and generate the scheduled report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/example_clustering.yaml")
    parser.add_argument("--output-dir", default="outputs/scheduled")
    args = parser.parse_args()
    run_scheduled_report(args.config, Path(args.output_dir))


if __name__ == "__main__":
    main()
