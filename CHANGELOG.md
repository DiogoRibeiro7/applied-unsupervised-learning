# Changelog

All notable changes to this project are documented here.

This project follows a practical versioning policy:

- Patch releases fix bugs, documentation, tests, and reproducibility issues.
- Minor releases add notebooks, modelling workflows, or reusable source modules.
- Major releases may reorganize notebooks, APIs, or package-level contracts.

## Unreleased

## 0.3.0 - 2026-08-17

Two new modules turn results into decisions: where to cut an anomaly ranking,
and how to explain, name and hand over a segmentation.

### Added

- `unsup_lab.thresholds` — four label-free strategies for choosing an operating
  point on anomaly scores (quantile, robust MAD, knee of the sorted curve, and
  Otsu's two-population split), plus offline scoring of a cut *after* it is
  chosen. On the sensor data the four rules demand between 116 and 322 alerts
  against the same 120 anomalies, and Otsu is both the strongest and among the
  weakest depending on whether the detector's scores really are two populations.
- `unsup_lab.explain` — a depth-limited surrogate tree that reports its own
  fidelity to the partition (six rules reproduce 99.1% of the customer
  segmentation), per-cluster feature deviations, generated segment names,
  exemplars, boundary members, and Markdown persona cards. The generated names
  recover every planted persona at purity 1.000 without labels being used to fit.
- Notebook 01 gained sections on explaining and naming a segmentation; notebook
  03 gained one on choosing an operating point.
- README now opens with the results the notebooks produced, including the ones
  that do not flatter the method, each figure taken from stored notebook output.
- Zenodo deposit records `language`, explanatory `notes` covering the project
  rename, and `references` crediting the fourteen primary sources the notebooks
  implement. Keywords expanded from 12 to 24.
- Version DOIs for 0.1.0 and 0.2.0 recorded in the README and `CITATION.cff`
  alongside the concept DOI.
- The roadmap gained a delivery order: which outstanding entries ship together,
  in what sequence, and what is deferred with the reason.

### Changed

- README gained a contents line, and its layout tree now lists `scripts/`,
  `configs/`, both licence files, the citation metadata and the lock file.
- The roadmap's HDBSCAN entry no longer claims an optional dependency is needed;
  it has been native in scikit-learn since 1.3.

### Deliberately not done

- Recommended business actions per segment. The persona cards carry the
  evidence and leave the action to a domain owner: the margin on an offer, the
  team's review capacity and last quarter's campaigns are not in the feature
  table, and generating confident advice without them would be the overreach
  this project argues against throughout.

## 0.2.0 - 2026-08-16

### Added

- Professional repository metadata, contribution guidance, security policy, issue
  templates, pull request template, and dependency update configuration.
- Repository rename to Applied Unsupervised Learning, including project
  metadata, documentation, citation, and Zenodo metadata updates.
- Deep-clustering CI coverage using the CPU PyTorch build pinned by the lock
  file.

### Changed

- Notebook 01 now includes a stronger clustering algorithm comparison.
- Dependency updates from Dependabot.

## 0.1.0

### Added

- Notebook-first unsupervised learning project covering clustering,
  dimensionality reduction, anomaly detection, topic modelling, recommender
  embeddings, stability analysis, graph methods, streaming drift, time series,
  shapelets, and deep clustering.
- Reusable `unsup_lab` package with tested utilities for data generation,
  evaluation, reporting, services, artifacts, tracking, and modelling workflows.
- CI quality gate for linting, type checking, tests, notebook smoke execution,
  and optional deep-clustering tests.
