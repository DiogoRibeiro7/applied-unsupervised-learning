# Changelog

All notable changes to this project are documented here.

This project follows a practical versioning policy:

- Patch releases fix bugs, documentation, tests, and reproducibility issues.
- Minor releases add notebooks, modelling workflows, or reusable source modules.
- Major releases may reorganize notebooks, APIs, or package-level contracts.

## Unreleased

## 0.2.1 - 2026-08-16

Documentation and archival metadata only; no source, notebook or dependency
changes, so the modelling results are identical to 0.2.0.

### Added

- README now opens with the results the notebooks produced, including the ones
  that do not flatter the method, each figure taken from stored notebook output.
- Zenodo deposit records `language`, explanatory `notes` covering the project
  rename, and `references` crediting the fourteen primary sources the notebooks
  implement. Keywords expanded from 12 to 24.
- Version DOIs for 0.1.0 and 0.2.0 recorded in the README and `CITATION.cff`
  alongside the concept DOI.

### Changed

- README gained a contents line, and its layout tree now lists `scripts/`,
  `configs/`, both licence files, the citation metadata and the lock file.

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
