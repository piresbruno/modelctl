# Changelog

This project follows semantic versioning.

## Unreleased

### Added

- Store Hugging Face model cards with generated model selections.
- Discover, select, validate, and serve llama.cpp `mmproj` and MTP companion
  GGUFs together with their primary model.
- `--mmproj` and `--mtp` overrides for ambiguous companion selections, including
  equivalent download queue fields.

## 0.2.0

### Added

- YAML download queues with configurable model-level concurrency.
- Strict queue schema validation and duplicate effective-name detection.
- Full preflight validation of Hugging Face sources, revisions, runtimes,
  quantizations, existing manifests, and model-root symlink support.
- Resumable queue execution that reuses valid published objects.
- Queue documentation and examples for multiple quantizations, MTP draft
  models, and DFlash draft models.

### Changed

- Package builds and CLI output now share one version source.

## 0.1.0

- Initial atomic Hugging Face model download, validation, publication,
  activation, local synchronization, and runtime command support.
