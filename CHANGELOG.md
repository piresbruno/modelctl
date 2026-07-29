# Changelog

This project follows semantic versioning.

## 0.9.3

### Fixed

- Detect GGUF draft models stored under an `MTP/` directory even when their
  filenames do not start with `mtp-`, and exclude them from primary GGUF
  quantization selection.

## 0.9.2

### Changed

- `sync-local` now displays aggregate transfer progress, speed, and ETA while
  copying an active NAS model into the local Hugging Face cache.

## 0.9.1

### Documentation

- Clarified Hugging Face cache defaults, deprecated local-root compatibility,
  partial-snapshot behavior, and record-only local deletion across the README
  and CLI help.

## 0.9.0

### Changed

- `sync-local` now publishes validated NAS files into the standard Hugging Face
  cache `blobs`/`snapshots`/`refs` layout, with offline ETag verification,
  resumable staging, and modelctl sidecar registrations.
- Local inventory, path, and serve workflows can resolve modelctl registrations
  from the Hugging Face cache using `--local` and `--cache-dir`.
- `delete-local` now removes only modelctl registration state; shared Hugging
  Face cache data is retained for explicit `hf cache rm` or `hf cache prune`.
- Existing `--root`, `MODELCTL_LOCAL_ROOT`, and saved local-root behavior remain
  deprecated cache-directory fallbacks.

## 0.8.1

### Documentation

- Documented how to compare Hugging Face cache repository IDs with repositories
  already present in a modelctl NAS store, using the `hf cache list` output
  format.

## 0.8.0

### Added

- `modelctl delete-local NAME` safely removes a synchronized model from the
  configured local store without modifying its NAS source.

## 0.7.0

### Changed

- `modelctl list` now displays only model name, runtime, and repository.
- The node-local model root can be persisted with `modelctl config
  set-local-root PATH`; `list --local` and `sync-local` use it automatically.

## 0.6.0

### Added

- `modelctl list` validates and displays active models in the configured NAS
  store, with optional JSON output.
- `modelctl list --local` displays models synchronized to the node-local store.

## 0.5.0

### Added

- `modelctl config set-root PATH` persists the default NAS model store so
  commands no longer require repeated `--root` arguments.
- `modelctl config get-root` prints the effective default model store.

## 0.4.0

### Added

- `sync-cards` command to backfill sidecar cards for active NAS models at their
  exact Hugging Face commits.
- Complete upstream `README.md` preservation plus generated `RUN.md` files with
  verified local commands and relevant upstream usage sections.
- Atomic card publication with provenance, detected instruction headings, and
  SHA-256 integrity metadata without modifying published model objects.

## 0.3.0

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
