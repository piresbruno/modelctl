# Changelog

This project follows semantic versioning.

## 0.13.0

### Added

- Add `--jobs N` to `push`: the transfer is split into up to N concurrent
  rsync streams into the same remote staging directory, sharply reducing
  wall time for many-file models over fast fabrics (defaults to the previous
  single-stream behavior when omitted). Repeating `--host` distributes the
  streams round-robin over several fabric interfaces of the same
  destination, so a node with two ConnectX-7 links can use both at once.

## 0.12.0

### Added

- `push` now always sources from the local Hugging Face cache: the model must
  have a modelctl registration there (`modelctl sync-local` or a prior
  copy), and the command builds a small transfer overlay from the registered
  snapshot (symlinks plus generated ETag metadata) before rsyncing it to the
  destination over ssh. This makes head-node fan-out over a fast
  interconnect the primary workflow, avoiding slow NAS reads on every node.
  The `--source-root` NAS-store option was removed.
- Local listings (`list --local`) now skip broken registrations whose cache
  data is missing, with an actionable warning, and `delete-local` can remove
  the stale records so they stop blocking cache-native workflows.

## 0.11.1

### Added

- `push` transfers only the retained per-file HF download metadata needed for
  ETag verification, excluding unrelated cache artifacts such as xet tree
  JSONs that can carry unreadable permissions on NAS mounts.
- `push` now creates the remote cache staging directory and proves the cache
  is writable over ssh before transferring anything, failing with an
  actionable error instead of an opaque rsync receiver failure. rsync's
  receiver creates the destination root with a single mkdir under
  `--files-from`, so the deep staging path must already exist.
- Document the full `push` workflow, prerequisites, options, fabric examples,
  and the `receive-cache` handshake and commit semantics in both the CLI help
  and the README.

## 0.11.0

### Added

- `push` now auto-discovers the remote `modelctl` binary, so a plain
  `uv tool install` on the destination host is sufficient for ssh
  deployments. Non-interactive ssh shells ignore shell rc files, so the
  probe checks the standard install locations (`~/.local/bin/modelctl`,
  `/usr/local/bin/modelctl`, `/usr/bin/modelctl`) explicitly and uses the
  resolved absolute path for the probe and commit commands. A new
  `--remote-modelctl PATH` option overrides discovery for custom installs.

## 0.10.0

### Added

- Add `push` (alias `sync-remote`) to copy one active NAS model into another
  host's Hugging Face cache over ssh. A JSON handshake against the remote
  probe resolves the canonical cache path, a resumable
  `rsync --files-from` transfer moves only the object's selected files over
  `ssh -o Compression=no`, and a remote `receive-cache` commit validates every
  transferred file against its retained Hugging Face ETag before publishing
  blobs, snapshot links, refs, and a local registration atomically. The remote
  cache directory defaults to the same path as the local host, and `--host`
  can target an interconnect interface such as ConnectX-7.
- Factor the `sync-local` publication tail into a shared
  `publish_synced_staging` used by both local cache sync and the remote
  receive path.

## 0.9.8

### Added

- Add an atomically maintained `catalog.json` whose model projection mirrors
  `list --json`, plus `catalog status`, `catalog refresh`, and `catalog path`
  commands for read-only integrations.

### Fixed

- Serialize catalog refreshes across concurrent activations, preserve the prior
  valid catalog after publication failure, and detect dirty or externally stale
  catalog state through active-reference fingerprints.

## 0.9.7

### Added

- Add dry-run-first `staging-audit`, `cleanup-staging`, `objects-audit`, and
  `gc-objects` commands for measuring and explicitly removing unpublished or
  unreachable store data.

### Fixed

- Exclude declared GGUF companions from primary quantization validation, retain
  successful activation history across failed update retries, and tolerate CIFS
  servers that retain only an empty quarantine directory after active repair.
- Report post-rsync validation and publication progress during `sync-local`,
  and avoid hashing newly published cache blobs a redundant second time.

## 0.9.6

### Added

- Add `doctor`, dry-run-first `repair-active`, and explicit
  `cleanup-quarantine` workflows for auditing and safely repairing malformed
  active references.

### Fixed

- Verify temporary and published active symlinks, restore the previous active
  reference after activation failure, and probe symlink support for direct
  downloads as well as queues.
- Warn when human-readable listings skip malformed active references while
  preserving machine-readable JSON output.

## 0.9.5

### Fixed

- `sync-local` now accepts a Hugging Face repository id and resolves it to a
  unique active NAS model name.

## 0.9.4

### Fixed

- Ignore hidden entries and non-symlinks, including macOS `.DS_Store` files,
  when listing active models.

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
