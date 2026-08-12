# Repository Guidelines

## Project overview

`modelctl` is a Python 3.11+ CLI for atomically downloading, validating,
publishing, activating, and synchronizing Hugging Face model snapshots. The
package uses a `src/` layout and exposes the `modelctl` command through
`modelctl.cli:main`.

## Repository map

- `src/modelctl/cli.py`: argument parsing, command dispatch, and user-facing
  output.
- `src/modelctl/generation.py`: Hugging Face source inspection and manifest
  generation.
- `src/modelctl/download_queue.py`: queue parsing, validation, and concurrent
  download orchestration.
- `src/modelctl/manifest.py`: immutable manifest models and parsing.
- `src/modelctl/operations.py`: update, activation, local sync, deletion, and
  serve-command workflows.
- `src/modelctl/layout.py`: managed-store paths, locks, and atomic symlinks.
- `src/modelctl/validation.py`: artifact checks and object metadata.
- `src/modelctl/state.py`: durable state-transition journals.
- `src/modelctl/cards.py`: model-card synchronization.
- `tests/`: pytest tests organized by the corresponding module.
- `README.md`: user documentation and CLI examples.
- `CHANGELOG.md`: release history.

## Development commands

Use `uv` and run commands from the repository root:

```bash
uv sync --extra test
uv run pytest
uv run pytest tests/test_operations.py
uv run pytest tests/test_operations.py::test_name
uv run modelctl --help
uv build
```

Run the smallest relevant test selection while iterating, then run the full
suite before handing off a change. Build the package when changing packaging,
versioning, entry points, or release metadata.

## Implementation conventions

- Follow the existing standard-library-first style: type annotations,
  `pathlib.Path`, frozen dataclasses for value objects, and explicit custom
  exceptions from `errors.py`.
- Preserve public CLI behavior. Human-readable output goes to stdout,
  actionable failures go through the existing `ModelctlError` handling, and
  `--json` output must remain machine-readable.
- Keep argument parsing and presentation in `cli.py`; put filesystem and
  domain behavior in the focused modules listed above.
- Pass external effects behind injectable callables where practical. Existing
  tests use fake Hugging Face APIs, snapshot functions, and subprocess runners
  and should not require network access.
- Validate untrusted manifest values and paths before filesystem use. Managed
  names must remain constrained, paths must be relative where required, and
  resolved active references must stay inside the managed model store.
- Do not add dependencies unless the standard library and current
  dependencies are insufficient. Declare runtime and test dependencies in
  `pyproject.toml` and refresh `uv.lock` together.
- Keep README examples, CLI help/epilog text, and behavior synchronized when a
  user-visible command or option changes. Add a changelog entry for notable
  user-facing changes.

## Atomicity and safety invariants

These are core product guarantees, not implementation details:

- Downloads remain in `.staging` until estimation, download, artifact
  validation, metadata writing, and object validation all succeed.
- Publish immutable objects before switching `active/NAME`; update the active
  reference last and atomically.
- Keep staging and published objects on the same filesystem so publication can
  use an atomic rename.
- A failed update must leave the previously active revision usable.
- Preserve resumable staging data after interrupted downloads.
- Validate existing content-addressed objects before reusing them.
- Journal state transitions around externally visible workflow stages.
- Local synchronization must exclude transient Hugging Face cache data,
  validate the copied object, and switch the local active reference last.
- Never follow an active symlink outside the managed `models/` tree. Deletion
  must not remove shared objects or NAS content when operating on a local
  store.
- Generate server commands only; `modelctl` must not start inference servers.
  Construct argv as a list and shell-quote only at the final display boundary.

When changing these workflows, include a failure-path test proving that
partially completed work cannot become active.

## Testing expectations

- Add or update tests for every behavior change and regression.
- Prefer `tmp_path`, `monkeypatch`, `capsys`, and small fakes over real network,
  NAS, or subprocess activity.
- Assert observable contracts: filesystem layout, symlink target, journal
  state, emitted argv/output, exception type, and preservation of the old
  active object on failure.
- Cover malformed manifests and queues, path traversal, ambiguous selections,
  partial downloads, invalid metadata, and subprocess failures when relevant.
- Keep tests deterministic and independent of user configuration. Override
  environment variables such as `XDG_CONFIG_HOME`, `MODELCTL_ROOT`, and
  `MODELCTL_LOCAL_ROOT` in tests that touch configuration.

## Change discipline

- Inspect `git status` before editing and preserve unrelated user changes.
- Keep changes narrowly scoped; avoid opportunistic rewrites of atomic
  filesystem code.
- Do not commit generated artifacts from `dist/`, caches, virtual
  environments, or platform metadata.
- Use concise, imperative commit subjects consistent with the existing
  history.
