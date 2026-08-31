from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from .errors import ModelctlError
from .operations import list_active_models, resolve_active_name, sync_local


class SyncQueueError(ModelctlError):
    pass


@dataclass(frozen=True)
class SyncQueueEntry:
    """One queue entry as written in the file: an active model name or a
    unique Hugging Face repository id."""

    identifier: str


@dataclass(frozen=True)
class PreparedSync:
    """An entry resolved against the source root to an active model name."""

    entry: SyncQueueEntry
    name: str


@dataclass(frozen=True)
class SyncQueueResult:
    entry: SyncQueueEntry
    name: str
    snapshot: Path | None = None
    error: Exception | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


def load_sync_queue(path: Path) -> list[SyncQueueEntry]:
    """Load a sync queue file: a YAML sequence of model identifiers.

    Entries are plain strings naming active NAS models (or unique Hugging
    Face repository ids). The file is read eagerly and fully validated
    before callers resolve anything against a store."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SyncQueueError(f"queue file not found: {path}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise SyncQueueError(f"cannot read queue file {path}: {exc}") from exc
    if document is None:
        raise SyncQueueError(f"queue file {path} is empty")
    if not isinstance(document, list):
        raise SyncQueueError(
            f"queue file {path} must be a YAML list of model names"
        )
    entries: list[SyncQueueEntry] = []
    for index, item in enumerate(document):
        if not isinstance(item, str) or not item.strip():
            raise SyncQueueError(
                f"queue file {path} entry {index + 1} must be a non-empty "
                "model name"
            )
        entries.append(SyncQueueEntry(item.strip()))
    if not entries:
        raise SyncQueueError(f"queue file {path} contains no entries")
    return entries


def prepare_sync_queue(
    source_root: Path, entries: list[SyncQueueEntry]
) -> list[PreparedSync]:
    """Resolve every entry against the source root before any transfer.

    All entries must resolve to distinct active models or nothing is
    prepared; the returned names are safe to sync concurrently."""
    active_names = {model.name for model in list_active_models(source_root)}
    prepared: list[PreparedSync] = []
    errors: list[str] = []
    resolved: dict[str, str] = {}
    for entry in entries:
        try:
            name = resolve_active_name(source_root, entry.identifier)
        except ModelctlError as exc:
            errors.append(f"{entry.identifier}: {exc}")
            continue
        if name not in active_names:
            errors.append(
                f"{entry.identifier}: model {name!r} has no active NAS "
                "reference; run 'modelctl list' to see active models"
            )
            continue
        if name in resolved:
            errors.append(
                f"{entry.identifier}: resolves to {name!r}, already selected "
                f"by {resolved[name]!r}"
            )
            continue
        resolved[name] = entry.identifier
        prepared.append(PreparedSync(entry, name))
    if errors:
        raise SyncQueueError(
            "sync queue preflight failed: " + "; ".join(errors)
        )
    return prepared


def _quiet_runner(base: Callable[..., object]) -> Callable[..., object]:
    """Wrap a runner so concurrent rsync progress does not interleave, and
    surface the tail of rsync stderr on failure."""

    def runner(command, check):
        try:
            return base(command, check=check, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or ""
            tail = [line for line in stderr.strip().splitlines() if line][-3:]
            detail = f": {' | '.join(tail)}" if tail else ""
            raise ModelctlError(
                f"rsync failed (exit {exc.returncode}){detail}"
            ) from exc

    return runner


def execute_sync_queue(
    source_root: Path,
    cache_dir: Path,
    prepared: list[PreparedSync],
    *,
    jobs: int = 1,
    rsync: str = "rsync",
    runner: Callable[..., object] = subprocess.run,
    progress: Callable[[str], None] | None = None,
) -> list[SyncQueueResult]:
    """Sync every prepared entry, continuing after individual failures.

    Results are returned in queue order regardless of completion order.
    With ``jobs > 1`` output is captured instead of streamed so concurrent
    transfers do not interleave progress lines."""
    if jobs < 1:
        raise SyncQueueError("jobs must be at least 1")
    if not prepared:
        return []

    effective_runner = runner
    effective_progress = progress
    if jobs > 1:
        effective_runner = _quiet_runner(runner)
        effective_progress = None

    def sync(item: PreparedSync) -> SyncQueueResult:
        try:
            snapshot = sync_local(
                source_root,
                cache_dir,
                item.name,
                rsync=rsync,
                runner=effective_runner,
                progress=effective_progress,
            )
            return SyncQueueResult(item.entry, item.name, snapshot=snapshot)
        except Exception as exc:
            return SyncQueueResult(item.entry, item.name, error=exc)

    worker_count = min(jobs, len(prepared))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(sync, prepared))


def run_sync_queue(
    source_root: Path,
    cache_dir: Path,
    entries: list[SyncQueueEntry],
    *,
    jobs: int = 1,
    rsync: str = "rsync",
    runner: Callable[..., object] = subprocess.run,
    progress: Callable[[str], None] | None = None,
) -> list[SyncQueueResult]:
    """Preflight the whole queue, then execute it."""
    prepared = prepare_sync_queue(source_root, entries)
    return execute_sync_queue(
        source_root,
        cache_dir,
        prepared,
        jobs=jobs,
        rsync=rsync,
        runner=runner,
        progress=progress,
    )
