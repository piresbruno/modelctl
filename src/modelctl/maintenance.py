from __future__ import annotations

import json
import os
import shutil
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator
from uuid import uuid4

from .errors import ModelctlError, ValidationError
from .layout import Layout, model_lock
from .validation import METADATA_FILE, validate_object


@dataclass(frozen=True)
class StoreEntryAudit:
    path: Path
    status: str
    bytes: int
    name: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload


def _tree_bytes(root: Path) -> int:
    total = 0
    for directory, _, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for filename in filenames:
            path = base / filename
            try:
                stat = path.lstat()
            except OSError:
                continue
            allocated = getattr(stat, "st_blocks", 0) * 512
            total += allocated or stat.st_size
    return total


def _object_directories(root: Path) -> Iterator[Path]:
    if not root.is_dir():
        return
    for owner in sorted(root.iterdir(), key=lambda path: path.name):
        if owner.name.startswith(".") or not owner.is_dir() or owner.is_symlink():
            continue
        for repository in sorted(owner.iterdir(), key=lambda path: path.name):
            if not repository.is_dir() or repository.is_symlink():
                continue
            for object_path in sorted(repository.iterdir(), key=lambda path: path.name):
                if object_path.is_dir() and not object_path.is_symlink():
                    yield object_path


def _current_staging_references(layout: Layout) -> dict[Path, tuple[str, str]]:
    references: dict[Path, tuple[str, str]] = {}
    if not layout.state.is_dir():
        return references
    for state_path in layout.state.glob("*.json"):
        try:
            document = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if document.get("operation") != "update":
            continue
        history = document.get("history")
        if not isinstance(history, list):
            continue
        event = next(
            (
                item
                for item in reversed(history)
                if isinstance(item, dict) and isinstance(item.get("staging"), str)
            ),
            None,
        )
        if event is None:
            continue
        path = Path(event["staging"])
        if not path.is_absolute():
            continue
        try:
            resolved = path.resolve(strict=False)
            resolved.relative_to(layout.staging.resolve(strict=True))
        except (OSError, ValueError):
            continue
        references[resolved] = (state_path.stem, str(document.get("state", "")))
    return references


def audit_staging(root: Path) -> list[StoreEntryAudit]:
    layout = Layout(root)
    references = _current_staging_references(layout)
    results: list[StoreEntryAudit] = []
    for path in _object_directories(layout.staging):
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(layout.staging.resolve(strict=True))
        published = layout.models / relative
        name: str | None = None
        state = ""
        if resolved in references:
            name, state = references[resolved]
        if published.is_dir():
            status = "published_duplicate"
            detail = f"canonical object exists at {published}"
        elif state == "PARTIAL_RESUMABLE":
            status = "resumable"
            detail = "latest update journal marks this download resumable"
        elif state == "FAILED_UNPUBLISHED":
            status = "failed_unpublished"
            detail = "latest update journal records failed validation"
        elif name is not None:
            status = "journal_referenced"
            detail = f"latest update journal state is {state or 'unknown'}"
        else:
            status = "orphaned"
            detail = "no current update journal references this staging path"
        results.append(StoreEntryAudit(path, status, _tree_bytes(path), name, detail))
    return results


def _safe_selected_path(base: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 3:
        raise ModelctlError(
            f"expected a three-part path relative to {base}: {value!r}"
        )
    candidate = base.joinpath(*relative.parts)
    try:
        candidate.resolve(strict=True).relative_to(base.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ModelctlError(f"unsafe or missing store path: {candidate}") from exc
    if candidate.is_symlink() or not candidate.is_dir():
        raise ModelctlError(f"store path is not a regular directory: {candidate}")
    return candidate


def cleanup_staging(
    root: Path, selections: list[str], *, apply: bool = False
) -> list[StoreEntryAudit]:
    if not selections:
        raise ModelctlError("cleanup-staging requires at least one audited relative path")
    layout = Layout(root)
    layout.prepare()
    selected = {
        _safe_selected_path(layout.staging, value).resolve(strict=True): value
        for value in selections
    }
    results_by_path = {
        item.path.resolve(strict=True): item for item in audit_staging(root)
    }
    results: list[StoreEntryAudit] = []
    allowed = {"failed_unpublished", "published_duplicate", "orphaned"}
    for path in selected:
        item = results_by_path.get(path)
        if item is None:
            raise ModelctlError(f"path is not an auditable staging object: {path}")
        if item.status not in allowed:
            raise ModelctlError(
                f"refusing to remove {item.status} staging data: {item.path}"
            )
        lock = model_lock(layout, item.name) if item.name is not None else nullcontext()
        with lock:
            current = {
                entry.path.resolve(strict=True): entry for entry in audit_staging(root)
            }.get(path)
            if current is None or current.status != item.status:
                raise ModelctlError(f"staging status changed while cleaning: {path}")
            if apply:
                quarantine = (
                    layout.staging / ".cleanup-quarantine" / f"{uuid4().hex}"
                )
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                os.rename(path, quarantine)
                if quarantine.exists():
                    shutil.rmtree(quarantine)
            results.append(current)
    return results


def _active_object_paths(layout: Layout) -> set[Path]:
    targets: set[Path] = set()
    if not layout.active.is_dir():
        return targets
    models = layout.models.resolve(strict=True)
    for reference in layout.active.iterdir():
        if not reference.is_symlink():
            continue
        try:
            target = reference.resolve(strict=True)
            target.relative_to(models)
        except (OSError, ValueError):
            continue
        targets.add(target)
    return targets


def audit_objects(root: Path) -> list[StoreEntryAudit]:
    layout = Layout(root)
    active = _active_object_paths(layout)
    results: list[StoreEntryAudit] = []
    for path in _object_directories(layout.models):
        resolved = path.resolve(strict=True)
        try:
            metadata = validate_object(resolved)
            name = metadata.get("name")
            if not isinstance(name, str):
                name = None
        except (ModelctlError, ValidationError, OSError) as exc:
            results.append(
                StoreEntryAudit(
                    path,
                    "invalid_object",
                    _tree_bytes(path),
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        status = "active" if resolved in active else "unreferenced"
        detail = "active reference target" if status == "active" else "no active reference"
        results.append(StoreEntryAudit(path, status, _tree_bytes(path), name, detail))
    return results


def cleanup_objects(
    root: Path, selections: list[str], *, apply: bool = False
) -> list[StoreEntryAudit]:
    if not selections:
        raise ModelctlError("gc-objects requires at least one audited relative path")
    layout = Layout(root)
    layout.prepare()
    selected = {
        _safe_selected_path(layout.models, value).resolve(strict=True): value
        for value in selections
    }
    audited = {item.path.resolve(strict=True): item for item in audit_objects(root)}
    results: list[StoreEntryAudit] = []
    for path in selected:
        item = audited.get(path)
        if item is None or item.status != "unreferenced" or item.name is None:
            status = item.status if item is not None else "unknown"
            raise ModelctlError(f"refusing to remove {status} object: {path}")
        with model_lock(layout, item.name):
            current = {
                entry.path.resolve(strict=True): entry for entry in audit_objects(root)
            }.get(path)
            if current is None or current.status != "unreferenced":
                raise ModelctlError(f"object status changed while cleaning: {path}")
            validate_object(path, expected_name=item.name)
            if apply:
                shutil.rmtree(path)
            results.append(current)
    return results
