from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar
from uuid import uuid4

from .errors import CatalogStaleViewError, ModelctlError
from .layout import Layout

CATALOG_SCHEMA = 3
CATALOG_FILE = "catalog.json"


class CatalogModel(Protocol):
    name: str
    runtime: str
    repo: str
    size_bytes: int


ModelT = TypeVar("ModelT", bound=CatalogModel)


@dataclass(frozen=True)
class CatalogRefresh:
    path: Path
    changed: bool
    generation: int
    models: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload


@dataclass(frozen=True)
class CatalogStatus:
    path: Path
    status: str
    detail: str = ""
    generation: int | None = None
    models: int | None = None
    dirty: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload


def catalog_path(root: Path) -> Path:
    return root / CATALOG_FILE


def _status_path(layout: Layout) -> Path:
    return layout.state / ".catalog.json"


def _dirty_path(layout: Layout) -> Path:
    return layout.state / ".catalog-dirty"


def _token_path(layout: Layout) -> Path:
    return layout.state / ".catalog-seq"


def _lock_path(layout: Layout) -> Path:
    return layout.locks / ".catalog.lock"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _content_hash(models: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json(models)).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir() and not path.is_symlink():
        raise ModelctlError(f"catalog destination is a directory: {path}")
    temp = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"
    try:
        descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp.unlink(missing_ok=True)


def _read_json_no_follow(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ModelctlError(f"missing or invalid catalog at {path}") from exc
    if not isinstance(document, dict):
        raise ModelctlError(f"catalog root must be an object: {path}")
    return document


def _validate_document(path: Path, document: dict[str, Any]) -> dict[str, Any]:
    if document.get("schema") not in (2, CATALOG_SCHEMA):
        raise ModelctlError(f"unsupported catalog schema at {path}")
    generation = document.get("generation")
    generated_at = document.get("generated_at")
    fingerprint = document.get("active_fingerprint")
    digest = document.get("content_sha256")
    seq = document.get("seq")
    models = document.get("models")
    if not isinstance(generation, int) or generation < 1:
        raise ModelctlError(f"catalog has an invalid generation at {path}")
    if not isinstance(generated_at, str) or not generated_at:
        raise ModelctlError(f"catalog has an invalid timestamp at {path}")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ModelctlError(f"catalog has an invalid active fingerprint at {path}")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ModelctlError(f"catalog has an invalid content hash at {path}")
    if document["schema"] >= 3 and (
        not isinstance(seq, int) or isinstance(seq, bool) or seq < 0
    ):
        raise ModelctlError(f"catalog has an invalid mutation token at {path}")
    if not isinstance(models, list):
        raise ModelctlError(f"catalog models must be a list at {path}")
    string_keys = {"name", "runtime", "repository"}
    if any(
        not isinstance(item, dict)
        or set(item) != string_keys | {"bytes"}
        or any(not isinstance(item[key], str) for key in string_keys)
        or not isinstance(item["bytes"], int)
        or isinstance(item["bytes"], bool)
        or item["bytes"] < 0
        for item in models
    ):
        raise ModelctlError(f"catalog contains an invalid model record at {path}")
    names = [item["name"] for item in models]
    if names != sorted(names) or len(names) != len(set(names)):
        raise ModelctlError(f"catalog model names are not unique and sorted at {path}")
    if digest != _content_hash(models):
        raise ModelctlError(f"catalog content hash does not match at {path}")
    return document


def load_catalog(root: Path) -> dict[str, Any]:
    path = catalog_path(root)
    return _validate_document(path, _read_json_no_follow(path))


def project_models(models: Iterable[CatalogModel]) -> list[dict[str, Any]]:
    projected = [
        {
            "name": model.name,
            "runtime": model.runtime,
            "repository": model.repo,
            "bytes": int(model.size_bytes),
        }
        for model in models
    ]
    projected.sort(key=lambda item: item["name"])
    names = [item["name"] for item in projected]
    if len(names) != len(set(names)):
        raise ModelctlError("active model inventory contains duplicate names")
    return projected


def _active_entries(root: Path) -> list[dict[str, str]]:
    """Canonical fingerprint entries for the active tree.

    Symlink values are client-independent: resolvable in-store targets use the
    store-relative object path, everything else uses the raw link text (the
    bytes stored in the symlink), never a client-resolved absolute path."""
    active = Layout(root).active
    entries: list[dict[str, str]] = []
    if active.is_dir():
        for path in sorted(active.iterdir(), key=lambda item: item.name):
            if path.name.startswith("."):
                continue
            try:
                if path.is_symlink():
                    raw = os.readlink(path)
                    target = Path(raw) if os.path.isabs(raw) else active / raw
                    try:
                        resolved = target.resolve()
                    except OSError:
                        kind, value = "broken", raw
                    else:
                        if resolved.exists():
                            kind = "object"
                            try:
                                value = resolved.relative_to(root).as_posix()
                            except ValueError:
                                value = raw
                        else:
                            kind, value = "broken", raw
                elif path.is_dir():
                    kind = "directory"
                    value = ""
                elif path.exists():
                    kind = "file"
                    value = ""
                else:
                    kind = "missing"
                    value = ""
            except OSError as exc:
                kind = "error"
                value = f"{type(exc).__name__}:{exc}"
            entries.append({"name": path.name, "kind": kind, "value": value})
    return entries


def active_fingerprint(root: Path) -> str:
    return hashlib.sha256(_canonical_json(_active_entries(root))).hexdigest()


@contextmanager
def catalog_lock(root: Path) -> Iterator[None]:
    layout = Layout(root)
    layout.locks.mkdir(parents=True, exist_ok=True)
    with _lock_path(layout).open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def mark_catalog_dirty_locked(root: Path, reason: str) -> None:
    layout = Layout(root)
    _atomic_json(
        _dirty_path(layout),
        {"schema": CATALOG_SCHEMA, "at": _utc_now(), "reason": reason},
    )


def _write_status(layout: Layout, status: str, **details: Any) -> None:
    _atomic_json(
        _status_path(layout),
        {"schema": CATALOG_SCHEMA, "status": status, "at": _utc_now(), **details},
    )


def catalog_token(root: Path) -> int:
    """Monotonic store mutation counter used for catalog invalidation.

    Every committed change to ``active/`` bumps the token under the catalog
    lock immediately after the change and before the matching catalog delta.
    Readers compare the token with the ``seq`` stamped in catalog.json; the
    token file is opened fresh on every read (close-to-open consistency), so
    a client whose directory view is stale still observes committed
    mutations. A missing or unreadable token file reads as 0."""
    try:
        document = _read_json_no_follow(_token_path(Layout(root)))
    except ModelctlError:
        return 0
    seq = document.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        return 0
    return seq


def bump_catalog_token_locked(root: Path) -> int:
    """Advance the mutation counter; the caller must hold catalog_lock."""
    token = catalog_token(root) + 1
    _atomic_json(
        _token_path(Layout(root)),
        {"schema": CATALOG_SCHEMA, "seq": token},
    )
    return token


def _load_previous_catalog(
    root: Path, path: Path
) -> dict[str, Any] | None:
    if not os.path.lexists(path):
        return None
    try:
        return load_catalog(root)
    except ModelctlError:
        return None


def _next_generation(previous: dict[str, Any] | None, unchanged: bool) -> int:
    if previous is None:
        return 1
    return (
        int(previous["generation"])
        if unchanged
        else int(previous["generation"]) + 1
    )


def _publish_catalog_document_locked(
    layout: Layout, path: Path, document: dict[str, Any]
) -> None:
    try:
        _atomic_json(path, document)
    except (ModelctlError, OSError) as exc:
        status_error: str | None = None
        try:
            _write_status(
                layout,
                "stale",
                error=f"{type(exc).__name__}: {exc}",
            )
        except (ModelctlError, OSError) as state_exc:
            status_error = f"; status update also failed: {state_exc}"
        raise ModelctlError(
            f"failed to refresh model catalog at {path}: {exc}{status_error or ''}"
        ) from exc


def _finalize_ready_locked(
    layout: Layout, generation: int, count: int, changed: bool
) -> None:
    try:
        _write_status(
            layout,
            "ready",
            generation=generation,
            models=count,
            changed=changed,
        )
        _dirty_path(layout).unlink(missing_ok=True)
    except OSError as exc:
        raise ModelctlError(f"failed to finalize model catalog state: {exc}") from exc


def _refresh_catalog_locked(
    root: Path,
    loader: Callable[[Path], list[ModelT]],
    *,
    preserve_if_empty: bool = False,
) -> tuple[list[ModelT], CatalogRefresh]:
    layout = Layout(root)
    layout.prepare()
    path = catalog_path(root)
    models = loader(root)
    projected = project_models(models)
    fingerprint = active_fingerprint(root)
    digest = _content_hash(projected)
    token = catalog_token(root)
    previous = _load_previous_catalog(root, path)
    if (
        preserve_if_empty
        and not projected
        and previous is not None
        and previous["models"]
    ):
        raise CatalogStaleViewError(
            f"0 active models visible but the catalog lists "
            f"{len(previous['models'])} model(s); refusing to overwrite; the "
            "store view on this client may be degraded (NFS/SMB mount); run "
            "'modelctl doctor'"
        )
    unchanged = (
        previous is not None
        and previous.get("seq") == token
        and previous["content_sha256"] == digest
        and previous["active_fingerprint"] == fingerprint
        and previous["models"] == projected
    )
    generation = _next_generation(previous, unchanged)
    if not unchanged:
        _publish_catalog_document_locked(
            layout,
            path,
            {
                "schema": CATALOG_SCHEMA,
                "generation": generation,
                "generated_at": _utc_now(),
                "active_fingerprint": fingerprint,
                "content_sha256": digest,
                "seq": token,
                "models": projected,
            },
        )
    _finalize_ready_locked(layout, generation, len(projected), not unchanged)
    return models, CatalogRefresh(path, not unchanged, generation, projected)


def commit_catalog_delta_locked(
    root: Path,
    loader: Callable[[Path], list[ModelT]],
    *,
    upsert: dict[str, Any] | None = None,
    remove: str | None = None,
) -> CatalogRefresh:
    """Commit one mutation's catalog delta; the caller holds catalog_lock.

    The caller bumps the mutation token after changing ``active/`` and before
    calling this, so on the happy path the stored catalog is exactly one
    token behind and the delta applies to its stored projection directly —
    the mutator's own record was validated by the mutation itself, and the
    sibling records come from the stored catalog, never from this client's
    view of other models. When the stored catalog is further behind (a
    previous mutation's delta write failed) or missing, the whole store is
    rescanned, and then the mutator's own change must be observable in the
    result: a store view that cannot see a freshly activated model, or that
    still shows a freshly deleted one, is degraded (NFS/SMB caching), and
    the write is refused with the dirty marker left in place so every client
    falls back to its own live scan instead of trusting a poisoned catalog."""
    layout = Layout(root)
    layout.prepare()
    path = catalog_path(root)
    token = catalog_token(root)
    previous = _load_previous_catalog(root, path)
    if (
        (upsert is not None or remove is not None)
        and previous is not None
        and token >= 1
        and previous.get("seq") == token - 1
    ):
        projected = [
            item
            for item in previous["models"]
            if not (remove is not None and item["name"] == remove)
        ]
        if upsert is not None:
            projected = [
                item for item in projected if item["name"] != upsert["name"]
            ]
            projected.append(dict(upsert))
        projected.sort(key=lambda item: item["name"])
    else:
        projected = project_models(loader(root))
        if upsert is not None and not any(
            item["name"] == upsert["name"] for item in projected
        ):
            mark_catalog_dirty_locked(root, "catalog delta refused")
            raise ModelctlError(
                f"activated model {upsert['name']!r} is not visible in this "
                "client's store view; refusing to write the catalog; the "
                "view may be degraded (NFS/SMB mount); run 'modelctl doctor'"
            )
        if remove is not None and any(
            item["name"] == remove for item in projected
        ):
            mark_catalog_dirty_locked(root, "catalog delta refused")
            raise ModelctlError(
                f"deleted model {remove!r} is still visible in this "
                "client's store view; refusing to write the catalog; the "
                "view may be degraded (NFS/SMB mount); run 'modelctl doctor'"
            )
    fingerprint = active_fingerprint(root)
    digest = _content_hash(projected)
    unchanged = (
        previous is not None
        and previous.get("seq") == token
        and previous["content_sha256"] == digest
        and previous["active_fingerprint"] == fingerprint
        and previous["models"] == projected
    )
    generation = _next_generation(previous, unchanged)
    if not unchanged:
        _publish_catalog_document_locked(
            layout,
            path,
            {
                "schema": CATALOG_SCHEMA,
                "generation": generation,
                "generated_at": _utc_now(),
                "active_fingerprint": fingerprint,
                "content_sha256": digest,
                "seq": token,
                "models": projected,
            },
        )
    _finalize_ready_locked(layout, generation, len(projected), not unchanged)
    return CatalogRefresh(path, not unchanged, generation, projected)


def refresh_catalog(
    root: Path,
    loader: Callable[[Path], list[ModelT]],
    *,
    preserve_if_empty: bool = False,
) -> tuple[list[ModelT], CatalogRefresh]:
    with catalog_lock(root):
        return _refresh_catalog_locked(root, loader, preserve_if_empty=preserve_if_empty)


def refresh_catalog_locked(
    root: Path,
    loader: Callable[[Path], list[ModelT]],
    *,
    preserve_if_empty: bool = False,
) -> tuple[list[ModelT], CatalogRefresh]:
    """Refresh while the caller holds :func:`catalog_lock`."""
    return _refresh_catalog_locked(root, loader, preserve_if_empty=preserve_if_empty)


def catalog_dirty(root: Path) -> bool:
    """True when an active-store mutation did not complete catalog refresh."""
    return _dirty_path(Layout(root)).exists()


def catalog_status(root: Path) -> CatalogStatus:
    layout = Layout(root)
    path = catalog_path(root)
    dirty = _dirty_path(layout).exists()
    if not os.path.lexists(path):
        return CatalogStatus(
            path, "missing", "catalog has not been generated", dirty=dirty
        )
    try:
        document = load_catalog(root)
    except ModelctlError as exc:
        return CatalogStatus(path, "invalid", str(exc), dirty=dirty)
    token = catalog_token(root)
    catalog_seq = document.get("seq")
    if dirty:
        return CatalogStatus(
            path,
            "dirty",
            "an active-store mutation has not completed catalog refresh",
            int(document["generation"]),
            len(document["models"]),
            True,
        )
    if catalog_seq != token:
        if catalog_seq is None:
            detail = "catalog predates mutation tracking; run 'modelctl catalog refresh'"
        elif token > catalog_seq:
            detail = (
                f"catalog is {token - catalog_seq} mutation(s) behind the "
                f"store (catalog seq {catalog_seq}, store seq {token})"
            )
        else:
            detail = (
                f"catalog seq {catalog_seq} is ahead of store seq {token}"
            )
        return CatalogStatus(
            path,
            "stale",
            detail,
            int(document["generation"]),
            len(document["models"]),
        )
    if document["active_fingerprint"] != active_fingerprint(root):
        return CatalogStatus(
            path,
            "stale",
            "active references differ from the catalog fingerprint",
            int(document["generation"]),
            len(document["models"]),
        )
    return CatalogStatus(
        path,
        "ready",
        generation=int(document["generation"]),
        models=len(document["models"]),
    )
