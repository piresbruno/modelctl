from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator

from huggingface_hub.utils import WeakFileLock

from .errors import ModelctlError, ValidationError
from .layout import Layout, _fsync_directory
from .manifest import validate_name
from .validation import ExpectedFile, validate_object

_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_ETAG_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_RECORD_SCHEMA = 1
_REFS_SCHEMA = 1
RECEIVE_PROTO = 1


@dataclass(frozen=True)
class CacheRecord:
    name: str
    repo: str
    revision: str
    commit: str
    metadata: dict[str, Any]
    files: dict[str, str]
    snapshot: Path


@dataclass(frozen=True)
class LocalDeleteResult:
    """Outcome of ``modelctl delete-local``.

    ``removed`` lists the cache paths deleted during this run and ``retained``
    explains, in human-readable form, anything deliberately kept (data shared
    with another registration, resumable staging, or stale registrations
    whose data cannot be identified safely)."""

    name: str
    record: Path
    snapshot: Path | None
    removed: tuple[Path, ...]
    retained: tuple[str, ...]


@dataclass(frozen=True)
class LocalCacheIssue:
    """One mismatch between a store object and a local Hugging Face cache."""

    check: str  # "file-set" | "blob-name" | "content-hash" | "refs"
    path: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"check": self.check, "path": self.path, "detail": self.detail}


@dataclass(frozen=True)
class LocalCacheRegistration:
    """Outcome of ``modelctl register-local``.

    ``verified`` means the local cache entry is equivalent to the active
    store object; ``registered`` means a record was actually written (never
    under ``--dry-run``, and only when every check passed)."""

    name: str
    repo: str
    revision: str
    commit: str
    cache: Path
    repository: Path
    snapshot: Path
    entrypoint: Path
    dry_run: bool
    verified: bool
    registered: bool
    selected_files: int
    lfs_files: int
    hashed_files: int
    hashed_bytes: int
    issues: tuple[LocalCacheIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": "register-local",
            "name": self.name,
            "repo": self.repo,
            "revision": self.revision,
            "commit": self.commit,
            "cache": str(self.cache),
            "repository": str(self.repository),
            "snapshot": str(self.snapshot),
            "entrypoint": str(self.entrypoint),
            "dry_run": self.dry_run,
            "verified": self.verified,
            "registered": self.registered,
            "files": {
                "selected": self.selected_files,
                "lfs_by_name": self.lfs_files,
                "hashed": self.hashed_files,
                "hash_bytes": self.hashed_bytes,
            },
            "issues": [issue.to_dict() for issue in self.issues],
        }


def canonical_cache(cache_dir: Path) -> Path:
    return cache_dir.expanduser().absolute().resolve(strict=False)


def _state_base() -> Path:
    selected = os.environ.get("XDG_STATE_HOME")
    return (
        Path(selected).expanduser()
        if selected
        else Path.home() / ".local" / "state"
    )


def state_root(cache_dir: Path) -> Path:
    cache = canonical_cache(cache_dir)
    identity = hashlib.sha256(os.fsencode(cache)).hexdigest()[:20]
    return _state_base() / "modelctl" / "hf-cache" / identity


def _repo_parts(repo: str) -> tuple[str, ...]:
    path = PurePosixPath(repo)
    if (
        path.is_absolute()
        or len(path.parts) not in {1, 2}
        or any(part in {"", ".", ".."} or "--" in part or "\\" in part for part in path.parts)
    ):
        raise ValidationError(f"unsafe Hugging Face repository id: {repo!r}")
    return path.parts


def repo_path(cache_dir: Path, repo: str) -> Path:
    parts = _repo_parts(repo)
    return canonical_cache(cache_dir) / f"models--{'--'.join(parts)}"


def _cache_staging_path(repository: Path, commit: str, files: dict[str, str]) -> Path:
    return repository / ".modelctl-staging" / f"{commit}-{_selection(files)}"


def staging_path_for(
    cache_dir: Path, repo: str, commit: str, files: dict[str, str]
) -> Path:
    """Deterministic cache staging path for a pending transfer.

    ``sync_cache`` and the SSH push protocol derive the same path from the same
    metadata, so the receiving side can verify exactly where transferred data
    was placed before publishing it.
    """
    return _cache_staging_path(
        repo_path(canonical_cache(cache_dir), repo), commit, files
    )


def _safe_relative(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or any(part in {"", ".", ".."} or "\\" in part for part in path.parts)
    ):
        raise ValidationError(f"unsafe {label}: {value!r}")
    return path


def _ensure_real_directory(path: Path, *, create: bool = True) -> None:
    if path.is_symlink():
        raise ValidationError(f"cache structural path must not be a symlink: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValidationError(f"cache structural path is not a directory: {path}")



def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _repo_lock(cache_dir: Path, repo: str) -> Path:
    identity = hashlib.sha256(repo.encode()).hexdigest()[:20]
    return state_root(cache_dir) / "locks" / f"repo-{identity}.lock"


def _name_lock(cache_dir: Path, name: str) -> Path:
    return state_root(cache_dir) / "locks" / f"name-{validate_name(name)}.lock"


def _record_path(cache_dir: Path, name: str) -> Path:
    return state_root(cache_dir) / "active" / f"{validate_name(name)}.json"


def _journal_path(cache_dir: Path, name: str) -> Path:
    return state_root(cache_dir) / "state" / f"{validate_name(name)}.json"


def _transition(
    cache_dir: Path,
    name: str,
    state: str,
    *,
    operation: str = "hf-cache-sync",
    **details: Any,
) -> None:
    path = _journal_path(cache_dir, name)
    history: list[dict[str, Any]] = []
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw.get("history"), list):
                history = raw["history"]
        except (OSError, ValueError):
            history = []
    history.append(
        {"state": state, "at": datetime.now(timezone.utc).isoformat(), **details}
    )
    _atomic_json(
        path,
        {"operation": operation, "state": state, "history": history},
    )


def _metadata_path(source: Path, filename: str) -> Path:
    relative = _safe_relative(filename, "repository file")
    return source / ".cache" / "huggingface" / "download" / Path(
        *relative.parts
    ).with_suffix(Path(relative.name).suffix + ".metadata")


def _read_etag(source: Path, filename: str, commit: str) -> str:
    path = _metadata_path(source, filename)
    try:
        with path.open(encoding="utf-8") as handle:
            metadata_commit = handle.readline().strip()
            etag = handle.readline().strip()
            timestamp = float(handle.readline().strip())
    except (OSError, ValueError) as exc:
        raise ValidationError(
            f"missing or invalid Hugging Face metadata for {filename!r} at {path}"
        ) from exc
    if metadata_commit != commit:
        raise ValidationError(
            f"Hugging Face metadata commit for {filename!r} is "
            f"{metadata_commit!r}, expected {commit!r}"
        )
    if not _ETAG_RE.fullmatch(etag):
        raise ValidationError(f"unsupported Hugging Face ETag for {filename!r}: {etag!r}")
    file_path = source.joinpath(*_safe_relative(filename, "repository file").parts)
    if file_path.stat().st_mtime > timestamp + 1:
        raise ValidationError(f"stale Hugging Face metadata for {filename!r}")
    return etag.lower()


def read_source_etag(source: Path, filename: str, commit: str) -> str:
    """Read the retained Hugging Face ETag for one managed object file."""
    return _read_etag(source, filename, commit)


def _sha256_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_digest(path: Path) -> str:
    """Git blob hash (``blob <size>\\0`` prefix + SHA-1) of a file's content."""
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_for_etag(path: Path, etag: str) -> str:
    if len(etag) == 64:
        return _sha256_digest(path)
    return _git_blob_digest(path)


def _verify_etag(path: Path, etag: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValidationError(f"cache blob is not a regular file: {path}")
    actual = _digest_for_etag(path, etag)
    if actual != etag:
        raise ValidationError(
            f"content hash mismatch for {path}: expected {etag}, got {actual}"
        )


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _selection(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]


def _write_file_list(path: Path, filenames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0".join(name.encode() for name in filenames) + b"\0")


def write_transfer_file_list(path: Path, filenames: list[str]) -> None:
    """Write a NUL-delimited file list for rsync ``--files-from --from0``."""
    _write_file_list(path, filenames)


def _snapshot_pointer(snapshot: Path, filename: str) -> Path:
    relative = _safe_relative(filename, "repository file")
    parent = snapshot
    for part in relative.parts[:-1]:
        parent = parent / part
        if parent.is_symlink():
            raise ValidationError(f"snapshot directory must not be a symlink: {parent}")
        parent.mkdir(exist_ok=True)
        if not parent.is_dir():
            raise ValidationError(f"snapshot path is not a directory: {parent}")
    try:
        parent.resolve(strict=True).relative_to(snapshot.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ValidationError(f"snapshot path escapes its revision: {parent}") from exc
    return parent / relative.name


def _atomic_symlink_text(target: str, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(f".{link.name}.tmp-{os.getpid()}")
    try:
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(target)
        os.replace(temporary, link)
        _fsync_directory(link.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _refs_path(cache_dir: Path) -> Path:
    return state_root(cache_dir) / "refs.json"


def _load_ref_ownership(cache_dir: Path) -> dict[str, dict[str, str]]:
    path = _refs_path(cache_dir)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelctlError(f"cannot read modelctl cache ref ownership at {path}") from exc
    refs = raw.get("refs")
    if (
        raw.get("schema") != _REFS_SCHEMA
        or not isinstance(refs, dict)
        or any(
            not isinstance(key, str)
            or not isinstance(value, dict)
            or not isinstance(value.get("commit"), str)
            for key, value in refs.items()
        )
    ):
        raise ModelctlError(f"unsupported modelctl cache ref ownership at {path}")
    return refs


def _publish_ref(
    cache_dir: Path, repository: Path, repo: str, revision: str, commit: str
) -> None:
    if _COMMIT_RE.fullmatch(revision):
        return
    relative = _safe_relative(revision, "Hugging Face revision")
    ref = repository / "refs" / Path(*relative.parts)
    key = f"{repo}@{relative.as_posix()}"
    ownership = _load_ref_ownership(cache_dir)
    previous = ownership.get(key)
    current = ref.read_text(encoding="utf-8") if ref.is_file() else None
    may_write = current is None or current == commit
    if previous and current == previous.get("commit"):
        may_write = True
    if not may_write:
        return
    ref.parent.mkdir(parents=True, exist_ok=True)
    temporary = ref.with_name(f".{ref.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(commit)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, ref)
        _fsync_directory(ref.parent)
    finally:
        temporary.unlink(missing_ok=True)
    if current is None or previous:
        ownership[key] = {"commit": commit}
        _atomic_json(
            _refs_path(cache_dir), {"schema": _REFS_SCHEMA, "refs": ownership}
        )


def _record_payload(
    cache_dir: Path,
    name: str,
    metadata: dict[str, Any],
    files: dict[str, str],
    snapshot: Path,
) -> dict[str, Any]:
    return {
        "schema": _RECORD_SCHEMA,
        "cache": str(canonical_cache(cache_dir)),
        "name": name,
        "repo": metadata["repo"],
        "revision": metadata["revision"],
        "commit": metadata["commit"],
        "files": files,
        "snapshot": str(snapshot),
        "metadata": metadata,
    }


def _active_store_object(source_root: Path, name: str) -> tuple[Path, dict[str, Any]]:
    """Resolve an active store object exactly like ``sync-local`` does.

    Returns the object directory and its validated metadata. Deliberately
    stat-only: object metadata never carries per-file SHA-256 digests, so
    ``validate_object`` sizes and contains files without reading content.
    """
    validate_name(name)
    source_layout = Layout(source_root)
    reference = source_layout.active_path(name)
    if not reference.is_symlink():
        raise ModelctlError(f"model {name!r} has no active NAS reference")
    try:
        source = reference.resolve(strict=True)
        source.relative_to(source_layout.models.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ModelctlError("active NAS reference escapes the models directory") from exc
    metadata = validate_object(source, expected_name=name)
    commit = str(metadata.get("commit", ""))
    if not _COMMIT_RE.fullmatch(commit):
        raise ValidationError(
            f"invalid Hugging Face commit in object metadata: {commit!r}"
        )
    return source, metadata


def _store_file_etags(source: Path, metadata: dict[str, Any]) -> dict[str, str]:
    """Read the retained Hugging Face ETag for every selected object file."""
    commit = str(metadata["commit"])
    return {
        item.path: read_source_etag(source, item.path, commit)
        for item in (ExpectedFile.from_dict(item) for item in metadata["files"])
    }


def sync_cache(
    source_root: Path,
    cache_dir: Path,
    name: str,
    *,
    rsync: str = "rsync",
    runner: Callable[..., Any] = subprocess.run,
    progress: Callable[[str], None] | None = None,
) -> Path:
    validate_name(name)
    cache = canonical_cache(cache_dir)
    source, metadata = _active_store_object(source_root, name)
    commit = str(metadata["commit"])
    expected = [ExpectedFile.from_dict(item) for item in metadata["files"]]
    files = _store_file_etags(source, metadata)
    repository = repo_path(cache, str(metadata.get("repo", "")))
    selection = _selection(files)
    staging = staging_path_for(cache, str(metadata.get("repo", "")), commit, files)
    file_list = staging.parent / f".{selection}.files"

    with _lock(_name_lock(cache, name)), _lock(
        _repo_lock(cache, str(metadata["repo"]))
    ):
        for path in (
            cache,
            repository,
            repository / "blobs",
            repository / "snapshots",
            repository / "refs",
        ):
            _ensure_real_directory(path)
        _ensure_real_directory(repository / ".modelctl-staging")
        staging.mkdir(parents=True, exist_ok=True)
        total_bytes = sum(item.size or 0 for item in expected)
        _transition(cache, name, "RSYNC_TO_CACHE_STAGING", staging=str(staging))
        _report(
            progress,
            f"sync-local: transferring {len(files)} files "
            f"({_format_bytes(total_bytes)}) to cache staging",
        )
        write_transfer_file_list(file_list, sorted(files))
        command = [
            rsync,
            "--archive",
            "--partial",
            "--delete",
            "--human-readable",
            "--info=progress2",
            "--from0",
            f"--files-from={file_list}",
            f"{source}/",
            f"{staging}/",
        ]
        try:
            runner(command, check=True)
        except BaseException as exc:
            _transition(cache, name, "PARTIAL_RESUMABLE", error=str(exc))
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise ModelctlError(f"rsync failed; staging is resumable: {exc}") from exc
        finally:
            file_list.unlink(missing_ok=True)
        return publish_synced_staging(
            cache, name, metadata, files, staging, progress=progress
        )


def _verify_snapshot_pointers(
    repository: Path, snapshot: Path, files: dict[str, str]
) -> None:
    """Prove every snapshot pointer is a symlink to exactly its own blob."""
    blobs = repository / "blobs"
    for filename, etag in files.items():
        pointer = snapshot.joinpath(*PurePosixPath(filename).parts)
        try:
            resolved = pointer.resolve(strict=True)
            resolved.relative_to(blobs.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise ValidationError(
                f"snapshot pointer escapes cache blobs: {pointer}"
            ) from exc
        if resolved != (blobs / etag).resolve(strict=True):
            raise ValidationError(f"snapshot pointer has unexpected target: {pointer}")


def _finalize_registration(
    cache: Path,
    name: str,
    metadata: dict[str, Any],
    files: dict[str, str],
    snapshot: Path,
    *,
    operation: str = "hf-cache-sync",
) -> Path:
    """Publish the ref, registration record, and journal for a snapshot.

    Callers must hold the per-name and per-repo cache locks. Never deletes
    anything; staging cleanup stays with the caller. Shared by
    ``publish_synced_staging`` (post-transfer) and ``register_local_cache``
    (no-transfer registration), so both write identical state.
    """
    repository = repo_path(cache, str(metadata["repo"]))
    _verify_snapshot_pointers(repository, snapshot, files)
    _publish_ref(
        cache,
        repository,
        str(metadata["repo"]),
        str(metadata["revision"]),
        str(metadata["commit"]),
    )
    _atomic_json(
        _record_path(cache, name),
        _record_payload(cache, name, metadata, files, snapshot),
    )
    _transition(
        cache,
        name,
        "READY_FOR_SERVICE_RESTART",
        operation=operation,
        snapshot=str(snapshot),
    )
    entrypoint = str(metadata["entrypoint"])
    return (
        snapshot if entrypoint == "." else snapshot.joinpath(*PurePosixPath(entrypoint).parts)
    ).resolve(strict=True)


def publish_synced_staging(
    cache_dir: Path,
    name: str,
    metadata: dict[str, Any],
    files: dict[str, str],
    staging: Path,
    *,
    progress: Callable[[str], None] | None = None,
    label: str = "sync-local",
) -> Path:
    """Validate a fully transferred staging directory and publish it as an HF
    cache snapshot, refs, blobs, and a registration record.

    Callers must hold the per-name and per-repo cache locks. The staging path
    must be the one derived from ``metadata`` and ``files`` so a misplaced or
    tampered transfer cannot be published."""
    cache = canonical_cache(cache_dir)
    repository = repo_path(cache, str(metadata["repo"]))
    commit = str(metadata["commit"])
    derived = _cache_staging_path(repository, commit, files)
    if staging != derived:
        raise ValidationError(
            f"staged transfer path {staging} does not match its derived path {derived}"
        )
    if staging.is_symlink() or not staging.is_dir():
        raise ValidationError(f"staged transfer is not a real directory: {staging}")
    for path in (
        cache,
        repository,
        repository / "blobs",
        repository / "snapshots",
        repository / "refs",
    ):
        _ensure_real_directory(path)
    total_bytes = sum(
        item.size or 0
        for item in (ExpectedFile.from_dict(entry) for entry in metadata["files"])
    )
    _transition(cache, name, "VALIDATING_STAGING", files=len(files), bytes=total_bytes)
    _report(
        progress,
        f"{label}: validating {len(files)} transferred files "
        f"({_format_bytes(total_bytes)})",
    )
    for index, (filename, etag) in enumerate(files.items(), start=1):
        _report(
            progress,
            f"{label}: validating [{index}/{len(files)}] {filename}",
        )
        _verify_etag(staging.joinpath(*PurePosixPath(filename).parts), etag)

    _transition(cache, name, "PUBLISHING_CACHE", files=len(files), bytes=total_bytes)
    _report(progress, f"{label}: publishing verified blobs and snapshot")
    blobs = repository / "blobs"
    hf_locks = cache / ".locks" / repository.name
    _ensure_real_directory(cache / ".locks")
    _ensure_real_directory(hf_locks)
    for index, (filename, etag) in enumerate(files.items(), start=1):
        staged = staging.joinpath(*PurePosixPath(filename).parts)
        blob = blobs / etag
        with WeakFileLock(hf_locks / f"{etag}.lock"):
            if blob.exists():
                _report(
                    progress,
                    f"{label}: checking existing blob "
                    f"[{index}/{len(files)}] {filename}",
                )
                _verify_etag(blob, etag)
                staged.unlink()
            else:
                # The staged file was verified immediately above. Atomic rename
                # preserves that same file while avoiding a second full hash pass.
                os.replace(staged, blob)
                _fsync_directory(blobs)

    snapshot = repository / "snapshots" / commit
    if snapshot.exists():
        _ensure_real_directory(snapshot, create=False)
        for filename, etag in files.items():
            pointer = _snapshot_pointer(snapshot, filename)
            target = os.path.relpath(blobs / etag, start=pointer.parent)
            if pointer.is_symlink():
                try:
                    matches = pointer.resolve(strict=True) == (blobs / etag)
                except OSError:
                    matches = False
                if not matches:
                    raise ValidationError(
                        f"snapshot path already points to different content: {pointer}"
                    )
            elif pointer.exists():
                raise ValidationError(f"snapshot path is not a symlink: {pointer}")
            else:
                _atomic_symlink_text(target, pointer)
    else:
        staged_snapshot = (
            repository / ".modelctl-staging" / f".snapshot-{commit}-{_selection(files)}"
        )
        if staged_snapshot.exists():
            shutil.rmtree(staged_snapshot)
        staged_snapshot.mkdir(parents=True)
        for filename, etag in files.items():
            staged_pointer = staged_snapshot.joinpath(*PurePosixPath(filename).parts)
            final_pointer = snapshot.joinpath(*PurePosixPath(filename).parts)
            staged_pointer.parent.mkdir(parents=True, exist_ok=True)
            staged_pointer.symlink_to(
                os.path.relpath(blobs / etag, start=final_pointer.parent)
            )
        try:
            os.rename(staged_snapshot, snapshot)
            _fsync_directory(snapshot.parent)
        except FileExistsError:
            shutil.rmtree(staged_snapshot)
            raise ModelctlError(
                f"cache snapshot appeared concurrently at {snapshot}; retry sync"
            )

    result = _finalize_registration(cache, name, metadata, files, snapshot)
    shutil.rmtree(staging, ignore_errors=True)
    return result


def _verify_local_cache(
    source: Path,
    cache: Path,
    metadata: dict[str, Any],
    files: dict[str, str],
    *,
    progress: Callable[[str], None] | None = None,
) -> list[LocalCacheIssue]:
    """Compare an active store object with a local cache entry, collecting
    every mismatch instead of failing on the first.

    LFS blobs (64-hex ETags) are proven by blob filename and size only —
    zero content reads. Small non-LFS files are git-blob-hashed on both
    sides. Read-only: nothing is created and no walk has side effects."""
    repository = repo_path(cache, str(metadata["repo"]))
    blobs = repository / "blobs"
    commit = str(metadata["commit"])
    snapshot = repository / "snapshots" / commit
    issues: list[LocalCacheIssue] = []

    revision = str(metadata["revision"])
    if not _COMMIT_RE.fullmatch(revision):
        relative = _safe_relative(revision, "Hugging Face revision")
        ref = repository / "refs" / Path(*relative.parts)
        if not ref.is_file():
            issues.append(
                LocalCacheIssue(
                    "refs",
                    relative.as_posix(),
                    f"local cache has no ref for revision {revision!r} "
                    f"(store commit is {commit!r}); re-download that revision "
                    "or run 'modelctl sync-local'",
                )
            )
        else:
            try:
                local_ref = ref.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise ModelctlError(f"cannot read local ref {ref}: {exc}") from exc
            if local_ref != commit:
                issues.append(
                    LocalCacheIssue(
                        "refs",
                        relative.as_posix(),
                        f"local ref points to {local_ref!r}, "
                        f"store commit is {commit!r}",
                    )
                )

    expected_sizes = {
        item.path: item.size
        for item in (ExpectedFile.from_dict(item) for item in metadata["files"])
    }
    lfs = sum(1 for etag in files.values() if len(etag) == 64)
    _report(
        progress,
        f"register-local: verifying {len(files)} files "
        f"({lfs} by blob identity, {len(files) - lfs} by content hash)",
    )
    for filename, etag in sorted(files.items()):
        relative = _safe_relative(filename, "repository file")
        parent = snapshot
        broken: str | None = None
        for part in relative.parts[:-1]:
            candidate = parent / part
            if candidate.is_symlink():
                broken = f"snapshot directory is a symlink: {candidate}"
                break
            if not candidate.is_dir():
                broken = f"snapshot path component is not a directory: {candidate}"
                break
            parent = candidate
        pointer = parent / relative.name
        if broken is not None:
            issues.append(LocalCacheIssue("file-set", filename, broken))
            continue
        if not pointer.is_symlink() and not pointer.exists():
            issues.append(
                LocalCacheIssue(
                    "file-set", filename, "missing from the local snapshot tree"
                )
            )
            continue
        if not pointer.is_symlink():
            issues.append(
                LocalCacheIssue(
                    "file-set", filename, "snapshot path is not a symlink into blobs"
                )
            )
            continue
        try:
            resolved = pointer.resolve(strict=True)
            resolved.relative_to(blobs.resolve(strict=True))
        except (OSError, ValueError):
            issues.append(
                LocalCacheIssue(
                    "file-set",
                    filename,
                    "snapshot pointer does not resolve inside the cache blobs",
                )
            )
            continue
        blob = blobs / resolved.name
        if not blob.is_file() or blob.is_symlink():
            issues.append(
                LocalCacheIssue(
                    "blob-name", filename, f"blob is not a regular file: {blob}"
                )
            )
            continue
        if resolved.name != etag:
            issues.append(
                LocalCacheIssue(
                    "blob-name",
                    filename,
                    f"local blob filename {resolved.name!r} != store ETag {etag!r}",
                )
            )
            continue
        expected_size = expected_sizes.get(filename)
        if expected_size is not None and blob.stat().st_size != expected_size:
            issues.append(
                LocalCacheIssue(
                    "blob-name",
                    filename,
                    f"local blob size {blob.stat().st_size} != "
                    f"store size {expected_size}",
                )
            )
            continue
        if len(etag) == 64:
            continue
        store_file = source.joinpath(*relative.parts)
        if not store_file.is_file() or store_file.is_symlink():
            issues.append(
                LocalCacheIssue(
                    "content-hash",
                    filename,
                    f"store file is not a regular file: {store_file}",
                )
            )
            continue
        store_hash = _git_blob_digest(store_file)
        blob_hash = _git_blob_digest(blob)
        if store_hash != etag or blob_hash != etag:
            issues.append(
                LocalCacheIssue(
                    "content-hash",
                    filename,
                    f"git-blob hashes disagree: store {store_hash}, "
                    f"local blob {blob_hash}, store ETag {etag}",
                )
            )
    return issues


def register_local_cache(
    source_root: Path,
    cache_dir: Path,
    name: str,
    *,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
) -> LocalCacheRegistration:
    """Register an existing local Hugging Face cache entry without transferring.

    Resolves the active store object like ``sync-local``, locates the local
    ``models--<owner>--<name>`` entry, and verifies equivalence: LFS blobs by
    filename and size (zero content reads), small files by content hash on
    both sides, every store file present in the snapshot tree, and the local
    ref pointing at the store commit. When equivalent, publishes the same
    registration record, ref, and journal ``sync-local`` writes — reusing
    ``_finalize_registration``. Any doubt refuses the registration and leaves
    the cache and any existing record untouched. Re-registering over an
    existing record follows sync-local parity: a verified run overwrites it.
    """
    cache = canonical_cache(cache_dir)
    source, metadata = _active_store_object(source_root, name)
    repo = str(metadata["repo"])
    commit = str(metadata["commit"])
    files = _store_file_etags(source, metadata)
    repository = repo_path(cache, repo)
    snapshot = repository / "snapshots" / commit

    if repository.is_symlink() or not repository.is_dir():
        raise ModelctlError(
            f"Hugging Face cache entry {repository.name!r} is not present in "
            f"{cache}; use 'modelctl sync-local' to transfer it"
        )
    blobs = repository / "blobs"
    if blobs.is_symlink() or not blobs.is_dir():
        raise ModelctlError(
            f"local cache entry {repository.name!r} has no usable blobs "
            f"directory: {blobs}"
        )
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ModelctlError(
            f"local cache entry {repository.name!r} has no snapshot for store "
            f"commit {commit!r}: {snapshot}"
        )

    expected_sizes = {
        item.path: item.size
        for item in (ExpectedFile.from_dict(item) for item in metadata["files"])
    }
    lfs = sum(1 for etag in files.values() if len(etag) == 64)
    entrypoint = str(metadata["entrypoint"])
    would_entrypoint = (
        snapshot
        if entrypoint == "."
        else snapshot.joinpath(*PurePosixPath(entrypoint).parts)
    )

    def result(
        issues: list[LocalCacheIssue],
        registered: bool = False,
        entrypoint_path: Path | None = None,
    ) -> LocalCacheRegistration:
        return LocalCacheRegistration(
            name=name,
            repo=repo,
            revision=str(metadata["revision"]),
            commit=commit,
            cache=cache,
            repository=repository,
            snapshot=snapshot,
            entrypoint=entrypoint_path or would_entrypoint,
            dry_run=dry_run,
            verified=not issues,
            registered=registered,
            selected_files=len(files),
            lfs_files=lfs,
            hashed_files=len(files) - lfs,
            hashed_bytes=sum(
                expected_sizes.get(filename) or 0
                for filename, etag in files.items()
                if len(etag) != 64
            ),
            issues=tuple(issues),
        )

    if dry_run:
        return result(
            _verify_local_cache(source, cache, metadata, files, progress=progress)
        )
    with _lock(_name_lock(cache, name)), _lock(_repo_lock(cache, repo)):
        issues = _verify_local_cache(
            source, cache, metadata, files, progress=progress
        )
        if issues:
            return result(issues)
        _report(progress, f"register-local: verified; registering {name!r}")
        record_path = _record_path(cache, name)
        previous_commit: str | None = None
        if record_path.exists():
            try:
                previous_commit = json.loads(
                    record_path.read_text(encoding="utf-8")
                ).get("commit")
            except (OSError, ValueError):
                previous_commit = None
        _transition(
            cache,
            name,
            "REGISTERING_LOCAL_CACHE",
            operation="hf-cache-register",
            repo=repo,
            commit=commit,
            previous_commit=previous_commit,
        )
        entrypoint_path = _finalize_registration(
            cache, name, metadata, files, snapshot, operation="hf-cache-register"
        )
        return result([], registered=True, entrypoint_path=entrypoint_path)


def receive_staged_cache(
    cache_dir: Path,
    name: str,
    *,
    probe: bool = False,
    staging: Path | None = None,
) -> dict[str, Any]:
    """Remote half of ``modelctl push``.

    With ``probe=True`` this prints a handshake and mutates nothing. With a
    ``staging`` directory it validates the staged transfer against its own
    metadata and publishes blobs, a snapshot, refs, and a registration record
    for the named model."""
    validate_name(name)
    cache = canonical_cache(cache_dir)
    if probe:
        return {"proto": RECEIVE_PROTO, "cache": str(cache)}
    if staging is None:
        raise ModelctlError(
            "receive-cache requires --staging; normally invoked by 'modelctl push'"
        )
    if staging.is_symlink() or not staging.is_dir():
        raise ValidationError(f"staged transfer is not a real directory: {staging}")
    metadata = validate_object(staging, expected_name=name)
    commit = str(metadata.get("commit", ""))
    if not _COMMIT_RE.fullmatch(commit):
        raise ValidationError(
            f"invalid Hugging Face commit in staged metadata: {commit!r}"
        )
    repo = str(metadata.get("repo", ""))
    with _lock(_name_lock(cache, name)), _lock(_repo_lock(cache, repo)):
        files = {
            item.path: _read_etag(staging, item.path, commit)
            for item in (ExpectedFile.from_dict(item) for item in metadata["files"])
        }
        derived = staging_path_for(cache, repo, commit, files)
        if Path(derived) != staging:
            raise ValidationError(
                f"staged transfer path {staging} does not match its metadata; "
                f"expected {derived}: re-run 'modelctl push'"
            )
        entrypoint = publish_synced_staging(
            cache, name, metadata, files, staging, label="receive-cache"
        )
    snapshot = repo_path(cache, repo) / "snapshots" / commit
    return {
        "proto": RECEIVE_PROTO,
        "cache": str(cache),
        "staging": str(staging),
        "snapshot": str(snapshot),
        "entrypoint": str(entrypoint),
    }


def load_record(cache_dir: Path, name: str) -> CacheRecord:
    path = _record_path(cache_dir, name)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelctlError(f"model {name!r} has no valid local cache record at {path}") from exc
    cache = canonical_cache(cache_dir)
    if raw.get("schema") != _RECORD_SCHEMA or raw.get("cache") != str(cache):
        raise ModelctlError(f"invalid or moved local cache record at {path}")
    if raw.get("name") != name or not isinstance(raw.get("files"), dict):
        raise ModelctlError(f"invalid local cache record at {path}")
    repository = repo_path(cache, str(raw.get("repo", "")))
    snapshot = Path(str(raw.get("snapshot", "")))
    for structural in (repository, repository / "snapshots", repository / "blobs"):
        _ensure_real_directory(structural, create=False)
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ModelctlError(
            f"local snapshot is not a real directory: {snapshot} (the cache "
            f"was pruned or moved after registration; run 'modelctl sync-local "
            f"{name}' to re-register the current revision, or 'modelctl "
            "delete-local "
            f"{name}' to drop this registration)"
        )
    try:
        snapshot.resolve(strict=True).relative_to(
            (repository / "snapshots").resolve(strict=True)
        )
    except (OSError, ValueError) as exc:
        raise ModelctlError(f"local snapshot escapes the selected cache: {snapshot}") from exc
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        raise ModelctlError(f"local cache record has invalid metadata at {path}")
    for key in ("name", "repo", "revision", "commit"):
        if metadata.get(key) != raw.get(key):
            raise ModelctlError(f"local cache record metadata mismatch for {key}")
    entrypoint = metadata.get("entrypoint")
    if not isinstance(entrypoint, str):
        raise ModelctlError(f"local cache record has invalid entrypoint at {path}")
    companions = metadata.get("companions", {})
    if not isinstance(companions, dict):
        raise ModelctlError(f"local cache record has invalid companions at {path}")
    blobs = repository / "blobs"
    for filename, etag in raw["files"].items():
        if not isinstance(filename, str) or not isinstance(etag, str):
            raise ModelctlError(f"local cache record has invalid files at {path}")
        pointer = snapshot.joinpath(*_safe_relative(filename, "repository file").parts)
        try:
            resolved = pointer.resolve(strict=True)
            resolved.relative_to(blobs.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise ModelctlError(f"invalid cached snapshot file: {pointer}") from exc
        if resolved != (blobs / etag).resolve(strict=True):
            raise ModelctlError(f"cached snapshot file points to wrong blob: {pointer}")
    if entrypoint != ".":
        entrypoint_path = _safe_relative(entrypoint, "entrypoint").as_posix()
        if entrypoint_path not in raw["files"]:
            raise ModelctlError("cached entrypoint is not in the selected file set")
    for kind, companion in companions.items():
        if not isinstance(kind, str) or not isinstance(companion, str):
            raise ModelctlError(f"local cache record has invalid companions at {path}")
        companion_path = _safe_relative(companion, "companion path").as_posix()
        if companion_path not in raw["files"]:
            raise ModelctlError(f"cached companion is not selected: {companion}")
    return CacheRecord(
        name=name,
        repo=str(raw["repo"]),
        revision=str(raw["revision"]),
        commit=str(raw["commit"]),
        metadata=metadata,
        files=dict(raw["files"]),
        snapshot=snapshot.resolve(strict=True),
    )


def cached_entrypoint(cache_dir: Path, name: str) -> Path:
    record = load_record(cache_dir, name)
    entrypoint = str(record.metadata["entrypoint"])
    return (
        record.snapshot
        if entrypoint == "."
        else record.snapshot.joinpath(*_safe_relative(entrypoint, "entrypoint").parts)
    ).resolve(strict=True)


def malformed_cached_records(cache_dir: Path) -> list[str]:
    """Names of local registrations whose cache data is broken.

    Mirrors ``malformed_active_references`` for NAS stores so local listings
    can warn about skipped records instead of failing."""
    active = state_root(cache_dir) / "active"
    if not active.exists():
        return []
    broken = []
    for path in sorted(active.glob("*.json")):
        try:
            load_record(cache_dir, path.stem)
        except ModelctlError:
            broken.append(path.stem)
    return broken


def list_records(cache_dir: Path) -> list[CacheRecord]:
    """Valid local registrations.

    Records whose snapshot or files are broken (for example the cache content
    was pruned or moved outside modelctl) are skipped so listings never fail
    on foreign or partial cache state."""
    active = state_root(cache_dir) / "active"
    if not active.exists():
        return []
    records = []
    for path in sorted(active.glob("*.json")):
        try:
            records.append(load_record(cache_dir, path.stem))
        except ModelctlError:
            continue
    return records


def _raw_records(cache: Path) -> dict[str, dict[str, Any]]:
    """Best-effort raw read of the remaining local registrations.

    Broken records are included so their pinned files and repositories can
    still be treated as shared instead of being deleted underneath them."""
    active = state_root(cache) / "active"
    records: dict[str, dict[str, Any]] = {}
    if not active.is_dir():
        return records
    for path in sorted(active.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(raw, dict):
            records[path.stem] = raw
    return records


def _shared_names(others: dict[str, dict[str, Any]], repo: str) -> list[str]:
    return sorted(
        other_name
        for other_name, raw in others.items()
        if isinstance(raw.get("repo"), str) and raw["repo"] == repo
    )


def _delete_repository(
    cache: Path, name: str, repo: str, repository: Path
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    """Remove an unshared repository: its snapshots, refs, blobs, and modelctl
    staging data, plus this repository's modelctl ref-ownership entries."""
    retained: list[str] = []
    state_dir = state_root(cache) / "state"
    if state_dir.is_dir():
        for journal in sorted(state_dir.glob("*.json")):
            if journal.stem == name:
                # The caller's own journal is removed after a successful
                # deletion; it does not block removing the data it describes.
                continue
            try:
                text = journal.read_text(encoding="utf-8")
            except OSError:
                continue
            if str(repository) in text:
                retained.append(
                    f"{repository} is referenced by a live or resumable "
                    f"transfer journal ({journal.name})"
                )
                break
    staging = repository / ".modelctl-staging"
    if staging.is_dir() and any(staging.iterdir()):
        retained.append(f"resumable staging data present under {staging}")
    if retained:
        return (), tuple(retained)
    shutil.rmtree(repository)
    _fsync_directory(repository.parent)
    ownership_path = _refs_path(cache)
    if ownership_path.exists():
        try:
            ownership = _load_ref_ownership(cache)
        except ModelctlError:
            retained.append(
                f"modelctl ref ownership at {ownership_path} could not be "
                "read; it may still list refs for the deleted repository"
            )
        else:
            remaining = {
                key: value
                for key, value in ownership.items()
                if not key.startswith(f"{repo}@")
            }
            if remaining != ownership:
                _atomic_json(
                    ownership_path, {"schema": _REFS_SCHEMA, "refs": remaining}
                )
    return (repository,), tuple(retained)


def _delete_pinned_data(
    cache: Path, record: CacheRecord, others: dict[str, dict[str, Any]]
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    """Remove only the data this registration pins from a repository that
    other local registrations still reference."""
    repository = repo_path(cache, record.repo)
    removed: list[Path] = []
    retained: list[str] = []

    sharers = _shared_names(others, record.repo)
    commit_sharers = sorted(
        other_name
        for other_name in sharers
        if others[other_name].get("commit") == record.commit
    )
    snapshot = record.snapshot
    if not commit_sharers and snapshot.is_dir():
        shutil.rmtree(snapshot)
        _fsync_directory(snapshot.parent)
        removed.append(snapshot)
    elif commit_sharers:
        retained.append(
            f"snapshot {snapshot} is still referenced by local "
            f"registration(s): {', '.join(commit_sharers)}"
        )

    if not _COMMIT_RE.fullmatch(record.revision) and not commit_sharers:
        relative = _safe_relative(record.revision, "Hugging Face revision")
        ref = repository / "refs" / Path(*relative.parts)
        try:
            points_here = (
                ref.is_file() and ref.read_text(encoding="utf-8") == record.commit
            )
        except OSError:
            points_here = False
        if points_here:
            ref.unlink()
            _fsync_directory(ref.parent)
            removed.append(ref)

    referenced_blobs: dict[str, list[str]] = {}
    for other_name in sharers:
        files = others[other_name].get("files")
        if not isinstance(files, dict):
            continue
        for etag in files.values():
            if isinstance(etag, str):
                referenced_blobs.setdefault(etag, []).append(other_name)
    for etag in dict.fromkeys(record.files.values()):
        blob = repository / "blobs" / etag
        if not blob.exists():
            continue
        if etag in referenced_blobs:
            retained.append(
                f"blob {blob} is still referenced by local registration(s): "
                f"{', '.join(referenced_blobs[etag])}"
            )
            continue
        blob.unlink()
        _fsync_directory(blob.parent)
        removed.append(blob)

    for directory in (
        repository / "blobs",
        repository / "refs",
        repository / "snapshots",
    ):
        try:
            directory.rmdir()
        except OSError:
            pass
    return tuple(removed), tuple(retained)


def _delete_error(name: str, exc: OSError) -> ModelctlError:
    return ModelctlError(
        f"could not delete local cache data for {name!r}: {exc}; the "
        "registration was kept so the command can be retried. Cache paths "
        "owned by another user (for example a container or service running "
        "as root against this cache) cannot be removed by the current user; "
        "fix ownership with 'sudo chown -R' or remove the leftovers with "
        "'sudo rm -rf', then re-run 'modelctl delete-local'",
    )


def delete_record(
    cache_dir: Path, name: str, *, keep_data: bool = False
) -> LocalDeleteResult:
    """Remove a local cache registration and, by default, its cache data.

    Unshared repositories are removed whole (snapshots, refs, blobs, and
    modelctl staging). Data shared with another local registration is pruned
    to what this registration alone pins, and anything still referenced —
    including resumable staging and stale registrations whose data cannot be
    identified safely — is retained and reported. The registration and
    journal are removed last, so a failed data deletion leaves them in place
    for a retry."""
    cache = canonical_cache(cache_dir)
    with _lock(_name_lock(cache, name)):
        path = _record_path(cache, name)
        if not path.exists():
            raise ModelctlError(
                f"model {name!r} has no local cache record at {path}"
            )
        try:
            record = load_record(cache, name)
        except ModelctlError:
            # A stale registration must still be removable so it stops
            # blocking listings; its data cannot be identified safely, so
            # cache data is never touched.
            record = None
        removed: tuple[Path, ...] = ()
        retained: tuple[str, ...] = ()
        if record is None:
            retained = (
                "registration was stale or malformed; "
                "cache data was left untouched",
            )
        elif keep_data:
            retained = (
                "--keep-data was set; snapshots, refs, and blobs were retained",
            )
        else:
            repository = repo_path(cache, record.repo)
            with _lock(_repo_lock(cache, record.repo)):
                # The record being deleted is unlinked only after a
                # successful data deletion, so exclude it from the share
                # analysis or it would count as its own sharer.
                others = _raw_records(cache)
                others.pop(name, None)
                try:
                    if _shared_names(others, record.repo):
                        removed, retained = _delete_pinned_data(
                            cache, record, others
                        )
                    else:
                        removed, retained = _delete_repository(
                            cache, name, record.repo, repository
                        )
                except OSError as exc:
                    raise _delete_error(name, exc) from exc
        path.unlink()
        _fsync_directory(path.parent)
        journal = _journal_path(cache, name)
        journal.unlink(missing_ok=True)
        if journal.parent.exists():
            _fsync_directory(journal.parent)
    return LocalDeleteResult(
        name=name,
        record=path,
        snapshot=record.snapshot if record is not None else None,
        removed=removed,
        retained=retained,
    )
