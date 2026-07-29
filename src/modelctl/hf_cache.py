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
from .layout import Layout
from .manifest import validate_name
from .validation import ExpectedFile, validate_object

_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_ETAG_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_RECORD_SCHEMA = 1
_REFS_SCHEMA = 1


@dataclass(frozen=True)
class CacheRecord:
    name: str
    repo: str
    revision: str
    commit: str
    metadata: dict[str, Any]
    files: dict[str, str]
    snapshot: Path


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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with temporary.open("rb") as handle:
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


def _transition(cache_dir: Path, name: str, state: str, **details: Any) -> None:
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
        {"operation": "hf-cache-sync", "state": state, "history": history},
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


def _digest_for_etag(path: Path, etag: str) -> str:
    if len(etag) == 64:
        digest = hashlib.sha256()
        prefix = b""
    else:
        digest = hashlib.sha1()
        prefix = f"blob {path.stat().st_size}\0".encode()
    digest.update(prefix)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_etag(path: Path, etag: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValidationError(f"cache blob is not a regular file: {path}")
    actual = _digest_for_etag(path, etag)
    if actual != etag:
        raise ValidationError(
            f"content hash mismatch for {path}: expected {etag}, got {actual}"
        )


def _selection(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]


def _write_file_list(path: Path, filenames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0".join(name.encode() for name in filenames) + b"\0")


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
        temporary.write_text(commit, encoding="utf-8")
        with temporary.open("rb") as handle:
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


def sync_cache(
    source_root: Path,
    cache_dir: Path,
    name: str,
    *,
    rsync: str = "rsync",
    runner: Callable[..., Any] = subprocess.run,
) -> Path:
    validate_name(name)
    cache = canonical_cache(cache_dir)
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
        raise ValidationError(f"invalid Hugging Face commit in object metadata: {commit!r}")
    expected = [ExpectedFile.from_dict(item) for item in metadata["files"]]
    files = {
        item.path: _read_etag(source, item.path, commit)
        for item in expected
    }
    repository = repo_path(cache, str(metadata.get("repo", "")))
    selection = _selection(files)
    staging = repository / ".modelctl-staging" / f"{commit}-{selection}"
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
        _transition(cache, name, "RSYNC_TO_CACHE_STAGING", staging=str(staging))
        _write_file_list(file_list, sorted(files))
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

        _transition(cache, name, "VALIDATING_STAGING")
        for filename, etag in files.items():
            _verify_etag(staging.joinpath(*PurePosixPath(filename).parts), etag)

        blobs = repository / "blobs"
        hf_locks = cache / ".locks" / repository.name
        _ensure_real_directory(cache / ".locks")
        _ensure_real_directory(hf_locks)
        for filename, etag in files.items():
            staged = staging.joinpath(*PurePosixPath(filename).parts)
            blob = blobs / etag
            with WeakFileLock(hf_locks / f"{etag}.lock"):
                if blob.exists():
                    _verify_etag(blob, etag)
                    staged.unlink()
                else:
                    os.replace(staged, blob)
                    _fsync_directory(blobs)
                    _verify_etag(blob, etag)

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
            staged_snapshot = repository / ".modelctl-staging" / f".snapshot-{commit}-{selection}"
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

        for filename, etag in files.items():
            pointer = snapshot.joinpath(*PurePosixPath(filename).parts)
            try:
                resolved = pointer.resolve(strict=True)
                resolved.relative_to(blobs.resolve(strict=True))
            except (OSError, ValueError) as exc:
                raise ValidationError(f"snapshot pointer escapes cache blobs: {pointer}") from exc
            if resolved != (blobs / etag).resolve(strict=True):
                raise ValidationError(f"snapshot pointer has unexpected target: {pointer}")

        _publish_ref(
            cache,
            repository,
            str(metadata["repo"]),
            str(metadata["revision"]),
            commit,
        )
        _atomic_json(
            _record_path(cache, name),
            _record_payload(cache, name, metadata, files, snapshot),
        )
        _transition(cache, name, "READY_FOR_SERVICE_RESTART", snapshot=str(snapshot))
        shutil.rmtree(staging, ignore_errors=True)
        entrypoint = str(metadata["entrypoint"])
        return (
            snapshot if entrypoint == "." else snapshot.joinpath(*PurePosixPath(entrypoint).parts)
        ).resolve(strict=True)


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
        raise ModelctlError(f"local snapshot is not a real directory: {snapshot}")
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


def list_records(cache_dir: Path) -> list[CacheRecord]:
    active = state_root(cache_dir) / "active"
    if not active.exists():
        return []
    return [load_record(cache_dir, path.stem) for path in sorted(active.glob("*.json"))]


def delete_record(cache_dir: Path, name: str) -> Path:
    with _lock(_name_lock(cache_dir, name)):
        record = load_record(cache_dir, name)
        path = _record_path(cache_dir, name)
        path.unlink()
        _fsync_directory(path.parent)
        journal = _journal_path(cache_dir, name)
        journal.unlink(missing_ok=True)
        if journal.parent.exists():
            _fsync_directory(journal.parent)
        return record.snapshot
