from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import yaml

from .errors import ModelctlError
from .generation import generate_manifest_document, infer_name, parse_hf_source
from .layout import Layout, atomic_symlink
from .manifest import validate_name


class DownloadQueueError(ModelctlError):
    pass


@dataclass(frozen=True)
class DownloadRequest:
    source: str
    name: str | None = None
    revision: str | None = None
    quantization: str | None = None
    runtime: str = "auto"
    force: bool = False


@dataclass(frozen=True)
class PreparedDownload:
    request: DownloadRequest
    document: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.document["name"])


@dataclass(frozen=True)
class DownloadResult:
    request: DownloadRequest
    manifest_path: Path | None = None
    active_path: Path | None = None
    error: Exception | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


_ALLOWED_KEYS = {
    "source",
    "name",
    "revision",
    "quantization",
    "runtime",
    "force",
}
_ALLOWED_RUNTIMES = {"auto", "vllm", "llama.cpp"}


def _optional_string(item: dict[str, Any], key: str, index: int) -> str | None:
    value = item.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise DownloadQueueError(
            f"queue entry {index} field {key!r} must be a non-empty string"
        )
    return value.strip()


def load_download_queue(path: Path) -> list[DownloadRequest]:
    """Load and strictly validate the local YAML queue structure."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DownloadQueueError(f"cannot read download queue {path}: {exc}") from exc

    if not isinstance(document, dict):
        raise DownloadQueueError(
            "download queue root must be a mapping with a 'downloads' list"
        )
    unknown_root = set(document).difference({"downloads"})
    if unknown_root:
        names = ", ".join(sorted(str(key) for key in unknown_root))
        raise DownloadQueueError(f"unknown queue field(s): {names}")
    items = document.get("downloads")
    if not isinstance(items, list) or not items:
        raise DownloadQueueError(
            "download queue field 'downloads' must be a non-empty list"
        )

    requests: list[DownloadRequest] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise DownloadQueueError(f"queue entry {index} must be a mapping")
        unknown = set(item).difference(_ALLOWED_KEYS)
        if unknown:
            names = ", ".join(sorted(str(key) for key in unknown))
            raise DownloadQueueError(
                f"queue entry {index} has unknown field(s): {names}"
            )
        source = _optional_string(item, "source", index)
        if source is None:
            raise DownloadQueueError(f"queue entry {index} requires source")
        runtime = _optional_string(item, "runtime", index) or "auto"
        if runtime not in _ALLOWED_RUNTIMES:
            raise DownloadQueueError(
                f"queue entry {index} runtime must be auto, vllm, or llama.cpp"
            )
        force = item.get("force", False)
        if not isinstance(force, bool):
            raise DownloadQueueError(
                f"queue entry {index} field 'force' must be true or false"
            )
        requests.append(
            DownloadRequest(
                source=source,
                name=_optional_string(item, "name", index),
                revision=_optional_string(item, "revision", index),
                quantization=_optional_string(item, "quantization", index),
                runtime=runtime,
                force=force,
            )
        )
    return requests


def _effective_names(requests: list[DownloadRequest]) -> list[str]:
    names: list[str] = []
    first_entry: dict[str, int] = {}
    duplicates: list[str] = []
    for index, request in enumerate(requests, start=1):
        try:
            repo, _ = parse_hf_source(request.source, request.revision)
            name = validate_name(request.name) if request.name else infer_name(repo)
        except ModelctlError as exc:
            raise DownloadQueueError(f"queue entry {index}: {exc}") from exc
        previous = first_entry.get(name)
        if previous is not None:
            duplicates.append(
                f"entries {previous} and {index} resolve to {name!r}"
            )
        else:
            first_entry[name] = index
        names.append(name)
    if duplicates:
        details = "; ".join(duplicates)
        raise DownloadQueueError(
            "duplicate queue model names: "
            f"{details}. Set a unique 'name' for every quantization or draft model."
        )
    return names


def validate_queue_root(root: Path) -> None:
    """Verify the store can create the atomic active references downloads need."""
    layout = Layout(root)
    try:
        layout.prepare()
    except OSError as exc:
        raise DownloadQueueError(
            f"cannot prepare model store root {root}: {exc}"
        ) from exc

    probe = layout.active / f"modelctl-symlink-probe-{os.getpid()}-{uuid4().hex}"
    try:
        atomic_symlink(layout.models, probe)
        target = probe.resolve(strict=True)
        if not probe.is_symlink() or target != layout.models.resolve(strict=True):
            raise OSError(
                "atomic symlink probe did not resolve to the models directory"
            )
    except OSError as exc:
        hint = (
            " CIFS mounts commonly require the 'mfsymlinks' mount option."
            if getattr(exc, "errno", None) in {1, 13, 45, 95}
            else ""
        )
        raise DownloadQueueError(
            f"model store root {root} does not support required atomic symlinks: "
            f"{exc}.{hint}"
        ) from exc
    finally:
        probe.unlink(missing_ok=True)


def prepare_download_queue(
    requests: list[DownloadRequest],
    *,
    jobs: int = 1,
    generator: Callable[..., dict[str, Any]] | None = None,
) -> list[PreparedDownload]:
    """Validate every remote selection before any model download starts."""
    if jobs < 1:
        raise DownloadQueueError("jobs must be at least 1")
    if not requests:
        raise DownloadQueueError("download queue is empty")
    effective_names = _effective_names(requests)
    generator = generator or generate_manifest_document

    def prepare(
        item: tuple[int, DownloadRequest],
    ) -> tuple[dict[str, Any] | None, Exception | None]:
        index, request = item
        try:
            document = generator(
                request.source,
                name=effective_names[index],
                revision=request.revision,
                quantization=request.quantization,
                runtime=request.runtime,
            )
            if (
                not isinstance(document, dict)
                or document.get("name") != effective_names[index]
            ):
                raise DownloadQueueError(
                    "manifest generator returned an invalid document"
                )
            return document, None
        except Exception as exc:
            return None, exc

    worker_count = min(jobs, len(requests))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        outcomes = list(executor.map(prepare, enumerate(requests)))

    errors = []
    prepared = []
    for index, (request, outcome) in enumerate(zip(requests, outcomes), start=1):
        document, error = outcome
        if error is not None:
            errors.append(
                f"entry {index} ({effective_names[index - 1]}): "
                f"{type(error).__name__}: {error}"
            )
        else:
            assert document is not None
            prepared.append(PreparedDownload(request, document))
    if errors:
        raise DownloadQueueError("queue preflight failed:\n  " + "\n  ".join(errors))
    return prepared


def validate_prepared_manifests(
    root: Path, prepared: list[PreparedDownload]
) -> None:
    """Reject existing manifest conflicts before any queue worker starts."""
    errors = []
    for index, item in enumerate(prepared, start=1):
        path = root / "manifests" / f"{item.name}.yaml"
        if not path.exists() or item.request.force:
            continue
        try:
            existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            errors.append(f"entry {index} ({item.name}): cannot read {path}: {exc}")
            continue
        if existing != item.document:
            errors.append(
                f"entry {index} ({item.name}): manifest already exists with "
                f"different settings at {path}; set force: true to replace it"
            )
    if errors:
        raise DownloadQueueError(
            "queue manifest preflight failed:\n  " + "\n  ".join(errors)
        )


def execute_download_queue(
    root: Path,
    prepared: list[PreparedDownload],
    *,
    jobs: int = 1,
    downloader: Callable[..., tuple[Path, Path]] | None = None,
) -> list[DownloadResult]:
    """Execute a queue only after prepare_download_queue has succeeded."""
    if jobs < 1:
        raise DownloadQueueError("jobs must be at least 1")
    if not prepared:
        return []

    def download(item: PreparedDownload) -> DownloadResult:
        request = item.request
        try:
            if downloader is None:
                from .generation import download_generated_manifest

                manifest_path, active_path = download_generated_manifest(
                    root,
                    item.document,
                    force_manifest=request.force,
                )
            else:
                manifest_path, active_path = downloader(
                    root,
                    request.source,
                    name=request.name,
                    revision=request.revision,
                    quantization=request.quantization,
                    runtime=request.runtime,
                    force_manifest=request.force,
                )
            return DownloadResult(request, manifest_path, active_path)
        except Exception as exc:
            return DownloadResult(request, error=exc)

    worker_count = min(jobs, len(prepared))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(download, prepared))


def run_download_queue(
    root: Path,
    requests: list[DownloadRequest],
    *,
    jobs: int = 1,
    generator: Callable[..., dict[str, Any]] | None = None,
    downloader: Callable[..., tuple[Path, Path]] | None = None,
) -> list[DownloadResult]:
    """Validate the root and all entries, then execute the prepared queue."""
    validate_queue_root(root)
    prepared = prepare_download_queue(requests, jobs=jobs, generator=generator)
    validate_prepared_manifests(root, prepared)
    return execute_download_queue(root, prepared, jobs=jobs, downloader=downloader)
