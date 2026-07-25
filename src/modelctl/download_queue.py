from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from .errors import ModelctlError


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
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DownloadQueueError(f"cannot read download queue {path}: {exc}") from exc

    if isinstance(document, dict):
        unknown_root = set(document).difference({"downloads"})
        if unknown_root:
            names = ", ".join(sorted(str(key) for key in unknown_root))
            raise DownloadQueueError(f"unknown queue field(s): {names}")
        items = document.get("downloads")
    else:
        items = document

    if not isinstance(items, list) or not items:
        raise DownloadQueueError(
            "download queue must be a non-empty list or a mapping with a downloads list"
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


def run_download_queue(
    root: Path,
    requests: list[DownloadRequest],
    *,
    jobs: int = 1,
    downloader: Callable[..., tuple[Path, Path]] | None = None,
) -> list[DownloadResult]:
    if jobs < 1:
        raise DownloadQueueError("jobs must be at least 1")
    if not requests:
        return []
    if downloader is None:
        from .generation import download_from_hf

        downloader = download_from_hf

    def download(request: DownloadRequest) -> DownloadResult:
        try:
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

    worker_count = min(jobs, len(requests))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(download, requests))
