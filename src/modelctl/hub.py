from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .errors import ModelctlError
from .manifest import ModelManifest
from .validation import ExpectedFile


def resolve_commit(manifest: ModelManifest, api: Any | None = None) -> str:
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    info = api.model_info(manifest.repo, revision=manifest.revision)
    commit = getattr(info, "sha", None)
    if not isinstance(commit, str) or not commit or any(c in commit for c in "/\\"):
        raise ModelctlError(
            f"Hugging Face did not resolve {manifest.repo}@{manifest.revision} to a commit"
        )
    return commit


def estimate_snapshot(
    manifest: ModelManifest,
    commit: str,
    staging: Path,
    snapshot: Callable[..., Any] | None = None,
) -> list[ExpectedFile]:
    if snapshot is None:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download
    staging.mkdir(parents=True, exist_ok=True)
    result = snapshot(
        repo_id=manifest.repo,
        revision=commit,
        allow_patterns=list(manifest.include) or None,
        local_dir=staging,
        dry_run=True,
    )
    if not isinstance(result, (list, tuple)):
        result = [result]
    expected: list[ExpectedFile] = []
    for item in result:
        filename = getattr(item, "filename", None)
        size = getattr(item, "file_size", None)
        if not isinstance(filename, str):
            raise ModelctlError("Hugging Face dry-run returned an invalid filename")
        expected.append(ExpectedFile(filename, int(size) if size is not None else None))
    expected.sort(key=lambda item: item.path)
    return expected


def download_snapshot(
    manifest: ModelManifest,
    commit: str,
    staging: Path,
    snapshot: Callable[..., Any] | None = None,
) -> None:
    if snapshot is None:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download
    snapshot(
        repo_id=manifest.repo,
        revision=commit,
        allow_patterns=list(manifest.include) or None,
        local_dir=staging,
    )
