from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .errors import ModelctlError, ValidationError
from .hub import download_snapshot, estimate_snapshot, resolve_commit
from .layout import Layout, assert_same_filesystem, atomic_symlink, model_lock
from .manifest import ModelManifest
from .state import StateJournal, SyncState, UpdateState
from .validation import (
    ExpectedFile,
    resolve_entrypoint,
    runtime_from_metadata,
    validate_artifacts,
    validate_expected,
    validate_object,
    write_metadata,
)


@dataclass(frozen=True)
class ActiveModel:
    name: str
    repo: str
    revision: str
    commit: str
    format: str
    runtime: str
    entrypoint: str
    path: Path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def update_model(
    root: Path,
    manifest: ModelManifest,
    *,
    api: Any | None = None,
    snapshot: Callable[..., Any] | None = None,
) -> Path:
    layout = Layout(root)
    layout.prepare()
    journal = StateJournal(layout.state_path(manifest.name), "update")
    with model_lock(layout, manifest.name):
        journal.transition(UpdateState.UNRESOLVED, revision=manifest.revision)
        commit = resolve_commit(manifest, api)
        journal.transition(UpdateState.RESOLVED_TO_COMMIT, commit=commit)
        final = layout.object_path(manifest, commit)
        staging = layout.staging_path(manifest, commit)

        if final.exists():
            try:
                validate_object(
                    final, expected_commit=commit, expected_name=manifest.name
                )
            except ValidationError as exc:
                journal.transition(
                    UpdateState.FAILED_UNPUBLISHED,
                    object=str(final),
                    error=f"existing object is invalid: {exc}",
                )
                raise
            journal.transition(UpdateState.READY_TO_ACTIVATE, object=str(final))
        else:
            journal.transition(
                UpdateState.DOWNLOADING_TO_STAGING, staging=str(staging)
            )
            try:
                expected = estimate_snapshot(
                    manifest, commit, staging, snapshot=snapshot
                )
                download_snapshot(manifest, commit, staging, snapshot=snapshot)
            except BaseException as exc:
                journal.transition(
                    UpdateState.PARTIAL_RESUMABLE,
                    staging=str(staging),
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise

            journal.transition(
                UpdateState.VALIDATING,
                files=len(expected),
                bytes=sum(item.size or 0 for item in expected),
            )
            try:
                validate_expected(staging, expected)
                validate_artifacts(staging, manifest, expected)
                entrypoint = resolve_entrypoint(staging, manifest, expected)
                write_metadata(staging, manifest, commit, expected, entrypoint)
                validate_object(
                    staging, expected_commit=commit, expected_name=manifest.name
                )
            except (ValidationError, OSError) as exc:
                journal.transition(
                    UpdateState.FAILED_UNPUBLISHED,
                    staging=str(staging),
                    error=str(exc),
                )
                raise

            assert_same_filesystem(staging, final)
            journal.transition(UpdateState.PUBLISHING_OBJECT, object=str(final))
            try:
                os.rename(staging, final)
                _fsync_directory(final.parent)
            except FileExistsError:
                validate_object(
                    final, expected_commit=commit, expected_name=manifest.name
                )

        journal.transition(UpdateState.UPDATING_REFERENCE, reference=str(layout.active_path(manifest.name)))
        atomic_symlink(final, layout.active_path(manifest.name))
        _fsync_directory(layout.active)
        journal.transition(UpdateState.ACTIVE_ON_NAS, object=str(final), commit=commit)
        metadata = validate_object(final, expected_commit=commit, expected_name=manifest.name)
        entrypoint = metadata["entrypoint"]
        return (final if entrypoint == "." else final / entrypoint).resolve(strict=True)


def _active_object(layout: Layout, name: str) -> tuple[Path, dict[str, Any]]:
    reference = layout.active_path(name)
    if not reference.is_symlink():
        raise ModelctlError(f"model {name!r} has no active reference at {reference}")
    try:
        object_path = reference.resolve(strict=True)
        object_path.relative_to(layout.models.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ModelctlError(
            f"active reference for {name!r} does not point into {layout.models}"
        ) from exc
    if not object_path.is_dir():
        raise ModelctlError(f"active object for {name!r} is not a directory")
    metadata = validate_object(object_path, expected_name=name)
    return object_path, metadata


def active_entrypoint(root: Path, name: str) -> Path:
    layout = Layout(root)
    object_path, metadata = _active_object(layout, name)
    entrypoint = metadata["entrypoint"]
    path = object_path if entrypoint == "." else object_path / entrypoint
    return path.resolve(strict=True)


def list_active_models(root: Path) -> list[ActiveModel]:
    layout = Layout(root)
    if not layout.active.exists():
        return []
    models = []
    for reference in sorted(layout.active.iterdir(), key=lambda path: path.name):
        object_path, metadata = _active_object(layout, reference.name)
        profile = runtime_from_metadata(metadata)
        entrypoint = metadata["entrypoint"]
        path = object_path if entrypoint == "." else object_path / entrypoint
        models.append(
            ActiveModel(
                name=reference.name,
                repo=str(metadata.get("repo", "")),
                revision=str(metadata.get("revision", "")),
                commit=str(metadata.get("commit", "")),
                format=str(metadata.get("format", "")),
                runtime=profile.kind,
                entrypoint=entrypoint,
                path=path.resolve(strict=True),
            )
        )
    return models


def serve_argv(root: Path, name: str) -> list[str]:
    layout = Layout(root)
    object_path, metadata = _active_object(layout, name)
    entrypoint = metadata["entrypoint"]
    path = (object_path if entrypoint == "." else object_path / entrypoint).resolve(
        strict=True
    )
    profile = runtime_from_metadata(metadata)
    substitutions = {"path": str(path), "name": name, "object": str(object_path)}
    custom = []
    for argument in profile.args:
        rendered = argument
        for key, value in substitutions.items():
            rendered = rendered.replace("{" + key + "}", value)
        custom.append(rendered)
    has_path_placeholder = any(
        "{path}" in argument or "{object}" in argument for argument in profile.args
    )
    companion_args: list[str] = []
    if profile.kind == "vllm":
        base = [profile.executable, "serve", str(path)]
    elif profile.kind in {"llama.cpp", "llama"}:
        base = [profile.executable, "--model", str(path)]
        companions = metadata.get("companions", {})
        mmproj = companions.get("mmproj")
        if mmproj:
            companion_args.extend(
                ["--mmproj", str((object_path / mmproj).resolve(strict=True))]
            )
        mtp = companions.get("mtp")
        if mtp:
            companion_args.extend(
                [
                    "--model-draft",
                    str((object_path / mtp).resolve(strict=True)),
                    "--spec-type",
                    "draft-mtp",
                ]
            )
    else:
        if not has_path_placeholder:
            raise ModelctlError(
                f"runtime {profile.kind!r} must include {{path}} in runtime.args"
            )
        base = [profile.executable]
    if has_path_placeholder:
        return [profile.executable, *custom, *companion_args]
    return [*base, *companion_args, *custom]


def serve_command(root: Path, name: str) -> str:
    return shlex.join(serve_argv(root, name))


def sync_local(
    source_root: Path,
    local_root: Path,
    name: str,
    *,
    rsync: str = "rsync",
    runner: Callable[..., Any] = subprocess.run,
) -> Path:
    source = Layout(source_root)
    local = Layout(local_root)
    local.prepare()
    journal = StateJournal(local.state_path(name), "local-sync")
    with model_lock(local, name):
        journal.transition(SyncState.NAS_REFERENCE_READ, source=str(source_root))
        source_object, source_metadata = _active_object(source, name)
        journal.transition(
            SyncState.SOURCE_VALIDATED, commit=source_metadata.get("commit")
        )
        try:
            relative = source_object.relative_to(source.models.resolve(strict=True))
        except ValueError as exc:
            raise ModelctlError("source object is outside the NAS models directory") from exc
        final = local.models / relative
        staging = local.staging / relative

        if final.exists():
            validate_object(
                final,
                expected_commit=source_metadata.get("commit"),
                expected_name=name,
            )
        else:
            assert_same_filesystem(staging, final)
            staging.mkdir(parents=True, exist_ok=True)
            journal.transition(
                SyncState.RSYNC_TO_LOCAL_STAGING, staging=str(staging)
            )
            command = [
                rsync,
                "--archive",
                "--delete",
                "--partial",
                "--exclude=/.cache/huggingface/",
                f"{source_object}/",
                f"{staging}/",
            ]
            try:
                runner(command, check=True)
            except BaseException as exc:
                journal.transition(
                    SyncState.PARTIAL_RESUMABLE,
                    staging=str(staging),
                    error=f"{type(exc).__name__}: {exc}",
                )
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                raise ModelctlError(f"rsync failed; staging is resumable: {exc}") from exc

            journal.transition(SyncState.LOCAL_VALIDATION, staging=str(staging))
            validate_object(
                staging,
                expected_commit=source_metadata.get("commit"),
                expected_name=name,
            )
            journal.transition(SyncState.LOCAL_OBJECT_PUBLICATION, object=str(final))
            try:
                os.rename(staging, final)
                _fsync_directory(final.parent)
            except FileExistsError:
                validate_object(
                    final,
                    expected_commit=source_metadata.get("commit"),
                    expected_name=name,
                )

        journal.transition(SyncState.LOCAL_REFERENCE_UPDATE, reference=str(local.active_path(name)))
        atomic_symlink(final, local.active_path(name))
        _fsync_directory(local.active)
        journal.transition(
            SyncState.READY_FOR_SERVICE_RESTART,
            object=str(final),
            commit=source_metadata.get("commit"),
        )
        entrypoint = source_metadata["entrypoint"]
        return (final if entrypoint == "." else final / entrypoint).resolve(strict=True)
