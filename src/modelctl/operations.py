from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .errors import ModelctlError, ValidationError
from .generation import parse_hf_source
from .hf_cache import (
    cached_entrypoint,
    delete_record,
    list_records,
    load_record,
    sync_cache,
)
from .hub import download_snapshot, estimate_snapshot, resolve_commit
from .layout import (
    Layout,
    assert_same_filesystem,
    atomic_symlink,
    model_lock,
    verify_symlink,
)
from .manifest import ModelManifest, validate_name
from .state import StateJournal, UpdateState
from .validation import (
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


def _reference_kind(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "directory"
    if path.exists():
        return "file"
    return "missing"


def _activate_reference(
    layout: Layout,
    name: str,
    target: Path,
    journal: StateJournal,
) -> None:
    reference = layout.active_path(name)
    previous: Path | None = None
    if os.path.lexists(reference):
        if not reference.is_symlink():
            error = f"active reference path is a {_reference_kind(reference)}: {reference}"
            journal.transition(
                UpdateState.FAILED_TO_ACTIVATE,
                object=str(target),
                reference=str(reference),
                observed_type=_reference_kind(reference),
                rollback="not-needed",
                error=error,
            )
            raise ModelctlError(
                f"{error}; run 'modelctl doctor' and 'modelctl repair-active'"
            )
        previous, _ = _active_object(layout, name)

    try:
        atomic_symlink(target, reference)
        verify_symlink(reference, target, managed_root=layout.models)
    except BaseException as exc:
        rollback = "not-needed"
        rollback_error: str | None = None
        try:
            if previous is not None:
                try:
                    verify_symlink(reference, previous, managed_root=layout.models)
                    rollback = "previous-reference-intact"
                except ModelctlError:
                    if os.path.lexists(reference):
                        if reference.is_dir() and not reference.is_symlink():
                            failed = layout.staging / ".failed-references" / name
                            failed.parent.mkdir(parents=True, exist_ok=True)
                            if os.path.lexists(failed):
                                failed = failed.with_name(
                                    f"{failed.name}-{os.getpid()}"
                                )
                            os.rename(reference, failed)
                        else:
                            reference.unlink(missing_ok=True)
                    atomic_symlink(previous, reference)
                    verify_symlink(reference, previous, managed_root=layout.models)
                    rollback = "previous-reference-restored"
            elif os.path.lexists(reference):
                if reference.is_dir() and not reference.is_symlink():
                    failed = layout.staging / ".failed-references" / name
                    failed.parent.mkdir(parents=True, exist_ok=True)
                    if os.path.lexists(failed):
                        failed = failed.with_name(f"{failed.name}-{os.getpid()}")
                    os.rename(reference, failed)
                    rollback = f"unexpected-reference-quarantined:{failed}"
                else:
                    reference.unlink(missing_ok=True)
                    rollback = "invalid-reference-removed"
        except Exception as recovery_exc:
            rollback = "failed"
            rollback_error = f"{type(recovery_exc).__name__}: {recovery_exc}"

        details: dict[str, Any] = {
            "object": str(target),
            "reference": str(reference),
            "observed_type": _reference_kind(reference),
            "rollback": rollback,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if rollback_error is not None:
            details["rollback_error"] = rollback_error
        journal.transition(UpdateState.FAILED_TO_ACTIVATE, **details)
        raise


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

        journal.transition(
            UpdateState.UPDATING_REFERENCE,
            reference=str(layout.active_path(manifest.name)),
        )
        _activate_reference(layout, manifest.name, final, journal)
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
        if reference.name.startswith(".") or not reference.is_symlink():
            continue
        try:
            object_path, metadata = _active_object(layout, reference.name)
        except (ModelctlError, ValidationError, OSError):
            continue
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


def delete_local(root: Path, name: str) -> Path:
    return delete_record(root, name)


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


def _resolve_active_name(source_root: Path, selector: str) -> str:
    if "/" not in selector:
        return validate_name(selector)

    repo, _ = parse_hf_source(selector)
    matches = [
        model.name for model in list_active_models(source_root) if model.repo == repo
    ]
    if not matches:
        raise ModelctlError(
            f"Hugging Face repository {repo!r} has no active NAS model; "
            "run 'modelctl list' to see active model names"
        )
    if len(matches) > 1:
        names = ", ".join(repr(name) for name in matches)
        raise ModelctlError(
            f"Hugging Face repository {repo!r} is active under multiple model "
            f"names: {names}; pass one of those names"
        )
    return matches[0]


def sync_local(
    source_root: Path,
    local_root: Path,
    name: str,
    *,
    rsync: str = "rsync",
    runner: Callable[..., Any] = subprocess.run,
) -> Path:
    resolved_name = _resolve_active_name(source_root, name)
    return sync_cache(
        source_root, local_root, resolved_name, rsync=rsync, runner=runner
    )


def list_cached_models(cache_dir: Path) -> list[ActiveModel]:
    models = []
    for record in list_records(cache_dir):
        metadata = record.metadata
        profile = runtime_from_metadata(metadata)
        entrypoint = str(metadata["entrypoint"])
        path = record.snapshot if entrypoint == "." else record.snapshot / entrypoint
        models.append(
            ActiveModel(
                record.name,
                record.repo,
                record.revision,
                record.commit,
                str(metadata.get("format", "")),
                profile.kind,
                entrypoint,
                path.resolve(strict=True),
            )
        )
    return models


def local_active_entrypoint(cache_dir: Path, name: str) -> Path:
    return cached_entrypoint(cache_dir, name)


def delete_cached(cache_dir: Path, name: str) -> Path:
    return delete_record(cache_dir, name)


def serve_cached_command(cache_dir: Path, name: str) -> str:
    record = load_record(cache_dir, name)
    metadata = record.metadata
    entrypoint = str(metadata["entrypoint"])
    path = (
        record.snapshot if entrypoint == "." else record.snapshot / entrypoint
    ).resolve(strict=True)
    profile = runtime_from_metadata(metadata)
    substitutions = {"path": str(path), "name": name, "object": str(record.snapshot)}
    custom = []
    for argument in profile.args:
        rendered = argument
        for key, value in substitutions.items():
            rendered = rendered.replace("{" + key + "}", value)
        custom.append(rendered)
    has_path = any("{path}" in argument or "{object}" in argument for argument in profile.args)
    companions = metadata.get("companions", {})
    companion_args = []
    if profile.kind == "vllm":
        base = [profile.executable, "serve", str(path)]
    elif profile.kind in {"llama.cpp", "llama"}:
        base = [profile.executable, "--model", str(path)]
        if companions.get("mmproj"):
            companion_args.extend(
                [
                    "--mmproj",
                    str((record.snapshot / companions["mmproj"]).resolve(strict=True)),
                ]
            )
        if companions.get("mtp"):
            companion_args.extend(
                [
                    "--model-draft",
                    str((record.snapshot / companions["mtp"]).resolve(strict=True)),
                    "--spec-type",
                    "draft-mtp",
                ]
            )
    else:
        if not has_path:
            raise ModelctlError(f"runtime {profile.kind!r} must include {{path}} in runtime.args")
        base = [profile.executable]
    argv = (
        [profile.executable, *custom, *companion_args]
        if has_path
        else [*base, *companion_args, *custom]
    )
    return shlex.join(argv)
