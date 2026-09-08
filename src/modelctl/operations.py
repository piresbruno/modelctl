from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from .catalog import (
    bump_catalog_token_locked,
    catalog_lock,
    commit_catalog_delta_locked,
    mark_catalog_dirty_locked,
    project_models,
)
from .errors import ModelctlError, ValidationError
from .generation import parse_hf_source
from .hf_cache import (
    LocalDeleteResult,
    cached_entrypoint,
    delete_record,
    list_records,
    load_record,
    sync_cache,
)
from .hub import download_snapshot, estimate_snapshot, resolve_commit
from .integrity import repair_reference_locked, repair_target
from .layout import (
    Layout,
    assert_same_filesystem,
    atomic_symlink,
    model_lock,
    verify_symlink,
)
from .maintenance import (
    StoreEntryAudit,
    audit_objects,
    audit_staging,
    cleanup_staging,
)
from .manifest import ModelManifest, validate_name
from .state import DeleteState, StateJournal, UpdateState
from .validation import (
    read_metadata,
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
    size_bytes: int


def _disk_usage(directory: Path) -> int:
    """On-disk usage of a directory tree in bytes (du-style).

    Sums ``st_blocks * 512`` per regular file; ``st_size`` is used when a
    filesystem reports zero blocks (some network filesystems do)."""
    total = 0
    stack = [directory]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        stat = entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        raise ModelctlError(
                            f"failed to measure disk usage of {directory}: {exc}"
                        ) from exc
                    if stat.st_mode & 0o170000 == 0o100000:
                        total += stat.st_blocks * 512 if stat.st_blocks else stat.st_size
                    elif stat.st_mode & 0o170000 == 0o040000:
                        stack.append(Path(entry.path))
        except OSError as exc:
            raise ModelctlError(
                f"failed to measure disk usage of {directory}: {exc}"
            ) from exc
    return total


@dataclass(frozen=True)
class DeleteResult:
    """Plan (dry-run) or outcome (--apply) of deleting a model from the store."""

    name: str
    repo: str
    reference: Path
    object: Path
    journals: tuple[Path, ...] = ()
    removed_staging: tuple[StoreEntryAudit, ...] = ()
    retained_staging: tuple[StoreEntryAudit, ...] = ()
    removed_objects: tuple[StoreEntryAudit, ...] = ()
    retained_objects: tuple[StoreEntryAudit, ...] = ()
    pruned_dirs: tuple[Path, ...] = ()
    applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "name": self.name,
            "repo": self.repo,
            "reference": str(self.reference),
            "object": str(self.object),
            "journals": [str(path) for path in self.journals],
            "removed_staging": [item.to_dict() for item in self.removed_staging],
            "retained_staging": [item.to_dict() for item in self.retained_staging],
            "removed_objects": [item.to_dict() for item in self.removed_objects],
            "retained_objects": [item.to_dict() for item in self.retained_objects],
            "pruned_dirs": [str(path) for path in self.pruned_dirs],
        }


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
            if reference.is_dir():
                try:
                    prior_target = repair_target(layout.root, name)
                except (ModelctlError, ValidationError, OSError) as exc:
                    error = (
                        f"active reference path is a "
                        f"{_reference_kind(reference)} that cannot be "
                        f"auto-repaired: {reference}"
                    )
                    journal.transition(
                        UpdateState.FAILED_TO_ACTIVATE,
                        object=str(target),
                        reference=str(reference),
                        observed_type=_reference_kind(reference),
                        rollback="not-needed",
                        error=error,
                    )
                    raise ModelctlError(
                        f"{error}; run 'modelctl doctor' and "
                        "'modelctl repair-active'"
                    ) from exc
                # A process outside modelctl replaced the active symlink with
                # a validated copy; restore the canonical symlink first so the
                # update has a well-formed previous reference to roll back to.
                repair_reference_locked(layout, name, prior_target)
            else:
                error = (
                    f"active reference path is a {_reference_kind(reference)}: "
                    f"{reference}"
                )
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
        with catalog_lock(root):
            mark_catalog_dirty_locked(root, f"activating {manifest.name}")
            _activate_reference(layout, manifest.name, final, journal)
            _fsync_directory(layout.active)
            journal.transition(
                UpdateState.ACTIVE_ON_NAS, object=str(final), commit=commit
            )
            try:
                bump_catalog_token_locked(root)
                metadata = validate_object(
                    final, expected_commit=commit, expected_name=manifest.name
                )
                record = project_models(
                    [_active_model_record(layout, manifest.name, final, metadata)]
                )[0]
                commit_catalog_delta_locked(root, catalog_models, upsert=record)
            except ModelctlError as exc:
                raise ModelctlError(
                    f"model {manifest.name!r} was activated, but {exc}"
                ) from exc
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


def _active_model_record(
    layout: Layout, name: str, object_path: Path, metadata: dict[str, Any]
) -> ActiveModel:
    profile = runtime_from_metadata(metadata)
    entrypoint = metadata["entrypoint"]
    path = object_path if entrypoint == "." else object_path / entrypoint
    return ActiveModel(
        name=name,
        repo=str(metadata.get("repo", "")),
        revision=str(metadata.get("revision", "")),
        commit=str(metadata.get("commit", "")),
        format=str(metadata.get("format", "")),
        runtime=profile.kind,
        entrypoint=entrypoint,
        path=path.resolve(strict=True),
        size_bytes=_disk_usage(object_path),
    )


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
        models.append(
            _active_model_record(layout, reference.name, object_path, metadata)
        )
    return models


def catalog_models(root: Path) -> list[ActiveModel]:
    """Active-model records for catalog generation.

    Like :func:`list_active_models`, but a validated regular directory at the
    active reference (a copy that replaced the symlink outside modelctl) still
    contributes a record. Regenerating the catalog therefore never drops a
    model that 'modelctl doctor' marks repairable; the copied directory passes
    the same evidence checks :func:`repair_target` uses."""
    layout = Layout(root)
    if not layout.active.exists():
        return []
    models = []
    for reference in sorted(layout.active.iterdir(), key=lambda path: path.name):
        name = reference.name
        if name.startswith("."):
            continue
        if reference.is_symlink():
            try:
                object_path, metadata = _active_object(layout, name)
            except (ModelctlError, ValidationError, OSError):
                continue
        elif reference.is_dir():
            try:
                repair_target(root, name)
                metadata = validate_object(reference, expected_name=name)
            except (ModelctlError, ValidationError, OSError):
                continue
            object_path = reference
        else:
            continue
        models.append(_active_model_record(layout, name, object_path, metadata))
    return models


def _active_metadata(
    layout: Layout, name: str, reference: Path
) -> tuple[Path, dict[str, Any]]:
    """Resolve an active reference to its object and metadata without
    revalidating object file content; publish-time validation and
    'modelctl doctor' remain the deep checks."""
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
    return object_path, read_metadata(object_path)


def scan_active_records(root: Path) -> tuple[list[ActiveModel], list[str]]:
    """Cheap live scan of active references for listings.

    Reads each reference's object metadata without revalidating object file
    content, and returns the records plus the names of entries that had to
    be skipped. Listing fallbacks use this so a degraded client view of a
    shared store never rewrites the shared catalog."""
    layout = Layout(root)
    if not layout.active.exists():
        return [], []
    models: list[ActiveModel] = []
    skipped: list[str] = []
    for reference in sorted(layout.active.iterdir(), key=lambda path: path.name):
        name = reference.name
        if name.startswith("."):
            continue
        try:
            if reference.is_symlink():
                object_path, metadata = _active_metadata(layout, name, reference)
            elif reference.is_dir():
                object_path = reference
                metadata = read_metadata(reference)
            else:
                skipped.append(name)
                continue
            models.append(_active_model_record(layout, name, object_path, metadata))
        except (ModelctlError, OSError):
            skipped.append(name)
            continue
    return models, skipped


def delete_local(
    root: Path, name: str, *, keep_data: bool = False
) -> LocalDeleteResult:
    return delete_record(root, name, keep_data=keep_data)


def _validate_delete_target(
    layout: Layout, name: str
) -> tuple[Path, Path, Path, dict[str, Any], PurePosixPath]:
    """Validate the active reference and return (reference, object, repo_dir,
    metadata, repo_parts)."""
    reference = layout.active_path(name)
    if not os.path.lexists(reference):
        raise ModelctlError(
            f"model {name!r} has no active reference at {reference}; "
            "run 'modelctl list' to see active models"
        )
    if not reference.is_symlink():
        raise ModelctlError(
            f"active reference path is a {_reference_kind(reference)}: {reference}; "
            "run 'modelctl doctor' and 'modelctl repair-active'"
        )
    try:
        object_path = reference.resolve(strict=True)
        object_path.relative_to(layout.models.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ModelctlError(
            f"active reference for {name!r} does not resolve inside "
            f"{layout.models}; run 'modelctl doctor'"
        ) from exc
    if not object_path.is_dir():
        raise ModelctlError(
            f"active object for {name!r} is not a directory; "
            "run 'modelctl doctor'"
        )
    metadata = validate_object(object_path, expected_name=name)
    repo = str(metadata.get("repo", ""))
    repo_parts = PurePosixPath(repo).parts
    if (
        not repo
        or len(repo_parts) != 2
        or any(part in {"", ".", ".."} for part in repo_parts)
    ):
        raise ModelctlError(
            f"active object metadata for {name!r} has an unusable repository "
            f"{repo!r}; run 'modelctl doctor'"
        )
    repo_dir = layout.models.joinpath(*repo_parts)
    return reference, object_path, repo_dir, metadata, repo_parts


def _plan_store_removal(
    layout: Layout,
    name: str,
    repo_parts: PurePosixPath,
    *,
    ignore_object_target: Path | None = None,
) -> tuple[
    tuple[StoreEntryAudit, ...],
    tuple[StoreEntryAudit, ...],
    tuple[StoreEntryAudit, ...],
    tuple[StoreEntryAudit, ...],
]:
    """Classify staging data and objects under the model's repository subtree.

    Staging references from *name*'s own update journal and the object target
    of the model's own active reference are ignored because both are removed
    before cleanup, keeping the dry-run plan and the applied outcome
    consistent.
    """
    removable_staging = {"failed_unpublished", "published_duplicate", "orphaned"}
    ignore = {ignore_object_target} if ignore_object_target is not None else set()
    staging: list[StoreEntryAudit] = []
    retained_staging: list[StoreEntryAudit] = []
    for item in audit_staging(layout.root, exclude_names={name}):
        if item.path.relative_to(layout.staging).parts[:2] != repo_parts:
            continue
        if item.status in removable_staging:
            staging.append(item)
        else:
            retained_staging.append(item)
    objects: list[StoreEntryAudit] = []
    retained_objects: list[StoreEntryAudit] = []
    for item in audit_objects(layout.root, ignore_active_targets=ignore):
        if item.path.relative_to(layout.models).parts[:2] != repo_parts:
            continue
        if item.status == "unreferenced" and item.name is not None:
            objects.append(item)
        else:
            retained_objects.append(item)
    return (
        tuple(staging),
        tuple(retained_staging),
        tuple(objects),
        tuple(retained_objects),
    )


def delete_model(root: Path, name: str, *, apply: bool = False) -> DeleteResult:
    """Plan or perform deletion of *name* from the managed model store.

    Dry-run (default) classifies everything that would be removed without
    touching the store. With ``apply`` the active reference is removed first
    (the inverse of the update flow's active-last rule), then the catalog is
    refreshed, then staging data and now-unreferenced objects of the model's
    repository are removed. A failure after deactivation leaves the store
    consistent and the data recoverable with the existing audit/gc commands;
    the delete journal at ``state/NAME.delete.json`` is retained as evidence.
    """
    validate_name(name)
    layout = Layout(root)
    layout.prepare()
    with model_lock(layout, name):
        reference, object_path, repo_dir, metadata, repo_parts = (
            _validate_delete_target(layout, name)
        )
        repo = str(metadata.get("repo", ""))
        journals = (
            layout.state_path(name),
            layout.state / f"{name}.delete.json",
        )
        plan = _plan_store_removal(
            layout,
            name,
            repo_parts,
            ignore_object_target=object_path.resolve(strict=True),
        )
        if not apply:
            return DeleteResult(
                name=name,
                repo=repo,
                reference=reference,
                object=object_path,
                journals=journals,
                removed_staging=plan[0],
                retained_staging=plan[1],
                removed_objects=plan[2],
                retained_objects=plan[3],
                applied=False,
            )

        journal = StateJournal(journals[1], "delete")
        journal.transition(
            DeleteState.PLANNED,
            reference=str(reference),
            object=str(object_path),
            repo=repo,
            staging=len(plan[0]),
            objects=len(plan[2]),
        )

        with catalog_lock(root):
            mark_catalog_dirty_locked(root, f"deleting {name}")
            reference.unlink()
            _fsync_directory(layout.active)
            journal.transition(
                DeleteState.REFERENCE_REMOVED,
                reference=str(reference),
                object=str(object_path),
            )
            try:
                bump_catalog_token_locked(root)
                commit_catalog_delta_locked(root, catalog_models, remove=name)
            except ModelctlError as exc:
                raise ModelctlError(
                    f"model {name!r} was deactivated, but {exc}"
                ) from exc
        journal.transition(DeleteState.CATALOG_REFRESHED)

        journals[0].unlink(missing_ok=True)

        staging, retained_staging, objects, retained_objects = _plan_store_removal(
            layout, name, repo_parts
        )
        removed_staging: list[StoreEntryAudit] = []
        if staging:
            selections = [
                str(item.path.relative_to(layout.staging)) for item in staging
            ]
            removed_staging = list(cleanup_staging(root, selections, apply=True))
        journal.transition(
            DeleteState.STAGING_REMOVED,
            removed=len(removed_staging),
            retained=len(retained_staging),
        )

        removed_objects: list[StoreEntryAudit] = []
        for candidate in objects:
            current = {
                item.path.resolve(strict=True): item for item in audit_objects(root)
            }.get(candidate.path.resolve(strict=True))
            if current is None or current.status != "unreferenced":
                raise ModelctlError(
                    f"object status changed while deleting: {candidate.path}"
                )
            validate_object(candidate.path, expected_name=candidate.name)
            try:
                shutil.rmtree(candidate.path)
            except OSError as exc:
                journal.transition(
                    DeleteState.FAILED,
                    object=str(candidate.path),
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise ModelctlError(
                    f"failed to remove object {candidate.path}: {exc}; the model "
                    "is deactivated and its data remains recoverable with "
                    "'modelctl objects-audit' and 'modelctl gc-objects'"
                ) from exc
            removed_objects.append(candidate)
        journal.transition(
            DeleteState.OBJECTS_REMOVED,
            removed=len(removed_objects),
            retained=len(retained_objects),
        )

        pruned_dirs: list[Path] = []
        for directory in (repo_dir, repo_dir.parent):
            if directory == layout.models or not directory.is_dir():
                continue
            try:
                directory.rmdir()
            except OSError:
                continue
            pruned_dirs.append(directory)

        journal.transition(DeleteState.COMPLETE)
        journals[1].unlink(missing_ok=True)
        return DeleteResult(
            name=name,
            repo=repo,
            reference=reference,
            object=object_path,
            journals=journals,
            removed_staging=tuple(removed_staging),
            retained_staging=retained_staging,
            removed_objects=tuple(removed_objects),
            retained_objects=retained_objects,
            pruned_dirs=tuple(pruned_dirs),
            applied=True,
        )


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


def resolve_active_name(source_root: Path, selector: str) -> str:
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
    progress: Callable[[str], None] | None = None,
) -> Path:
    resolved_name = resolve_active_name(source_root, name)
    return sync_cache(
        source_root,
        local_root,
        resolved_name,
        rsync=rsync,
        runner=runner,
        progress=progress,
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
                0,
            )
        )
    return models


def local_active_entrypoint(cache_dir: Path, name: str) -> Path:
    return cached_entrypoint(cache_dir, name)


def delete_cached(
    cache_dir: Path, name: str, *, keep_data: bool = False
) -> LocalDeleteResult:
    return delete_record(cache_dir, name, keep_data=keep_data)


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
