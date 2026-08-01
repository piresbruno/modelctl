from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .errors import ModelctlError, ValidationError
from .layout import Layout, atomic_symlink, model_lock, verify_symlink
from .manifest import load_manifest, validate_name
from .state import StateJournal
from .validation import validate_object


@dataclass(frozen=True)
class ActiveReferenceAudit:
    name: str
    status: str
    reference: Path
    object: Path | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reference"] = str(self.reference)
        payload["object"] = str(self.object) if self.object is not None else None
        return payload


@dataclass(frozen=True)
class RepairResult:
    name: str
    status: str
    reference: Path
    object: Path
    quarantine: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("reference", "object", "quarantine"):
            value = payload[key]
            payload[key] = str(value) if value is not None else None
        return payload


def _latest_active_event(layout: Layout, name: str) -> dict[str, Any]:
    path = layout.state_path(name)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelctlError(f"missing or invalid update journal at {path}") from exc
    history = document.get("history")
    if not isinstance(history, list):
        raise ModelctlError(f"update journal has no history: {path}")
    for event in reversed(history):
        if (
            isinstance(event, dict)
            and event.get("state") == "ACTIVE_ON_NAS"
            and isinstance(event.get("object"), str)
            and isinstance(event.get("commit"), str)
        ):
            return event
    raise ModelctlError(f"update journal has no successful activation: {path}")


def _journal_object(layout: Layout, name: str) -> tuple[Path, str]:
    event = _latest_active_event(layout, name)
    raw = Path(event["object"])
    if not raw.is_absolute():
        raise ModelctlError(f"journal object path is not absolute: {raw}")
    try:
        object_path = raw.resolve(strict=True)
        object_path.relative_to(layout.models.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ModelctlError(
            f"journal object for {name!r} is outside the managed model store"
        ) from exc
    return object_path, str(event["commit"])


def repair_target(root: Path, name: str) -> Path:
    """Validate independent evidence and return a safe repair target."""
    validate_name(name)
    layout = Layout(root)
    reference = layout.active_path(name)
    if reference.is_symlink() or not reference.is_dir():
        raise ModelctlError(
            f"active entry for {name!r} is not a regular directory: {reference}"
        )

    active_metadata = validate_object(reference, expected_name=name)
    object_path, commit = _journal_object(layout, name)
    if active_metadata.get("commit") != commit:
        raise ModelctlError(
            f"active directory commit does not match the update journal for {name!r}"
        )
    canonical_metadata = validate_object(
        object_path, expected_name=name, expected_commit=commit
    )
    if active_metadata != canonical_metadata:
        raise ModelctlError(
            f"active directory metadata differs from the canonical object for {name!r}"
        )

    manifest = load_manifest(root, name)
    expected_object = layout.object_path(manifest, commit).resolve(strict=True)
    if expected_object != object_path:
        raise ModelctlError(
            f"manifest-derived object {expected_object} does not match journal object "
            f"{object_path} for {name!r}"
        )
    return object_path


def _audit_symlink(layout: Layout, reference: Path) -> ActiveReferenceAudit:
    name = reference.name
    try:
        resolved = reference.resolve(strict=True)
    except OSError as exc:
        return ActiveReferenceAudit(name, "broken_symlink", reference, detail=str(exc))
    try:
        resolved.relative_to(layout.models.resolve(strict=True))
    except (OSError, ValueError):
        return ActiveReferenceAudit(
            name,
            "outside_store_symlink",
            reference,
            resolved,
            "reference resolves outside the managed models directory",
        )
    try:
        metadata = validate_object(resolved, expected_name=name)
    except (ModelctlError, ValidationError, OSError) as exc:
        return ActiveReferenceAudit(
            name, "invalid_object", reference, resolved, str(exc)
        )
    try:
        expected, commit = _journal_object(layout, name)
    except ModelctlError as exc:
        return ActiveReferenceAudit(
            name, "missing_journal", reference, resolved, str(exc)
        )
    if expected != resolved or metadata.get("commit") != commit:
        return ActiveReferenceAudit(
            name,
            "journal_object_mismatch",
            reference,
            resolved,
            f"journal expects {expected}@{commit}",
        )
    return ActiveReferenceAudit(name, "valid", reference, resolved)


def audit_active_references(root: Path) -> list[ActiveReferenceAudit]:
    layout = Layout(root)
    if not layout.active.exists():
        return []
    results: list[ActiveReferenceAudit] = []
    for reference in sorted(layout.active.iterdir(), key=lambda item: item.name):
        name = reference.name
        if name.startswith("."):
            results.append(
                ActiveReferenceAudit(name, "hidden_foreign_entry", reference)
            )
        elif reference.is_symlink():
            results.append(_audit_symlink(layout, reference))
        elif reference.is_dir():
            try:
                target = repair_target(root, name)
            except (ModelctlError, ValidationError, OSError) as exc:
                results.append(
                    ActiveReferenceAudit(
                        name, "regular_directory", reference, detail=str(exc)
                    )
                )
            else:
                results.append(
                    ActiveReferenceAudit(
                        name,
                        "repairable_directory",
                        reference,
                        target,
                        "validated duplicate of canonical object",
                    )
                )
        else:
            results.append(ActiveReferenceAudit(name, "foreign_entry", reference))
    return results


def malformed_active_references(root: Path) -> list[ActiveReferenceAudit]:
    """Return entries that cannot be safely consumed as active references."""
    return [
        item
        for item in audit_active_references(root)
        if item.status not in {"valid", "hidden_foreign_entry", "missing_journal"}
    ]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _quarantine_path(layout: Layout, name: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        layout.staging
        / ".active-quarantine"
        / name
        / f"{stamp}-{uuid4().hex[:12]}"
    )


def repair_active_reference(root: Path, name: str, *, apply: bool = False) -> RepairResult:
    layout = Layout(root)
    layout.prepare()
    reference = layout.active_path(name)
    with model_lock(layout, name):
        target = repair_target(root, name)
        if not apply:
            return RepairResult(name, "dry-run", reference, target)

        quarantine = _quarantine_path(layout, name)
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        repair_journal = StateJournal(
            layout.state / "repairs" / f"{name}-{uuid4().hex}.json",
            "repair-active",
        )
        repair_journal.transition(
            "AUDITED", reference=str(reference), object=str(target)
        )
        os.rename(reference, quarantine)
        _fsync_directory(reference.parent)
        repair_journal.transition("QUARANTINED", quarantine=str(quarantine))
        try:
            atomic_symlink(target, reference)
            verify_symlink(reference, target, managed_root=layout.models)
            _fsync_directory(reference.parent)
        except BaseException as exc:
            rollback_error: str | None = None
            try:
                if os.path.lexists(reference):
                    if reference.is_dir() and not reference.is_symlink():
                        failed = quarantine.with_name(
                            f"{quarantine.name}-failed-publication"
                        )
                        os.rename(reference, failed)
                    else:
                        reference.unlink(missing_ok=True)
                os.rename(quarantine, reference)
                _fsync_directory(reference.parent)
            except Exception as recovery_exc:
                rollback_error = f"{type(recovery_exc).__name__}: {recovery_exc}"
            repair_journal.transition(
                "ROLLED_BACK" if rollback_error is None else "ROLLBACK_FAILED",
                error=f"{type(exc).__name__}: {exc}",
                rollback_error=rollback_error,
            )
            raise ModelctlError(
                f"failed to repair active reference for {name!r}: {exc}"
                + (f"; rollback failed: {rollback_error}" if rollback_error else "")
            ) from exc

        if quarantine.is_dir():
            validate_object(quarantine, expected_name=name)
            repair_journal.transition(
                "REPAIRED",
                reference=str(reference),
                object=str(target),
                quarantine=str(quarantine),
            )
            return RepairResult(name, "repaired", reference, target, quarantine)

        repair_journal.transition(
            "REPAIRED_QUARANTINE_MISSING",
            reference=str(reference),
            object=str(target),
            expected_quarantine=str(quarantine),
        )
        return RepairResult(name, "repaired-no-quarantine", reference, target)


def cleanup_quarantine(root: Path, name: str, *, apply: bool = False) -> list[Path]:
    """Validate then optionally remove quarantined duplicates for a repaired model."""
    validate_name(name)
    layout = Layout(root)
    reference = layout.active_path(name)
    target, _ = _journal_object(layout, name)
    verify_symlink(reference, target, managed_root=layout.models)
    validate_object(target, expected_name=name)
    parent = layout.staging / ".active-quarantine" / name
    if not parent.exists():
        return []
    quarantines = sorted(
        path for path in parent.iterdir() if path.is_dir() and not path.is_symlink()
    )
    for path in quarantines:
        metadata = validate_object(path, expected_name=name)
        canonical = validate_object(target, expected_name=name)
        if metadata != canonical:
            raise ModelctlError(
                f"quarantine metadata differs from active object: {path}"
            )
    if apply:
        for path in quarantines:
            shutil.rmtree(path)
        try:
            parent.rmdir()
        except OSError:
            pass
    return quarantines
