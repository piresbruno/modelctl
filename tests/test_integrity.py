import json
import shutil
from pathlib import Path

import pytest

from modelctl.errors import ModelctlError
from modelctl.integrity import (
    audit_active_references,
    cleanup_quarantine,
    repair_active_reference,
)
from modelctl.layout import Layout, atomic_symlink
from modelctl.manifest import parse_manifest
from modelctl.validation import ExpectedFile, write_metadata


def _regular_active_copy(root: Path, name: str = "demo") -> tuple[Path, Path]:
    manifest = parse_manifest({"repo": "org/demo", "runtime": "vllm"}, name)
    layout = Layout(root)
    layout.prepare()
    commit = "a" * 40
    object_path = layout.object_path(manifest, commit)
    object_path.mkdir(parents=True)
    (object_path / "config.json").write_bytes(b"{}")
    write_metadata(
        object_path,
        manifest,
        commit,
        [ExpectedFile("config.json", 2)],
        ".",
    )
    manifests = root / "manifests"
    manifests.mkdir(exist_ok=True)
    (manifests / f"{name}.yaml").write_text(
        f"name: {name}\nrepo: org/demo\nruntime: vllm\n"
    )
    state = {
        "operation": "update",
        "state": "ACTIVE_ON_NAS",
        "history": [
            {
                "state": "ACTIVE_ON_NAS",
                "at": "2026-01-01T00:00:00+00:00",
                "object": str(object_path),
                "commit": commit,
            }
        ],
    }
    layout.state_path(name).write_text(json.dumps(state))
    reference = layout.active_path(name)
    shutil.copytree(object_path, reference)
    return object_path, reference


def test_audit_classifies_valid_repairable_and_foreign_entries(tmp_path):
    object_path, reference = _regular_active_copy(tmp_path)
    (tmp_path / "active" / ".DS_Store").write_bytes(b"metadata")
    (tmp_path / "active" / "notes.txt").write_text("foreign")

    results = {item.name: item for item in audit_active_references(tmp_path)}

    assert results["demo"].status == "repairable_directory"
    assert results["demo"].object == object_path.resolve()
    assert results[".DS_Store"].status == "hidden_foreign_entry"
    assert results["notes.txt"].status == "foreign_entry"
    assert reference.is_dir() and not reference.is_symlink()


def test_audit_classifies_broken_outside_and_journal_mismatch(tmp_path):
    layout = Layout(tmp_path)
    layout.prepare()
    (layout.active / "broken").symlink_to("../models/missing")
    outside = tmp_path / "outside"
    outside.mkdir()
    (layout.active / "outside").symlink_to(outside)

    object_path, reference = _regular_active_copy(tmp_path, "mismatch")
    shutil.rmtree(reference)
    atomic_symlink(object_path, reference)
    state = json.loads(layout.state_path("mismatch").read_text())
    state["history"][-1]["commit"] = "b" * 40
    layout.state_path("mismatch").write_text(json.dumps(state))

    results = {item.name: item.status for item in audit_active_references(tmp_path)}
    assert results["broken"] == "broken_symlink"
    assert results["outside"] == "outside_store_symlink"
    assert results["mismatch"] == "journal_object_mismatch"


def test_repair_is_dry_run_by_default_and_apply_quarantines_copy(tmp_path):
    object_path, reference = _regular_active_copy(tmp_path)

    dry_run = repair_active_reference(tmp_path, "demo")
    assert dry_run.status == "dry-run"
    assert reference.is_dir() and not reference.is_symlink()

    repaired = repair_active_reference(tmp_path, "demo", apply=True)
    assert repaired.status == "repaired"
    assert repaired.quarantine is not None
    assert repaired.quarantine.is_dir()
    assert reference.is_symlink()
    assert reference.resolve() == object_path.resolve()
    assert {item.name: item.status for item in audit_active_references(tmp_path)}[
        "demo"
    ] == "valid"

    quarantines = cleanup_quarantine(tmp_path, "demo")
    assert quarantines == [repaired.quarantine]
    assert repaired.quarantine.exists()
    assert cleanup_quarantine(tmp_path, "demo", apply=True) == [repaired.quarantine]
    assert not repaired.quarantine.exists()


def test_repair_reports_when_filesystem_does_not_retain_quarantine(
    tmp_path, monkeypatch
):
    object_path, reference = _regular_active_copy(tmp_path)
    real_atomic_symlink = atomic_symlink

    def publish_then_drop_quarantine(target, link):
        real_atomic_symlink(target, link)
        parent = tmp_path / ".staging" / ".active-quarantine" / "demo"
        for path in parent.iterdir():
            if path.is_dir():
                shutil.rmtree(path)

    monkeypatch.setattr(
        "modelctl.integrity.atomic_symlink", publish_then_drop_quarantine
    )
    result = repair_active_reference(tmp_path, "demo", apply=True)

    assert result.status == "repaired-no-quarantine"
    assert result.quarantine is None
    assert reference.is_symlink()
    assert reference.resolve() == object_path.resolve()
    repair_journal = next((tmp_path / "state" / "repairs").glob("*.json"))
    assert json.loads(repair_journal.read_text())["state"] == (
        "REPAIRED_QUARANTINE_MISSING"
    )


def test_repair_refuses_metadata_mismatch(tmp_path):
    _, reference = _regular_active_copy(tmp_path)
    metadata_path = reference / ".modelctl.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["revision"] = "different"
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ModelctlError, match="metadata differs"):
        repair_active_reference(tmp_path, "demo", apply=True)
    assert reference.is_dir() and not reference.is_symlink()


def test_repair_rolls_back_directory_when_symlink_publication_fails(
    tmp_path, monkeypatch
):
    _, reference = _regular_active_copy(tmp_path)

    def fail_publication(target, link):
        link.mkdir()
        (link / "unexpected").write_text("preserve me")
        raise OSError("simulated filesystem failure")

    monkeypatch.setattr("modelctl.integrity.atomic_symlink", fail_publication)
    with pytest.raises(ModelctlError, match="failed to repair"):
        repair_active_reference(tmp_path, "demo", apply=True)

    assert reference.is_dir() and not reference.is_symlink()
    assert (reference / "config.json").read_bytes() == b"{}"
    repair_journals = list((tmp_path / "state" / "repairs").glob("*.json"))
    assert len(repair_journals) == 1
    assert json.loads(repair_journals[0].read_text())["state"] == "ROLLED_BACK"
