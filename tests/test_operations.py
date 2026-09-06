import contextlib
import hashlib
import json
import os
import shlex
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub import scan_cache_dir

from modelctl import catalog, hf_cache
from modelctl.catalog import catalog_status, load_catalog
from modelctl.errors import ModelctlError, ValidationError
from modelctl.hf_cache import list_records, load_record, malformed_cached_records, state_root
from modelctl.layout import Layout, atomic_symlink
from modelctl.manifest import parse_manifest
from modelctl.operations import (
    active_entrypoint,
    delete_local,
    delete_model,
    list_active_models,
    list_cached_models,
    serve_command,
    sync_local,
    update_model,
)
from modelctl.operations import _disk_usage
from modelctl.state import UpdateState
from modelctl.validation import ExpectedFile, write_metadata


@pytest.fixture(autouse=True)
def isolate_modelctl_state(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))


class FakeApi:
    def __init__(self, sha):
        self.sha = sha
        self.calls = []

    def model_info(self, repo, revision):
        self.calls.append((repo, revision))
        return SimpleNamespace(sha=self.sha)


class FakeSnapshot:
    def __init__(self, files, active_must_not_exist=None):
        self.files = files
        self.calls = []
        self.active_must_not_exist = active_must_not_exist

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("dry_run"):
            return [
                SimpleNamespace(filename=name, file_size=len(content))
                for name, content in self.files.items()
            ]
        if self.active_must_not_exist is not None:
            assert not self.active_must_not_exist.exists()
        root = Path(kwargs["local_dir"])
        for name, content in self.files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            metadata = root / ".cache" / "huggingface" / "download" / f"{name}.metadata"
            metadata.parent.mkdir(parents=True, exist_ok=True)
            metadata.write_text(f"{kwargs['revision']}\n{digest}\n{time.time()}\n")
        (root / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        return str(root)


def _manifest(name="demo", **overrides):
    raw = {"repo": "org/demo", "runtime": "vllm", **overrides}
    return parse_manifest(raw, name)


def test_update_publishes_before_switching_reference_and_reuses_valid_object(tmp_path):
    active = tmp_path / "active" / "demo"
    snapshot = FakeSnapshot(
        {"config.json": b"{}", "model.safetensors": b"weights"}, active
    )
    result = update_model(
        tmp_path, _manifest(), api=FakeApi("a" * 40), snapshot=snapshot
    )

    assert result == active.resolve()
    assert active.is_symlink()
    assert (active.resolve() / ".cache" / "huggingface").is_dir()
    states = [
        item["state"]
        for item in json.loads((tmp_path / "state" / "demo.json").read_text())["history"]
    ]
    assert states[-3:] == [
        UpdateState.PUBLISHING_OBJECT,
        UpdateState.UPDATING_REFERENCE,
        UpdateState.ACTIVE_ON_NAS,
    ]
    assert snapshot.calls[0]["dry_run"] is True
    assert "dry_run" not in snapshot.calls[1]
    [record] = load_catalog(tmp_path)["models"]
    assert {
        key: value for key, value in record.items() if key != "bytes"
    } == {"name": "demo", "runtime": "vllm", "repository": "org/demo"}
    assert isinstance(record["bytes"], int) and record["bytes"] > 0

    def should_not_download(**kwargs):
        raise AssertionError("valid content-addressed object should be reused")

    reused = update_model(
        tmp_path, _manifest(), api=FakeApi("a" * 40), snapshot=should_not_download
    )
    assert reused == result


def test_update_auto_repairs_dereferenced_active_copy(tmp_path):
    """A process that replaces the active symlink with a validated copy must
    not block the next activation of the same model."""
    manifest = _manifest()
    snapshot = FakeSnapshot({"config.json": b"{}", "model.safetensors": b"weights"})
    first = update_model(tmp_path, manifest, api=FakeApi("a" * 40), snapshot=snapshot)

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    (manifest_dir / "demo.yaml").write_text(
        "name: demo\nrepo: org/demo\nruntime: vllm\n"
    )

    reference = tmp_path / "active" / "demo"
    reference.unlink()
    shutil.copytree(first, reference)
    assert reference.is_dir() and not reference.is_symlink()

    def should_not_download(**kwargs):
        raise AssertionError("valid content-addressed object should be reused")

    second = update_model(
        tmp_path, manifest, api=FakeApi("a" * 40), snapshot=should_not_download
    )
    assert second == first
    assert reference.is_symlink()
    assert reference.resolve() == first

    repairs = list((tmp_path / "state" / "repairs").glob("demo-*.json"))
    assert len(repairs) == 1
    repair = json.loads(repairs[0].read_text())
    assert repair["state"] == "REPAIRED"
    quarantine = Path(repair["history"][-1]["quarantine"])
    assert quarantine.is_dir()
    assert (quarantine / "config.json").is_file()
    states = [
        item["state"]
        for item in json.loads((tmp_path / "state" / "demo.json").read_text())["history"]
    ]
    assert states[-1] == UpdateState.ACTIVE_ON_NAS
    [record] = load_catalog(tmp_path)["models"]
    assert record["name"] == "demo"
    assert catalog_status(tmp_path).status == "ready"


def test_update_fails_with_guidance_when_copy_cannot_be_auto_repaired(tmp_path):
    manifest = _manifest()
    snapshot = FakeSnapshot({"config.json": b"{}", "model.safetensors": b"weights"})
    first = update_model(tmp_path, manifest, api=FakeApi("a" * 40), snapshot=snapshot)

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    (manifest_dir / "demo.yaml").write_text(
        "name: demo\nrepo: org/demo\nruntime: vllm\n"
    )

    reference = tmp_path / "active" / "demo"
    reference.unlink()
    shutil.copytree(first, reference)
    (reference / ".modelctl.json").unlink()

    def should_not_download(**kwargs):
        raise AssertionError("valid content-addressed object should be reused")

    with pytest.raises(ModelctlError, match="cannot be auto-repaired"):
        update_model(
            tmp_path, manifest, api=FakeApi("a" * 40), snapshot=should_not_download
        )
    assert reference.is_dir() and not reference.is_symlink()
    assert (reference / "config.json").is_file()
    history = json.loads((tmp_path / "state" / "demo.json").read_text())["history"]
    assert history[-1]["state"] == UpdateState.FAILED_TO_ACTIVATE


def test_concurrent_updates_publish_complete_catalog(tmp_path):
    def update(name, commit):
        return update_model(
            tmp_path,
            _manifest(name),
            api=FakeApi(commit * 40),
            snapshot=FakeSnapshot({"config.json": name.encode()}),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda item: update(*item),
                [("alpha", "a"), ("beta", "b")],
            )
        )

    assert len(results) == 2
    assert [item["name"] for item in load_catalog(tmp_path)["models"]] == [
        "alpha",
        "beta",
    ]


def test_catalog_failure_does_not_roll_back_activated_model(tmp_path, monkeypatch):
    update_model(
        tmp_path,
        _manifest("old"),
        api=FakeApi("a" * 40),
        snapshot=FakeSnapshot({"config.json": b"old"}),
    )
    catalog_path = tmp_path / "catalog.json"
    previous = catalog_path.read_bytes()
    real_atomic_json = catalog._atomic_json

    def fail_catalog_write(path, payload):
        if path == catalog_path:
            raise OSError("simulated catalog failure")
        real_atomic_json(path, payload)

    monkeypatch.setattr(catalog, "_atomic_json", fail_catalog_write)
    with pytest.raises(ModelctlError, match="was activated.*catalog"):
        update_model(
            tmp_path,
            _manifest("new"),
            api=FakeApi("b" * 40),
            snapshot=FakeSnapshot({"config.json": b"new"}),
        )

    assert (tmp_path / "active" / "new").is_symlink()
    assert catalog_path.read_bytes() == previous
    assert catalog_status(tmp_path).status == "dirty"


def test_failed_new_revision_does_not_change_active_reference(tmp_path):
    update_model(
        tmp_path,
        _manifest(),
        api=FakeApi("a" * 40),
        snapshot=FakeSnapshot({"config.json": b"old"}),
    )
    old_target = (tmp_path / "active" / "demo").resolve()

    class BadSnapshot:
        def __call__(self, **kwargs):
            if kwargs.get("dry_run"):
                return [SimpleNamespace(filename="config.json", file_size=10)]
            (Path(kwargs["local_dir"]) / "config.json").write_bytes(b"bad")

    with pytest.raises(ValidationError, match="size mismatch"):
        update_model(
            tmp_path,
            _manifest(),
            api=FakeApi("b" * 40),
            snapshot=BadSnapshot(),
        )
    assert (tmp_path / "active" / "demo").resolve() == old_target
    state = json.loads((tmp_path / "state" / "demo.json").read_text())
    assert state["state"] == UpdateState.FAILED_UNPUBLISHED
    assert not any(path.name.startswith("b" * 40) for path in (tmp_path / "models").rglob("*"))


def test_activation_failure_restores_previous_active_reference(tmp_path, monkeypatch):
    update_model(
        tmp_path,
        _manifest(),
        api=FakeApi("a" * 40),
        snapshot=FakeSnapshot({"config.json": b"old"}),
    )
    reference = tmp_path / "active" / "demo"
    old_target = reference.resolve()
    real_atomic_symlink = atomic_symlink
    calls = 0

    def malformed_publication(target, link):
        nonlocal calls
        calls += 1
        if calls == 1:
            link.unlink()
            link.mkdir()
            (link / "unexpected").write_text("preserved")
            raise ModelctlError("simulated invalid publication")
        real_atomic_symlink(target, link)

    monkeypatch.setattr("modelctl.operations.atomic_symlink", malformed_publication)
    with pytest.raises(ModelctlError, match="simulated invalid publication"):
        update_model(
            tmp_path,
            _manifest(),
            api=FakeApi("b" * 40),
            snapshot=FakeSnapshot({"config.json": b"new"}),
        )

    assert reference.is_symlink()
    assert reference.resolve() == old_target
    state = json.loads((tmp_path / "state" / "demo.json").read_text())
    assert state["state"] == UpdateState.FAILED_TO_ACTIVATE
    assert state["history"][-1]["rollback"] == "previous-reference-restored"
    failed = tmp_path / ".staging" / ".failed-references" / "demo"
    assert (failed / "unexpected").read_text() == "preserved"


def test_update_refuses_to_overwrite_regular_active_directory(tmp_path):
    active = tmp_path / "active" / "demo"
    active.mkdir(parents=True)
    (active / "keep").write_text("data")

    with pytest.raises(ModelctlError, match="active reference path is a directory"):
        update_model(
            tmp_path,
            _manifest(),
            api=FakeApi("a" * 40),
            snapshot=FakeSnapshot({"config.json": b"new"}),
        )

    assert (active / "keep").read_text() == "data"
    state = json.loads((tmp_path / "state" / "demo.json").read_text())
    assert state["state"] == UpdateState.FAILED_TO_ACTIVATE


def test_path_and_serve_command_use_resolved_entrypoint_and_shell_escape(tmp_path):
    root = tmp_path / "root with spaces"
    layout = Layout(root)
    layout.prepare()
    manifest = _manifest(runtime={"type": "llama.cpp", "args": ["--ctx-size", "8192"]}, format="gguf", entrypoint="model file.gguf")
    object_path = layout.object_path(manifest, "c" * 40)
    object_path.mkdir(parents=True)
    (object_path / "model file.gguf").write_bytes(b"gguf")
    write_metadata(
        object_path,
        manifest,
        "c" * 40,
        [ExpectedFile("model file.gguf", 4)],
        "model file.gguf",
    )
    atomic_symlink(object_path, layout.active_path("demo"))

    assert active_entrypoint(root, "demo") == (object_path / "model file.gguf").resolve()
    command = serve_command(root, "demo")
    assert shlex.split(command) == [
        "llama-server",
        "--model",
        str((object_path / "model file.gguf").resolve()),
        "--ctx-size",
        "8192",
    ]


def test_serve_command_includes_mmproj_and_mtp_companions(tmp_path):
    layout = Layout(tmp_path)
    layout.prepare()
    manifest = _manifest(
        runtime="llama.cpp",
        format="gguf",
        entrypoint="model.gguf",
        companions={
            "mmproj": "mmproj-F16.gguf",
            "mtp": "mtp-model.gguf",
        },
    )
    object_path = layout.object_path(manifest, "e" * 40)
    object_path.mkdir(parents=True)
    names = ["model.gguf", "mmproj-F16.gguf", "mtp-model.gguf"]
    for name in names:
        (object_path / name).write_bytes(b"gguf")
    write_metadata(
        object_path,
        manifest,
        "e" * 40,
        [ExpectedFile(name, 4) for name in names],
        "model.gguf",
    )
    atomic_symlink(object_path, layout.active_path("demo"))

    argv = shlex.split(serve_command(tmp_path, "demo"))
    assert argv == [
        "llama-server",
        "--model",
        str((object_path / "model.gguf").resolve()),
        "--mmproj",
        str((object_path / "mmproj-F16.gguf").resolve()),
        "--model-draft",
        str((object_path / "mtp-model.gguf").resolve()),
        "--spec-type",
        "draft-mtp",
    ]


def test_local_sync_resolves_hugging_face_repository_to_active_name(
    tmp_path, monkeypatch
):
    source_root = tmp_path / "nas"
    cache = tmp_path / "hub"
    calls = []
    model = SimpleNamespace(name="custom-name", repo="org/demo")

    monkeypatch.setattr(
        "modelctl.operations.list_active_models", lambda root: [model]
    )

    def fake_sync(source, destination, name, *, rsync, runner, progress):
        calls.append((source, destination, name, rsync, runner, progress))
        return destination / "snapshot"

    monkeypatch.setattr("modelctl.operations.sync_cache", fake_sync)
    result = sync_local(
        source_root, cache, "org/demo", rsync="custom-rsync", runner=shutil.copy
    )

    assert result == cache / "snapshot"
    assert calls == [
        (source_root, cache, "custom-name", "custom-rsync", shutil.copy, None)
    ]


def test_local_sync_rejects_ambiguous_hugging_face_repository(
    tmp_path, monkeypatch
):
    models = [
        SimpleNamespace(name="demo-fp16", repo="org/demo"),
        SimpleNamespace(name="demo-q4", repo="org/demo"),
    ]
    monkeypatch.setattr(
        "modelctl.operations.list_active_models", lambda root: models
    )

    with pytest.raises(ModelctlError, match="multiple model names.*demo-fp16.*demo-q4"):
        sync_local(tmp_path / "nas", tmp_path / "hub", "org/demo")


def test_local_sync_reports_unknown_hugging_face_repository(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "modelctl.operations.list_active_models", lambda root: []
    )

    with pytest.raises(ModelctlError, match="has no active NAS model"):
        sync_local(tmp_path / "nas", tmp_path / "hub", "org/missing")


def test_local_sync_publishes_hf_cache_and_updates_record_last(
    tmp_path, monkeypatch
):
    nas = tmp_path / "nas"
    cache = tmp_path / "hub"
    update_model(
        nas,
        _manifest(),
        api=FakeApi("d" * 40),
        snapshot=FakeSnapshot({"config.json": b"data"}),
    )

    def fake_rsync(command, check):
        assert check is True
        assert "--human-readable" in command
        assert "--info=progress2" in command
        source = Path(command[-2].removesuffix("/"))
        destination = Path(command[-1].removesuffix("/"))
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "config.json", destination / "config.json")

    verified = []
    real_verify = hf_cache._verify_etag

    def track_verify(path, etag):
        verified.append(path)
        real_verify(path, etag)

    monkeypatch.setattr("modelctl.hf_cache._verify_etag", track_verify)
    progress = []
    result = sync_local(
        nas, cache, "demo", runner=fake_rsync, progress=progress.append
    )
    snapshot = cache / "models--org--demo" / "snapshots" / ("d" * 40)
    assert result == snapshot.resolve()
    assert len(verified) == 1
    assert ".modelctl-staging" in verified[0].parts
    assert verified[0].name == "config.json"
    assert progress == [
        "sync-local: transferring 1 files (4.00 B) to cache staging",
        "sync-local: validating 1 transferred files (4.00 B)",
        "sync-local: validating [1/1] config.json",
        "sync-local: publishing verified blobs and snapshot",
    ]
    assert (snapshot / "config.json").is_symlink()
    assert (cache / "models--org--demo" / "refs" / "main").read_text() == "d" * 40
    info = scan_cache_dir(cache)
    assert not info.warnings
    assert next(iter(info.repos)).repo_id == "org/demo"
    assert load_record(cache, "demo").snapshot == snapshot.resolve()
    state = json.loads((state_root(cache) / "state" / "demo.json").read_text())
    assert state["state"] == "READY_FOR_SERVICE_RESTART"
    assert [item["state"] for item in state["history"]][-3:] == [
        "VALIDATING_STAGING",
        "PUBLISHING_CACHE",
        "READY_FOR_SERVICE_RESTART",
    ]
    inventory = list_cached_models(cache)
    assert [(item.name, item.repo, item.path) for item in inventory] == [
        ("demo", "org/demo", snapshot.resolve())
    ]


def test_delete_local_deletes_cache_data_and_preserves_nas(tmp_path):
    nas = tmp_path / "nas"
    cache = tmp_path / "hub"
    update_model(
        nas,
        _manifest(),
        api=FakeApi("d" * 40),
        snapshot=FakeSnapshot({"config.json": b"data"}),
    )

    def copy(command, check):
        source = Path(command[-2].removesuffix("/"))
        destination = Path(command[-1].removesuffix("/"))
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "config.json", destination / "config.json")

    snapshot = sync_local(nas, cache, "demo", runner=copy)
    nas_object = (nas / "active" / "demo").resolve()
    repository = cache / "models--org--demo"
    result = delete_local(cache, "demo")
    assert result.snapshot == snapshot.resolve()
    assert result.removed == (repository,)
    assert result.retained == ()
    assert not repository.exists()
    assert not (state_root(cache) / "active" / "demo.json").exists()
    refs = json.loads((state_root(cache) / "refs.json").read_text())
    assert refs == {"schema": 1, "refs": {}}
    assert (nas / "active" / "demo").resolve() == nas_object
    assert nas_object.exists()


def test_delete_local_keep_data_preserves_cache(tmp_path):
    nas = tmp_path / "nas"
    cache = tmp_path / "hub"
    update_model(
        nas,
        _manifest(),
        api=FakeApi("d" * 40),
        snapshot=FakeSnapshot({"config.json": b"data"}),
    )

    def copy(command, check):
        source = Path(command[-2].removesuffix("/"))
        destination = Path(command[-1].removesuffix("/"))
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "config.json", destination / "config.json")

    snapshot = sync_local(nas, cache, "demo", runner=copy)
    repository = cache / "models--org--demo"
    result = delete_local(cache, "demo", keep_data=True)
    assert result.snapshot == snapshot.resolve()
    assert result.removed == ()
    assert result.retained == (
        "--keep-data was set; snapshots, refs, and blobs were retained",
    )
    assert snapshot.exists()
    assert repository.exists()
    assert not (state_root(cache) / "active" / "demo.json").exists()


def test_delete_local_keeps_registration_when_data_deletion_fails(
    tmp_path, monkeypatch
):
    nas = tmp_path / "nas"
    cache = tmp_path / "hub"
    update_model(
        nas,
        _manifest(),
        api=FakeApi("d" * 40),
        snapshot=FakeSnapshot({"config.json": b"data"}),
    )

    def copy(command, check):
        source = Path(command[-2].removesuffix("/"))
        destination = Path(command[-1].removesuffix("/"))
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "config.json", destination / "config.json")

    sync_local(nas, cache, "demo", runner=copy)
    repository = cache / "models--org--demo"
    record_path = state_root(cache) / "active" / "demo.json"

    real_rmtree = shutil.rmtree

    def denied(path, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr("modelctl.hf_cache.shutil.rmtree", denied)
    with pytest.raises(ModelctlError, match="could not delete local cache data"):
        delete_local(cache, "demo")
    assert record_path.exists()
    assert repository.exists()
    assert load_record(cache, "demo").snapshot.is_dir()

    monkeypatch.setattr("modelctl.hf_cache.shutil.rmtree", real_rmtree)
    result = delete_local(cache, "demo")
    assert result.removed == (repository,)
    assert not repository.exists()
    assert not record_path.exists()


def test_delete_local_reports_data_shared_with_other_registration(tmp_path):
    nas = tmp_path / "nas"
    cache = tmp_path / "hub"

    def copy(command, check):
        source = Path(command[-2].removesuffix("/"))
        destination = Path(command[-1].removesuffix("/"))
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "config.json", destination / "config.json")

    for name in ("demo", "demo2"):
        update_model(
            nas,
            _manifest(name=name),
            api=FakeApi("d" * 40),
            snapshot=FakeSnapshot({"config.json": b"data"}),
        )
        sync_local(nas, cache, name, runner=copy)

    repository = cache / "models--org--demo"
    result = delete_local(cache, "demo")
    assert result.removed == ()
    assert len(result.retained) == 2
    assert all(
        note.startswith(("snapshot ", "blob " )) for note in result.retained
    )
    assert all("demo2" in note for note in result.retained)
    assert repository.exists()
    assert load_record(cache, "demo2").snapshot.is_dir()

    result = delete_local(cache, "demo2")
    assert result.removed == (repository,)
    assert result.retained == ()
    assert not repository.exists()


def test_delete_local_only_removes_one_shared_record(tmp_path):
    cache = tmp_path / "hub"
    with pytest.raises(ModelctlError, match="no local cache record"):
        delete_local(cache, "demo")


def _broken_record(cache, name="broken"):
    active = state_root(cache) / "active"
    active.mkdir(parents=True, exist_ok=True)
    (active / f"{name}.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "cache": str(cache),
                "name": name,
                "repo": "org/broken",
                "revision": "main",
                "commit": "e" * 40,
                "files": {"config.json": "e" * 40},
                "snapshot": str(
                    cache / "models--org--broken" / "snapshots" / ("e" * 40)
                ),
                "metadata": {
                    "name": name,
                    "repo": "org/broken",
                    "revision": "main",
                    "commit": "e" * 40,
                    "entrypoint": ".",
                    "companions": {},
                },
            }
        )
    )


def test_load_record_hints_recovery_for_missing_snapshot(tmp_path):
    cache = tmp_path / "hub"
    repository = cache / "models--org--broken"
    (repository / "blobs").mkdir(parents=True)
    (repository / "snapshots").mkdir()
    (repository / "refs").mkdir()
    _broken_record(cache)
    with pytest.raises(ModelctlError, match="sync-local"):
        load_record(cache, "broken")


def test_local_listing_skips_broken_registration(tmp_path):
    cache = tmp_path / "hub"
    _broken_record(cache)
    assert list_records(cache) == []
    assert malformed_cached_records(cache) == ["broken"]


def test_delete_local_removes_broken_registration(tmp_path):
    cache = tmp_path / "hub"
    _broken_record(cache)
    result = delete_local(cache, "broken")
    assert result.snapshot is None
    assert result.removed == ()
    assert result.retained == (
        "registration was stale or malformed; cache data was left untouched",
    )
    assert not (state_root(cache) / "active" / "broken.json").exists()
    with pytest.raises(ModelctlError, match="no local cache record"):
        delete_local(cache, "broken")


def test_interrupted_local_sync_keeps_previous_cache_record(tmp_path):
    nas = tmp_path / "nas"
    cache = tmp_path / "hub"
    manifest = _manifest()
    update_model(
        nas,
        manifest,
        api=FakeApi("1" * 40),
        snapshot=FakeSnapshot({"config.json": b"old"}),
    )

    def copy(command, check):
        source = Path(command[-2].removesuffix("/"))
        destination = Path(command[-1].removesuffix("/"))
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "config.json", destination / "config.json")

    sync_local(nas, cache, "demo", runner=copy)
    old_record = load_record(cache, "demo")
    update_model(
        nas,
        manifest,
        api=FakeApi("2" * 40),
        snapshot=FakeSnapshot({"config.json": b"new"}),
    )

    def interrupted(command, check):
        raise OSError("connection lost")

    with pytest.raises(ModelctlError, match="resumable"):
        sync_local(nas, cache, "demo", runner=interrupted)
    assert load_record(cache, "demo").commit == old_record.commit
    assert load_record(cache, "demo").snapshot == old_record.snapshot
    state = json.loads((state_root(cache) / "state" / "demo.json").read_text())
    assert state["state"] == "PARTIAL_RESUMABLE"



def test_sync_preserves_foreign_ref_and_publishes_detached_snapshot(tmp_path):
    nas = tmp_path / "nas"
    cache = tmp_path / "hub"
    update_model(
        nas,
        _manifest(),
        api=FakeApi("d" * 40),
        snapshot=FakeSnapshot({"config.json": b"data"}),
    )
    repository = cache / "models--org--demo"
    (repository / "blobs").mkdir(parents=True)
    (repository / "snapshots").mkdir()
    (repository / "snapshots" / ("f" * 40)).mkdir()
    (repository / "refs").mkdir()
    (repository / "refs" / "main").write_text("f" * 40)

    def copy(command, check):
        source = Path(command[-2].removesuffix("/"))
        destination = Path(command[-1].removesuffix("/"))
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "config.json", destination / "config.json")

    result = sync_local(nas, cache, "demo", runner=copy)
    assert result == (repository / "snapshots" / ("d" * 40)).resolve()
    assert (repository / "refs" / "main").read_text() == "f" * 40
    assert not scan_cache_dir(cache).warnings


def test_sync_rejects_mismatched_existing_blob_without_active_record(tmp_path):
    nas = tmp_path / "nas"
    cache = tmp_path / "hub"
    update_model(
        nas,
        _manifest(),
        api=FakeApi("d" * 40),
        snapshot=FakeSnapshot({"config.json": b"data"}),
    )
    source = (nas / "active" / "demo").resolve()
    metadata_path = source / ".cache" / "huggingface" / "download" / "config.json.metadata"
    etag = metadata_path.read_text().splitlines()[1]
    blob = cache / "models--org--demo" / "blobs" / etag
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"evil")

    def copy(command, check):
        source_dir = Path(command[-2].removesuffix("/"))
        destination = Path(command[-1].removesuffix("/"))
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir / "config.json", destination / "config.json")

    with pytest.raises(ValidationError, match="content hash mismatch"):
        sync_local(nas, cache, "demo", runner=copy)
    assert blob.read_bytes() == b"evil"
    with pytest.raises(ModelctlError, match="no valid local cache record"):
        load_record(cache, "demo")


def test_sync_accepts_sha256_lfs_etag(tmp_path):
    nas = tmp_path / "nas"
    cache = tmp_path / "hub"
    content = b"weights"
    update_model(
        nas,
        _manifest(),
        api=FakeApi("d" * 40),
        snapshot=FakeSnapshot({"model.safetensors": content}),
    )
    source = (nas / "active" / "demo").resolve()
    metadata_path = source / ".cache" / "huggingface" / "download" / "model.safetensors.metadata"
    metadata_path.write_text(f"{'d' * 40}\n{hashlib.sha256(content).hexdigest()}\n{time.time()}\n")

    def copy(command, check):
        source_dir = Path(command[-2].removesuffix("/"))
        destination = Path(command[-1].removesuffix("/"))
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir / "model.safetensors", destination / "model.safetensors")

    snapshot = sync_local(nas, cache, "demo", runner=copy)
    assert (snapshot / "model.safetensors").read_bytes() == content


def _store_snapshot(root: Path) -> list[tuple[str, bytes | None]]:
    items: list[tuple[str, bytes | None]] = []
    for sub in ("active", "models", "state", ".staging", "catalog.json"):
        path = root / sub
        if path.is_file():
            items.append((sub, path.read_bytes()))
        elif path.is_dir():
            for item in sorted(path.rglob("*")):
                relative = str(item.relative_to(root))
                items.append((relative, item.read_bytes() if item.is_file() else None))
    return items


def test_delete_dry_run_leaves_store_unchanged(tmp_path):
    update_model(
        tmp_path, _manifest(), api=FakeApi("a" * 40), snapshot=FakeSnapshot({"config.json": b"{}"})
    )
    before = _store_snapshot(tmp_path)
    result = delete_model(tmp_path, "demo")

    assert result.applied is False
    assert [item.name for item in result.removed_objects] == ["demo"]
    assert result.removed_objects[0].status == "unreferenced"
    assert result.journals == (
        tmp_path / "state" / "demo.json",
        tmp_path / "state" / "demo.delete.json",
    )
    assert _store_snapshot(tmp_path) == before
    assert (tmp_path / "active" / "demo").is_symlink()


def test_delete_apply_removes_reference_journal_objects_and_empty_dirs(tmp_path):
    update_model(
        tmp_path, _manifest(), api=FakeApi("a" * 40), snapshot=FakeSnapshot({"config.json": b"{}"})
    )
    result = delete_model(tmp_path, "demo", apply=True)

    assert result.applied is True
    assert not (tmp_path / "active" / "demo").exists()
    assert not (tmp_path / "state" / "demo.json").exists()
    assert not (tmp_path / "state" / "demo.delete.json").exists()
    for item in result.removed_objects:
        assert not item.path.exists()
    assert not (tmp_path / "models" / "org" / "demo").exists()
    assert not (tmp_path / "models" / "org").exists()
    assert load_catalog(tmp_path)["models"] == []


def test_delete_keeps_objects_referenced_by_other_active_models(tmp_path):
    update_model(
        tmp_path,
        _manifest("demo"),
        api=FakeApi("a" * 40),
        snapshot=FakeSnapshot({"config.json": b"demo"}),
    )
    update_model(
        tmp_path,
        _manifest("beta"),
        api=FakeApi("b" * 40),
        snapshot=FakeSnapshot({"config.json": b"beta"}),
    )
    result = delete_model(tmp_path, "demo", apply=True)

    assert not (tmp_path / "active" / "demo").exists()
    assert (tmp_path / "active" / "beta").is_symlink()
    assert (tmp_path / "active" / "beta").resolve().is_dir()
    assert [item.name for item in result.removed_objects] == ["demo"]
    assert [item.name for item in result.retained_objects] == ["beta"]
    assert (tmp_path / "models" / "org" / "demo").exists()
    assert [item["name"] for item in load_catalog(tmp_path)["models"]] == ["beta"]


def test_delete_removes_eligible_staging_and_retains_live_data(tmp_path):
    update_model(
        tmp_path, _manifest(), api=FakeApi("a" * 40), snapshot=FakeSnapshot({"config.json": b"{}"})
    )
    layout = Layout(tmp_path)
    failed = layout.staging / "org" / "demo" / "commit--failed"
    failed.mkdir(parents=True)
    (failed / "data").write_bytes(b"x")
    (tmp_path / "state" / "stale.json").write_text(
        json.dumps(
            {
                "operation": "update",
                "state": "FAILED_UNPUBLISHED",
                "history": [{"state": "FAILED_UNPUBLISHED", "staging": str(failed)}],
            }
        )
    )
    resumable = layout.staging / "org" / "demo" / "commit--partial"
    resumable.mkdir(parents=True)
    (resumable / "data").write_bytes(b"x")
    (tmp_path / "state" / "other.json").write_text(
        json.dumps(
            {
                "operation": "update",
                "state": "PARTIAL_RESUMABLE",
                "history": [{"state": "PARTIAL_RESUMABLE", "staging": str(resumable)}],
            }
        )
    )

    plan = delete_model(tmp_path, "demo")
    result = delete_model(tmp_path, "demo", apply=True)

    assert [item.path for item in plan.removed_staging] == [
        item.path for item in result.removed_staging
    ]
    assert [item.path for item in result.removed_staging] == [failed]
    assert not failed.exists()
    assert resumable.exists()
    assert [item.status for item in result.retained_staging] == ["resumable"]


def test_delete_retains_invalid_objects(tmp_path):
    update_model(
        tmp_path, _manifest(), api=FakeApi("a" * 40), snapshot=FakeSnapshot({"config.json": b"{}"})
    )
    junk = tmp_path / "models" / "org" / "demo" / "junk"
    junk.mkdir(parents=True)
    (junk / "config.json").write_bytes(b"x")

    result = delete_model(tmp_path, "demo", apply=True)

    assert junk.exists()
    assert [item.status for item in result.retained_objects] == ["invalid_object"]
    assert not (tmp_path / "models" / "org" / "demo").exists() is False
    assert (tmp_path / "models" / "org" / "demo").exists()


def test_delete_refuses_unsafe_references(tmp_path):
    layout = Layout(tmp_path)
    layout.prepare()
    with pytest.raises(ModelctlError, match="no active reference"):
        delete_model(tmp_path, "demo")

    reference = layout.active_path("demo")
    reference.mkdir()
    with pytest.raises(ModelctlError, match="repair-active"):
        delete_model(tmp_path, "demo")
    reference.rmdir()

    reference.symlink_to(tmp_path / "missing-target")
    with pytest.raises(ModelctlError, match="doctor"):
        delete_model(tmp_path, "demo")
    reference.unlink()

    outside = tmp_path / "outside"
    outside.mkdir()
    reference.symlink_to(outside)
    with pytest.raises(ModelctlError, match="doctor"):
        delete_model(tmp_path, "demo")


def test_delete_failure_keeps_objects_and_delete_journal(tmp_path, monkeypatch):
    update_model(
        tmp_path, _manifest(), api=FakeApi("a" * 40), snapshot=FakeSnapshot({"config.json": b"{}"})
    )
    object_path = (tmp_path / "active" / "demo").resolve()

    import modelctl.operations as operations_module

    def fail_rmtree(path, *args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(operations_module.shutil, "rmtree", fail_rmtree)
    with pytest.raises(ModelctlError, match="disk unavailable"):
        delete_model(tmp_path, "demo", apply=True)

    assert not (tmp_path / "active" / "demo").exists()
    assert object_path.is_dir()
    assert not (tmp_path / "state" / "demo.json").exists()
    document = json.loads((tmp_path / "state" / "demo.delete.json").read_text())
    assert document["history"][-1]["state"] == "FAILED"
    assert "disk unavailable" in document["history"][-1]["error"]

    monkeypatch.undo()
    with pytest.raises(ModelctlError, match="no active reference"):
        delete_model(tmp_path, "demo", apply=True)


def test_delete_catalog_failure_retains_delete_journal_evidence(tmp_path, monkeypatch):
    update_model(
        tmp_path, _manifest(), api=FakeApi("a" * 40), snapshot=FakeSnapshot({"config.json": b"{}"})
    )
    object_path = (tmp_path / "active" / "demo").resolve()

    import modelctl.operations as operations_module

    def fail_refresh(root, listing):
        raise ModelctlError("catalog boom")

    monkeypatch.setattr(operations_module, "refresh_catalog_locked", fail_refresh)
    with pytest.raises(ModelctlError, match="was deactivated.*catalog boom"):
        delete_model(tmp_path, "demo", apply=True)

    assert not (tmp_path / "active" / "demo").exists()
    assert object_path.is_dir()
    document = json.loads((tmp_path / "state" / "demo.delete.json").read_text())
    assert document["history"][-1]["state"] == "REFERENCE_REMOVED"


class _FakeDirEntry:
    def __init__(self, path, mode, blocks, size):
        self.path = path
        self._stat = os.stat_result(
            (mode, 1, 0, 1, 0, 0, size, 0, 0, 0),
            {"st_blksize": 4096, "st_blocks": blocks},
        )

    def stat(self, follow_symlinks=False):
        return self._stat


def _fake_scandir(entries_by_dir):
    @contextlib.contextmanager
    def fake_scandir(path):
        if str(path) not in entries_by_dir:
            raise PermissionError(13, "permission denied")
        yield iter(entries_by_dir[str(path)])

    return fake_scandir


def test_disk_usage_sums_blocks_with_size_fallback(tmp_path, monkeypatch):
    file_mode = 0o100644
    dir_mode = 0o040755
    entries_by_dir = {
        str(tmp_path): [
            _FakeDirEntry(str(tmp_path / "a.bin"), file_mode, blocks=16, size=100),
            _FakeDirEntry(str(tmp_path / "sub"), dir_mode, blocks=0, size=0),
        ],
        str(tmp_path / "sub"): [
            _FakeDirEntry(str(tmp_path / "sub" / "b.bin"), file_mode, blocks=0, size=2048),
        ],
    }
    monkeypatch.setattr(
        "modelctl.operations.os.scandir", _fake_scandir(entries_by_dir)
    )

    assert _disk_usage(tmp_path) == 16 * 512 + 2048


def test_disk_usage_wraps_os_errors(tmp_path, monkeypatch):
    @contextlib.contextmanager
    def fail_scandir(path):
        raise PermissionError(13, f"permission denied: {path}")
        yield  # pragma: no cover

    monkeypatch.setattr("modelctl.operations.os.scandir", fail_scandir)

    with pytest.raises(ModelctlError, match="failed to measure disk usage"):
        _disk_usage(tmp_path)


def test_list_active_models_reports_object_disk_usage(tmp_path):
    update_model(
        tmp_path,
        _manifest(),
        api=FakeApi("a" * 40),
        snapshot=FakeSnapshot({"config.json": b"{}" * 100}),
    )

    [model] = list_active_models(tmp_path)
    assert model.size_bytes > 0


def test_failed_size_measurement_does_not_roll_back_activated_model(
    tmp_path, monkeypatch
):
    update_model(
        tmp_path,
        _manifest("old"),
        api=FakeApi("a" * 40),
        snapshot=FakeSnapshot({"config.json": b"old"}),
    )
    catalog_path = tmp_path / "catalog.json"
    previous = catalog_path.read_bytes()

    def fail_measure(directory):
        raise ModelctlError(f"failed to measure disk usage of {directory}: boom")

    monkeypatch.setattr("modelctl.operations._disk_usage", fail_measure)
    with pytest.raises(ModelctlError, match="was activated"):
        update_model(
            tmp_path,
            _manifest("new"),
            api=FakeApi("b" * 40),
            snapshot=FakeSnapshot({"config.json": b"new"}),
        )

    assert (tmp_path / "active" / "new").is_symlink()
    assert catalog_path.read_bytes() == previous
    assert catalog_status(tmp_path).status == "dirty"
