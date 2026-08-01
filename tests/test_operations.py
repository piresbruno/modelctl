import hashlib
import json
import shlex
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub import scan_cache_dir

from modelctl.errors import ModelctlError, ValidationError
from modelctl.hf_cache import load_record, state_root
from modelctl.layout import Layout, atomic_symlink
from modelctl.manifest import parse_manifest
from modelctl.operations import (
    active_entrypoint,
    delete_local,
    list_cached_models,
    serve_command,
    sync_local,
    update_model,
)
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

    def should_not_download(**kwargs):
        raise AssertionError("valid content-addressed object should be reused")

    reused = update_model(
        tmp_path, _manifest(), api=FakeApi("a" * 40), snapshot=should_not_download
    )
    assert reused == result


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

    def fake_sync(source, destination, name, *, rsync, runner):
        calls.append((source, destination, name, rsync, runner))
        return destination / "snapshot"

    monkeypatch.setattr("modelctl.operations.sync_cache", fake_sync)
    result = sync_local(
        source_root, cache, "org/demo", rsync="custom-rsync", runner=shutil.copy
    )

    assert result == cache / "snapshot"
    assert calls == [
        (source_root, cache, "custom-name", "custom-rsync", shutil.copy)
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


def test_local_sync_publishes_hf_cache_and_updates_record_last(tmp_path):
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

    result = sync_local(nas, cache, "demo", runner=fake_rsync)
    snapshot = cache / "models--org--demo" / "snapshots" / ("d" * 40)
    assert result == snapshot.resolve()
    assert (snapshot / "config.json").is_symlink()
    assert (cache / "models--org--demo" / "refs" / "main").read_text() == "d" * 40
    info = scan_cache_dir(cache)
    assert not info.warnings
    assert next(iter(info.repos)).repo_id == "org/demo"
    assert load_record(cache, "demo").snapshot == snapshot.resolve()
    state = json.loads((state_root(cache) / "state" / "demo.json").read_text())
    assert state["state"] == "READY_FOR_SERVICE_RESTART"
    inventory = list_cached_models(cache)
    assert [(item.name, item.repo, item.path) for item in inventory] == [
        ("demo", "org/demo", snapshot.resolve())
    ]


def test_delete_local_unregisters_model_and_preserves_cache_and_nas(tmp_path):
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
    assert delete_local(cache, "demo") == snapshot
    assert snapshot.exists()
    assert not (state_root(cache) / "active" / "demo.json").exists()
    assert (nas / "active" / "demo").resolve() == nas_object
    assert nas_object.exists()


def test_delete_local_only_removes_one_shared_record(tmp_path):
    cache = tmp_path / "hub"
    with pytest.raises(ModelctlError, match="no valid local cache record"):
        delete_local(cache, "demo")


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
