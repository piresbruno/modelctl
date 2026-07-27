import json
import shlex
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelctl.errors import ModelctlError, ValidationError
from modelctl.layout import Layout, atomic_symlink
from modelctl.manifest import parse_manifest
from modelctl.operations import (
    active_entrypoint,
    delete_local,
    list_active_models,
    serve_command,
    sync_local,
    update_model,
)
from modelctl.state import SyncState, UpdateState
from modelctl.validation import ExpectedFile, write_metadata


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


def test_local_sync_excludes_hf_cache_and_updates_reference_last(tmp_path):
    nas = tmp_path / "nas"
    local = tmp_path / "local"
    manifest = _manifest()
    update_model(
        nas,
        manifest,
        api=FakeApi("d" * 40),
        snapshot=FakeSnapshot({"config.json": b"data"}),
    )

    def fake_rsync(command, check):
        assert check is True
        source = Path(command[-2].removesuffix("/"))
        destination = Path(command[-1].removesuffix("/"))
        for child in source.iterdir():
            if child.name == ".cache":
                continue
            target = destination / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)

    result = sync_local(nas, local, "demo", runner=fake_rsync)
    assert result == (local / "active" / "demo").resolve()
    assert not (result / ".cache" / "huggingface").exists()
    state = json.loads((local / "state" / "demo.json").read_text())
    assert state["state"] == SyncState.READY_FOR_SERVICE_RESTART
    assert [event["state"] for event in state["history"]][-2:] == [
        SyncState.LOCAL_REFERENCE_UPDATE,
        SyncState.READY_FOR_SERVICE_RESTART,
    ]
    inventory = list_active_models(local)
    assert len(inventory) == 1
    assert inventory[0].name == "demo"
    assert inventory[0].repo == "org/demo"
    assert inventory[0].commit == "d" * 40
    assert inventory[0].path == result


def test_delete_local_removes_local_model_and_preserves_nas(tmp_path):
    nas = tmp_path / "nas"
    local = tmp_path / "local"
    update_model(
        nas,
        _manifest(),
        api=FakeApi("d" * 40),
        snapshot=FakeSnapshot({"config.json": b"data"}),
    )

    def copy(command, check):
        assert check is True
        source = Path(command[-2].removesuffix("/"))
        destination = Path(command[-1].removesuffix("/"))
        shutil.copytree(source, destination, dirs_exist_ok=True)

    sync_local(nas, local, "demo", runner=copy)
    local_object = (local / "active" / "demo").resolve()
    nas_object = (nas / "active" / "demo").resolve()

    assert delete_local(local, "demo") == local_object
    assert not (local / "active" / "demo").exists()
    assert not local_object.exists()
    assert not (local / "state" / "demo.json").exists()
    assert (nas / "active" / "demo").resolve() == nas_object
    assert nas_object.exists()


def test_delete_local_refuses_object_used_by_another_active_name(tmp_path):
    update_model(
        tmp_path,
        _manifest(),
        api=FakeApi("d" * 40),
        snapshot=FakeSnapshot({"config.json": b"data"}),
    )
    object_path = (tmp_path / "active" / "demo").resolve()
    atomic_symlink(object_path, tmp_path / "active" / "alias")

    with pytest.raises(ModelctlError, match="also active as 'alias'"):
        delete_local(tmp_path, "demo")

    assert (tmp_path / "active" / "demo").is_symlink()
    assert object_path.exists()


def test_interrupted_local_sync_keeps_previous_local_reference(tmp_path):
    nas = tmp_path / "nas"
    local = tmp_path / "local"
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
        for child in source.iterdir():
            if child.name != ".cache":
                target = destination / child.name
                shutil.copy2(child, target)

    sync_local(nas, local, "demo", runner=copy)
    old_target = (local / "active" / "demo").resolve()
    update_model(
        nas,
        manifest,
        api=FakeApi("2" * 40),
        snapshot=FakeSnapshot({"config.json": b"new"}),
    )

    def interrupted(command, check):
        raise OSError("connection lost")

    with pytest.raises(ModelctlError, match="resumable"):
        sync_local(nas, local, "demo", runner=interrupted)
    assert (local / "active" / "demo").resolve() == old_target
    state = json.loads((local / "state" / "demo.json").read_text())
    assert state["state"] == SyncState.PARTIAL_RESUMABLE
