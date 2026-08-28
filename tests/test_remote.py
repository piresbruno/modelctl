import hashlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelctl.errors import ModelctlError, ValidationError
from modelctl.hf_cache import (
    canonical_cache,
    load_record,
    read_source_etag,
    receive_staged_cache,
    staging_path_for,
    state_root,
)
from modelctl.manifest import parse_manifest
from modelctl.operations import update_model
from modelctl.remote import push_model

COMMIT = "d" * 40


@pytest.fixture(autouse=True)
def isolate_modelctl_state(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))


class FakeApi:
    def __init__(self, sha):
        self.sha = sha

    def model_info(self, repo, revision):
        return SimpleNamespace(sha=self.sha)


class FakeSnapshot:
    def __init__(self, files):
        self.files = files

    def __call__(self, **kwargs):
        if kwargs.get("dry_run"):
            return [
                SimpleNamespace(filename=name, file_size=len(content))
                for name, content in self.files.items()
            ]
        root = Path(kwargs["local_dir"])
        for name, content in self.files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            digest = hashlib.sha1(
                f"blob {len(content)}\0".encode() + content
            ).hexdigest()
            metadata = (
                root / ".cache" / "huggingface" / "download" / f"{name}.metadata"
            )
            metadata.parent.mkdir(parents=True, exist_ok=True)
            metadata.write_text(f"{kwargs['revision']}\n{digest}\n{time.time()}\n")
        (root / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        return str(root)


def _nas_object(tmp_path, name="demo", files=None):
    nas = tmp_path / "nas"
    manifest = parse_manifest({"repo": "org/demo", "runtime": "vllm"}, name)
    update_model(
        nas,
        manifest,
        api=FakeApi(COMMIT),
        snapshot=FakeSnapshot(files or {"config.json": b"data"}),
    )
    return nas


def _stage_transfer(nas, cache, name="demo", *, staging=None):
    object_path = (nas / "active" / name).resolve()
    metadata = json.loads((object_path / ".modelctl.json").read_text())
    files = {
        item["path"]: read_source_etag(object_path, item["path"], metadata["commit"])
        for item in metadata["files"]
    }
    if staging is None:
        staging = staging_path_for(cache, metadata["repo"], metadata["commit"], files)
    staging.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(object_path, staging)
    return staging


def test_receive_probe_is_read_only(tmp_path):
    cache = tmp_path / "hub"
    payload = receive_staged_cache(cache, "demo", probe=True)
    assert payload["proto"] == 1
    assert payload["cache"] == str(canonical_cache(cache))
    assert not cache.exists()


def test_receive_requires_staging(tmp_path):
    with pytest.raises(ModelctlError, match="--staging"):
        receive_staged_cache(tmp_path / "hub", "demo")


def test_receive_rejects_missing_staging_dir(tmp_path):
    with pytest.raises(ValidationError, match="not a real directory"):
        receive_staged_cache(tmp_path / "hub", "demo", staging=tmp_path / "nope")


def test_receive_rejects_wrong_name(tmp_path):
    nas = _nas_object(tmp_path)
    cache = tmp_path / "hub"
    staging = _stage_transfer(nas, cache)
    with pytest.raises(ValidationError, match="belongs to"):
        receive_staged_cache(cache, "other", staging=staging)


def test_receive_rejects_misplaced_staging(tmp_path):
    nas = _nas_object(tmp_path)
    cache = tmp_path / "hub"
    object_path = (nas / "active" / "demo").resolve()
    wrong = tmp_path / "elsewhere"
    shutil.copytree(object_path, wrong / "staging")
    with pytest.raises(ValidationError, match="does not match"):
        receive_staged_cache(cache, "demo", staging=wrong / "staging")


def test_receive_publishes_validated_staged_transfer(tmp_path):
    nas = _nas_object(tmp_path)
    cache = tmp_path / "hub"
    staging = _stage_transfer(nas, cache)
    payload = receive_staged_cache(cache, "demo", staging=staging)
    snapshot = cache / "models--org--demo" / "snapshots" / COMMIT
    assert payload["proto"] == 1
    assert payload["cache"] == str(canonical_cache(cache))
    assert payload["snapshot"] == str(snapshot)
    assert Path(payload["entrypoint"]).resolve() == snapshot.resolve()
    assert not staging.exists()
    assert (snapshot / "config.json").is_symlink()
    assert (cache / "models--org--demo" / "blobs").exists()
    assert (cache / "models--org--demo" / "refs" / "main").read_text() == COMMIT
    assert load_record(cache, "demo").snapshot == snapshot.resolve()
    state = json.loads((state_root(cache) / "state" / "demo.json").read_text())
    assert state["state"] == "READY_FOR_SERVICE_RESTART"
    assert [item["state"] for item in state["history"]][-3:] == [
        "VALIDATING_STAGING",
        "PUBLISHING_CACHE",
        "READY_FOR_SERVICE_RESTART",
    ]


def test_receive_rejects_corrupted_transfer_without_publishing(tmp_path):
    nas = _nas_object(tmp_path)
    cache = tmp_path / "hub"
    staging = _stage_transfer(nas, cache)
    (staging / "config.json").write_bytes(b"tampered!")
    with pytest.raises(ValidationError, match="size mismatch"):
        receive_staged_cache(cache, "demo", staging=staging)
    repository = cache / "models--org--demo"
    assert not (repository / "snapshots").exists()
    assert not (repository / "blobs").exists()
    assert not (state_root(cache) / "active" / "demo.json").exists()
    assert staging.exists()


class FakeRunner:
    def __init__(self, cache):
        self.cache = cache
        self.calls = []
        self.file_list = []

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        for argument in command:
            if argument.startswith("--files-from="):
                path = Path(argument.split("=", 1)[1])
                self.file_list = [
                    name.decode() for name in path.read_bytes().split(b"\0") if name
                ]
        if any(
            isinstance(argument, str) and "cd; for p in " in argument
            for argument in command
        ):
            return SimpleNamespace(
                returncode=0, stdout="/usr/local/bin/modelctl\n", stderr=""
            )
        if "receive-cache" in command and "--probe" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"proto": 1, "cache": str(self.cache)}),
                stderr="",
            )
        if "receive-cache" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "proto": 1,
                        "cache": str(self.cache),
                        "snapshot": str(
                            self.cache / "models--org--demo" / "snapshots" / COMMIT
                        ),
                    }
                ),
                stderr="",
            )
        source = Path(command[-2].removesuffix("/"))
        destination = Path(command[-1].split(":", 1)[1].removesuffix("/"))
        destination.mkdir(parents=True, exist_ok=True)
        for entry in source.rglob("*"):
            if entry.is_file():
                target = destination / entry.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(entry, target)
        return SimpleNamespace(returncode=0)


def test_push_orchestrates_probe_rsync_and_commit(tmp_path):
    nas = _nas_object(tmp_path)
    cache = tmp_path / "hub"
    object_path = (nas / "active" / "demo").resolve()
    metadata = json.loads((object_path / ".modelctl.json").read_text())
    files = {
        item["path"]: read_source_etag(object_path, item["path"], metadata["commit"])
        for item in metadata["files"]
    }
    staging = staging_path_for(cache, metadata["repo"], metadata["commit"], files)

    fake = FakeRunner(cache)
    result = push_model(nas, cache, "demo", host="node-b", runner=fake)

    assert result == str(cache / "models--org--demo" / "snapshots" / COMMIT)
    assert fake.calls[0][:2] == ["ssh", "node-b"]
    assert fake.calls[0][2:4] == ["sh", "-c"]
    script = fake.calls[0][4]
    assert "$HOME/.local/bin/modelctl" in script
    assert fake.calls[1] == [
        "ssh",
        "node-b",
        "/usr/local/bin/modelctl",
        "receive-cache",
        "--probe",
        "demo",
        "--cache-dir",
        str(cache),
    ]
    rsync = fake.calls[2]
    assert rsync[0] == "rsync"
    for flag in ("--archive", "--partial", "--delete", "--from0", "-e"):
        assert flag in rsync
    assert rsync[rsync.index("-e") + 1] == "ssh -o Compression=no"
    assert rsync[-2] == f"{object_path}/"
    assert rsync[-1] == f"node-b:{staging}/"
    assert fake.calls[3] == [
        "ssh",
        "node-b",
        "/usr/local/bin/modelctl",
        "receive-cache",
        "demo",
        "--cache-dir",
        str(cache),
        "--staging",
        str(staging),
    ]
    assert sorted(fake.file_list) == [
        ".cache/huggingface/download/config.json.metadata",
        ".modelctl.json",
        "config.json",
    ]


def test_push_uses_port_identity_and_fabric_host(tmp_path):
    nas = _nas_object(tmp_path)
    cache = tmp_path / "hub"
    fake = FakeRunner(cache)
    push_model(
        nas,
        cache,
        "demo",
        host="user@10.0.0.2",
        port=2222,
        identity="/keys/connectx",
        runner=fake,
    )
    assert fake.calls[1] == [
        "ssh",
        "-p",
        "2222",
        "-i",
        "/keys/connectx",
        "user@10.0.0.2",
        "/usr/local/bin/modelctl",
        "receive-cache",
        "--probe",
        "demo",
        "--cache-dir",
        str(cache),
    ]
    assert fake.calls[2][fake.calls[2].index("-e") + 1] == (
        "ssh -p 2222 -i /keys/connectx -o Compression=no"
    )


def test_push_uses_remote_modelctl_override(tmp_path):
    nas = _nas_object(tmp_path)
    cache = tmp_path / "hub"
    fake = FakeRunner(cache)
    push_model(
        nas,
        cache,
        "demo",
        host="node-b",
        remote_modelctl="/opt/modelctl/bin/modelctl",
        runner=fake,
    )
    assert fake.calls[0][2:4] != ["sh", "-c"]
    assert fake.calls[0] == [
        "ssh",
        "node-b",
        "/opt/modelctl/bin/modelctl",
        "receive-cache",
        "--probe",
        "demo",
        "--cache-dir",
        str(cache),
    ]


def test_push_fails_when_remote_modelctl_missing(tmp_path):
    nas = _nas_object(tmp_path)

    def not_found(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    with pytest.raises(ModelctlError, match="--remote-modelctl"):
        push_model(nas, tmp_path / "hub", "demo", host="node-b", runner=not_found)


def test_push_resolves_repository_id_to_active_name(tmp_path):
    nas = _nas_object(tmp_path)
    cache = tmp_path / "hub"
    fake = FakeRunner(cache)
    push_model(nas, cache, "org/demo", host="node-b", runner=fake)
    probe = fake.calls[1]
    assert "--probe" in probe
    assert probe[probe.index("--probe") + 1] == "demo"


def test_push_fails_when_remote_unreachable(tmp_path):
    nas = _nas_object(tmp_path)

    def refused(command, **kwargs):
        return SimpleNamespace(
            returncode=255,
            stdout="",
            stderr="ssh: connect to host node-b port 22: Connection refused",
        )

    with pytest.raises(ModelctlError, match="node-b"):
        push_model(nas, tmp_path / "hub", "demo", host="node-b", runner=refused)


def test_push_fails_on_protocol_mismatch(tmp_path):
    nas = _nas_object(tmp_path)

    def old_probe(command, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"proto": 99, "cache": ""}),
            stderr="",
        )

    with pytest.raises(ModelctlError, match="incompatible"):
        push_model(nas, tmp_path / "hub", "demo", host="node-b", runner=old_probe)


def test_push_reports_resumable_rsync_failure(tmp_path):
    nas = _nas_object(tmp_path)

    def fail_rsync(command, **kwargs):
        if any(
            isinstance(argument, str) and "cd; for p in " in argument
            for argument in command
        ):
            return SimpleNamespace(
                returncode=0, stdout="/usr/local/bin/modelctl\n", stderr=""
            )
        if "receive-cache" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"proto": 1, "cache": str(tmp_path / "hub")}),
                stderr="",
            )
        raise RuntimeError("rsync error 23")

    with pytest.raises(ModelctlError, match="resumable"):
        push_model(nas, tmp_path / "hub", "demo", host="node-b", runner=fail_rsync)


def test_push_end_to_end_through_real_receive_cli(tmp_path, monkeypatch):
    """The ssh argv contract: the remote 'modelctl receive-cache' invocation
    published by push must be a valid local modelctl invocation. The ssh
    transport is emulated locally, so this exercises the real probe and commit
    CLI paths plus the rsync file-list placement."""
    import subprocess
    import sys

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
    nas = _nas_object(
        tmp_path, files={"config.json": b"data", "README.md": b"#"}
    )
    cache = tmp_path / "hub"

    class LocalSsh:
        def __init__(self):
            self.calls = []
            self.file_list = []

        def __call__(self, command, **kwargs):
            self.calls.append(command)
            if command[0] == "rsync":
                source = Path(command[-2].removesuffix("/"))
                destination = Path(command[-1].split(":", 1)[1].removesuffix("/"))
                list_arg = next(
                    argument
                    for argument in command
                    if argument.startswith("--files-from=")
                )
                names = [
                    name.decode()
                    for name in Path(list_arg.split("=", 1)[1])
                    .read_bytes()
                    .split(b"\0")
                    if name
                ]
                self.file_list = names
                for name in names:
                    staged = destination / name
                    staged.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source / name, staged)
                return SimpleNamespace(returncode=0)
            if any(
                isinstance(argument, str) and "cd; for p in " in argument
                for argument in command
            ):
                return SimpleNamespace(
                    returncode=0, stdout="/usr/local/bin/modelctl\n", stderr=""
                )
            remote_args = [
                argument.strip("'") for argument in command[command.index("node-b") + 1 :]
            ]
            if remote_args and remote_args[0] != "receive-cache":
                remote_args = remote_args[1:]
            return subprocess.run(
                [sys.executable, "-m", "modelctl", *remote_args],
                capture_output=True,
                text=True,
            )

    local_ssh = LocalSsh()
    result = push_model(nas, cache, "demo", host="node-b", runner=local_ssh)

    snapshot = cache / "models--org--demo" / "snapshots" / COMMIT
    assert result == str(snapshot)
    assert (snapshot / "config.json").is_symlink()
    assert (snapshot / "README.md").is_symlink()
    assert (cache / "models--org--demo" / "refs" / "main").read_text() == COMMIT
    assert load_record(cache, "demo").snapshot == snapshot.resolve()
    assert sorted(local_ssh.file_list) == [
        ".cache/huggingface/download/README.md.metadata",
        ".cache/huggingface/download/config.json.metadata",
        ".modelctl.json",
        "README.md",
        "config.json",
    ]
