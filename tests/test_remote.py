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
from modelctl.operations import sync_local, update_model
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


def _local_cache(tmp_path, name="demo", files=None):
    """Build a NAS object and sync it into a local HF cache registration."""
    nas = _nas_object(tmp_path, name=name, files=files)
    cache = tmp_path / "hub"

    def copy(command, check):
        source = Path(command[-2].removesuffix("/"))
        destination = Path(command[-1].removesuffix("/"))
        list_arg = next(
            argument for argument in command if argument.startswith("--files-from=")
        )
        names = [
            entry.decode()
            for entry in Path(list_arg.split("=", 1)[1])
            .read_bytes()
            .split(b"\0")
            if entry
        ]
        for entry in names:
            staged = destination / entry
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / entry, staged)

    sync_local(nas, cache, name, runner=copy)
    return cache


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
        self.file_lists = []

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        for argument in command:
            if argument.startswith("--files-from="):
                path = Path(argument.split("=", 1)[1])
                names = [
                    name.decode() for name in path.read_bytes().split(b"\0") if name
                ]
                self.file_list = names
                self.file_lists.append(names)
        if any(
            isinstance(argument, str) and "cd; for p in " in argument
            for argument in command
        ):
            return SimpleNamespace(
                returncode=0, stdout="/usr/local/bin/modelctl\n", stderr=""
            )
        if any(
            isinstance(argument, str) and "mkdir -p" in argument
            for argument in command
        ):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
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
        for name in self.file_list:
            staged = destination / name
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, staged)
        return SimpleNamespace(returncode=0)


def test_push_orchestrates_probe_rsync_and_commit(tmp_path):
    cache = _local_cache(tmp_path)
    record = load_record(cache, "demo")
    staging = staging_path_for(cache, record.repo, record.commit, record.files)

    fake = FakeRunner(cache)
    messages = []
    result = push_model(
        cache, "demo", host="node-b", runner=fake, progress=messages.append
    )

    assert result == str(cache / "models--org--demo" / "snapshots" / COMMIT)
    assert messages[3] == (
        f"push: transferring 1 files (+2 metadata files) to node-b:{staging}"
    )
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
    prepare = fake.calls[2]
    assert prepare[:2] == ["ssh", "node-b"]
    assert prepare[2:4] == ["sh", "-c"]
    assert "mkdir -p" in prepare[4]
    assert prepare[5:7] == ["sh", str(staging)]
    assert prepare[7] == str(cache)
    rsync = fake.calls[3]
    assert rsync[0] == "rsync"
    for flag in (
        "--archive",
        "--copy-links",
        "--partial",
        "--delete",
        "--from0",
        "-e",
    ):
        assert flag in rsync
    assert rsync[rsync.index("-e") + 1] == "ssh -o Compression=no"
    overlay = Path(rsync[-2].removesuffix("/"))
    assert overlay.name.startswith("modelctl-push-cache-")
    assert rsync[-1] == f"node-b:{staging}/"
    assert fake.calls[4] == [
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
    assert not overlay.exists()


def test_push_jobs_splits_transfer_into_parallel_streams(tmp_path):
    cache = _local_cache(
        tmp_path,
        files={"a.bin": b"1", "b.bin": b"2", "c.bin": b"3", "d.bin": b"4"},
    )
    fake = FakeRunner(cache)
    push_model(cache, "demo", host="node-b", jobs=3, runner=fake)

    rsync_calls = [call for call in fake.calls if call[0] == "rsync"]
    assert len(rsync_calls) == 3
    assert all(call[-1] == rsync_calls[0][-1] for call in rsync_calls)
    assert all(
        Path(call[-2].removesuffix("/")).name.startswith("modelctl-push-cache-")
        for call in rsync_calls
    )
    commit = fake.calls[-1]
    assert commit[2] == "/usr/local/bin/modelctl"
    assert "receive-cache" in commit
    transferred = set()
    for names in fake.file_lists:
        assert names
        assert names == sorted(names)
        transferred.update(names)
    assert transferred == {
        ".cache/huggingface/download/a.bin.metadata",
        ".cache/huggingface/download/b.bin.metadata",
        ".cache/huggingface/download/c.bin.metadata",
        ".cache/huggingface/download/d.bin.metadata",
        ".modelctl.json",
        "a.bin",
        "b.bin",
        "c.bin",
        "d.bin",
    }


def test_push_distributes_streams_across_fabric_hosts(tmp_path):
    cache = _local_cache(
        tmp_path,
        files={"a.bin": b"1", "b.bin": b"2", "c.bin": b"3", "d.bin": b"4"},
    )
    fake = FakeRunner(cache)
    push_model(
        cache,
        "demo",
        host="node-b",
        job_hosts=["node-c"],
        jobs=2,
        runner=fake,
    )

    rsync_calls = [call for call in fake.calls if call[0] == "rsync"]
    assert len(rsync_calls) == 2
    destinations = [call[-1].split(":", 1)[0] for call in rsync_calls]
    assert sorted(destinations) == ["node-b", "node-c"]
    assert destinations[0] != destinations[1]
    # probe and commit still go through the control host
    assert fake.calls[1][1] == "node-b"
    assert fake.calls[-1][1] == "node-b"


def test_push_round_robins_streams_over_three_hosts(tmp_path):
    cache = _local_cache(
        tmp_path,
        files={"a.bin": b"1", "b.bin": b"2", "c.bin": b"3", "d.bin": b"4"},
    )
    fake = FakeRunner(cache)
    push_model(
        cache,
        "demo",
        host="h1",
        job_hosts=["h2", "h3"],
        jobs=5,
        runner=fake,
    )
    rsync_calls = [call for call in fake.calls if call[0] == "rsync"]
    hosts = [call[-1].split(":", 1)[0] for call in rsync_calls]
    assert hosts == ["h1", "h2", "h3", "h1", "h2"]


def test_push_jobs_clamps_streams_to_file_count(tmp_path):
    cache = _local_cache(
        tmp_path,
        files={"a.bin": b"1", "b.bin": b"2", "c.bin": b"3", "d.bin": b"4"},
    )
    fake = FakeRunner(cache)
    push_model(cache, "demo", host="node-b", jobs=50, runner=fake)
    rsync_calls = [call for call in fake.calls if call[0] == "rsync"]
    assert len(rsync_calls) == 9  # 4 files + 4 metadata + .modelctl.json
    assert all(len(names) == 1 for names in fake.file_lists)


def test_push_jobs_reports_resumable_error_when_stream_fails(tmp_path):
    cache = _local_cache(
        tmp_path,
        files={"a.bin": b"1", "b.bin": b"2", "c.bin": b"3", "d.bin": b"4"},
    )

    def failing(command, **kwargs):
        if command[0] == "rsync":
            raise RuntimeError("stream transfer failed")
        if any(
            isinstance(argument, str) and "cd; for p in " in argument
            for argument in command
        ):
            return SimpleNamespace(
                returncode=0, stdout="/usr/local/bin/modelctl\n", stderr=""
            )
        if any(
            isinstance(argument, str) and "mkdir -p" in argument
            for argument in command
        ):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "--probe" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"proto": 1, "cache": str(cache)}),
                stderr="",
            )
        if "receive-cache" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"proto": 1, "cache": str(cache), "snapshot": "/snap"}
                ),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(ModelctlError, match="resumable"):
        push_model(cache, "demo", host="node-b", jobs=2, runner=failing)

def test_push_requires_local_cache_record(tmp_path):
    cache = tmp_path / "hub"
    fake = FakeRunner(cache)
    with pytest.raises(ModelctlError, match="no valid local cache record"):
        push_model(cache, "demo", host="node-b", runner=fake)


def test_push_uses_port_identity_and_fabric_host(tmp_path):
    cache = _local_cache(tmp_path)
    fake = FakeRunner(cache)
    push_model(
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
    assert fake.calls[3][fake.calls[3].index("-e") + 1] == (
        "ssh -p 2222 -i /keys/connectx -o Compression=no"
    )


def test_push_uses_remote_modelctl_override(tmp_path):
    cache = _local_cache(tmp_path)
    fake = FakeRunner(cache)
    push_model(
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
    def not_found(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    with pytest.raises(ModelctlError, match="--remote-modelctl"):
        push_model(
            _local_cache(tmp_path), "demo", host="node-b", runner=not_found
        )


def test_push_resolves_repository_id_to_registered_name(tmp_path):
    cache = _local_cache(tmp_path)
    fake = FakeRunner(cache)
    push_model(cache, "org/demo", host="node-b", runner=fake)
    probe = fake.calls[1]
    assert "--probe" in probe
    assert probe[probe.index("--probe") + 1] == "demo"


def test_push_fails_when_remote_unreachable(tmp_path):
    def refused(command, **kwargs):
        return SimpleNamespace(
            returncode=255,
            stdout="",
            stderr="ssh: connect to host node-b port 22: Connection refused",
        )

    with pytest.raises(ModelctlError, match="node-b"):
        push_model(
            _local_cache(tmp_path), "demo", host="node-b", runner=refused
        )


def test_push_fails_on_protocol_mismatch(tmp_path):
    def old_probe(command, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"proto": 99, "cache": ""}),
            stderr="",
        )

    with pytest.raises(ModelctlError, match="incompatible"):
        push_model(
            _local_cache(tmp_path), "demo", host="node-b", runner=old_probe
        )


def test_push_reports_resumable_rsync_failure(tmp_path):
    cache = _local_cache(tmp_path)

    def fail_rsync(command, **kwargs):
        if any(
            isinstance(argument, str) and "cd; for p in " in argument
            for argument in command
        ):
            return SimpleNamespace(
                returncode=0, stdout="/usr/local/bin/modelctl\n", stderr=""
            )
        if any(
            isinstance(argument, str) and "mkdir -p" in argument
            for argument in command
        ):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "receive-cache" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"proto": 1, "cache": str(cache)}),
                stderr="",
            )
        raise RuntimeError("rsync error 23")

    with pytest.raises(ModelctlError, match="resumable"):
        push_model(cache, "demo", host="node-b", runner=fail_rsync)


def test_push_fails_when_remote_cache_not_writable(tmp_path):
    cache = _local_cache(tmp_path)

    def deny_cache(command, **kwargs):
        if any(
            isinstance(argument, str) and "mkdir -p" in argument
            for argument in command
        ):
            return SimpleNamespace(
                returncode=1, stdout="", stderr="mkdir: cannot create directory: Permission denied"
            )
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
                stdout=json.dumps({"proto": 1, "cache": str(cache)}),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(ModelctlError, match="not writable on node-b"):
        push_model(cache, "demo", host="node-b", runner=deny_cache)


def test_push_end_to_end_through_real_receive_cli(tmp_path, monkeypatch):
    """The ssh argv contract: the remote 'modelctl receive-cache' invocation
    published by push must be a valid local modelctl invocation. The ssh
    transport is emulated locally, so this exercises the real probe and commit
    CLI paths plus the rsync file-list placement."""
    import subprocess
    import sys

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
    cache = _local_cache(
        tmp_path, files={"config.json": b"data", "README.md": b"#"}
    )

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
            if remote_args and remote_args[0] == "sh":
                return subprocess.run(
                    ["sh", "-c", remote_args[2], "sh", *remote_args[4:]],
                    capture_output=True,
                    text=True,
                )
            if remote_args and remote_args[0] != "receive-cache":
                remote_args = remote_args[1:]
            return subprocess.run(
                [sys.executable, "-m", "modelctl", *remote_args],
                capture_output=True,
                text=True,
            )

    local_ssh = LocalSsh()
    result = push_model(cache, "demo", host="node-b", runner=local_ssh)

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
