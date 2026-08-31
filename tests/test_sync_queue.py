import hashlib
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelctl.errors import ModelctlError
from modelctl.hf_cache import load_record
from modelctl.manifest import parse_manifest
from modelctl.operations import update_model
from modelctl.sync_queue import (
    SyncQueueEntry,
    SyncQueueError,
    execute_sync_queue,
    load_sync_queue,
    prepare_sync_queue,
)


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
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
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
            metadata.write_text(
                f"{kwargs['revision']}\n{digest}\n{time.time()}\n"
            )
        (root / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        return str(root)


def _nas_model(nas, name, repo, sha):
    update_model(
        nas,
        parse_manifest({"repo": repo, "runtime": "vllm"}, name),
        api=FakeApi(sha),
        snapshot=FakeSnapshot({"config.json": b"data"}),
    )


def _copy_runner(fail_for=None):
    def runner(command, check, **kwargs):
        source = Path(command[-2].removesuffix("/"))
        destination = Path(command[-1].removesuffix("/"))
        if fail_for and fail_for in str(destination):
            raise subprocess.CalledProcessError(1, command, stderr="boom\n")
        destination.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.is_file():
                shutil.copy2(item, destination / item.name)

    return runner


def _two_model_store(tmp_path):
    nas = tmp_path / "nas"
    _nas_model(nas, "alpha", "org/alpha", "d" * 40)
    _nas_model(nas, "beta", "org/beta", "e" * 40)
    return nas


def test_load_sync_queue_reads_name_list(tmp_path):
    queue_file = tmp_path / "models.txt"
    queue_file.write_text("- alpha\n- beta\n", encoding="utf-8")
    entries = load_sync_queue(queue_file)
    assert entries == [SyncQueueEntry("alpha"), SyncQueueEntry("beta")]


def test_load_sync_queue_strips_and_requires_non_empty_names(tmp_path):
    queue_file = tmp_path / "models.txt"
    queue_file.write_text("- alpha \n- '  '\n", encoding="utf-8")
    with pytest.raises(SyncQueueError, match="entry 2"):
        load_sync_queue(queue_file)


def test_load_sync_queue_rejects_empty_file(tmp_path):
    queue_file = tmp_path / "models.txt"
    queue_file.write_text("", encoding="utf-8")
    with pytest.raises(SyncQueueError, match="empty"):
        load_sync_queue(queue_file)


def test_load_sync_queue_rejects_non_list_document(tmp_path):
    queue_file = tmp_path / "models.txt"
    queue_file.write_text("models:\n  - alpha\n", encoding="utf-8")
    with pytest.raises(SyncQueueError, match="YAML list"):
        load_sync_queue(queue_file)


def test_load_sync_queue_rejects_non_string_entries(tmp_path):
    queue_file = tmp_path / "models.txt"
    queue_file.write_text("- alpha\n- 3\n", encoding="utf-8")
    with pytest.raises(SyncQueueError, match="entry 2"):
        load_sync_queue(queue_file)


def test_load_sync_queue_rejects_missing_file(tmp_path):
    with pytest.raises(SyncQueueError, match="not found"):
        load_sync_queue(tmp_path / "missing.txt")


def test_prepare_sync_queue_resolves_names_and_repositories(tmp_path):
    nas = _two_model_store(tmp_path)
    prepared = prepare_sync_queue(
        nas,
        [SyncQueueEntry("alpha"), SyncQueueEntry("org/beta")],
    )
    assert [(item.name, item.entry.identifier) for item in prepared] == [
        ("alpha", "alpha"),
        ("beta", "org/beta"),
    ]


def test_prepare_sync_queue_rejects_unknown_names_before_any_transfer(
    tmp_path,
):
    nas = _two_model_store(tmp_path)
    with pytest.raises(SyncQueueError, match="ghost"):
        prepare_sync_queue(nas, [SyncQueueEntry("alpha"), SyncQueueEntry("ghost")])


def test_prepare_sync_queue_rejects_unknown_repositories(tmp_path):
    nas = _two_model_store(tmp_path)
    with pytest.raises(SyncQueueError, match="org/missing"):
        prepare_sync_queue(nas, [SyncQueueEntry("org/missing")])


def test_prepare_sync_queue_rejects_duplicate_identifiers(tmp_path):
    nas = _two_model_store(tmp_path)
    with pytest.raises(SyncQueueError, match="already selected"):
        prepare_sync_queue(
            nas, [SyncQueueEntry("alpha"), SyncQueueEntry("org/alpha")]
        )


def test_execute_sync_queue_syncs_every_entry_in_order(tmp_path):
    nas = _two_model_store(tmp_path)
    cache = tmp_path / "hub"
    prepared = prepare_sync_queue(
        nas, [SyncQueueEntry("alpha"), SyncQueueEntry("beta")]
    )
    results = execute_sync_queue(
        nas, cache, prepared, runner=_copy_runner()
    )
    assert [item.name for item in results] == ["alpha", "beta"]
    assert all(item.succeeded for item in results)
    for item in results:
        assert load_record(cache, item.name).snapshot.is_dir()


def test_execute_sync_queue_continues_after_failure(tmp_path):
    nas = _two_model_store(tmp_path)
    cache = tmp_path / "hub"
    prepared = prepare_sync_queue(
        nas, [SyncQueueEntry("alpha"), SyncQueueEntry("beta")]
    )
    results = execute_sync_queue(
        nas,
        cache,
        prepared,
        runner=_copy_runner(fail_for="e" * 40),
    )
    assert [item.name for item in results] == ["alpha", "beta"]
    assert results[0].succeeded
    assert not results[1].succeeded
    assert isinstance(results[1].error, ModelctlError)
    assert "staging is resumable" in str(results[1].error)
    assert load_record(cache, "alpha").snapshot.is_dir()
    with pytest.raises(ModelctlError, match="no valid local cache record"):
        load_record(cache, "beta")


def test_execute_sync_queue_with_jobs_captures_output_and_reports_stderr(
    tmp_path,
):
    nas = _two_model_store(tmp_path)
    cache = tmp_path / "hub"
    prepared = prepare_sync_queue(
        nas, [SyncQueueEntry("alpha"), SyncQueueEntry("beta")]
    )
    results = execute_sync_queue(
        nas,
        cache,
        prepared,
        jobs=2,
        runner=_copy_runner(fail_for="e" * 40),
    )
    assert [item.name for item in results] == ["alpha", "beta"]
    assert results[0].succeeded
    assert not results[1].succeeded
    assert "rsync failed (exit 1)" in str(results[1].error)
    assert "boom" in str(results[1].error)
    assert load_record(cache, "alpha").snapshot.is_dir()


def test_execute_sync_queue_rejects_invalid_jobs(tmp_path):
    nas = _two_model_store(tmp_path)
    prepared = prepare_sync_queue(nas, [SyncQueueEntry("alpha")])
    with pytest.raises(SyncQueueError, match="jobs must be at least 1"):
        execute_sync_queue(nas, tmp_path / "hub", prepared, jobs=0)
