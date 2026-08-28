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
