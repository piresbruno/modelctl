from types import SimpleNamespace

import pytest

from modelctl import catalog
from modelctl.catalog import (
    catalog_lock,
    catalog_status,
    load_catalog,
    mark_catalog_dirty_locked,
    refresh_catalog,
    refresh_catalog_locked,
)
from modelctl.errors import ModelctlError
from modelctl.layout import Layout


def _model(name: str, repo: str = "org/model", runtime: str = "vllm"):
    return SimpleNamespace(name=name, repo=repo, runtime=runtime)


def test_refresh_writes_exact_projection_and_skips_unchanged_catalog(tmp_path):
    models = [_model("zeta"), _model("alpha", runtime="llama.cpp")]

    _, first = refresh_catalog(tmp_path, lambda root: models)
    document = load_catalog(tmp_path)
    stat = first.path.stat()
    content = first.path.read_bytes()

    assert first.changed is True
    assert first.generation == 1
    assert document["models"] == [
        {"name": "alpha", "runtime": "llama.cpp", "repository": "org/model"},
        {"name": "zeta", "runtime": "vllm", "repository": "org/model"},
    ]

    _, second = refresh_catalog(tmp_path, lambda root: list(reversed(models)))
    assert second.changed is False
    assert second.generation == 1
    assert second.path.read_bytes() == content
    assert second.path.stat().st_ino == stat.st_ino
    assert second.path.stat().st_mtime_ns == stat.st_mtime_ns
    assert catalog_status(tmp_path).status == "ready"


def test_catalog_status_detects_dirty_and_external_active_changes(tmp_path):
    layout = Layout(tmp_path)
    layout.prepare()
    refresh_catalog(tmp_path, lambda root: [])

    with catalog_lock(tmp_path):
        mark_catalog_dirty_locked(tmp_path, "test mutation")
    assert catalog_status(tmp_path).status == "dirty"

    refresh_catalog(tmp_path, lambda root: [])
    (layout.active / "unexpected").mkdir()
    status = catalog_status(tmp_path)
    assert status.status == "stale"
    assert "fingerprint" in status.detail


def test_failed_refresh_preserves_previous_catalog_and_dirty_marker(
    tmp_path, monkeypatch
):
    refresh_catalog(tmp_path, lambda root: [_model("old")])
    path = tmp_path / "catalog.json"
    previous = path.read_bytes()
    real_atomic_json = catalog._atomic_json

    def fail_catalog_write(destination, payload):
        if destination == path:
            raise OSError("simulated catalog publication failure")
        real_atomic_json(destination, payload)

    monkeypatch.setattr(catalog, "_atomic_json", fail_catalog_write)
    with catalog_lock(tmp_path):
        mark_catalog_dirty_locked(tmp_path, "activating new")
        with pytest.raises(ModelctlError, match="failed to refresh"):
            refresh_catalog_locked(
                tmp_path, lambda root: [_model("old"), _model("new")]
            )

    assert path.read_bytes() == previous
    assert (tmp_path / "state" / ".catalog-dirty").is_file()
    assert catalog_status(tmp_path).status == "dirty"


def test_catalog_reader_refuses_symlink(tmp_path):
    target = tmp_path / "outside.json"
    target.write_text("{}")
    (tmp_path / "catalog.json").symlink_to(target)

    with pytest.raises(ModelctlError, match="missing or invalid catalog"):
        load_catalog(tmp_path)
