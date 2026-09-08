import json
import os
import shutil
from types import SimpleNamespace

import pytest

from modelctl import catalog
from modelctl.catalog import (
    active_fingerprint,
    bump_catalog_token_locked,
    catalog_lock,
    catalog_status,
    catalog_token,
    commit_catalog_delta_locked,
    catalog_dirty,
    load_catalog,
    mark_catalog_dirty_locked,
    refresh_catalog,
    refresh_catalog_locked,
)
from modelctl.errors import CatalogStaleViewError, ModelctlError
from modelctl.layout import Layout, atomic_symlink
from modelctl.manifest import ModelManifest, RuntimeProfile, parse_manifest
from modelctl.operations import catalog_models, list_active_models
from modelctl.validation import ExpectedFile, write_metadata


def _model(name: str, repo: str = "org/model", runtime: str = "vllm", size: int = 1024):
    return SimpleNamespace(name=name, repo=repo, runtime=runtime, size_bytes=size)


def test_catalog_status_reports_mutation_counter_behind(tmp_path):
    refresh_catalog(tmp_path, lambda root: [_model("alpha")])
    with catalog_lock(tmp_path):
        bump_catalog_token_locked(tmp_path)
    status = catalog_status(tmp_path)
    assert status.status == "stale"
    assert "mutation(s) behind" in status.detail


def _real_active_model(root, name: str) -> None:
    layout = Layout(root)
    layout.prepare()
    manifest = ModelManifest(
        name=name, repo=f"org/{name}", runtime=RuntimeProfile("vllm", "vllm")
    )
    object_path = layout.models / "org" / name / "c0ffee"
    object_path.mkdir(parents=True)
    (object_path / "weights.bin").write_bytes(b"x" * 4096)
    write_metadata(
        object_path, manifest, "c0ffee", [ExpectedFile("weights.bin", 4096)], "."
    )
    atomic_symlink(object_path, layout.active_path(name))


def test_refresh_writes_exact_projection_and_skips_unchanged_catalog(tmp_path):
    models = [_model("zeta"), _model("alpha", runtime="llama.cpp")]

    _, first = refresh_catalog(tmp_path, lambda root: models)
    document = load_catalog(tmp_path)
    stat = first.path.stat()
    content = first.path.read_bytes()

    assert first.changed is True
    assert first.generation == 1
    assert document["models"] == [
        {"name": "alpha", "runtime": "llama.cpp", "repository": "org/model", "bytes": 1024},
        {"name": "zeta", "runtime": "vllm", "repository": "org/model", "bytes": 1024},
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


def test_active_fingerprint_normalizes_symlink_rendering(tmp_path):
    layout = Layout(tmp_path)
    layout.prepare()
    obj = layout.models / "org" / "model" / "hash"
    obj.mkdir(parents=True)
    atomic_symlink(obj, layout.active_path("demo"))
    relative_fp = active_fingerprint(tmp_path)

    (layout.active / "demo").unlink()
    os.symlink(str(obj), layout.active_path("demo"))
    absolute_fp = active_fingerprint(tmp_path)

    assert relative_fp == absolute_fp

    (layout.active / "gone").symlink_to(tmp_path / "models" / "nope")
    assert active_fingerprint(tmp_path) != relative_fp


def test_active_fingerprint_out_of_root_targets_use_raw_text(tmp_path):
    layout = Layout(tmp_path)
    layout.prepare()
    raw = os.path.relpath("/dev/null", layout.active)
    os.symlink(raw, layout.active_path("demo"))

    entries = catalog._active_entries(tmp_path)

    assert entries == [{"name": "demo", "kind": "object", "value": raw}]


def test_refresh_preserve_if_empty_keeps_nonempty_catalog(tmp_path):
    refresh_catalog(tmp_path, lambda root: [_model("alpha")])
    document = load_catalog(tmp_path)

    with pytest.raises(CatalogStaleViewError, match="refusing to overwrite"):
        refresh_catalog(tmp_path, lambda root: [], preserve_if_empty=True)

    reloaded = load_catalog(tmp_path)
    assert len(reloaded["models"]) == 1
    assert reloaded["generation"] == document["generation"]


def test_refresh_preserve_allows_growth_and_empty_start(tmp_path):
    _, first = refresh_catalog(tmp_path, lambda root: [], preserve_if_empty=True)
    assert first.changed is True
    assert first.models == []

    _, second = refresh_catalog(
        tmp_path, lambda root: [_model("alpha")], preserve_if_empty=True
    )
    assert second.changed is True

    _, third = refresh_catalog(tmp_path, lambda root: [])
    assert third.changed is True
    assert third.models == []


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


def test_refresh_keeps_repairable_directory_models(tmp_path):
    """Regenerating the catalog must not drop a model whose active symlink was
    replaced by a validated copy; doctor marks it repairable and repair-active
    converges the reference."""
    name = "demo"
    layout = Layout(tmp_path)
    layout.prepare()
    manifest = parse_manifest({"repo": "org/demo", "runtime": "vllm"}, name)
    commit = "a" * 40
    object_path = layout.object_path(manifest, commit)
    object_path.mkdir(parents=True)
    (object_path / "config.json").write_bytes(b"{}")
    write_metadata(
        object_path, manifest, commit, [ExpectedFile("config.json", 2)], "."
    )
    (tmp_path / "manifests").mkdir(exist_ok=True)
    (tmp_path / "manifests" / f"{name}.yaml").write_text(
        "name: demo\nrepo: org/demo\nruntime: vllm\n"
    )
    layout.state_path(name).write_text(
        json.dumps(
            {
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
        )
    )
    reference = layout.active_path(name)
    shutil.copytree(object_path, reference)

    assert list_active_models(tmp_path) == []
    models, _ = refresh_catalog(tmp_path, catalog_models)
    assert [model.name for model in models] == [name]
    [record] = load_catalog(tmp_path)["models"]
    assert {
        key: value for key, value in record.items() if key != "bytes"
    } == {"name": "demo", "runtime": "vllm", "repository": "org/demo"}
    assert record["bytes"] > 0
    # The catalog fingerprints the current active state, so it is ready even
    # while the reference is a repairable directory; doctor flags the state.
    assert catalog_status(tmp_path).status == "ready"


def test_load_catalog_reader_refuses_symlink(tmp_path):
    target = tmp_path / "outside.json"
    target.write_text("{}")
    (tmp_path / "catalog.json").symlink_to(target)

    with pytest.raises(ModelctlError, match="missing or invalid catalog"):
        load_catalog(tmp_path)


def test_load_catalog_rejects_invalid_bytes(tmp_path):
    refresh_catalog(tmp_path, lambda root: [_model("alpha")])
    path = tmp_path / "catalog.json"
    document = json.loads(path.read_text())
    document["models"][0]["bytes"] = -1
    path.write_text(json.dumps(document))

    with pytest.raises(ModelctlError, match="invalid model record"):
        load_catalog(tmp_path)


def test_load_catalog_rejects_non_integer_bytes(tmp_path):
    refresh_catalog(tmp_path, lambda root: [_model("alpha")])
    path = tmp_path / "catalog.json"
    document = json.loads(path.read_text())
    document["models"][0]["bytes"] = "many"
    path.write_text(json.dumps(document))

    with pytest.raises(ModelctlError, match="invalid model record"):
        load_catalog(tmp_path)


def test_token_starts_at_zero_and_bumps_under_lock(tmp_path):
    assert catalog_token(tmp_path) == 0
    with catalog_lock(tmp_path):
        assert bump_catalog_token_locked(tmp_path) == 1
    with catalog_lock(tmp_path):
        assert bump_catalog_token_locked(tmp_path) == 2
    assert catalog_token(tmp_path) == 2


def test_refresh_stamps_mutation_token(tmp_path):
    with catalog_lock(tmp_path):
        bump_catalog_token_locked(tmp_path)
    refresh_catalog(tmp_path, lambda root: [_model("alpha")])
    document = load_catalog(tmp_path)
    assert document["seq"] == catalog_token(tmp_path) == 1


def test_load_catalog_accepts_schema2_without_seq(tmp_path):
    refresh_catalog(tmp_path, lambda root: [_model("alpha")])
    path = tmp_path / "catalog.json"
    document = json.loads(path.read_text())
    document["schema"] = 2
    del document["seq"]
    path.write_text(json.dumps(document))

    assert load_catalog(tmp_path)["schema"] == 2
    status = catalog_status(tmp_path)
    assert status.status == "stale"
    assert "predates mutation tracking" in status.detail


def test_delta_upsert_and_remove_keep_projection_sorted(tmp_path):
    refresh_catalog(tmp_path, lambda root: [_model("alpha")])
    with catalog_lock(tmp_path):
        token = bump_catalog_token_locked(tmp_path)
        commit_catalog_delta_locked(
            tmp_path,
            lambda root: [],
            upsert={
                "name": "beta",
                "runtime": "vllm",
                "repository": "org/beta",
                "bytes": 5,
            },
        )
    document = load_catalog(tmp_path)
    assert [item["name"] for item in document["models"]] == ["alpha", "beta"]
    assert document["seq"] == token
    assert document["generation"] == 2

    with catalog_lock(tmp_path):
        bump_catalog_token_locked(tmp_path)
        commit_catalog_delta_locked(tmp_path, lambda root: [], remove="alpha")
    document = load_catalog(tmp_path)
    assert [item["name"] for item in document["models"]] == ["beta"]
    assert document["generation"] == 3
    assert catalog_status(tmp_path).status == "ready"



def test_delta_upsert_preserves_siblings_without_rescanning(tmp_path):
    """The delta path must trust the stored catalog for sibling records: a
    mutator whose view of OTHER models is degraded still publishes its own
    validated record without dropping anyone."""
    refresh_catalog(tmp_path, lambda root: [_model("alpha")])

    def forbidden_scan(root):
        raise AssertionError("delta commit must not rescan the store")

    with catalog_lock(tmp_path):
        bump_catalog_token_locked(tmp_path)
        commit_catalog_delta_locked(
            tmp_path,
            forbidden_scan,
            upsert={
                "name": "beta",
                "runtime": "vllm",
                "repository": "org/beta",
                "bytes": 5,
            },
        )
    document = load_catalog(tmp_path)
    assert [item["name"] for item in document["models"]] == ["alpha", "beta"]

def test_delta_rescans_when_stored_catalog_missed_mutations(tmp_path):
    refresh_catalog(tmp_path, lambda root: [_model("alpha")])
    # Two advances: one mutation whose delta write failed, then ours. The
    # stored catalog is more than one token behind, so the delta rescans.
    with catalog_lock(tmp_path):
        bump_catalog_token_locked(tmp_path)
        token = bump_catalog_token_locked(tmp_path)
    with catalog_lock(tmp_path):
        result = commit_catalog_delta_locked(
            tmp_path,
            lambda root: [_model("alpha"), _model("beta")],
            upsert={
                "name": "beta",
                "runtime": "vllm",
                "repository": "org/beta",
                "bytes": 5,
            },
        )
    assert result.changed
    document = load_catalog(tmp_path)
    assert [item["name"] for item in document["models"]] == ["alpha", "beta"]
    assert document["seq"] == token


def test_delta_refuses_upsert_invisible_in_degraded_view(tmp_path):
    refresh_catalog(tmp_path, lambda root: [_model("alpha")])
    previous = (tmp_path / "catalog.json").read_bytes()
    with catalog_lock(tmp_path):
        bump_catalog_token_locked(tmp_path)
        bump_catalog_token_locked(tmp_path)
        with pytest.raises(ModelctlError, match="not visible in this client"):
            commit_catalog_delta_locked(
                tmp_path,
                lambda root: [_model("alpha")],
                upsert={
                    "name": "beta",
                    "runtime": "vllm",
                    "repository": "org/beta",
                    "bytes": 5,
                },
            )
    assert catalog_dirty(tmp_path)
    assert (tmp_path / "catalog.json").read_bytes() == previous


def test_delta_refuses_removal_still_visible_in_degraded_view(tmp_path):
    refresh_catalog(tmp_path, lambda root: [_model("alpha")])
    previous = (tmp_path / "catalog.json").read_bytes()
    with catalog_lock(tmp_path):
        bump_catalog_token_locked(tmp_path)
        bump_catalog_token_locked(tmp_path)
        with pytest.raises(ModelctlError, match="still visible in this client"):
            commit_catalog_delta_locked(
                tmp_path, lambda root: [_model("alpha")], remove="alpha"
            )
    assert catalog_dirty(tmp_path)
    assert (tmp_path / "catalog.json").read_bytes() == previous


def test_delta_refuses_upsert_when_real_scan_cannot_see_model(tmp_path):
    """The rescan loader is the real catalog_models scan: unreadable object
    metadata (root squash, transient EIO) makes the fresh model invisible and
    the delta must refuse to poison the shared catalog."""
    _real_active_model(tmp_path, "alpha")
    refresh_catalog(tmp_path, catalog_models)
    _real_active_model(tmp_path, "beta")
    beta_metadata = (
        Layout(tmp_path).models / "org" / "beta" / "c0ffee" / ".modelctl.json"
    )
    os.chmod(beta_metadata, 0)
    try:
        with catalog_lock(tmp_path):
            bump_catalog_token_locked(tmp_path)
            bump_catalog_token_locked(tmp_path)
            with pytest.raises(ModelctlError, match="not visible in this client"):
                commit_catalog_delta_locked(
                    tmp_path,
                    catalog_models,
                    upsert={
                        "name": "beta",
                        "runtime": "vllm",
                        "repository": "org/beta",
                        "bytes": 4096,
                    },
                )
        assert catalog_dirty(tmp_path)
        names = [item["name"] for item in load_catalog(tmp_path)["models"]]
        assert names == ["alpha"]
    finally:
        os.chmod(beta_metadata, 0o644)
