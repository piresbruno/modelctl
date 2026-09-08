import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelctl import __version__
from modelctl.cards import CardResult
from modelctl.catalog import active_fingerprint, load_catalog
from modelctl.cli import DEFAULT_ROOT, _cache_dir, _format_size, _local_root, _root, build_parser, run
from modelctl.errors import ModelctlError
from modelctl.catalog import (
    bump_catalog_token_locked,
    catalog_lock,
    catalog_token,
    mark_catalog_dirty_locked,
)
from modelctl.integrity import ActiveReferenceAudit, RepairResult
from modelctl.sync_queue import PreparedSync, SyncQueueEntry, SyncQueueResult
from modelctl.hf_cache import LocalDeleteResult, state_root
from modelctl.layout import Layout, atomic_symlink
from modelctl.maintenance import StoreEntryAudit
from modelctl.manifest import parse_manifest
from modelctl.validation import ExpectedFile, write_metadata


def _active_model(root):
    manifest = parse_manifest({"repo": "org/model", "runtime": "vllm"}, "demo")
    layout = Layout(root)
    layout.prepare()
    object_path = layout.object_path(manifest, "e" * 40)
    object_path.mkdir(parents=True)
    (object_path / "config.json").write_bytes(b"{}")
    write_metadata(
        object_path,
        manifest,
        "e" * 40,
        [ExpectedFile("config.json", 2)],
        ".",
    )
    atomic_symlink(object_path, layout.active_path("demo"))
    layout.state_path("demo").write_text(json.dumps({
        "operation": "update",
        "state": "ACTIVE_ON_NAS",
        "history": [{
            "state": "ACTIVE_ON_NAS",
            "object": str(object_path),
            "commit": "e" * 40,
        }],
    }))
    return object_path.resolve()


def test_path_prints_only_resolved_entrypoint(tmp_path, capsys):
    object_path = _active_model(tmp_path)
    assert run(["path", "demo", "--root", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert captured.out == f"{object_path}\n"
    assert captured.err == ""


def test_serve_command_prints_without_starting_server(tmp_path, capsys):
    object_path = _active_model(tmp_path)
    assert run(["serve-command", "demo", "--root", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert captured.out == f"vllm serve {object_path}\n"
    assert captured.err == ""


def test_config_saves_and_uses_default_root(tmp_path, monkeypatch, capsys):
    config_home = tmp_path / "config"
    model_root = tmp_path / "nas" / "models"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.delenv("MODELCTL_ROOT", raising=False)

    assert run(["config", "set-root", str(model_root)]) == 0
    output = capsys.readouterr().out
    assert f"root: {model_root}\n" in output
    assert _root(None) == model_root

    assert run(["config", "get-root"]) == 0
    assert capsys.readouterr().out == f"{model_root}\n"

    local_root = tmp_path / "local" / "models"
    assert run(["config", "set-local-root", str(local_root)]) == 0
    capsys.readouterr()
    assert _local_root(None) == local_root
    assert _root(None) == model_root
    assert run(["config", "get-local-root"]) == 0
    assert capsys.readouterr().out == f"{local_root}\n"


def test_root_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("MODELCTL_ROOT", raising=False)
    assert _root(None) == Path(DEFAULT_ROOT)

    saved = tmp_path / "saved"
    run(["config", "set-root", str(saved)])
    environment = tmp_path / "environment"
    monkeypatch.setenv("MODELCTL_ROOT", str(environment))
    assert _root(None) == environment

    explicit = tmp_path / "explicit"
    assert _root(str(explicit)) == explicit


def test_list_prints_active_models(tmp_path, capsys):
    _active_model(tmp_path)
    assert run(["list", "--root", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "NAME" in output
    assert "RUNTIME" in output
    assert "REPOSITORY" in output
    assert "demo" in output
    assert "vllm" in output
    assert "org/model" in output
    assert "COMMIT" not in output
    assert "PATH" not in output


def test_list_json_is_machine_readable(tmp_path, capsys):
    _active_model(tmp_path)
    assert run(["catalog", "refresh", "--root", str(tmp_path)]) == 0
    capsys.readouterr()
    assert run(["list", "--root", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    [record] = payload
    assert {
        key: value for key, value in record.items() if key != "bytes"
    } == {"name": "demo", "runtime": "vllm", "repository": "org/model"}
    assert isinstance(record["bytes"], int) and record["bytes"] > 0
    assert load_catalog(tmp_path)["models"] == payload


def test_catalog_commands_report_refresh_status_and_path(tmp_path, capsys):
    _active_model(tmp_path)

    assert run(["catalog", "refresh", "--root", str(tmp_path)]) == 0
    assert "catalog: updated" in capsys.readouterr().out
    assert run(["catalog", "status", "--root", str(tmp_path)]) == 0
    assert "[ready]" in capsys.readouterr().out
    assert run(["catalog", "path", "--root", str(tmp_path)]) == 0
    assert capsys.readouterr().out == f"{tmp_path / 'catalog.json'}\n"


def test_catalog_status_json_reports_stale_external_change(tmp_path, capsys):
    _active_model(tmp_path)
    run(["catalog", "refresh", "--root", str(tmp_path)])
    capsys.readouterr()
    (tmp_path / "active" / "foreign").mkdir()

    assert run([
        "catalog",
        "status",
        "--root",
        str(tmp_path),
        "--json",
    ]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "stale"


def test_list_ignores_hidden_entries_and_non_symlinks(tmp_path, capsys):
    object_path = _active_model(tmp_path)
    active = tmp_path / "active"
    (active / ".DS_Store").write_bytes(b"metadata")
    (active / ".hidden-model").symlink_to(object_path)
    (active / "notes.txt").write_text("not a managed reference")

    assert run(["list", "--root", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [model["name"] for model in payload] == ["demo"]


def test_list_warns_when_malformed_active_references_are_skipped(tmp_path, capsys):
    _active_model(tmp_path)
    (tmp_path / "active" / "copied-model").mkdir()

    assert run(["list", "--root", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert "demo" in captured.out
    assert "skipped 1 malformed active reference" in captured.err
    assert "modelctl doctor" in captured.err


def test_list_reads_catalog_without_live_scan(tmp_path, monkeypatch, capsys):
    _active_model(tmp_path)
    assert run(["catalog", "refresh", "--root", str(tmp_path)]) == 0
    capsys.readouterr()

    def fail_live_scan(root):
        raise AssertionError("live scan must not run when the catalog is valid")

    monkeypatch.setattr("modelctl.cli.scan_active_records", fail_live_scan)
    assert run(["list", "--root", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [model["name"] for model in payload] == ["demo"]


def test_list_size_column_shows_disk_usage(tmp_path, capsys):
    _active_model(tmp_path)
    assert run(["catalog", "refresh", "--root", str(tmp_path)]) == 0
    capsys.readouterr()
    assert run(["list", "--root", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    [record] = load_catalog(tmp_path)["models"]
    assert "SIZE" in captured.out
    assert "REPOSITORY" in captured.out
    assert _format_size(record["bytes"]) in captured.out


def test_list_warns_when_catalog_is_dirty(tmp_path, capsys):
    _active_model(tmp_path)
    assert run(["catalog", "refresh", "--root", str(tmp_path)]) == 0
    capsys.readouterr()
    with catalog_lock(tmp_path):
        mark_catalog_dirty_locked(tmp_path, "test mutation")

    assert run(["list", "--root", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert "demo" in captured.out
    assert "catalog may be stale" in captured.err
    assert "modelctl catalog refresh" in captured.err


def test_list_falls_back_to_live_scan_without_catalog(tmp_path, capsys):
    _active_model(tmp_path)
    assert run(["catalog", "refresh", "--root", str(tmp_path)]) == 0
    capsys.readouterr()
    (tmp_path / "catalog.json").unlink()

    assert run(["list", "--root", str(tmp_path), "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert [model["name"] for model in payload] == ["demo"]
    assert "missing or invalid" in captured.err
    assert "listing a live scan" in captured.err
    assert not (tmp_path / "catalog.json").exists()


def test_list_treats_schema1_catalog_as_missing(tmp_path, capsys):
    _active_model(tmp_path)
    assert run(["catalog", "refresh", "--root", str(tmp_path)]) == 0
    capsys.readouterr()
    path = tmp_path / "catalog.json"
    document = json.loads(path.read_text())
    document["schema"] = 1
    path.write_text(json.dumps(document))

    assert run(["list", "--root", str(tmp_path), "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert [model["name"] for model in payload] == ["demo"]
    assert "listing a live scan" in captured.err
    assert json.loads(path.read_text())["schema"] == 1


def test_list_shows_live_scan_without_rewriting_stale_catalog(tmp_path, capsys):
    _active_model(tmp_path)
    assert run(["catalog", "refresh", "--root", str(tmp_path)]) == 0
    capsys.readouterr()
    catalog_bytes = (tmp_path / "catalog.json").read_bytes()
    (tmp_path / "active" / "extra").write_text("x")

    assert run(["list", "--root", str(tmp_path), "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert [model["name"] for model in payload] == ["demo"]
    assert "does not match the live active references" in captured.err
    assert (tmp_path / "catalog.json").read_bytes() == catalog_bytes


def test_list_keeps_last_known_catalog_on_degraded_view(tmp_path, capsys):
    _active_model(tmp_path)
    assert run(["catalog", "refresh", "--root", str(tmp_path)]) == 0
    capsys.readouterr()
    (tmp_path / "active" / "demo").unlink()

    assert run(["list", "--root", str(tmp_path), "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert [model["name"] for model in payload] == ["demo"]
    assert "keeping the last known catalog" in captured.err
    assert len(load_catalog(tmp_path)["models"]) == 1


def test_list_never_writes_catalog_when_behind_the_store(tmp_path, capsys):
    _active_model(tmp_path)
    assert run(["catalog", "refresh", "--root", str(tmp_path)]) == 0
    capsys.readouterr()
    catalog_bytes = (tmp_path / "catalog.json").read_bytes()
    with catalog_lock(tmp_path):
        bump_catalog_token_locked(tmp_path)

    assert run(["list", "--root", str(tmp_path), "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert [model["name"] for model in payload] == ["demo"]
    assert "mutation(s) behind the store" in captured.err
    assert (tmp_path / "catalog.json").read_bytes() == catalog_bytes


def test_catalog_refresh_refuses_empty_and_force_overrides(tmp_path, capsys):
    _active_model(tmp_path)
    assert run(["catalog", "refresh", "--root", str(tmp_path)]) == 0
    capsys.readouterr()
    (tmp_path / "active" / "demo").unlink()

    assert run(["catalog", "refresh", "--root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "refusing to overwrite" in captured.err
    assert len(load_catalog(tmp_path)["models"]) == 1

    assert run(["catalog", "refresh", "--root", str(tmp_path), "--force"]) == 0
    captured = capsys.readouterr()
    assert "0 model(s)" in captured.out
    assert load_catalog(tmp_path)["models"] == []


def test_doctor_reports_malformed_references(tmp_path, capsys):
    _active_model(tmp_path)
    (tmp_path / "active" / "copied-model").mkdir()

    assert run(["doctor", "--root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "[regular_directory] copied-model" in captured.out
    assert "doctor: 1 healthy/ignored, 0 warning(s), 1 malformed" in captured.out


def test_doctor_hints_repair_for_repairable_directory(tmp_path, monkeypatch, capsys):
    reference = tmp_path / "active" / "demo"
    monkeypatch.setattr(
        "modelctl.cli.audit_active_references",
        lambda root: [
            ActiveReferenceAudit(
                "demo",
                "repairable_directory",
                reference,
                tmp_path / "models" / "demo",
                "validated duplicate of canonical object",
            )
        ],
    )

    assert run(["doctor", "--root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "[repairable_directory] demo" in out
    assert "modelctl repair-active --apply" in out
    assert "outside modelctl" in out


def test_repair_active_is_dry_run_by_default(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_repair(root, name, *, apply):
        calls.append((root, name, apply))
        return RepairResult(
            name,
            "dry-run",
            root / "active" / name,
            root / "models" / name,
        )

    monkeypatch.setattr("modelctl.cli.repair_active_reference", fake_repair)
    assert run([
        "repair-active",
        "demo",
        "--root",
        str(tmp_path),
    ]) == 0
    assert calls == [(tmp_path.absolute(), "demo", False)]
    assert "[dry-run] demo" in capsys.readouterr().out


def test_staging_audit_reports_relative_paths_and_sizes(tmp_path, monkeypatch, capsys):
    path = tmp_path / ".staging" / "org" / "repo" / "object"
    monkeypatch.setattr(
        "modelctl.cli.audit_staging",
        lambda root: [
            StoreEntryAudit(path, "orphaned", 1024, detail="no journal")
        ],
    )

    assert run(["staging-audit", "--root", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "[orphaned] org/repo/object" in output
    assert "1.00 KiB" in output


def test_cleanup_staging_is_dry_run_by_default(tmp_path, monkeypatch, capsys):
    calls = []
    path = tmp_path / ".staging" / "org" / "repo" / "object"

    def fake_cleanup(root, selections, *, apply):
        calls.append((root, selections, apply))
        return [StoreEntryAudit(path, "orphaned", 1024)]

    monkeypatch.setattr("modelctl.cli.cleanup_staging", fake_cleanup)
    assert run([
        "cleanup-staging",
        "org/repo/object",
        "--root",
        str(tmp_path),
    ]) == 0
    assert calls == [(tmp_path.absolute(), ["org/repo/object"], False)]
    assert "[would remove] org/repo/object" in capsys.readouterr().out


def test_download_checks_symlink_support_before_remote_work(
    tmp_path, monkeypatch, capsys
):
    calls = []

    def fake_validate(root):
        calls.append(("validate", root))

    def fake_download(root, source, **kwargs):
        calls.append(("download", root, source))
        return root / "manifests" / "demo.yaml", root / "models" / "demo"

    monkeypatch.setattr("modelctl.cli.validate_queue_root", fake_validate)
    monkeypatch.setattr("modelctl.cli.download_from_hf", fake_download)
    assert run(["download", "org/demo", "--root", str(tmp_path)]) == 0
    assert calls == [
        ("validate", tmp_path.absolute()),
        ("download", tmp_path.absolute(), "org/demo"),
    ]
    assert "manifest:" in capsys.readouterr().out


def _download_pipeline(files, sha):
    """Drive the real download -> activate -> catalog-refresh pipeline with a
    network-free Hugging Face API and snapshot writer."""

    class FakeApi:
        def model_info(self, repo, revision):
            return SimpleNamespace(
                sha=sha,
                siblings=[
                    SimpleNamespace(rfilename=name) for name in sorted(files)
                ],
            )

    def snapshot(**kwargs):
        if kwargs.get("dry_run"):
            return [
                SimpleNamespace(filename=name, file_size=len(content))
                for name, content in files.items()
            ]
        root = Path(kwargs["local_dir"])
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return str(root)

    return FakeApi, snapshot


def test_download_refreshes_catalog_and_keeps_model_available_in_list(
    tmp_path, monkeypatch, capsys
):
    from modelctl.generation import download_from_hf as real_download_from_hf

    files = {
        "config.json": b"{}",
        "model.safetensors": b"weights",
        "README.md": b"# Model card\n",
    }
    FakeApi, snapshot = _download_pipeline(files, "a" * 40)

    def fake_download(root, source, **kwargs):
        return real_download_from_hf(
            root, source, api=FakeApi(), snapshot=snapshot, **kwargs
        )

    monkeypatch.setattr("modelctl.cli.download_from_hf", fake_download)
    assert run(
        ["download", "org/demo", "--name", "demo", "--root", str(tmp_path)]
    ) == 0
    capsys.readouterr()

    # A process outside modelctl replaces the active symlink with a copy.
    reference = tmp_path / "active" / "demo"
    target = reference.resolve()
    reference.unlink()
    shutil.copytree(target, reference)
    assert reference.is_dir() and not reference.is_symlink()

    # Re-downloading the same model auto-repairs its reference without error.
    assert run(
        ["download", "org/demo", "--name", "demo", "--root", str(tmp_path)]
    ) == 0
    capsys.readouterr()
    assert reference.is_symlink()
    assert reference.resolve() == target
    assert [record["name"] for record in load_catalog(tmp_path)["models"]] == [
        "demo"
    ]

    # A process dereferences the model again; downloading a different model
    # must keep the repairable copy available in the catalog and in list.
    reference.unlink()
    shutil.copytree(target, reference)
    assert run(
        ["download", "org/fresh", "--name", "fresh", "--root", str(tmp_path)]
    ) == 0
    capsys.readouterr()
    assert {record["name"] for record in load_catalog(tmp_path)["models"]} == {
        "demo",
        "fresh",
    }

    assert run(["list", "--root", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {item["name"] for item in payload} == {"demo", "fresh"}


def test_queue_download_refreshes_catalog_and_list_with_dereferenced_model(
    tmp_path, monkeypatch, capsys
):
    import modelctl.download_queue as download_queue_module
    import modelctl.generation as generation_module

    files = {
        "config.json": b"{}",
        "model.safetensors": b"weights",
        "README.md": b"# Model card\n",
    }
    FakeApi, writer = _download_pipeline(files, "b" * 40)
    real_generate = generation_module.generate_manifest_document
    real_download_generated = generation_module.download_generated_manifest

    def fake_generate(source, **kwargs):
        return real_generate(source, api=FakeApi(), **kwargs)

    def fake_download_generated(root, document, *, force_manifest=False, api=None, snapshot=None):
        return real_download_generated(
            root,
            document,
            force_manifest=force_manifest,
            api=FakeApi(),
            snapshot=writer,
        )

    monkeypatch.setattr(
        download_queue_module, "generate_manifest_document", fake_generate
    )
    monkeypatch.setattr(
        generation_module, "download_generated_manifest", fake_download_generated
    )

    first = tmp_path / "first.yaml"
    first.write_text(
        "downloads:\n"
        "  - source: org/alpha\n"
        "    name: alpha\n"
        "  - source: org/beta\n"
        "    name: beta\n"
    )
    assert run(["queue", str(first), "--root", str(tmp_path)]) == 0
    capsys.readouterr()
    assert {record["name"] for record in load_catalog(tmp_path)["models"]} == {
        "alpha",
        "beta",
    }

    # A process outside modelctl replaces alpha's active symlink with a copy;
    # the next queue run must keep alpha available in the catalog and list.
    alpha_reference = tmp_path / "active" / "alpha"
    alpha_target = alpha_reference.resolve()
    alpha_reference.unlink()
    shutil.copytree(alpha_target, alpha_reference)
    assert alpha_reference.is_dir() and not alpha_reference.is_symlink()

    second = tmp_path / "second.yaml"
    second.write_text("downloads:\n  - source: org/beta\n    name: beta\n")
    assert run(["queue", str(second), "--root", str(tmp_path)]) == 0
    capsys.readouterr()
    assert {record["name"] for record in load_catalog(tmp_path)["models"]} == {
        "alpha",
        "beta",
    }
    assert run(["list", "--root", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {item["name"] for item in payload} == {"alpha", "beta"}


def test_list_local_uses_hf_cache(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "hub"
    model = type("Model", (), {"name": "demo", "runtime": "vllm", "repo": "org/model"})()
    calls = []

    def fake_list(selected):
        calls.append(selected)
        return [model]

    monkeypatch.setattr("modelctl.cli.list_cached_models", fake_list)
    assert run(["list", "--local", "--cache-dir", str(cache)]) == 0
    assert calls == [cache]
    assert "demo" in capsys.readouterr().out


def test_list_empty_store(tmp_path, capsys):
    assert run(["list", "--root", str(tmp_path)]) == 0
    assert capsys.readouterr().out == f"No active models in {tmp_path}.\n"


def test_sync_local_uses_saved_nas_and_local_roots(tmp_path, monkeypatch, capsys):
    nas = tmp_path / "nas"
    local = tmp_path / "local"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    run(["config", "set-root", str(nas)])
    run(["config", "set-local-root", str(local)])
    capsys.readouterr()
    calls = []

    def fake_sync(source_root, local_root, name, *, rsync, progress):
        calls.append((source_root, local_root, name, rsync, callable(progress)))
        return local_root / "active" / name

    monkeypatch.setattr("modelctl.cli.sync_local", fake_sync)
    assert run(["sync-local", "demo"]) == 0
    assert calls == [(nas, local, "demo", "rsync", True)]
    assert capsys.readouterr().out == f"{local / 'active' / 'demo'}\n"


def test_sync_local_accepts_hugging_face_repository(
    tmp_path, monkeypatch, capsys
):
    nas = tmp_path / "nas"
    cache = tmp_path / "hub"
    calls = []

    def fake_sync(source_root, local_root, name, *, rsync, progress):
        calls.append((source_root, local_root, name, rsync, callable(progress)))
        return local_root / "snapshot"

    monkeypatch.setattr("modelctl.cli.sync_local", fake_sync)
    assert run([
        "sync-local",
        "unsloth/DeepSeek-V4-Flash-0731",
        "--source-root",
        str(nas),
        "--cache-dir",
        str(cache),
    ]) == 0
    assert calls == [
        (
            nas.absolute(),
            cache,
            "unsloth/DeepSeek-V4-Flash-0731",
            "rsync",
            True,
        )
    ]
    assert capsys.readouterr().out == f"{cache / 'snapshot'}\n"


def test_sync_local_queue_preflights_and_reports_results(
    tmp_path, monkeypatch, capsys
):
    queue_file = tmp_path / "models.txt"
    queue_file.write_text("- alpha\n- beta\n", encoding="utf-8")
    nas = tmp_path / "nas"
    cache = tmp_path / "hub"
    calls = []

    def fake_prepare(source_root, entries):
        calls.append(("prepare", source_root, [e.identifier for e in entries]))
        return [
            PreparedSync(entry, entry.identifier)
            for entry in entries
        ]

    def fake_execute(source_root, cache_dir, prepared, *, jobs, rsync):
        calls.append(("execute", source_root, cache_dir, len(prepared), jobs))
        return [
            SyncQueueResult(
                prepared[0].entry,
                "alpha",
                snapshot=cache / "snapshots" / "alpha",
            ),
            SyncQueueResult(
                prepared[1].entry,
                "beta",
                error=ModelctlError("rsync failed; staging is resumable"),
            ),
        ]

    monkeypatch.setattr("modelctl.cli.prepare_sync_queue", fake_prepare)
    monkeypatch.setattr("modelctl.cli.execute_sync_queue", fake_execute)
    assert run([
        "sync-local",
        "--queue",
        str(queue_file),
        "--source-root",
        str(nas),
        "--cache-dir",
        str(cache),
    ]) == 1
    assert calls == [
        ("prepare", nas, ["alpha", "beta"]),
        ("execute", nas, cache, 2, 1),
    ]
    out = capsys.readouterr().out
    assert "preflight: validating source store and 2 queue entries" in out
    assert "[1/2] ready: alpha <- alpha" in out
    assert "[2/2] ready: beta <- beta" in out
    assert "preflight complete: starting 2 sync(s), up to 1 concurrent" in out
    assert "[1/2] complete: alpha" in out
    assert "[2/2] failed: beta: ModelctlError: rsync failed" in out
    assert "summary: 1 complete, 1 failed" in out


def test_sync_local_queue_rejects_name_and_queue_together(tmp_path):
    queue_file = tmp_path / "models.txt"
    queue_file.write_text("- alpha\n", encoding="utf-8")
    with pytest.raises(ModelctlError, match="not both"):
        run(["sync-local", "alpha", "--queue", str(queue_file)])


def test_sync_local_jobs_requires_queue(tmp_path):
    with pytest.raises(ModelctlError, match="--jobs requires --queue"):
        run(["sync-local", "alpha", "--jobs", "2"])


def test_sync_local_requires_name_or_queue(tmp_path):
    with pytest.raises(ModelctlError, match="provide MODEL_OR_REPO or --queue"):
        run(["sync-local"])


def test_delete_local_deletes_cache_data_by_default(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "hub"
    record = cache / "state" / "active" / "demo.json"
    repository = cache / "models--org--demo"
    calls = []

    def fake_delete(selected, name, *, keep_data=False):
        calls.append((selected, name, keep_data))
        return LocalDeleteResult(
            name=name,
            record=record,
            snapshot=repository,
            removed=(repository,),
            retained=(),
        )

    monkeypatch.setattr("modelctl.cli.delete_cached", fake_delete)
    assert run(["delete-local", "demo", "--cache-dir", str(cache)]) == 0
    assert calls == [(cache, "demo", False)]
    assert capsys.readouterr().out == (
        f"deleted: {repository}\nunregistered: {record}\n"
    )


def test_delete_local_keep_data_flag_retains_cache(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "hub"
    record = cache / "state" / "active" / "demo.json"
    calls = []

    def fake_delete(selected, name, *, keep_data=False):
        calls.append((selected, name, keep_data))
        return LocalDeleteResult(
            name=name,
            record=record,
            snapshot=None,
            removed=(),
            retained=("--keep-data was set; snapshots, refs, and blobs were retained",),
        )

    monkeypatch.setattr("modelctl.cli.delete_cached", fake_delete)
    assert run(
        ["delete-local", "demo", "--cache-dir", str(cache), "--keep-data"]
    ) == 0
    assert calls == [(cache, "demo", True)]
    assert capsys.readouterr().out == (
        "retained: --keep-data was set; snapshots, refs, and blobs were retained\n"
        f"unregistered: {record} (cache data retained)\n"
    )


def test_sync_cards_prints_results_and_summary(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_sync(root, names, *, force=False):
        calls.append((root, names, force))
        return [
            CardResult("a", "updated", root / "cards" / "a"),
            CardResult("b", "unavailable", message="no README.md"),
        ]

    monkeypatch.setattr("modelctl.cli.sync_model_cards", fake_sync)
    assert run(["sync-cards", "a", "b", "--force", "--root", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert calls == [(tmp_path.absolute(), ["a", "b"], True)]
    assert "[updated] a:" in output
    assert "[unavailable] b: no README.md" in output
    assert "1 updated, 0 unchanged, 1 unavailable, 0 failed" in output


def test_cli_version_uses_package_version(capsys):
    assert __version__ == "0.19.0"
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"modelctl {__version__}\n"


def test_queue_help_documents_format_concurrency_and_examples(capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["queue", "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "source: Qwen/Qwen3-8B" in output
    assert "quantization: Q4_K_M" in output
    assert "modelctl queue downloads.yaml --jobs 2" in output
    assert "No transfer starts unless every queue" in output
    assert "unique effective model" in output
    assert "mfsymlinks" in output
    assert "no fixed jobs limit" in output


def test_top_level_help_has_description_and_examples(capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "Atomically download, validate, publish" in output
    assert "examples:" in output
    assert "modelctl download Qwen/Qwen3-8B" in output
    assert "modelctl COMMAND --help" in output


@pytest.mark.parametrize(
    ("command", "example"),
    [
        ("download", "modelctl download Qwen/Qwen3-8B"),
        ("config", "modelctl config set-root"),
        ("delete-local", "modelctl delete-local qwen3-8b-vllm"),
        ("list", "modelctl list --local"),
        ("manifest", "modelctl manifest Qwen/Qwen3-8B"),
        ("queue", "modelctl queue downloads.yaml"),
        ("sync-cards", "modelctl sync-cards qwen3-8b model-q4"),
        ("update", "modelctl update qwen3-8b-vllm"),
        ("path", "MODEL_PATH=$(modelctl path"),
        ("serve-command", "modelctl serve-command model-q4"),
        ("sync-local", "modelctl sync-local qwen3-8b-vllm"),
        ("push", "modelctl push qwen3-8b-vllm --host node-b"),
        ("sync-remote", "modelctl sync-remote qwen3-8b-vllm --host node-b"),
        ("receive-cache", "modelctl receive-cache --probe qwen3-8b-vllm"),
    ],
)
def test_subcommand_help_has_description_and_examples(command, example, capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args([command, "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "examples:" in output
    assert example in output
    if command == "sync-local":
        assert "unsloth/DeepSeek-V4-Flash-0731" in output
        assert "active model name or its Hugging Face repository id" in output


def test_hf_cache_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("MODELCTL_LOCAL_ROOT", str(tmp_path / "legacy"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "explicit-env"))
    assert _cache_dir(None) == tmp_path / "explicit-env"
    assert _cache_dir(str(tmp_path / "argument")) == tmp_path / "argument"


def test_push_uses_local_default_cache_path(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    calls = []

    def fake_push(local_cache, name, **kwargs):
        calls.append((local_cache, name, kwargs))
        return local_cache / "snapshot"

    monkeypatch.setattr("modelctl.cli.push_model", fake_push)
    assert run(["push", "demo", "--host", "node-b"]) == 0
    local_cache, name, options = calls[0]
    assert local_cache == tmp_path / "hub"
    assert name == "demo"
    assert options["host"] == "node-b"
    assert options["ssh"] == "ssh"
    assert options["rsync"] == "rsync"
    assert capsys.readouterr().out == f"{tmp_path / 'hub' / 'snapshot'}\n"


def test_push_passes_explicit_cache_dir_and_ssh_options(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "custom-cache"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "ignored"))
    calls = []

    def fake_push(local_cache, name, **kwargs):
        calls.append((local_cache, name, kwargs))
        return local_cache / "snapshot"

    monkeypatch.setattr("modelctl.cli.push_model", fake_push)
    assert run([
        "push", "demo", "--host", "node-b",
        "--cache-dir", str(cache),
        "--port", "2222",
        "--identity", "/keys/id",
        "--remote-modelctl", "/opt/modelctl/bin/modelctl",
        "--jobs", "4",
    ]) == 0
    local_cache, name, options = calls[0]
    assert local_cache == cache
    assert name == "demo"
    assert options["port"] == 2222
    assert options["identity"] == "/keys/id"
    assert options["remote_modelctl"] == "/opt/modelctl/bin/modelctl"
    assert options["jobs"] == 4


def test_push_repeated_host_splits_control_and_stream_hosts(
    tmp_path, monkeypatch, capsys
):
    cache = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    calls = []

    def fake_push(local_cache, name, **kwargs):
        calls.append((local_cache, name, kwargs))
        return local_cache / "snapshot"

    monkeypatch.setattr("modelctl.cli.push_model", fake_push)
    assert run([
        "push", "demo", "--host", "node-b", "--host", "node-c", "--jobs", "6",
    ]) == 0
    _, name, options = calls[0]
    assert name == "demo"
    assert options["host"] == "node-b"
    assert options["job_hosts"] == ["node-c"]
    assert options["jobs"] == 6


def test_receive_cache_probe_prints_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert run(["receive-cache", "--probe", "demo"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"proto": 1, "cache": str(tmp_path / "hub")}


def test_list_local_warns_and_skips_broken_registration(
    tmp_path, monkeypatch, capsys
):
    cache = tmp_path / "hub"
    active = state_root(cache) / "active"
    active.mkdir(parents=True, exist_ok=True)
    (active / "broken.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "cache": str(cache),
                "name": "broken",
                "repo": "org/broken",
                "revision": "main",
                "commit": "e" * 40,
                "files": {"config.json": "e" * 40},
                "snapshot": str(
                    cache / "models--org--broken" / "snapshots" / ("e" * 40)
                ),
                "metadata": {
                    "name": "broken",
                    "repo": "org/broken",
                    "revision": "main",
                    "commit": "e" * 40,
                    "entrypoint": ".",
                    "companions": {},
                },
            }
        )
    )
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    assert run(["list", "--local"]) == 0
    captured = capsys.readouterr()
    assert "No active models" in captured.out
    assert "delete-local" in captured.err
    assert "broken" in captured.err


def test_sync_rejects_cache_dir_and_legacy_root_together(tmp_path):
    with pytest.raises(ModelctlError, match="cannot be combined"):
        run([
            "sync-local",
            "demo",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--root",
            str(tmp_path / "legacy"),
        ])


def test_path_local_uses_cache_record_resolver(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "hub"
    result = cache / "models--org--demo" / "snapshots" / ("d" * 40)
    calls = []

    def fake_path(selected, name):
        calls.append((selected, name))
        return result

    monkeypatch.setattr("modelctl.cli.local_active_entrypoint", fake_path)
    assert run(["path", "demo", "--local", "--cache-dir", str(cache)]) == 0
    assert calls == [(cache, "demo")]
    assert capsys.readouterr().out == f"{result}\n"


class _FakeTty(io.StringIO):
    def isatty(self):
        return True


def test_delete_dry_run_is_default_and_mutates_nothing(tmp_path, capsys):
    _active_model(tmp_path)
    assert run(["delete", "demo", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "delete (dry-run): demo (org/model)" in out
    assert "would remove" in out
    assert "MANAGED ROOT STORE" not in out
    assert (tmp_path / "active" / "demo").is_symlink()
    assert (tmp_path / "models" / "org" / "model").exists()


def test_delete_apply_requires_typed_confirmation(tmp_path, monkeypatch, capsys):
    _active_model(tmp_path)
    object_path = (tmp_path / "active" / "demo").resolve()

    # Non-interactive stdin without --yes refuses before mutating anything.
    with pytest.raises(ModelctlError, match="--yes"):
        run(["delete", "demo", "--root", str(tmp_path), "--apply"])
    capsys.readouterr()
    assert (tmp_path / "active" / "demo").is_symlink()

    # Any answer other than a line reading 'yes' aborts without mutating.
    monkeypatch.setattr("sys.stdin", _FakeTty("n\n"))
    with pytest.raises(ModelctlError, match="aborted"):
        run(["delete", "demo", "--root", str(tmp_path), "--apply"])
    capsys.readouterr()
    assert (tmp_path / "active" / "demo").is_symlink()

    # Typing 'yes' confirms the NAS root-store deletion.
    monkeypatch.setattr("sys.stdin", _FakeTty("YES\n"))
    assert run(["delete", "demo", "--root", str(tmp_path), "--apply"]) == 0
    out = capsys.readouterr().out
    assert "MANAGED ROOT STORE" in out
    assert "delete-local" in out
    assert "removed" in out
    assert not (tmp_path / "active" / "demo").exists()
    assert not object_path.exists()
    assert not (tmp_path / "state" / "demo.json").exists()
    assert load_catalog(tmp_path)["models"] == []


def test_delete_apply_yes_skips_confirmation(tmp_path, capsys):
    _active_model(tmp_path)
    assert run(["delete", "demo", "--root", str(tmp_path), "--apply", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "MANAGED ROOT STORE" not in out
    assert "delete (apply): demo (org/model)" in out
    assert not (tmp_path / "active" / "demo").exists()


def test_delete_json_is_machine_readable(tmp_path, capsys):
    _active_model(tmp_path)
    assert (
        run(["delete", "demo", "--root", str(tmp_path), "--apply", "--yes", "--json"])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["applied"] is True
    assert payload["name"] == "demo"
    assert payload["repo"] == "org/model"
    assert [item["name"] for item in payload["removed_objects"]] == ["demo"]
    assert payload["removed_objects"][0]["status"] == "unreferenced"
    assert str(tmp_path / "state" / "demo.json") in payload["journals"]
