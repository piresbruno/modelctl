import json
from pathlib import Path

import pytest

from modelctl import __version__
from modelctl.cards import CardResult
from modelctl.errors import ModelctlError
from modelctl.cli import DEFAULT_ROOT, _cache_dir, _local_root, _root, build_parser, run
from modelctl.layout import Layout, atomic_symlink
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
    assert run(["list", "--root", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "name": "demo",
            "runtime": "vllm",
            "repository": "org/model",
        }
    ]


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

    def fake_sync(source_root, local_root, name, *, rsync):
        calls.append((source_root, local_root, name, rsync))
        return local_root / "active" / name

    monkeypatch.setattr("modelctl.cli.sync_local", fake_sync)
    assert run(["sync-local", "demo"]) == 0
    assert calls == [(nas, local, "demo", "rsync")]
    assert capsys.readouterr().out == f"{local / 'active' / 'demo'}\n"


def test_delete_local_uses_hf_cache_and_retains_data(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "hub"
    snapshot = cache / "models--org--demo" / "snapshots" / ("e" * 40)
    calls = []

    def fake_delete(selected, name):
        calls.append((selected, name))
        return snapshot

    monkeypatch.setattr("modelctl.cli.delete_cached", fake_delete)
    assert run(["delete-local", "demo", "--cache-dir", str(cache)]) == 0
    assert calls == [(cache, "demo")]
    assert capsys.readouterr().out == f"unregistered: {snapshot} (cache data retained)\n"


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
    assert __version__ == "0.9.1"
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"modelctl {__version__}\n"


def test_queue_help_documents_format_concurrency_and_examples(capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["queue", "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "downloads:" in output
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
    ],
)
def test_subcommand_help_has_description_and_examples(command, example, capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args([command, "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "examples:" in output
    assert example in output



def test_hf_cache_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("MODELCTL_LOCAL_ROOT", str(tmp_path / "legacy"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "explicit-env"))
    assert _cache_dir(None) == tmp_path / "explicit-env"
    assert _cache_dir(str(tmp_path / "argument")) == tmp_path / "argument"


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
