import pytest

from modelctl import __version__
from modelctl.cli import build_parser, run
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


def test_cli_version_uses_package_version(capsys):
    assert __version__ == "0.2.0"
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
        ("manifest", "modelctl manifest Qwen/Qwen3-8B"),
        ("queue", "modelctl queue downloads.yaml"),
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
