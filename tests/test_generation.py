from types import SimpleNamespace

import pytest
import yaml

from modelctl.errors import ManifestError
from modelctl.generation import (
    download_from_hf,
    generate_manifest_document,
    parse_hf_source,
    write_generated_manifest,
)


class FakeApi:
    def __init__(self, files):
        self.files = files
        self.calls = []

    def model_info(self, repo, revision):
        self.calls.append((repo, revision))
        return SimpleNamespace(
            sha="f" * 40,
            siblings=[SimpleNamespace(rfilename=value) for value in self.files],
        )


def test_parses_hf_ids_and_urls():
    assert parse_hf_source("Qwen/Qwen3-8B") == ("Qwen/Qwen3-8B", "main")
    assert parse_hf_source("Qwen/Qwen3-8B@v2") == ("Qwen/Qwen3-8B", "v2")
    assert parse_hf_source(
        "https://huggingface.co/Qwen/Qwen3-8B/tree/refs%2Fpr%2F7"
    ) == ("Qwen/Qwen3-8B", "refs/pr/7")


def test_generates_filtered_vllm_manifest():
    api = FakeApi(
        [
            "config.json",
            "tokenizer.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "pytorch_model.bin",
            "README.md",
        ]
    )
    document = generate_manifest_document(
        "https://huggingface.co/Qwen/Qwen3-8B", api=api
    )
    assert document["name"] == "qwen3-8b"
    assert document["format"] == "safetensors"
    assert document["runtime"]["type"] == "vllm"
    assert "*.safetensors" in document["include"]
    assert "*.bin" not in document["include"]
    assert api.calls == [("Qwen/Qwen3-8B", "main")]


def test_multiple_gguf_quantizations_require_selection():
    api = FakeApi(["Model-Q4_K_M.gguf", "Model-Q8_0.gguf"])
    with pytest.raises(ManifestError, match="--quantization"):
        generate_manifest_document("org/model-GGUF", api=api)

    document = generate_manifest_document(
        "org/model-GGUF", quantization="q4_k_m", api=api
    )
    assert document["format"] == "gguf"
    assert document["include"] == ["Model-Q4_K_M.gguf"]
    assert document["entrypoint"] == "Model-Q4_K_M.gguf"
    assert document["runtime"]["type"] == "llama.cpp"


def test_selects_every_shard_and_first_shard_entrypoint():
    files = [
        f"weights/Model-Q4_K_M-{index:05d}-of-00003.gguf"
        for index in range(1, 4)
    ] + ["Model-Q8_0.gguf"]
    document = generate_manifest_document(
        "org/model-GGUF", quantization="Q4_K_M", api=FakeApi(files)
    )
    assert document["include"] == files[:3]
    assert document["entrypoint"].endswith("00001-of-00003.gguf")


def test_manifest_writer_refuses_overwrite_without_force(tmp_path):
    document = generate_manifest_document(
        "org/model", api=FakeApi(["config.json", "model.safetensors"])
    )
    path = write_generated_manifest(tmp_path, document)
    assert yaml.safe_load(path.read_text()) == document
    with pytest.raises(ManifestError, match="--force"):
        write_generated_manifest(tmp_path, document)

    changed = {**document, "revision": "v2"}
    assert write_generated_manifest(tmp_path, changed, force=True) == path
    assert yaml.safe_load(path.read_text())["revision"] == "v2"


def test_download_generates_manifest_and_activates_model(tmp_path):
    api = FakeApi(["config.json", "model.safetensors"])

    def snapshot(**kwargs):
        files = {"config.json": b"{}", "model.safetensors": b"weights"}
        if kwargs.get("dry_run"):
            return [
                SimpleNamespace(filename=name, file_size=len(content))
                for name, content in files.items()
            ]
        root = kwargs["local_dir"]
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return str(root)

    manifest_path, active = download_from_hf(
        tmp_path,
        "https://huggingface.co/org/model",
        name="my-model",
        api=api,
        snapshot=snapshot,
    )
    assert manifest_path == (tmp_path / "manifests" / "my-model.yaml").absolute()
    assert manifest_path.is_file()
    assert active == (tmp_path / "active" / "my-model").resolve()
    assert (active / "model.safetensors").read_bytes() == b"weights"

    def no_download(**kwargs):
        raise AssertionError("an identical published object should be reused")

    _, reused = download_from_hf(
        tmp_path,
        "org/model",
        name="my-model",
        api=api,
        snapshot=no_download,
    )
    assert reused == active
