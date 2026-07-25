import pytest

from modelctl.errors import ManifestError
from modelctl.manifest import parse_manifest


def test_parses_gguf_runtime_and_include():
    manifest = parse_manifest(
        {
            "repo": "org/model-GGUF",
            "revision": "v1",
            "format": "gguf",
            "include": "*Q4_K_M*.gguf",
        },
        "model",
    )
    assert manifest.include == ("*Q4_K_M*.gguf",)
    assert manifest.runtime.kind == "llama.cpp"
    assert manifest.runtime.executable == "llama-server"


def test_rejects_default_vllm_gguf_profile():
    with pytest.raises(ManifestError, match="GGUF with vLLM"):
        parse_manifest(
            {"repo": "org/model", "format": "gguf", "runtime": "vllm"},
            "model",
        )


def test_selects_named_entry_from_combined_manifest():
    manifest = parse_manifest(
        {"models": {"a": {"repo": "org/a"}, "b": {"repo": "org/b"}}},
        "b",
    )
    assert manifest.repo == "org/b"


def test_rejects_unsafe_entrypoint():
    with pytest.raises(ManifestError, match="relative path"):
        parse_manifest({"repo": "org/model", "entrypoint": "../outside"}, "model")
