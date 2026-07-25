from pathlib import Path

import pytest

from modelctl.errors import ValidationError
from modelctl.manifest import parse_manifest
from modelctl.validation import ExpectedFile, resolve_entrypoint, validate_expected


def _touch(root: Path, names: list[str]) -> list[ExpectedFile]:
    expected = []
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"gguf")
        expected.append(ExpectedFile(name, 4))
    return expected


def test_sharded_gguf_requires_every_shard_and_uses_first(tmp_path):
    names = [f"Model-Q4_K_M-{i:05d}-of-00004.gguf" for i in range(1, 5)]
    expected = _touch(tmp_path, names)
    manifest = parse_manifest(
        {"repo": "org/model", "format": "gguf", "include": "*Q4_K_M*"},
        "model",
    )
    validate_expected(tmp_path, expected)
    assert resolve_entrypoint(tmp_path, manifest, expected) == names[0]


def test_sharded_gguf_rejects_missing_shard(tmp_path):
    names = [
        "Model-Q4_K_M-00001-of-00003.gguf",
        "Model-Q4_K_M-00003-of-00003.gguf",
    ]
    expected = _touch(tmp_path, names)
    manifest = parse_manifest(
        {"repo": "org/model", "format": "gguf", "include": "*Q4_K_M*"},
        "model",
    )
    with pytest.raises(ValidationError, match="00002"):
        resolve_entrypoint(tmp_path, manifest, expected)


def test_independent_quantizations_are_ambiguous(tmp_path):
    expected = _touch(tmp_path, ["Model-Q4_K_M.gguf", "Model-Q8_0.gguf"])
    manifest = parse_manifest(
        {"repo": "org/model", "format": "gguf", "include": "*.gguf"},
        "model",
    )
    with pytest.raises(ValidationError, match="exactly one quantization"):
        resolve_entrypoint(tmp_path, manifest, expected)
