from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import ManifestError

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class RuntimeProfile:
    kind: str
    executable: str
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelManifest:
    name: str
    repo: str
    revision: str = "main"
    include: tuple[str, ...] = ()
    format: str = "auto"
    entrypoint: str | None = None
    runtime: RuntimeProfile = field(
        default_factory=lambda: RuntimeProfile("vllm", "vllm")
    )


def validate_name(name: str) -> str:
    if not _NAME_RE.fullmatch(name):
        raise ManifestError(
            f"invalid model name {name!r}; use letters, digits, '.', '_' or '-'"
        )
    return name


def _relative_path(value: str, field_name: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not value:
        raise ManifestError(f"{field_name} must be a non-empty relative path")
    return path.as_posix()


def _runtime(raw: Any, model_format: str) -> RuntimeProfile:
    default_kind = "llama.cpp" if model_format == "gguf" else "vllm"
    if raw is None:
        kind = default_kind
        return RuntimeProfile(kind, "llama-server" if kind == "llama.cpp" else "vllm")
    if isinstance(raw, str):
        kind = raw
        executable = "llama-server" if kind in {"llama.cpp", "llama"} else "vllm"
        return RuntimeProfile(kind, executable)
    if not isinstance(raw, dict):
        raise ManifestError("runtime must be a string or mapping")
    kind = str(raw.get("type", raw.get("kind", default_kind)))
    executable = str(
        raw.get(
            "executable",
            raw.get("command", "llama-server" if kind in {"llama.cpp", "llama"} else "vllm"),
        )
    )
    args = raw.get("args", ())
    if isinstance(args, str) or not isinstance(args, (list, tuple)):
        raise ManifestError("runtime.args must be a list of command arguments")
    return RuntimeProfile(kind, executable, tuple(str(item) for item in args))


def _select_document(document: Any, name: str) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ManifestError("manifest root must be a mapping")
    models = document.get("models")
    if isinstance(models, dict):
        try:
            selected = models[name]
        except KeyError as exc:
            raise ManifestError(f"model {name!r} is not present in manifest") from exc
    elif isinstance(models, list):
        selected = next(
            (item for item in models if isinstance(item, dict) and item.get("name") == name),
            None,
        )
        if selected is None:
            raise ManifestError(f"model {name!r} is not present in manifest")
    else:
        selected = document
    if not isinstance(selected, dict):
        raise ManifestError(f"manifest entry for {name!r} must be a mapping")
    return selected


def parse_manifest(document: Any, name: str) -> ModelManifest:
    validate_name(name)
    raw = _select_document(document, name)
    declared_name = str(raw.get("name", name))
    if declared_name != name:
        raise ManifestError(
            f"manifest declares model {declared_name!r}, not requested model {name!r}"
        )
    repo = raw.get("repo", raw.get("repo_id", raw.get("repository")))
    if not isinstance(repo, str) or not repo.strip():
        raise ManifestError("manifest requires a Hugging Face 'repo'")
    repo_path = PurePosixPath(repo)
    if repo_path.is_absolute() or ".." in repo_path.parts:
        raise ManifestError("repo must be a Hugging Face repository id")

    include = raw.get("include", ())
    if isinstance(include, str):
        include = (include,)
    if not isinstance(include, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in include
    ):
        raise ManifestError("include must be a string or list of non-empty patterns")

    model_format = str(raw.get("format", "auto")).lower()
    if model_format not in {"auto", "hf", "safetensors", "gguf"}:
        raise ManifestError("format must be auto, hf, safetensors, or gguf")
    entrypoint = raw.get("entrypoint")
    if entrypoint is not None:
        if not isinstance(entrypoint, str):
            raise ManifestError("entrypoint must be a relative path")
        entrypoint = _relative_path(entrypoint, "entrypoint")

    runtime_raw = raw.get("runtime")
    if runtime_raw is None and isinstance(raw.get("serve"), dict):
        runtime_raw = raw["serve"]
    runtime = _runtime(runtime_raw, model_format)
    if model_format == "gguf" and runtime.kind == "vllm":
        raise ManifestError(
            "GGUF with vLLM is not supported by the version 1 runtime profile"
        )
    return ModelManifest(
        name=name,
        repo=repo.strip("/"),
        revision=str(raw.get("revision", "main")),
        include=tuple(include),
        format=model_format,
        entrypoint=entrypoint,
        runtime=runtime,
    )


def manifest_path(root: Path, name: str, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    for suffix in (".yaml", ".yml", ".json"):
        candidate = root / "manifests" / f"{name}{suffix}"
        if candidate.is_file():
            return candidate
    raise ManifestError(
        f"manifest for {name!r} not found under {root / 'manifests'}"
    )


def load_manifest(root: Path, name: str, explicit: Path | None = None) -> ModelManifest:
    path = manifest_path(root, name, explicit)
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            document = json.loads(text)
        else:
            import yaml

            document = yaml.safe_load(text)
    except (OSError, ValueError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    return parse_manifest(document, name)
