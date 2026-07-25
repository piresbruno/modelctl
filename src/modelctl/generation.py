from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

from .errors import ManifestError, ModelctlError
from .manifest import parse_manifest, validate_name

_SHARD_RE = re.compile(
    r"^(?P<base>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$", re.I
)
_SAFE_NAME_RE = re.compile(r"[^a-z0-9._-]+")
_STANDARD_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.model",
    "*.txt",
    "*.tiktoken",
    "*.jinja",
    "*.py",
    "*.vocab",
    "*.merges",
    "*.bpe",
)


def parse_hf_source(source: str, revision: str | None = None) -> tuple[str, str]:
    """Return a model repository id and revision from an HF id or URL."""
    value = source.strip()
    if value.startswith(("huggingface.co/", "www.huggingface.co/", "hf.co/")):
        value = "https://" + value

    inferred_revision: str | None = None
    if value.startswith("hf://"):
        parsed = urlparse(value)
        parts = [parsed.netloc, *parsed.path.strip("/").split("/")]
        if parts[0] == "models":
            parts = parts[1:]
        if len(parts) < 2:
            raise ManifestError(f"invalid Hugging Face model URL: {source!r}")
        repo = "/".join(parts[:2])
    elif "://" in value:
        parsed = urlparse(value)
        host = (parsed.hostname or "").lower()
        if host not in {"huggingface.co", "www.huggingface.co", "hf.co"}:
            raise ManifestError("only huggingface.co model URLs are supported")
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        if parts and parts[0] in {"models", "datasets", "spaces"}:
            if parts[0] != "models":
                raise ManifestError("the source must identify a Hugging Face model repository")
            parts = parts[1:]
        if len(parts) < 2:
            raise ManifestError(f"invalid Hugging Face model URL: {source!r}")
        repo = "/".join(parts[:2])
        if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
            inferred_revision = parts[3]
    else:
        plain = value
        if "@" in plain:
            plain, inferred_revision = plain.rsplit("@", 1)
        parts = plain.strip("/").split("/")
        if len(parts) != 2 or not all(parts):
            raise ManifestError(
                "Hugging Face id must have the form 'owner/model'"
            )
        repo = "/".join(parts)

    repo_path = PurePosixPath(repo)
    if len(repo_path.parts) != 2 or ".." in repo_path.parts:
        raise ManifestError(f"invalid Hugging Face model id: {repo!r}")
    return repo, revision or inferred_revision or "main"


def infer_name(repo: str) -> str:
    name = _SAFE_NAME_RE.sub("-", repo.rsplit("/", 1)[-1].lower()).strip("-.")
    if not name:
        raise ManifestError(f"cannot infer a model name from {repo!r}; pass --name")
    return validate_name(name)


def _sibling_names(info: Any) -> list[str]:
    siblings = getattr(info, "siblings", None)
    if not siblings:
        raise ModelctlError("Hugging Face returned no files for this model repository")
    names = []
    for sibling in siblings:
        value = (
            sibling.get("rfilename")
            if isinstance(sibling, dict)
            else getattr(sibling, "rfilename", None)
        )
        if isinstance(value, str) and value:
            names.append(value)
    if not names:
        raise ModelctlError("Hugging Face returned no usable model filenames")
    return sorted(set(names))


def _gguf_groups(files: list[str]) -> list[list[str]]:
    groups: dict[tuple[str, str, int], list[tuple[int, str]]] = {}
    standalone: list[list[str]] = []
    for value in files:
        if not value.lower().endswith(".gguf"):
            continue
        if Path(value).name.lower().startswith("mmproj"):
            continue
        match = _SHARD_RE.match(Path(value).name)
        if match:
            key = (
                str(Path(value).parent),
                match.group("base"),
                int(match.group("count")),
            )
            groups.setdefault(key, []).append((int(match.group("index")), value))
        else:
            standalone.append([value])

    result = standalone
    for (_, _, count), shards in groups.items():
        indices = {index for index, _ in shards}
        required = set(range(1, count + 1))
        if indices != required:
            missing = sorted(required.difference(indices))
            raise ManifestError(
                "Hugging Face repository has an incomplete GGUF group; missing shard(s) "
                + ", ".join(f"{index:05d}" for index in missing)
            )
        result.append([value for _, value in sorted(shards)])
    return sorted(result, key=lambda group: group[0].lower())


def _select_gguf(files: list[str], quantization: str | None) -> list[str]:
    groups = _gguf_groups(files)
    if not groups:
        raise ManifestError("llama.cpp runtime requires at least one model GGUF file")
    if quantization:
        needle = quantization.lower()
        groups = [
            group
            for group in groups
            if any(needle in Path(value).name.lower() for value in group)
        ]
        if not groups:
            raise ManifestError(
                f"no GGUF quantization matches {quantization!r}"
            )
    if len(groups) != 1:
        examples = ", ".join(Path(group[0]).name for group in groups[:6])
        raise ManifestError(
            "multiple GGUF quantizations are available; pass --quantization. "
            f"Candidates: {examples}"
        )
    return groups[0]


def generate_manifest_document(
    source: str,
    *,
    name: str | None = None,
    revision: str | None = None,
    quantization: str | None = None,
    runtime: str = "auto",
    api: Any | None = None,
) -> dict[str, Any]:
    repo, selected_revision = parse_hf_source(source, revision)
    model_name = validate_name(name) if name else infer_name(repo)
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    info = api.model_info(repo, revision=selected_revision)
    commit = getattr(info, "sha", None)
    if not isinstance(commit, str) or not commit:
        raise ModelctlError(
            f"Hugging Face did not resolve {repo}@{selected_revision} to a commit"
        )
    files = _sibling_names(info)
    has_safetensors = any(
        value.lower().endswith(".safetensors") for value in files
    )
    has_bin = any(value.lower().endswith(".bin") for value in files)
    has_standard_weights = has_safetensors or has_bin
    has_gguf = any(value.lower().endswith(".gguf") for value in files)

    selected_runtime = runtime
    if selected_runtime == "auto":
        if has_standard_weights:
            selected_runtime = "vllm"
        elif has_gguf:
            selected_runtime = "llama.cpp"
        else:
            raise ManifestError(
                "cannot infer a runtime: repository has no safetensors, bin, or GGUF weights"
            )

    document: dict[str, Any] = {
        "name": model_name,
        "repo": repo,
        "revision": selected_revision,
    }
    if selected_runtime == "llama.cpp":
        selected = _select_gguf(files, quantization)
        document.update(
            {
                "format": "gguf",
                "include": selected,
                "entrypoint": selected[0],
                "runtime": {"type": "llama.cpp", "executable": "llama-server"},
            }
        )
    elif selected_runtime == "vllm":
        if not has_standard_weights:
            raise ManifestError("vLLM runtime requires safetensors or bin weights")
        patterns = [
            pattern
            for pattern in _STANDARD_PATTERNS
            if any(fnmatch.fnmatch(filename, pattern) for filename in files)
        ]
        if not has_safetensors:
            patterns.append("*.bin")
        document.update(
            {
                "format": "safetensors" if has_safetensors else "hf",
                "include": patterns,
                "runtime": {"type": "vllm", "executable": "vllm"},
            }
        )
    else:
        raise ManifestError("runtime must be auto, vllm, or llama.cpp")

    # Validate generated output with the same parser used by update.
    parse_manifest(document, model_name)
    return document


def write_generated_manifest(
    root: Path,
    document: dict[str, Any],
    *,
    output: Path | None = None,
    force: bool = False,
) -> Path:
    import yaml

    name = str(document["name"])
    path = output or root / "manifests" / f"{name}.yaml"
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    temp = path.parent / f".{path.name}.tmp-{os.getpid()}"
    try:
        temp.write_text(text, encoding="utf-8")
        with temp.open("rb") as handle:
            os.fsync(handle.fileno())
        if force:
            os.replace(temp, path)
        else:
            try:
                os.link(temp, path)
            except FileExistsError as exc:
                raise ManifestError(
                    f"manifest already exists at {path}; pass --force to replace it"
                ) from exc
    finally:
        temp.unlink(missing_ok=True)
    return path


def download_from_hf(
    root: Path,
    source: str,
    *,
    name: str | None = None,
    revision: str | None = None,
    quantization: str | None = None,
    runtime: str = "auto",
    force_manifest: bool = False,
    api: Any | None = None,
    snapshot: Any | None = None,
) -> tuple[Path, Path]:
    """Generate a manifest, then download, validate, publish, and activate it."""
    import yaml

    from .operations import update_model

    document = generate_manifest_document(
        source,
        name=name,
        revision=revision,
        quantization=quantization,
        runtime=runtime,
        api=api,
    )
    manifest_name = str(document["name"])
    path = root / "manifests" / f"{manifest_name}.yaml"
    if path.exists() and not force_manifest:
        try:
            existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ManifestError(f"cannot read existing manifest {path}: {exc}") from exc
        if existing != document:
            raise ManifestError(
                f"manifest already exists with different settings at {path}; "
                "pass --force to replace it"
            )
        manifest_path = path.absolute()
    else:
        manifest_path = write_generated_manifest(
            root, document, force=force_manifest
        )
    manifest = parse_manifest(document, manifest_name)
    active = update_model(root, manifest, api=api, snapshot=snapshot)
    return manifest_path, active
