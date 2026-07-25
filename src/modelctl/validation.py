from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .errors import ValidationError
from .manifest import ModelManifest, RuntimeProfile

METADATA_FILE = ".modelctl.json"
_SHARD_RE = re.compile(r"^(?P<base>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$", re.I)


@dataclass(frozen=True)
class ExpectedFile:
    path: str
    size: int | None = None
    sha256: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExpectedFile":
        return cls(str(raw["path"]), raw.get("size"), raw.get("sha256"))


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not value:
        raise ValidationError(f"unsafe repository path: {value!r}")
    return path


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_expected(directory: Path, expected: Iterable[ExpectedFile]) -> None:
    root = directory.resolve()
    seen = set()
    for item in expected:
        relative = _safe_relative(item.path)
        if relative.as_posix() in seen:
            raise ValidationError(f"duplicate expected file: {item.path}")
        seen.add(relative.as_posix())
        path = directory.joinpath(*relative.parts)
        if not path.is_file():
            raise ValidationError(f"missing expected file: {item.path}")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise ValidationError(f"file escapes object directory: {item.path}") from exc
        stat = path.stat()
        if item.size is not None and stat.st_size != item.size:
            raise ValidationError(
                f"size mismatch for {item.path}: expected {item.size}, got {stat.st_size}"
            )
        if item.sha256 is not None and _digest(path) != item.sha256.lower():
            raise ValidationError(f"SHA-256 mismatch for {item.path}")
    if not seen:
        raise ValidationError("the selected snapshot contains no files")


def _gguf_files(directory: Path, expected: Iterable[ExpectedFile]) -> list[str]:
    return sorted(item.path for item in expected if item.path.lower().endswith(".gguf"))


def resolve_entrypoint(
    directory: Path, manifest: ModelManifest, expected: list[ExpectedFile]
) -> str:
    ggufs = _gguf_files(directory, expected)
    is_gguf = manifest.format == "gguf" or (manifest.format == "auto" and bool(ggufs))
    if not is_gguf:
        entrypoint = manifest.entrypoint or "."
        path = directory if entrypoint == "." else directory / entrypoint
        if not path.exists():
            raise ValidationError(f"entrypoint does not exist: {entrypoint}")
        return entrypoint
    if manifest.runtime.kind == "vllm":
        raise ValidationError(
            "GGUF with vLLM is outside the version 1 default workflow"
        )
    if not ggufs:
        raise ValidationError("GGUF manifest selected no .gguf files")

    selected = manifest.entrypoint
    if selected is None:
        primary = [path for path in ggufs if not Path(path).name.lower().startswith("mmproj")]
        groups: dict[tuple[str, int, str], list[tuple[int, str]]] = {}
        standalone: list[str] = []
        for value in primary:
            filename = Path(value).name
            match = _SHARD_RE.match(filename)
            if match:
                key = (match.group("base"), int(match.group("count")), str(Path(value).parent))
                groups.setdefault(key, []).append((int(match.group("index")), value))
            else:
                standalone.append(value)
        if len(groups) + len(standalone) != 1:
            raise ValidationError(
                "GGUF include patterns must select exactly one quantization; set entrypoint explicitly"
            )
        if groups:
            _, shards = next(iter(groups.items()))
            selected = min(shards)[1]
        else:
            selected = standalone[0]
    if selected not in ggufs:
        raise ValidationError(f"GGUF entrypoint was not downloaded: {selected}")

    primary_files = {
        value for value in ggufs if not Path(value).name.lower().startswith("mmproj")
    }
    match = _SHARD_RE.match(Path(selected).name)
    if match:
        count = int(match.group("count"))
        base = match.group("base")
        folder = Path(selected).parent
        expected_names = {
            (folder / f"{base}-{index:05d}-of-{count:05d}.gguf").as_posix()
            for index in range(1, count + 1)
        }
        missing = expected_names.difference(ggufs)
        if missing:
            raise ValidationError(
                "incomplete sharded GGUF; missing " + ", ".join(sorted(missing))
            )
        extra = primary_files.difference(expected_names)
        if extra:
            raise ValidationError(
                "GGUF include patterns selected additional quantizations: "
                + ", ".join(sorted(extra))
            )
        first = (folder / f"{base}-00001-of-{count:05d}.gguf").as_posix()
        selected = first
    else:
        extra = primary_files.difference({selected})
        if extra:
            raise ValidationError(
                "GGUF include patterns selected additional quantizations: "
                + ", ".join(sorted(extra))
            )
    return selected


def write_metadata(
    directory: Path,
    manifest: ModelManifest,
    commit: str,
    expected: list[ExpectedFile],
    entrypoint: str,
) -> None:
    payload = {
        "schema": 1,
        "name": manifest.name,
        "repo": manifest.repo,
        "revision": manifest.revision,
        "commit": commit,
        "include": list(manifest.include),
        "format": manifest.format,
        "entrypoint": entrypoint,
        "runtime": asdict(manifest.runtime),
        "files": [asdict(item) for item in expected],
    }
    path = directory / METADATA_FILE
    temp = directory / f".{METADATA_FILE}.tmp-{os.getpid()}"
    try:
        temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        with temp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_metadata(directory: Path) -> dict[str, Any]:
    path = directory / METADATA_FILE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValidationError(f"missing or invalid object metadata at {path}") from exc
    if raw.get("schema") != 1 or not isinstance(raw.get("files"), list):
        raise ValidationError(f"unsupported object metadata at {path}")
    return raw


def validate_object(
    directory: Path,
    *,
    expected_commit: str | None = None,
    expected_name: str | None = None,
) -> dict[str, Any]:
    metadata = read_metadata(directory)
    if expected_commit is not None and metadata.get("commit") != expected_commit:
        raise ValidationError(
            f"object commit is {metadata.get('commit')!r}, expected {expected_commit!r}"
        )
    if expected_name is not None and metadata.get("name") != expected_name:
        raise ValidationError(
            f"object belongs to {metadata.get('name')!r}, expected {expected_name!r}"
        )
    expected = [ExpectedFile.from_dict(item) for item in metadata["files"]]
    validate_expected(directory, expected)
    entrypoint = metadata.get("entrypoint")
    if not isinstance(entrypoint, str):
        raise ValidationError("object metadata has no entrypoint")
    target = directory if entrypoint == "." else directory / entrypoint
    if not target.exists():
        raise ValidationError(f"object entrypoint is missing: {entrypoint}")
    return metadata


def runtime_from_metadata(raw: dict[str, Any]) -> RuntimeProfile:
    runtime = raw.get("runtime")
    if not isinstance(runtime, dict):
        raise ValidationError("object metadata has no runtime profile")
    args = runtime.get("args", ())
    return RuntimeProfile(
        kind=str(runtime.get("kind", "")),
        executable=str(runtime.get("executable", "")),
        args=tuple(str(item) for item in args),
    )
