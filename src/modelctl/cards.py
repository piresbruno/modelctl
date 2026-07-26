from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from .errors import ModelctlError
from .layout import Layout, assert_same_filesystem, atomic_symlink, model_lock
from .manifest import validate_name
from .operations import _active_object

_CARD_METADATA = "card.json"
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_USAGE_TERMS = (
    "usage",
    "quickstart",
    "quick start",
    "getting started",
    "inference",
    "running the model",
    "run the model",
    "transformers",
    "vllm",
    "llama.cpp",
    "llama cpp",
)
_MAX_CARD_BYTES = 10 * 1024 * 1024


class CardUnavailable(ModelctlError):
    """The selected Hugging Face revision has no root README model card."""


@dataclass(frozen=True)
class CardResult:
    name: str
    status: str
    path: Path | None = None
    message: str | None = None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_card(path: Path) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ModelctlError(f"cannot read model card at {path}: {exc}") from exc
    if not data:
        raise ModelctlError(f"model card at {path} is empty")
    if len(data) > _MAX_CARD_BYTES:
        raise ModelctlError(
            f"model card at {path} exceeds the {_MAX_CARD_BYTES // (1024 * 1024)} MiB limit"
        )
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ModelctlError(f"model card at {path} is not UTF-8 Markdown") from exc
    return data


def _heading_ranges(markdown: str) -> list[tuple[int, int, str]]:
    lines = markdown.splitlines(keepends=True)
    headings: list[tuple[int, int, str]] = []
    fence: str | None = None
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        marker = stripped[:3]
        if marker in {"```", "~~~"}:
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            continue
        if fence is not None:
            continue
        match = _HEADING_RE.match(line.rstrip("\r\n"))
        if match:
            headings.append((index, len(match.group(1)), match.group(2).strip()))

    ranges: list[tuple[int, int, str]] = []
    for position, (start, level, title) in enumerate(headings):
        end = len(lines)
        for next_start, next_level, _ in headings[position + 1 :]:
            if next_level <= level:
                end = next_start
                break
        ranges.append((start, end, title))
    return ranges


def extract_usage_sections(markdown: str) -> tuple[list[str], str]:
    """Return upstream usage heading names and their complete Markdown blocks."""
    lines = markdown.splitlines(keepends=True)
    selected: list[tuple[int, int, str]] = []
    for start, end, title in _heading_ranges(markdown):
        normalized = re.sub(r"\s+", " ", title.lower())
        if not any(term in normalized for term in _USAGE_TERMS):
            continue
        if any(parent_start <= start and end <= parent_end for parent_start, parent_end, _ in selected):
            continue
        selected.append((start, end, title))
    sections = "\n".join("".join(lines[start:end]).rstrip() for start, end, _ in selected)
    return [title for _, _, title in selected], sections


def _run_document(
    *,
    root: Path,
    name: str,
    metadata: dict[str, Any],
    source_url: str,
    card_text: str,
) -> tuple[bytes, list[str]]:
    headings, sections = extract_usage_sections(card_text)
    quoted_name = shlex.quote(name)
    quoted_root = shlex.quote(str(root))
    runtime = metadata.get("runtime", {})
    runtime_kind = str(runtime.get("kind", "unknown")) if isinstance(runtime, dict) else "unknown"
    executable = str(runtime.get("executable", "unknown")) if isinstance(runtime, dict) else "unknown"

    lines = [
        f"# Running {name}",
        "",
        f"- Hugging Face model: [{metadata['repo']}]({source_url})",
        f"- Revision: `{metadata['commit']}`",
        f"- Runtime: `{runtime_kind}`",
        f"- Runtime executable: `{executable}`",
        f"- Entrypoint: `{metadata['entrypoint']}`",
        "",
        "## Resolve the active model",
        "",
        "```bash",
        f"modelctl path {quoted_name} --root {quoted_root}",
        "```",
        "",
        "## Generate the server command",
        "",
        "Review the command before running it:",
        "",
        "```bash",
        f"modelctl serve-command {quoted_name} --root {quoted_root}",
        "```",
    ]
    companions = metadata.get("companions", {})
    if isinstance(companions, dict) and companions:
        lines.extend(["", "## Companion files", ""])
        for kind, path in sorted(companions.items()):
            lines.append(f"- `{kind}`: `{path}`")
    if sections:
        lines.extend(
            [
                "",
                "## Upstream usage instructions",
                "",
                "The following sections are copied from the model card at the exact revision above.",
                "",
                sections,
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Upstream usage instructions",
                "",
                (
                    "The Hugging Face model card does not contain a recognized usage or "
                    "inference section. Consult the complete `README.md` in this directory "
                    "before running the model."
                ),
            ]
        )
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8"), headings


def _current_card(cards: Path, name: str, repo: str, commit: str) -> Path | None:
    reference = cards / name
    if not reference.is_symlink():
        return None
    try:
        directory = reference.resolve(strict=True)
        directory.relative_to((cards / ".objects").resolve(strict=True))
        raw = json.loads((directory / _CARD_METADATA).read_text(encoding="utf-8"))
        readme = (directory / "README.md").read_bytes()
        run = (directory / "RUN.md").read_bytes()
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        raw.get("schema") != 1
        or raw.get("name") != name
        or raw.get("repo") != repo
        or raw.get("commit") != commit
        or raw.get("sha256", {}).get("README.md") != _sha256(readme)
        or raw.get("sha256", {}).get("RUN.md") != _sha256(run)
    ):
        return None
    return directory


def _published_card_matches(
    directory: Path,
    *,
    name: str,
    repo: str,
    commit: str,
    readme: bytes,
    run: bytes,
) -> bool:
    try:
        raw = json.loads((directory / _CARD_METADATA).read_text(encoding="utf-8"))
        stored_readme = (directory / "README.md").read_bytes()
        stored_run = (directory / "RUN.md").read_bytes()
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        raw.get("schema") == 1
        and raw.get("name") == name
        and raw.get("repo") == repo
        and raw.get("commit") == commit
        and stored_readme == readme
        and stored_run == run
        and raw.get("sha256", {}).get("README.md") == _sha256(stored_readme)
        and raw.get("sha256", {}).get("RUN.md") == _sha256(stored_run)
    )


def _publish_card(
    root: Path,
    name: str,
    metadata: dict[str, Any],
    readme: bytes,
    *,
    source: str,
    source_url: str,
) -> Path:
    cards = root / "cards"
    objects = cards / ".objects" / name
    staging_root = cards / ".staging"
    objects.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    run, headings = _run_document(
        root=root,
        name=name,
        metadata=metadata,
        source_url=source_url,
        card_text=readme.decode("utf-8"),
    )
    identity = hashlib.sha256(readme + b"\0" + run).hexdigest()[:12]
    final = objects / f"{metadata['commit']}--{identity}"
    staging = staging_root / f"{name}-{os.getpid()}-{uuid4().hex}"
    assert_same_filesystem(staging, final)
    staging.mkdir(parents=True)
    try:
        (staging / "README.md").write_bytes(readme)
        (staging / "RUN.md").write_bytes(run)
        payload = {
            "schema": 1,
            "name": name,
            "repo": metadata["repo"],
            "revision": metadata.get("revision"),
            "commit": metadata["commit"],
            "source": source,
            "source_url": source_url,
            "fetched_at": datetime.now(UTC).isoformat(),
            "instruction_sections": headings,
            "sha256": {"README.md": _sha256(readme), "RUN.md": _sha256(run)},
        }
        (staging / _CARD_METADATA).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        for path in staging.iterdir():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        _fsync_directory(staging)
        if final.exists() and _published_card_matches(
            final,
            name=name,
            repo=str(metadata["repo"]),
            commit=str(metadata["commit"]),
            readme=readme,
            run=run,
        ):
            shutil.rmtree(staging)
        else:
            if final.exists():
                final = objects / f"{metadata['commit']}--{identity}-{uuid4().hex[:8]}"
            os.rename(staging, final)
            _fsync_directory(objects)
        atomic_symlink(final, cards / name)
        _fsync_directory(cards)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return (cards / name).resolve(strict=True)


def _source_identity(metadata: dict[str, Any], name: str) -> tuple[str, str]:
    repo = metadata.get("repo")
    commit = metadata.get("commit")
    repo_path = PurePosixPath(repo) if isinstance(repo, str) else None
    if (
        repo_path is None
        or repo_path.is_absolute()
        or ".." in repo_path.parts
        or len(repo_path.parts) != 2
    ):
        raise ModelctlError(f"active object for {name!r} has invalid repository metadata")
    if (
        not isinstance(commit, str)
        or not commit
        or commit in {".", ".."}
        or any(character in commit for character in "/\\")
    ):
        raise ModelctlError(f"active object for {name!r} has invalid commit metadata")
    return repo, commit


def sync_model_card(
    root: Path,
    name: str,
    *,
    force: bool = False,
    downloader: Callable[..., Any] | None = None,
) -> CardResult:
    name = validate_name(name)
    layout = Layout(root)
    cards = root / "cards"

    # Do not hold the model update lock during a potentially slow network request.
    with model_lock(layout, name):
        object_path, metadata = _active_object(layout, name)
        repo, commit = _source_identity(metadata, name)
        current = _current_card(cards, name, repo, commit)
        if current is not None and not force:
            return CardResult(name, "unchanged", current)
        model_card = metadata.get("model_card")
        if isinstance(model_card, str):
            readme = _read_card(object_path / model_card)
            source = "model-object"
        else:
            readme = None
            source = "huggingface"

    if readme is None:
        if downloader is None:
            from huggingface_hub import hf_hub_download

            downloader = hf_hub_download
        try:
            downloaded = downloader(
                repo_id=repo,
                filename="README.md",
                revision=commit,
            )
        except Exception as exc:
            from huggingface_hub.errors import EntryNotFoundError

            if isinstance(exc, EntryNotFoundError):
                raise CardUnavailable(
                    f"{repo}@{commit} has no root README.md model card"
                ) from exc
            raise ModelctlError(
                f"cannot fetch the model card for {repo}@{commit}: {exc}"
            ) from exc
        readme = _read_card(Path(downloaded))

    source_url = f"https://huggingface.co/{repo}/tree/{commit}"
    with model_lock(layout, name):
        _, current_metadata = _active_object(layout, name)
        if _source_identity(current_metadata, name) != (repo, commit):
            raise ModelctlError(
                f"active model {name!r} changed while its card was being fetched; retry"
            )
        current = _current_card(cards, name, repo, commit)
        if current is not None and not force:
            return CardResult(name, "unchanged", current)
        path = _publish_card(
            root,
            name,
            current_metadata,
            readme,
            source=source,
            source_url=source_url,
        )
    return CardResult(name, "updated", path)


def active_model_names(root: Path) -> list[str]:
    active = Layout(root).active
    if not active.is_dir():
        return []
    names: list[str] = []
    for path in active.iterdir():
        if path.name.startswith(".") or not path.is_symlink():
            continue
        try:
            names.append(validate_name(path.name))
        except ModelctlError:
            continue
    return sorted(names)


def sync_model_cards(
    root: Path,
    names: Iterable[str] | None = None,
    *,
    force: bool = False,
    downloader: Callable[..., Any] | None = None,
) -> list[CardResult]:
    selected = list(names) if names else active_model_names(root)
    results: list[CardResult] = []
    for name in selected:
        try:
            results.append(
                sync_model_card(root, name, force=force, downloader=downloader)
            )
        except CardUnavailable as exc:
            results.append(CardResult(name, "unavailable", message=str(exc)))
        except Exception as exc:  # noqa: BLE001 - batch processing must isolate models
            results.append(
                CardResult(name, "failed", message=f"{type(exc).__name__}: {exc}")
            )
    return results
