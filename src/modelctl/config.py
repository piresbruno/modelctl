from __future__ import annotations

import json
import os
from pathlib import Path

from .errors import ModelctlError


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base).expanduser() / "modelctl" / "config.json"
    return Path.home() / ".config" / "modelctl" / "config.json"


def _load() -> dict[str, object]:
    path = config_path()
    if not path.exists():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelctlError(f"cannot read configuration {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ModelctlError(f"configuration {path} must contain a JSON object")
    return document


def _load_path(key: str) -> str | None:
    value = _load().get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ModelctlError(f"configuration {config_path()} has no valid {key}")
    return value


def load_root() -> str | None:
    return _load_path("root")


def load_local_root() -> str | None:
    return _load_path("local_root")


def _save_path(key: str, root: Path) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        document = _load()
        document[key] = str(root)
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        os.replace(temporary, path)
    except OSError as exc:
        raise ModelctlError(f"cannot write configuration {path}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return path


def save_root(root: Path) -> Path:
    return _save_path("root", root)


def save_local_root(root: Path) -> Path:
    return _save_path("local_root", root)
