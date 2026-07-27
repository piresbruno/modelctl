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


def load_root() -> str | None:
    path = config_path()
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelctlError(f"cannot read configuration {path}: {exc}") from exc
    root = document.get("root") if isinstance(document, dict) else None
    if not isinstance(root, str) or not root.strip():
        raise ModelctlError(f"configuration {path} has no valid root")
    return root


def save_root(root: Path) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps({"root": str(root)}, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        os.replace(temporary, path)
    except OSError as exc:
        raise ModelctlError(f"cannot write configuration {path}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return path
