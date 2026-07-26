from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator

from .errors import ModelctlError
from .manifest import ModelManifest, validate_name


@dataclass(frozen=True)
class Layout:
    root: Path

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def staging(self) -> Path:
        return self.root / ".staging"

    @property
    def active(self) -> Path:
        return self.root / "active"

    @property
    def state(self) -> Path:
        return self.root / "state"

    @property
    def locks(self) -> Path:
        return self.root / ".locks"

    def prepare(self) -> None:
        for path in (self.models, self.staging, self.active, self.state, self.locks):
            path.mkdir(parents=True, exist_ok=True)

    def object_path(self, manifest: ModelManifest, commit: str) -> Path:
        repo = PurePosixPath(manifest.repo)
        settings = {
            "name": manifest.name,
            "include": manifest.include,
            "format": manifest.format,
            "entrypoint": manifest.entrypoint,
            "runtime": {
                "kind": manifest.runtime.kind,
                "executable": manifest.runtime.executable,
                "args": manifest.runtime.args,
            },
        }
        # Keep object identities stable for pre-companion manifests.
        if manifest.model_card:
            settings["model_card"] = manifest.model_card
        if manifest.companions:
            settings["companions"] = manifest.companions
        selection = hashlib.sha256(
            json.dumps(settings, sort_keys=True).encode()
        ).hexdigest()[:12]
        return self.models.joinpath(*repo.parts, f"{commit}--{selection}")

    def staging_path(self, manifest: ModelManifest, commit: str) -> Path:
        relative = self.object_path(manifest, commit).relative_to(self.models)
        return self.staging / relative

    def active_path(self, name: str) -> Path:
        return self.active / validate_name(name)

    def state_path(self, name: str) -> Path:
        return self.state / f"{validate_name(name)}.json"


@contextmanager
def model_lock(layout: Layout, name: str) -> Iterator[None]:
    import fcntl

    layout.locks.mkdir(parents=True, exist_ok=True)
    path = layout.locks / f"{validate_name(name)}.lock"
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def assert_same_filesystem(staging: Path, final: Path) -> None:
    staging.parent.mkdir(parents=True, exist_ok=True)
    final.parent.mkdir(parents=True, exist_ok=True)
    if staging.parent.stat().st_dev != final.parent.stat().st_dev:
        raise ModelctlError(
            f"staging and final object paths are not on the same filesystem: "
            f"{staging.parent} vs {final.parent}"
        )


def atomic_symlink(target: Path, link: Path) -> None:
    """Atomically replace link with a symlink to target.

    A relative target keeps a root movable and makes the temporary symlink valid
    before os.replace publishes it.
    """
    link.parent.mkdir(parents=True, exist_ok=True)
    temp = link.parent / f".{link.name}.tmp-{os.getpid()}"
    try:
        temp.unlink(missing_ok=True)
        relative = os.path.relpath(target, start=link.parent)
        temp.symlink_to(relative, target_is_directory=target.is_dir())
        os.replace(temp, link)
    finally:
        temp.unlink(missing_ok=True)
