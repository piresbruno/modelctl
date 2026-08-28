from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from .errors import ModelctlError, ValidationError
from .hf_cache import (
    RECEIVE_PROTO,
    read_source_etag,
    staging_path_for,
    write_transfer_file_list,
)
from .operations import active_object, resolve_active_name
from .validation import ExpectedFile


def _ssh_argv(
    ssh: str,
    host: str,
    port: int | None,
    identity: str | None,
    remote_args: list[str],
) -> list[str]:
    argv = [ssh]
    if port is not None:
        argv += ["-p", str(port)]
    if identity is not None:
        argv += ["-i", str(identity)]
    argv.append(host)
    # ssh concatenates argv with spaces and the remote shell re-splits it, so
    # every remote argument must be shell-quoted exactly once.
    argv.extend(shlex.quote(str(argument)) for argument in remote_args)
    return argv


def _ssh_transport(ssh: str, port: int | None, identity: str | None) -> str:
    parts = [ssh]
    if port is not None:
        parts += ["-p", str(port)]
    if identity is not None:
        parts += ["-i", str(identity)]
    parts += ["-o", "Compression=no"]
    return shlex.join(parts)


def _run_ssh(
    ssh: str,
    host: str,
    port: int | None,
    identity: str | None,
    remote_args: list[str],
    runner: Callable[..., Any],
) -> str:
    argv = _ssh_argv(ssh, host, port, identity, remote_args)
    try:
        result = runner(argv, check=False, capture_output=True, text=True, errors="replace")
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise ModelctlError(f"cannot reach {host}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise ModelctlError(f"remote command failed on {host}: {detail}")
    return result.stdout


def _remote_payload(stdout: str, host: str, *, command: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except (ValueError, TypeError) as exc:
        raise ModelctlError(
            f"{host} returned no valid JSON for 'modelctl {command}'"
        ) from exc
    if not isinstance(payload, dict):
        raise ModelctlError(
            f"{host} returned an invalid payload for 'modelctl {command}'"
        )
    if payload.get("proto") != RECEIVE_PROTO:
        raise ModelctlError(
            f"{host} runs an incompatible modelctl receive-cache protocol "
            f"{payload.get('proto')!r}; install a compatible modelctl on the remote"
        )
    return payload


def _push_file_list(source: Path, metadata: dict[str, Any]) -> list[str]:
    """Every file rsync must move: the selected repository files, the object
    metadata journal, and retained Hugging Face download metadata."""
    names = [str(item["path"]) for item in metadata["files"]]
    names.append(".modelctl.json")
    cache_root = source / ".cache"
    if cache_root.is_dir():
        for path in sorted(cache_root.rglob("*")):
            if path.is_file():
                names.append(path.relative_to(source).as_posix())
    return sorted(names)


def _remote_command(
    name: str,
    cache_dir: Path,
    *,
    probe: bool = False,
    staging: Path | None = None,
) -> list[str]:
    args = ["modelctl", "receive-cache"]
    if probe:
        args.append("--probe")
    args += [str(name), "--cache-dir", str(cache_dir)]
    if staging is not None:
        args += ["--staging", str(staging)]
    return args


def push_model(
    source_root: Path,
    remote_cache_dir: Path,
    name: str,
    *,
    host: str,
    port: int | None = None,
    identity: str | None = None,
    ssh: str = "ssh",
    rsync: str = "rsync",
    runner: Callable[..., Any] = subprocess.run,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Copy one active NAS model into another host's Hugging Face cache over ssh.

    The source is always the local managed store. The remote runs
    ``modelctl receive-cache``, rsync transfers the object's selected files
    into the remote cache staging area, and the remote validates and publishes
    blobs, a snapshot, refs, and a registration record. Interrupted transfers
    remain resumable in remote staging.
    """
    resolved = resolve_active_name(source_root, name)
    object_path, metadata = active_object(source_root, resolved)
    commit = str(metadata.get("commit", ""))
    if not commit:
        raise ValidationError("active object metadata has no commit")
    expected = [ExpectedFile.from_dict(item) for item in metadata["files"]]
    files = {
        item.path: read_source_etag(object_path, item.path, commit)
        for item in expected
    }
    probe_args = _remote_command(resolved, remote_cache_dir, probe=True)
    if progress is not None:
        progress(f"push: probing {host} for receive-cache support")
    stdout = _run_ssh(ssh, host, port, identity, probe_args, runner)
    payload = _remote_payload(stdout, host, command="receive-cache --probe")
    if not isinstance(payload.get("cache"), str) or not payload["cache"]:
        raise ModelctlError(f"{host} returned no cache path in its probe payload")
    remote_cache = Path(payload["cache"])
    staging = staging_path_for(remote_cache, str(metadata["repo"]), commit, files)

    names = _push_file_list(object_path, metadata)
    descriptor, list_path = tempfile.mkstemp(prefix="modelctl-push-", suffix=".files")
    os.close(descriptor)
    try:
        write_transfer_file_list(Path(list_path), names)
        if progress is not None:
            progress(
                f"push: transferring {len(files)} selected files "
                f"(+{len(names) - len(files)} metadata files) to {host}:{staging}"
            )
        command = [
            rsync,
            "--archive",
            "--partial",
            "--delete",
            "--human-readable",
            "--info=progress2",
            "--from0",
            f"--files-from={list_path}",
            "-e",
            _ssh_transport(ssh, port, identity),
            f"{object_path}/",
            f"{host}:{shlex.quote(str(staging) + '/')}",
        ]
        try:
            runner(command, check=True)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise ModelctlError(
                f"rsync to {host} failed; remote staging is resumable: {exc}"
            ) from exc
    finally:
        Path(list_path).unlink(missing_ok=True)

    commit_args = _remote_command(resolved, remote_cache_dir, staging=staging)
    if progress is not None:
        progress(f"push: validating and publishing on {host}")
    stdout = _run_ssh(ssh, host, port, identity, commit_args, runner)
    payload = _remote_payload(stdout, host, command="receive-cache")
    if not isinstance(payload.get("snapshot"), str) or not payload["snapshot"]:
        raise ModelctlError(f"{host} returned no snapshot path after publication")
    return payload["snapshot"]
