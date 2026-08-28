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


def _prepare_remote_staging(
    ssh: str,
    host: str,
    port: int | None,
    identity: str | None,
    staging: Path,
    cache: Path,
    runner: Callable[..., Any],
    progress: Callable[[str], None] | None,
) -> None:
    """Create the remote staging directory and verify the cache is writable.

    rsync's receiver creates the destination root with a single mkdir (no
    ``-p``) under ``--files-from``, so the deep staging path must already
    exist or the transfer fails with an opaque ENOENT. Creating it also proves
    the cache tree is writable before any bytes move.
    """
    if progress is not None:
        progress(f"push: preparing {cache} on {host}")
    script = 'mkdir -p "$1" && test -w "$2"'
    argv = _ssh_argv(
        ssh,
        host,
        port,
        identity,
        ["sh", "-c", script, "sh", str(staging), str(cache)],
    )
    try:
        result = runner(
            argv, check=False, capture_output=True, text=True, errors="replace"
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise ModelctlError(f"cannot reach {host}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise ModelctlError(
            f"cache directory {cache} is not writable on {host}: {detail}; "
            "create it and make it writable by the ssh user, or use the "
            "default cache path"
        )


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
    metadata journal, and the retained per-file Hugging Face download metadata
    used for remote ETag verification.

    Other Hugging Face cache artifacts (for example xet tree JSONs under
    ``.cache/huggingface/trees``) are not needed by ``receive-cache`` and can
    carry unreadable permissions on NAS mounts, so they are excluded.
    """
    names = [str(item["path"]) for item in metadata["files"]]
    names.append(".modelctl.json")
    metadata_root = source / ".cache" / "huggingface" / "download"
    if metadata_root.is_dir():
        for path in sorted(metadata_root.rglob("*")):
            if path.is_file():
                names.append(path.relative_to(source).as_posix())
    return sorted(names)


_REMOTE_MODELCTL_CANDIDATES = (
    '"$HOME/.local/bin/modelctl"',
    "/usr/local/bin/modelctl",
    "/usr/bin/modelctl",
)


def _discover_remote_modelctl(
    ssh: str,
    host: str,
    port: int | None,
    identity: str | None,
    runner: Callable[..., Any],
) -> str:
    """Locate the remote modelctl binary without relying on ssh PATH setup.

    Non-interactive ssh commands run without the user's shell rc files, so
    ``uv tool install`` binaries under ``~/.local/bin`` are invisible to PATH.
    Probe the standard install locations explicitly; ``$HOME`` expands inside
    the remote ``sh -c`` script.
    """
    script = (
        "cd; for p in "
        + " ".join(_REMOTE_MODELCTL_CANDIDATES)
        + "; do [ -x \"$p\" ] && { echo \"$p\"; exit 0; }; done; exit 1"
    )
    argv = _ssh_argv(ssh, host, port, identity, ["sh", "-c", script])
    try:
        result = runner(
            argv, check=False, capture_output=True, text=True, errors="replace"
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise ModelctlError(f"cannot reach {host}: {exc}") from exc
    if result.returncode != 0:
        raise ModelctlError(
            f"modelctl is not installed on {host}; install it with "
            "'uv tool install modelctl' on the remote or pass "
            "--remote-modelctl PATH"
        )
    path = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not path:
        raise ModelctlError(f"modelctl not found on {host}")
    return path


def _remote_command(
    modelctl_bin: str,
    name: str,
    cache_dir: Path,
    *,
    probe: bool = False,
    staging: Path | None = None,
) -> list[str]:
    args = [str(modelctl_bin), "receive-cache"]
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
    remote_modelctl: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Copy one active NAS model into another host's Hugging Face cache over ssh.

    The source is always the local managed store. The remote
    ``modelctl receive-cache`` binary is auto-discovered (or taken from
    ``remote_modelctl``), rsync transfers the object's selected files into the
    remote cache staging area, and the remote validates and publishes blobs, a
    snapshot, refs, and a registration record. Interrupted transfers remain
    resumable in remote staging.
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
    if remote_modelctl is not None:
        modelctl_bin = str(remote_modelctl)
    else:
        modelctl_bin = _discover_remote_modelctl(ssh, host, port, identity, runner)
    probe_args = _remote_command(modelctl_bin, resolved, remote_cache_dir, probe=True)
    if progress is not None:
        progress(f"push: probing {host} for receive-cache support")
    stdout = _run_ssh(ssh, host, port, identity, probe_args, runner)
    payload = _remote_payload(stdout, host, command="receive-cache --probe")
    if not isinstance(payload.get("cache"), str) or not payload["cache"]:
        raise ModelctlError(f"{host} returned no cache path in its probe payload")
    remote_cache = Path(payload["cache"])
    staging = staging_path_for(remote_cache, str(metadata["repo"]), commit, files)
    _prepare_remote_staging(ssh, host, port, identity, staging, remote_cache, runner, progress)

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

    commit_args = _remote_command(modelctl_bin, resolved, remote_cache_dir, staging=staging)
    if progress is not None:
        progress(f"push: validating and publishing on {host}")
    stdout = _run_ssh(ssh, host, port, identity, commit_args, runner)
    payload = _remote_payload(stdout, host, command="receive-cache")
    if not isinstance(payload.get("snapshot"), str) or not payload["snapshot"]:
        raise ModelctlError(f"{host} returned no snapshot path after publication")
    return payload["snapshot"]
