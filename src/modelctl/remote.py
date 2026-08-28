from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from .errors import ModelctlError
from .generation import parse_hf_source
from .hf_cache import (
    RECEIVE_PROTO,
    list_records,
    load_record,
    staging_path_for,
    write_transfer_file_list,
)
from .manifest import validate_name


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


def _cache_source(
    cache_dir: Path, selector: str
) -> tuple[str, dict[str, Any], dict[str, str], str, Path]:
    """Resolve a local HF cache registration and build a transfer overlay.

    Returns ``(name, metadata, files, commit, overlay)``. The overlay is a
    small directory whose entries are symlinks to the registered snapshot
    files plus a generated ``.modelctl.json`` and retained ETag metadata, so
    the remote ``receive-cache`` path validates and publishes the transfer
    unchanged. Callers must remove the overlay afterwards.
    """
    if "/" not in selector:
        record = load_record(cache_dir, validate_name(selector))
    else:
        repo, _ = parse_hf_source(selector)
        matches = [item for item in list_records(cache_dir) if item.repo == repo]
        if not matches:
            raise ModelctlError(
                f"repository {repo!r} has no local cache registration on this "
                "host; run 'modelctl sync-local' first"
            )
        if len(matches) > 1:
            names = ", ".join(repr(item.name) for item in matches)
            raise ModelctlError(
                f"repository {repo!r} is registered under multiple names: {names}"
            )
        record = matches[0]
    overlay = Path(tempfile.mkdtemp(prefix="modelctl-push-cache-"))
    commit = record.commit
    try:
        for path, etag in record.files.items():
            relative = PurePosixPath(path)
            source = record.snapshot.joinpath(*relative.parts)
            link = overlay.joinpath(*relative.parts)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(source)
        metadata = dict(record.metadata)
        metadata["commit"] = commit
        (overlay / ".modelctl.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        metadata_root = overlay / ".cache" / "huggingface" / "download"
        for path, etag in record.files.items():
            relative = PurePosixPath(path)
            target = record.snapshot.joinpath(*relative.parts)
            metadata_target = metadata_root.joinpath(*relative.parts).with_suffix(
                relative.suffix + ".metadata"
            )
            metadata_target.parent.mkdir(parents=True, exist_ok=True)
            timestamp = target.stat().st_mtime + 60
            metadata_target.write_text(
                f"{commit}\n{etag}\n{timestamp}\n", encoding="utf-8"
            )
        return record.name, metadata, dict(record.files), commit, overlay
    except BaseException:
        shutil.rmtree(overlay, ignore_errors=True)
        raise


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


def _run_rsync_streams(
    commands: list[list[str]], runner: Callable[..., Any], host: str
) -> None:
    """Run one or more rsync argv lists, concurrently."""
    if len(commands) == 1:
        try:
            runner(commands[0], check=True)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise ModelctlError(
                f"rsync to {host} failed; remote staging is resumable: {exc}"
            ) from exc
        return
    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=len(commands)) as pool:
        futures = [pool.submit(runner, command, check=True) for command in commands]
        for future in futures:
            try:
                future.result()
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                errors.append(exc)
    if errors:
        raise ModelctlError(
            f"rsync to {host} failed; remote staging is resumable: {errors[0]}"
        ) from errors[0]


def push_model(
    local_cache_dir: Path,
    name: str,
    *,
    host: str,
    port: int | None = None,
    identity: str | None = None,
    ssh: str = "ssh",
    rsync: str = "rsync",
    remote_modelctl: str | None = None,
    jobs: int = 1,
    runner: Callable[..., Any] = subprocess.run,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Copy one model from the local Hugging Face cache into another host's
    Hugging Face cache over ssh.

    The model must already have a modelctl registration in the local cache
    (for example from ``modelctl sync-local`` or an earlier push); push reads
    that snapshot and transfers only its files through the remote
    ``receive-cache`` validation and publication path. With ``jobs`` greater
    than one the transfer is split into that many independent rsync streams
    into the same staging directory, which helps many-file models saturate a
    fast fabric. Interrupted transfers remain resumable in remote staging.
    """
    resolved, metadata, files, commit, overlay = _cache_source(local_cache_dir, name)
    if progress is not None:
        progress(f"push: reading {resolved} from the local Hugging Face cache")
    try:
        if remote_modelctl is not None:
            modelctl_bin = str(remote_modelctl)
        else:
            modelctl_bin = _discover_remote_modelctl(ssh, host, port, identity, runner)
        probe_args = _remote_command(
            modelctl_bin, resolved, local_cache_dir, probe=True
        )
        if progress is not None:
            progress(f"push: probing {host} for receive-cache support")
        stdout = _run_ssh(ssh, host, port, identity, probe_args, runner)
        payload = _remote_payload(stdout, host, command="receive-cache --probe")
        if not isinstance(payload.get("cache"), str) or not payload["cache"]:
            raise ModelctlError(f"{host} returned no cache path in its probe payload")
        remote_cache = Path(payload["cache"])
        staging = staging_path_for(remote_cache, str(metadata["repo"]), commit, files)
        _prepare_remote_staging(
            ssh, host, port, identity, staging, remote_cache, runner, progress
        )

        names = _push_file_list(overlay, metadata)
        stream_count = min(max(jobs, 1), len(names))
        chunks = [names[index::stream_count] for index in range(stream_count)]
        commands: list[list[str]] = []
        list_paths: list[Path] = []
        try:
            for index, chunk in enumerate(chunks):
                descriptor, list_path = tempfile.mkstemp(
                    prefix=f"modelctl-push-{index}-", suffix=".files"
                )
                os.close(descriptor)
                list_paths.append(Path(list_path))
                write_transfer_file_list(Path(list_path), chunk)
                commands.append(
                    [
                        rsync,
                        "--archive",
                        "--copy-links",
                        "--partial",
                        "--delete",
                        "--human-readable",
                        "--info=progress2",
                        "--from0",
                        f"--files-from={list_path}",
                        "-e",
                        _ssh_transport(ssh, port, identity),
                        f"{overlay}/",
                        f"{host}:{shlex.quote(str(staging) + '/')}",
                    ]
                )
            if progress is not None:
                progress(
                    f"push: transferring {len(files)} files "
                    f"(+{len(names) - len(files)} metadata files) to {host}:{staging} "
                    f"with {len(commands)} rsync stream(s)"
                )
            _run_rsync_streams(commands, runner, host)
        finally:
            for list_path in list_paths:
                list_path.unlink(missing_ok=True)

        commit_args = _remote_command(
            modelctl_bin, resolved, local_cache_dir, staging=staging
        )
        if progress is not None:
            progress(f"push: validating and publishing on {host}")
        stdout = _run_ssh(ssh, host, port, identity, commit_args, runner)
        payload = _remote_payload(stdout, host, command="receive-cache")
        if not isinstance(payload.get("snapshot"), str) or not payload["snapshot"]:
            raise ModelctlError(f"{host} returned no snapshot path after publication")
        return payload["snapshot"]
    finally:
        shutil.rmtree(overlay, ignore_errors=True)
