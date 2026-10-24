"""Throttled download-progress emission for the irori hub (IRI-021).

During a real (non-dry-run) download a daemon thread watches the staging
directory and emits machine-readable progress lines on stderr:

    progress: {"bytes_done": 123, "bytes_total": 456, "entry": 1, "name": "..."}

- ``bytes_done``: bytes present under the staging directory (incomplete and
  completed files alike). ``st_size`` is backend-agnostic — it tracks regular
  writes, xet materialization, and resume-from-``.incomplete`` alike — and is
  capped at ``bytes_total`` when a total is known.
- ``bytes_total``: the sum of the dry-run estimate's known file sizes;
  omitted when the estimate carried none.
- ``name``/``entry``: irori row matching (the hub prefers the store name,
  then the 1-based batch index, then the single running row). ``name`` is set
  by :meth:`DownloadProgressEmitter.configure` from the manifest; ``entry``
  is fixed at construction by the queue worker.
- ``speed_bps``: bytes delta over the sample interval; present from the
  second sample onward and only while the byte count grows.

Emission is throttled to one line per interval (default 1s) per model — job
output frames are persisted, so flooding would bloat the job record. The
emitter is inert unless :meth:`start` is called; dry-run paths never
construct one, so human and ``--json`` output are unchanged.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import IO, Callable

DEFAULT_INTERVAL = 1.0


def _staging_bytes(staging: Path) -> int:
    """Sum the sizes of every regular file under ``staging`` (best effort)."""
    total = 0
    try:
        for path in staging.rglob("*"):
            try:
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


class DownloadProgressEmitter:
    """Watch a staging directory and emit throttled ``progress:`` lines."""

    def __init__(
        self,
        *,
        entry: int | None = None,
        stream: IO[str] | None = None,
        interval: float = DEFAULT_INTERVAL,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._entry = entry
        self._stream = stream if stream is not None else sys.stderr
        self._interval = max(interval, 0.05)
        self._clock = clock
        self._staging: Path | None = None
        self._name: str | None = None
        self._total: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def configure(self, *, staging: Path, name: str, total_bytes: int | None) -> None:
        """Bind the watched directory, the store name, and the byte total."""
        self._staging = staging
        self._name = name
        self._total = total_bytes if total_bytes is not None and total_bytes > 0 else None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="modelctl-progress", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _loop(self) -> None:
        previous: int | None = None
        previous_at: float | None = None
        first = True
        while True:
            if not first and self._stop.wait(self._interval):
                return
            first = False
            if self._staging is None:
                continue
            now = self._clock()
            done = _staging_bytes(self._staging)
            if self._total is not None:
                done = min(done, self._total)
            payload: dict[str, object] = {"bytes_done": done}
            if self._name is not None:
                payload["name"] = self._name
            if self._entry is not None:
                payload["entry"] = self._entry
            if self._total is not None:
                payload["bytes_total"] = self._total
            if previous is not None and previous_at is not None and done > previous:
                elapsed = max(now - previous_at, 1e-9)
                payload["speed_bps"] = max(1, int((done - previous) / elapsed))
            self._emit(payload)
            previous = done
            previous_at = now

    def _emit(self, payload: dict[str, object]) -> None:
        try:
            print(
                f"progress: {json.dumps(payload, separators=(',', ':'))}",
                file=self._stream,
                flush=True,
            )
        except (OSError, ValueError):
            # The consumer pipe is gone (agent restart mid-download); dropping
            # progress beats crashing the download thread.
            self._stop.set()
