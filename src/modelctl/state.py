from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any


class UpdateState(StrEnum):
    UNRESOLVED = "UNRESOLVED"
    RESOLVED_TO_COMMIT = "RESOLVED_TO_COMMIT"
    READY_TO_ACTIVATE = "READY_TO_ACTIVATE"
    DOWNLOADING_TO_STAGING = "DOWNLOADING_TO_STAGING"
    PARTIAL_RESUMABLE = "PARTIAL_RESUMABLE"
    VALIDATING = "VALIDATING"
    FAILED_UNPUBLISHED = "FAILED_UNPUBLISHED"
    PUBLISHING_OBJECT = "PUBLISHING_OBJECT"
    UPDATING_REFERENCE = "UPDATING_REFERENCE"
    FAILED_TO_ACTIVATE = "FAILED_TO_ACTIVATE"
    ACTIVE_ON_NAS = "ACTIVE_ON_NAS"


class DeleteState(StrEnum):
    PLANNED = "PLANNED"
    REFERENCE_REMOVED = "REFERENCE_REMOVED"
    CATALOG_REFRESHED = "CATALOG_REFRESHED"
    STAGING_REMOVED = "STAGING_REMOVED"
    OBJECTS_REMOVED = "OBJECTS_REMOVED"
    FAILED = "FAILED"
    COMPLETE = "COMPLETE"


class SyncState(StrEnum):
    NAS_REFERENCE_READ = "NAS_REFERENCE_READ"
    SOURCE_VALIDATED = "SOURCE_VALIDATED"
    RSYNC_TO_LOCAL_STAGING = "RSYNC_TO_LOCAL_STAGING"
    PARTIAL_RESUMABLE = "PARTIAL_RESUMABLE"
    LOCAL_VALIDATION = "LOCAL_VALIDATION"
    LOCAL_OBJECT_PUBLICATION = "LOCAL_OBJECT_PUBLICATION"
    LOCAL_REFERENCE_UPDATE = "LOCAL_REFERENCE_UPDATE"
    READY_FOR_SERVICE_RESTART = "READY_FOR_SERVICE_RESTART"


@dataclass
class StateJournal:
    path: Path
    operation: str
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Continue a valid journal so failed retries retain activation evidence."""
        if self.history or not self.path.is_file():
            return
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        existing = document.get("history")
        if document.get("operation") != self.operation or not isinstance(existing, list):
            return
        if not all(
            isinstance(event, dict) and isinstance(event.get("state"), str)
            for event in existing
        ):
            return
        self.history.extend(dict(event) for event in existing)

    def transition(self, state: StrEnum | str, **details: Any) -> None:
        event = {
            "state": str(state),
            "at": datetime.now(timezone.utc).isoformat(),
            **details,
        }
        self.history.append(event)
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "operation": self.operation,
            "state": self.history[-1]["state"] if self.history else None,
            "history": self.history,
        }
        temp = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        try:
            with temp.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.path)
        finally:
            temp.unlink(missing_ok=True)
