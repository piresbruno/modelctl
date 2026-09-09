import json

from modelctl.state import StateJournal


def test_state_journal_preserves_existing_history(tmp_path):
    path = tmp_path / "demo.json"
    path.write_text(
        json.dumps(
            {
                "operation": "update",
                "state": "ACTIVE_ON_NAS",
                "history": [
                    {
                        "state": "ACTIVE_ON_NAS",
                        "at": "2026-01-01T00:00:00+00:00",
                        "object": "/models/demo",
                        "commit": "a" * 40,
                    }
                ],
            }
        )
    )

    journal = StateJournal(path, "update")
    journal.transition("FAILED_UNPUBLISHED", error="validation failed")

    document = json.loads(path.read_text())
    assert [event["state"] for event in document["history"]] == [
        "ACTIVE_ON_NAS",
        "FAILED_UNPUBLISHED",
    ]


def test_state_journal_does_not_mix_operations(tmp_path):
    path = tmp_path / "demo.json"
    path.write_text(
        json.dumps(
            {
                "operation": "sync",
                "state": "READY",
                "history": [{"state": "READY"}],
            }
        )
    )

    journal = StateJournal(path, "update")
    journal.transition("UNRESOLVED")

    document = json.loads(path.read_text())
    assert document["operation"] == "update"
    assert [event["state"] for event in document["history"]] == ["UNRESOLVED"]


def test_state_journal_write_survives_hostile_fsync(tmp_path, monkeypatch):
    """SMB servers deny fsync on read-only descriptors; journal writes must
    flush through the writable handle (regression: EACCES on CIFS)."""
    import errno
    import fcntl
    import os

    seen_modes = []
    real_fsync = os.fsync

    def hostile_fsync(fd):
        mode = fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE
        if mode == os.O_RDONLY:
            raise PermissionError(errno.EACCES, "Permission denied")
        seen_modes.append(mode)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", hostile_fsync)

    path = tmp_path / "demo.json"
    journal = StateJournal(path, "update")
    journal.transition("UNRESOLVED", revision="main")

    document = json.loads(path.read_text())
    assert [event["state"] for event in document["history"]] == ["UNRESOLVED"]
    assert seen_modes == [os.O_WRONLY]
