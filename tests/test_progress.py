"""IRI-021: throttled staging-size progress emission for the irori hub."""

import io
import json
import time
from pathlib import Path

from modelctl.progress import DownloadProgressEmitter, _staging_bytes


def _lines(stream: io.StringIO) -> list[dict]:
    return [
        json.loads(line[len("progress: ") :])
        for line in stream.getvalue().splitlines()
        if line.startswith("progress: ")
    ]


def test_staging_bytes_counts_nested_files_and_ignores_missing(tmp_path):
    staging = tmp_path / "staging"
    (staging / "sub").mkdir(parents=True)
    (staging / "sub" / "model.safetensors.incomplete").write_bytes(b"x" * 100)
    (staging / "top.bin").write_bytes(b"y" * 50)
    (staging / ".cache" / "huggingface").mkdir(parents=True)
    (staging / ".cache" / "huggingface" / "download.meta").write_bytes(b"z" * 10)
    assert _staging_bytes(staging) == 160


def test_emits_shape_with_name_entry_and_total(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "model.safetensors.incomplete").write_bytes(b"x" * 100)
    (staging / "tokenizer.json").write_bytes(b"y" * 50)
    stream = io.StringIO()
    emitter = DownloadProgressEmitter(entry=2, stream=stream, interval=0.05)
    emitter.configure(staging=staging, name="demo", total_bytes=200)
    emitter.start()
    time.sleep(0.3)
    emitter.stop()

    lines = _lines(stream)
    assert lines, "emitter produced no lines"
    assert all(line["name"] == "demo" for line in lines)
    assert all(line["entry"] == 2 for line in lines)
    assert all(line["bytes_total"] == 200 for line in lines)
    assert all(0 <= line["bytes_done"] <= 200 for line in lines)
    assert lines[-1]["bytes_done"] == 150


def test_caps_bytes_done_at_the_total(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "big.bin").write_bytes(b"x" * 500)
    stream = io.StringIO()
    emitter = DownloadProgressEmitter(stream=stream, interval=0.05)
    emitter.configure(staging=staging, name="demo", total_bytes=100)
    emitter.start()
    time.sleep(0.15)
    emitter.stop()
    assert _lines(stream) and all(line["bytes_done"] <= 100 for line in _lines(stream))


def test_throttles_to_one_line_per_interval(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    stream = io.StringIO()
    emitter = DownloadProgressEmitter(stream=stream, interval=10.0)
    emitter.configure(staging=staging, name="demo", total_bytes=None)
    emitter.start()
    time.sleep(0.3)
    emitter.stop()
    assert len(_lines(stream)) == 1


def test_no_emission_without_configure(tmp_path):
    stream = io.StringIO()
    emitter = DownloadProgressEmitter(stream=stream, interval=0.05)
    emitter.start()
    time.sleep(0.2)
    emitter.stop()
    assert stream.getvalue() == ""


def test_speed_bps_present_once_bytes_grow(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "weights.bin").write_bytes(b"x" * 10)
    stream = io.StringIO()
    emitter = DownloadProgressEmitter(stream=stream, interval=0.05)
    emitter.configure(staging=staging, name="demo", total_bytes=None)
    emitter.start()
    time.sleep(0.12)
    (staging / "weights.bin").write_bytes(b"x" * 10_000)
    time.sleep(0.15)
    emitter.stop()

    lines = _lines(stream)
    assert lines
    assert lines[-1]["bytes_done"] == 10_000
    assert any(line.get("speed_bps", 0) > 0 for line in lines)


def test_omits_total_when_unknown(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "a.bin").write_bytes(b"x" * 10)
    stream = io.StringIO()
    emitter = DownloadProgressEmitter(stream=stream, interval=0.05)
    emitter.configure(staging=staging, name="demo", total_bytes=None)
    emitter.start()
    time.sleep(0.15)
    emitter.stop()
    lines = _lines(stream)
    assert lines and all("bytes_total" not in line for line in lines)


def test_stop_is_idempotent_and_joins(tmp_path):
    emitter = DownloadProgressEmitter(stream=io.StringIO(), interval=0.05)
    emitter.configure(staging=tmp_path, name="demo", total_bytes=None)
    emitter.start()
    time.sleep(0.05)
    emitter.stop()
    emitter.stop()
    assert emitter._thread is None


def test_missing_staging_dir_is_tolerated(tmp_path):
    stream = io.StringIO()
    emitter = DownloadProgressEmitter(stream=stream, interval=0.05)
    emitter.configure(staging=tmp_path / "gone", name="demo", total_bytes=10)
    emitter.start()
    time.sleep(0.15)
    emitter.stop()
    assert _lines(stream) and all(line["bytes_done"] == 0 for line in _lines(stream))
