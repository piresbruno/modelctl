import threading
from pathlib import Path

import pytest

from modelctl.download_queue import (
    DownloadQueueError,
    DownloadRequest,
    load_download_queue,
    run_download_queue,
)


def test_loads_mapping_or_list_queue_and_validates_fields(tmp_path):
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text(
        """downloads:
  - source: Qwen/Qwen3-8B
    name: qwen3
    revision: v2
  - source: org/model-GGUF
    quantization: Q4_K_M
    runtime: llama.cpp
    force: true
"""
    )
    requests = load_download_queue(mapping)
    assert requests == [
        DownloadRequest("Qwen/Qwen3-8B", name="qwen3", revision="v2"),
        DownloadRequest(
            "org/model-GGUF",
            quantization="Q4_K_M",
            runtime="llama.cpp",
            force=True,
        ),
    ]

    direct_list = tmp_path / "list.yaml"
    direct_list.write_text("- source: org/model\n")
    assert load_download_queue(direct_list) == [DownloadRequest("org/model")]

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("downloads:\n  - source: org/model\n    quantizaton: Q4\n")
    with pytest.raises(DownloadQueueError, match="quantizaton"):
        load_download_queue(invalid)


def test_queue_limits_concurrency_and_preserves_result_order(tmp_path):
    active = 0
    maximum = 0
    lock = threading.Lock()
    barrier = threading.Barrier(3)
    calls = []

    def downloader(root, source, **kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            calls.append((source, kwargs))
        barrier.wait(timeout=2)
        with lock:
            active -= 1
        return root / "manifests" / f"{source}.yaml", root / "active" / source

    requests = [DownloadRequest(f"model-{index}") for index in range(3)]
    results = run_download_queue(tmp_path, requests, jobs=3, downloader=downloader)

    assert maximum == 3
    assert [result.request for result in results] == requests
    assert all(result.succeeded for result in results)
    assert all(call[1]["runtime"] == "auto" for call in calls)
    assert all(call[1]["force_manifest"] is False for call in calls)


def test_queue_records_failure_and_continues(tmp_path):
    completed = []

    def downloader(root, source, **kwargs):
        if source == "org/bad":
            raise RuntimeError("download failed")
        completed.append(source)
        return Path("manifest.yaml"), Path("active")

    results = run_download_queue(
        tmp_path,
        [DownloadRequest("org/bad"), DownloadRequest("org/good")],
        jobs=1,
        downloader=downloader,
    )

    assert not results[0].succeeded
    assert isinstance(results[0].error, RuntimeError)
    assert results[1].succeeded
    assert completed == ["org/good"]


def test_queue_rejects_nonpositive_jobs(tmp_path):
    with pytest.raises(DownloadQueueError, match="at least 1"):
        run_download_queue(tmp_path, [DownloadRequest("org/model")], jobs=0)
