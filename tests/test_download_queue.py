import errno
import threading
from pathlib import Path

import pytest

import modelctl.download_queue as queue_module
from modelctl.download_queue import (
    DownloadQueueError,
    DownloadRequest,
    load_download_queue,
    prepare_download_queue,
    run_download_queue,
    validate_queue_root,
)


def _generator(source, *, name, revision, quantization, runtime, mmproj, mtp):
    return {
        "name": name,
        "repo": source,
        "revision": revision or "main",
        "format": "gguf" if runtime == "llama.cpp" else "safetensors",
        "include": [quantization] if quantization else ["*.safetensors"],
        "runtime": {"type": runtime, "executable": runtime},
    }


def test_loads_strict_mapping_queue_and_validates_fields(tmp_path):
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text(
        """downloads:
  - source: Qwen/Qwen3-8B
    name: qwen3
    revision: v2
  - source: org/model-GGUF
    quantization: Q4_K_M
    runtime: llama.cpp
    mmproj: mmproj-F16.gguf
    mtp: mtp-model.gguf
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
            mmproj="mmproj-F16.gguf",
            mtp="mtp-model.gguf",
            force=True,
        ),
    ]

    direct_list = tmp_path / "list.yaml"
    direct_list.write_text("- source: org/model\n")
    with pytest.raises(DownloadQueueError, match="root must be a mapping"):
        load_download_queue(direct_list)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("downloads:\n  - source: org/model\n    quantizaton: Q4\n")
    with pytest.raises(DownloadQueueError, match="quantizaton"):
        load_download_queue(invalid)


def test_preflight_rejects_duplicate_effective_names_before_remote_calls():
    called = False

    def generator(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    requests = [
        DownloadRequest("org/model-GGUF", quantization="Q4"),
        DownloadRequest("org/model-GGUF", quantization="Q8"),
    ]
    with pytest.raises(DownloadQueueError, match="unique 'name'"):
        prepare_download_queue(requests, generator=generator)
    assert called is False


def test_remote_preflight_finishes_before_any_download(tmp_path):
    downloaded = []

    def generator(source, **kwargs):
        if source == "org/bad":
            raise RuntimeError("quantization does not exist")
        return _generator(source, **kwargs)

    def downloader(root, source, **kwargs):
        downloaded.append(source)
        return Path("manifest"), Path("active")

    with pytest.raises(DownloadQueueError, match="quantization does not exist"):
        run_download_queue(
            tmp_path,
            [
                DownloadRequest("org/good", name="good"),
                DownloadRequest("org/bad", name="bad"),
            ],
            generator=generator,
            downloader=downloader,
        )
    assert downloaded == []


def test_existing_manifest_conflict_fails_before_download(tmp_path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "model.yaml").write_text(
        "name: model\nrepo: org/other\nrevision: main\n"
    )
    downloaded = []

    def downloader(root, source, **kwargs):
        downloaded.append(source)
        return Path("manifest"), Path("active")

    with pytest.raises(DownloadQueueError, match="manifest already exists"):
        run_download_queue(
            tmp_path,
            [DownloadRequest("org/model", name="model")],
            generator=_generator,
            downloader=downloader,
        )
    assert downloaded == []


def test_root_preflight_reports_cifs_symlink_requirement(tmp_path, monkeypatch):
    def unsupported(target, link):
        raise OSError(errno.EOPNOTSUPP, "Operation not supported")

    monkeypatch.setattr(queue_module, "atomic_symlink", unsupported)
    with pytest.raises(DownloadQueueError, match="mfsymlinks"):
        validate_queue_root(tmp_path)


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

    requests = [DownloadRequest(f"org/model-{index}") for index in range(3)]
    results = run_download_queue(
        tmp_path,
        requests,
        jobs=3,
        generator=_generator,
        downloader=downloader,
    )

    assert maximum == 3
    assert [result.request for result in results] == requests
    assert all(result.succeeded for result in results)
    assert all(call[1]["runtime"] == "auto" for call in calls)
    assert all(call[1]["mmproj"] is None for call in calls)
    assert all(call[1]["mtp"] is None for call in calls)
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
        [
            DownloadRequest("org/bad", name="bad"),
            DownloadRequest("org/good", name="good"),
        ],
        jobs=1,
        generator=_generator,
        downloader=downloader,
    )

    assert not results[0].succeeded
    assert isinstance(results[0].error, RuntimeError)
    assert results[1].succeeded
    assert completed == ["org/good"]


def test_queue_rejects_nonpositive_jobs(tmp_path):
    with pytest.raises(DownloadQueueError, match="at least 1"):
        run_download_queue(
            tmp_path,
            [DownloadRequest("org/model")],
            jobs=0,
            generator=_generator,
        )
