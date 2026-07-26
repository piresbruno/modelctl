import json
from pathlib import Path

from huggingface_hub.errors import EntryNotFoundError

from modelctl.cards import extract_usage_sections, sync_model_card, sync_model_cards
from modelctl.layout import Layout, atomic_symlink
from modelctl.manifest import parse_manifest
from modelctl.validation import ExpectedFile, write_metadata


def _active_model(root: Path, *, name="demo", card: bytes | None = None):
    raw = {"repo": "org/demo", "runtime": "vllm"}
    if card is not None:
        raw["model_card"] = "README.md"
        raw["include"] = ["config.json", "README.md"]
    manifest = parse_manifest(raw, name)
    layout = Layout(root)
    layout.prepare()
    commit = "a" * 40
    object_path = layout.object_path(manifest, commit)
    object_path.mkdir(parents=True)
    (object_path / "config.json").write_bytes(b"{}")
    expected = [ExpectedFile("config.json", 2)]
    if card is not None:
        (object_path / "README.md").write_bytes(card)
        expected.append(ExpectedFile("README.md", len(card)))
    write_metadata(object_path, manifest, commit, expected, ".")
    atomic_symlink(object_path, layout.active_path(name))
    return object_path, commit


def test_fetches_exact_commit_and_builds_run_instructions(tmp_path):
    _, commit = _active_model(tmp_path)
    upstream = b"---\nlicense: apache-2.0\n---\n# Demo\n\n## Usage\n\n```python\nprint('run')\n```\n"
    downloaded = tmp_path / "downloaded.md"
    downloaded.write_bytes(upstream)
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs)
        return downloaded

    result = sync_model_card(tmp_path, "demo", downloader=downloader)

    assert result.status == "updated"
    assert calls == [
        {"repo_id": "org/demo", "filename": "README.md", "revision": commit}
    ]
    card_dir = tmp_path / "cards" / "demo"
    assert card_dir.is_symlink()
    assert (card_dir / "README.md").read_bytes() == upstream
    run = (card_dir / "RUN.md").read_text()
    assert "modelctl path demo --root" in run
    assert "modelctl serve-command demo --root" in run
    assert "## Usage" in run
    assert "print('run')" in run
    assert "license: apache-2.0" not in run
    metadata = json.loads((card_dir / "card.json").read_text())
    assert metadata["commit"] == commit
    assert metadata["source"] == "huggingface"
    assert metadata["instruction_sections"] == ["Usage"]


def test_reuses_card_in_active_object_without_hugging_face(tmp_path):
    upstream = b"# Demo\n\n## Inference\n\nUse the official example.\n"
    _active_model(tmp_path, card=upstream)

    def no_download(**kwargs):
        raise AssertionError("the local validated model card should be reused")

    result = sync_model_card(tmp_path, "demo", downloader=no_download)
    metadata = json.loads((result.path / "card.json").read_text())
    assert result.status == "updated"
    assert metadata["source"] == "model-object"
    assert (result.path / "README.md").read_bytes() == upstream


def test_matching_sidecar_is_idempotent(tmp_path):
    _active_model(tmp_path)
    downloaded = tmp_path / "downloaded.md"
    downloaded.write_text("# Demo\n")
    calls = 0

    def downloader(**kwargs):
        nonlocal calls
        calls += 1
        return downloaded

    assert sync_model_card(tmp_path, "demo", downloader=downloader).status == "updated"
    assert sync_model_card(tmp_path, "demo", downloader=downloader).status == "unchanged"
    assert calls == 1


def test_corrupt_published_card_is_replaced_atomically(tmp_path):
    _active_model(tmp_path)
    downloaded = tmp_path / "downloaded.md"
    downloaded.write_text("# Demo\n")

    def downloader(**kwargs):
        return downloaded

    first = sync_model_card(tmp_path, "demo", downloader=downloader)
    first_target = first.path
    (first_target / "README.md").write_text("corrupt\n")

    repaired = sync_model_card(tmp_path, "demo", downloader=downloader)
    assert repaired.status == "updated"
    assert repaired.path != first_target
    assert (repaired.path / "README.md").read_text() == "# Demo\n"
    assert (tmp_path / "cards" / "demo").resolve() == repaired.path


def test_missing_upstream_readme_is_reported_as_unavailable(tmp_path):
    _active_model(tmp_path)

    def downloader(**kwargs):
        raise EntryNotFoundError("missing")

    results = sync_model_cards(tmp_path, ["demo"], downloader=downloader)
    assert results[0].status == "unavailable"
    assert "no root README.md" in results[0].message


def test_batch_continues_after_a_failure(tmp_path):
    _active_model(tmp_path, name="a")
    _active_model(tmp_path, name="b")
    downloaded = tmp_path / "downloaded.md"
    downloaded.write_text("# Demo\n")

    def downloader(**kwargs):
        if kwargs["revision"] == "a" * 40 and not getattr(downloader, "used", False):
            downloader.used = True
            raise OSError("network down")
        return downloaded

    results = sync_model_cards(tmp_path, ["a", "b"], downloader=downloader)
    assert [result.status for result in results] == ["failed", "updated"]


def test_usage_extraction_ignores_frontmatter_and_fenced_headings():
    markdown = """---
title: Demo
---
# Demo

```markdown
## Usage inside an example
```

## Quick start

Run this.

### vLLM

Nested instructions.

## Evaluation

Not instructions.
"""
    headings, sections = extract_usage_sections(markdown)
    assert headings == ["Quick start"]
    assert "Run this." in sections
    assert "Nested instructions." in sections
    assert "Usage inside an example" not in sections
    assert "Not instructions." not in sections
