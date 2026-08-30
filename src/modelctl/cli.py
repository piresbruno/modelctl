from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .cards import sync_model_cards
from .catalog import catalog_path, catalog_status, refresh_catalog
from .config import load_local_root, load_root, save_local_root, save_root
from .download_queue import (
    DownloadQueueError,
    execute_download_queue,
    load_download_queue,
    prepare_download_queue,
    validate_prepared_manifests,
    validate_queue_root,
)
from .errors import ModelctlError
from .generation import (
    download_from_hf,
    generate_manifest_document,
    write_generated_manifest,
)
from .integrity import (
    audit_active_references,
    cleanup_quarantine,
    malformed_active_references,
    repair_active_reference,
)
from .hf_cache import malformed_cached_records, receive_staged_cache
from .maintenance import (
    audit_objects,
    audit_staging,
    cleanup_objects,
    cleanup_staging,
)
from .manifest import load_manifest, validate_name
from .operations import (
    DeleteResult,
    active_entrypoint,
    delete_cached,
    delete_model,
    list_active_models,
    list_cached_models,
    local_active_entrypoint,
    serve_cached_command,
    serve_command,
    sync_local,
    update_model,
)
from .remote import push_model

DEFAULT_ROOT = "/var/lib/llm-models"
HELP_FORMATTER = argparse.RawDescriptionHelpFormatter


def _root(value: str | None) -> Path:
    selected = value or os.environ.get("MODELCTL_ROOT") or load_root() or DEFAULT_ROOT
    return Path(selected).expanduser().absolute()


def _local_root(value: str | None) -> Path:
    selected = (
        value
        or os.environ.get("MODELCTL_LOCAL_ROOT")
        or load_local_root()
        or DEFAULT_ROOT
    )
    return Path(selected).expanduser().absolute()


def _cache_dir(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().absolute()
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser().absolute()
    if os.environ.get("HF_HOME"):
        return (Path(os.environ["HF_HOME"]).expanduser() / "hub").absolute()
    legacy = os.environ.get("MODELCTL_LOCAL_ROOT") or load_local_root()
    if legacy:
        return Path(legacy).expanduser().absolute()
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return (cache_home / "huggingface" / "hub").absolute()


def _selected_cache(args: argparse.Namespace) -> Path:
    explicit = getattr(args, "cache_dir", None)
    legacy = getattr(args, "local_root", None)
    if explicit and legacy:
        raise ModelctlError("--cache-dir and deprecated --root cannot be combined")
    return _cache_dir(explicit or legacy)


def _format_size(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        metavar="PATH",
        help=(
            "model store root (default: $MODELCTL_ROOT, saved configuration, "
            f"or {DEFAULT_ROOT})"
        ),
    )


def _add_local_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-dir", metavar="PATH",
        help=("Hugging Face hub cache directory (default: HF_HUB_CACHE, "
              "HF_HOME/hub, or the HF default)"),
    )
    parser.add_argument(
        "--root", dest="local_root", metavar="PATH",
        help="deprecated alias for --cache-dir",
    )


def _print_models(root: Path, *, json_output: bool, local: bool = False) -> None:
    if local:
        models = list_cached_models(root)
    else:
        try:
            models, _ = refresh_catalog(root, list_active_models)
        except ModelctlError as exc:
            models = list_active_models(root)
            print(
                f"warning: live listing succeeded but catalog refresh failed: {exc}",
                file=sys.stderr,
            )
    if json_output:
        payload = [
            {"name": model.name, "runtime": model.runtime, "repository": model.repo}
            for model in models
        ]
        print(json.dumps(payload, indent=2))
        return
    malformed = (
        malformed_cached_records(root) if local else malformed_active_references(root)
    )
    warning = _malformed_warning(malformed, local=local)
    if not models:
        print(f"No active models in {root}.")
        if warning:
            print(warning, file=sys.stderr)
        return
    rows = [(model.name, model.runtime, model.repo) for model in models]
    headers = ("NAME", "RUNTIME", "REPOSITORY")
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    template = "  ".join(f"{{{index}:<{width}}}" for index, width in enumerate(widths))
    print(template.format(*headers))
    for row in rows:
        print(template.format(*row))
    if warning:
        print(warning, file=sys.stderr)


def _malformed_warning(malformed: list[str], *, local: bool) -> str | None:
    """Actionable warning naming skipped malformed entries (local listings
    name the stale registrations so they can be deleted)."""
    if not malformed:
        return None
    if local:
        names = ", ".join(sorted(set(malformed)))
        return (
            f"warning: skipped {len(malformed)} malformed local "
            f"registration(s): {names}; use 'modelctl delete-local NAME' "
            "to remove them"
        )
    return (
        f"warning: skipped {len(malformed)} malformed active reference(s); "
        "run 'modelctl doctor' for details"
    )


def _confirm_root_delete(root: Path, plan: DeleteResult, *, yes: bool) -> None:
    """Require an explicit typed confirmation that the NAS root store (not the
    local Hugging Face cache) is about to be deleted from."""
    if yes:
        return
    if not sys.stdin.isatty():
        raise ModelctlError(
            "deleting from the managed root store requires --yes when stdin is "
            "not interactive"
        )
    print(
        f"This permanently deletes model '{plan.name}' from the MANAGED ROOT "
        f"STORE ({root}) on the NAS: its active reference, journals, staging "
        "data, and published model objects on disk."
    )
    print(
        "This does NOT touch your local Hugging Face cache (use 'modelctl "
        "delete-local' for that). The deleted data cannot be recovered except "
        "by re-downloading."
    )
    answer = input("Type 'yes' to confirm: ")
    if answer.strip().lower() != "yes":
        raise ModelctlError("deletion aborted; the model store was not modified")


def _print_delete_result(root: Path, result: DeleteResult, *, apply: bool) -> None:
    action = "removed" if apply else "would remove"
    mode = "apply" if apply else "dry-run"
    print(f"delete ({mode}): {result.name} ({result.repo})")
    print(f"  reference: {result.reference} [{action}]")
    for path in result.journals:
        print(f"  journal: {path} [{action}]")
    staging_base = root / ".staging"
    models_base = root / "models"
    for item in result.removed_staging:
        print(
            f"  [{action}] staging {item.path.relative_to(staging_base)} "
            f"({_format_size(item.bytes)})"
        )
    for item in result.retained_staging:
        print(
            f"  [retained] staging {item.path.relative_to(staging_base)} "
            f"({item.status}; {item.detail})"
        )
    for item in result.removed_objects:
        print(
            f"  [{action}] object {item.path.relative_to(models_base)} "
            f"({_format_size(item.bytes)})"
        )
    for item in result.retained_objects:
        print(
            f"  [retained] object {item.path.relative_to(models_base)} "
            f"({item.status}; {item.detail})"
        )
    for path in result.pruned_dirs:
        print(f"  [{action}] empty directory {path.relative_to(root)}")
    total = sum(item.bytes for item in result.removed_staging) + sum(
        item.bytes for item in result.removed_objects
    )
    print(
        f"delete: {len(result.removed_staging)} staging path(s), "
        f"{len(result.removed_objects)} object(s), {_format_size(total)} {action}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modelctl",
        description=(
            "Atomically download, validate, publish, and synchronize Hugging Face "
            "models for NAS-backed inference hosts."
        ),
        formatter_class=HELP_FORMATTER,
        epilog="""examples:
  modelctl download Qwen/Qwen3-8B --root /mnt/nas/llm-models
  modelctl download OWNER/MODEL-GGUF --quantization Q4_K_M --root /mnt/nas/llm-models
  modelctl queue downloads.yaml --jobs 2 --root /mnt/nas/llm-models
  modelctl sync-cards --root /mnt/nas/llm-models
  modelctl path qwen3-8b --root /mnt/nas/llm-models
  modelctl serve-command qwen3-8b --root /mnt/nas/llm-models
  modelctl sync-local qwen3-8b --source-root /mnt/nas/llm-models --cache-dir ~/.cache/huggingface/hub
  modelctl push qwen3-8b --host node-b

Use 'modelctl COMMAND --help' for command-specific examples.
Use 'modelctl config set-root PATH' to save the NAS root once.""",
    )
    parser.add_argument(
        "--version", action="version", version=f"modelctl {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    config = commands.add_parser(
        "config",
        help="save or display modelctl configuration",
        description="Save the default model store root or display its current value.",
        formatter_class=HELP_FORMATTER,
        epilog="""examples:
  modelctl config set-root /mnt/nas/llm-models
  modelctl config get-root
  # Deprecated fallback; prefer HF_HUB_CACHE or --cache-dir:
  modelctl config set-local-root /srv/huggingface/hub
  modelctl config get-local-root

An explicit --root takes precedence over MODELCTL_ROOT, which takes precedence
over the saved NAS root. Local cache commands prefer --cache-dir,
HF_HUB_CACHE, and HF_HOME/hub. MODELCTL_LOCAL_ROOT and the saved local root are
deprecated compatibility fallbacks.""",
    )
    config_commands = config.add_subparsers(dest="config_command", required=True)
    set_root = config_commands.add_parser("set-root", help="save the default model root")
    set_root.add_argument("path", metavar="PATH")
    config_commands.add_parser("get-root", help="print the effective default model root")
    set_local_root = config_commands.add_parser(
        "set-local-root", help="save a deprecated fallback Hugging Face cache path"
    )
    set_local_root.add_argument("path", metavar="PATH")
    config_commands.add_parser(
        "get-local-root", help="print the effective local Hugging Face cache path"
    )

    list_models = commands.add_parser(
        "list",
        help="list active NAS models or local HF-cache registrations",
        description=(
            "Validate and list active models. The configured model store is used "
            "by default; --local selects modelctl registrations in the HF cache."
        ),
        formatter_class=HELP_FORMATTER,
        epilog="""examples:
  modelctl list
  modelctl list --local
  modelctl list --root /srv/models
  modelctl list --json

Only active, validated models are listed. Published objects that are not active
and incomplete staging downloads are excluded.""",
    )
    list_location = list_models.add_mutually_exclusive_group()
    list_location.add_argument(
        "--local",
        action="store_true",
        help="list modelctl registrations in the local Hugging Face cache",
    )
    list_location.add_argument("--root", metavar="PATH", help="model store root")
    list_models.add_argument(
        "--cache-dir", metavar="PATH", help="Hugging Face cache used with --local"
    )
    list_models.add_argument("--json", action="store_true", help="emit JSON")

    catalog = commands.add_parser(
        "catalog",
        help="inspect or refresh the generated active-model catalog",
        description=(
            "Manage ROOT/catalog.json, a derived catalog whose models array mirrors "
            "modelctl list --json. Active references remain authoritative."
        ),
    )
    catalog_commands = catalog.add_subparsers(dest="catalog_command", required=True)
    catalog_status_command = catalog_commands.add_parser(
        "status", help="check catalog schema, dirty state, and active fingerprint"
    )
    catalog_status_command.add_argument(
        "--json", action="store_true", help="emit JSON"
    )
    _add_root(catalog_status_command)
    catalog_refresh_command = catalog_commands.add_parser(
        "refresh", help="rebuild the catalog from validated active references"
    )
    catalog_refresh_command.add_argument(
        "--json", action="store_true", help="emit JSON"
    )
    _add_root(catalog_refresh_command)
    catalog_path_command = catalog_commands.add_parser(
        "path", help="print the generated catalog path"
    )
    _add_root(catalog_path_command)

    doctor = commands.add_parser(
        "doctor",
        help="audit active NAS references and canonical objects",
        description=(
            "Classify every active-store entry without following unsafe paths or "
            "modifying the model store."
        ),
        formatter_class=HELP_FORMATTER,
        epilog="""examples:
  modelctl doctor --root /mnt/nas/llm-models
  modelctl doctor --root /mnt/nas/llm-models --json

Use repair-active for regular directories that doctor marks repairable.""",
    )
    doctor.add_argument("--json", action="store_true", help="emit JSON")
    _add_root(doctor)

    repair = commands.add_parser(
        "repair-active",
        help="replace validated active directory copies with symlinks",
        description=(
            "Cross-check the manifest, update journal, active-directory metadata, "
            "and canonical object before quarantining a directory and publishing "
            "the required symlink. Dry-run is the default."
        ),
        formatter_class=HELP_FORMATTER,
        epilog="""examples:
  modelctl repair-active --root /mnt/nas/llm-models
  modelctl repair-active MODEL_NAME --root /mnt/nas/llm-models --apply

Quarantined directories are retained until cleanup-quarantine is explicitly run.""",
    )
    repair.add_argument("names", nargs="*", metavar="NAME")
    repair.add_argument(
        "--apply", action="store_true", help="apply repairs (default: dry-run)"
    )
    repair.add_argument("--json", action="store_true", help="emit JSON")
    _add_root(repair)

    cleanup = commands.add_parser(
        "cleanup-quarantine",
        help="remove validated quarantined active-directory copies",
        description=(
            "Verify the repaired active symlink and every quarantined copy before "
            "optionally deleting quarantine data. Dry-run is the default."
        ),
        formatter_class=HELP_FORMATTER,
        epilog="""examples:
  modelctl cleanup-quarantine MODEL_NAME --root /mnt/nas/llm-models
  modelctl cleanup-quarantine MODEL_NAME --root /mnt/nas/llm-models --apply""",
    )
    cleanup.add_argument("name", metavar="NAME")
    cleanup.add_argument(
        "--apply", action="store_true", help="delete validated copies"
    )
    cleanup.add_argument("--json", action="store_true", help="emit JSON")
    _add_root(cleanup)

    staging_audit = commands.add_parser(
        "staging-audit",
        help="classify unpublished and resumable staging data",
        description=(
            "Report staging objects as resumable, failed, published duplicates, "
            "journal-referenced, or orphaned without modifying the store."
        ),
    )
    staging_audit.add_argument("--json", action="store_true", help="emit JSON")
    _add_root(staging_audit)

    staging_cleanup = commands.add_parser(
        "cleanup-staging",
        help="remove explicitly selected audited staging objects",
        description=(
            "Re-audit explicitly selected paths and remove only failed, published-"
            "duplicate, or orphaned staging objects. Dry-run is the default."
        ),
    )
    staging_cleanup.add_argument("paths", nargs="+", metavar="OWNER/REPO/OBJECT")
    staging_cleanup.add_argument(
        "--apply", action="store_true", help="delete selected staging data"
    )
    staging_cleanup.add_argument("--json", action="store_true", help="emit JSON")
    _add_root(staging_cleanup)

    objects_audit = commands.add_parser(
        "objects-audit",
        help="classify active and unreferenced immutable objects",
        description="Validate published objects and report active reachability.",
    )
    objects_audit.add_argument("--json", action="store_true", help="emit JSON")
    _add_root(objects_audit)

    object_cleanup = commands.add_parser(
        "gc-objects",
        help="remove explicitly selected unreferenced immutable objects",
        description=(
            "Revalidate explicitly selected objects and prove they are not active. "
            "Dry-run is the default."
        ),
    )
    object_cleanup.add_argument("paths", nargs="+", metavar="OWNER/REPO/OBJECT")
    object_cleanup.add_argument(
        "--apply", action="store_true", help="delete selected unreferenced objects"
    )
    object_cleanup.add_argument("--json", action="store_true", help="emit JSON")
    _add_root(object_cleanup)

    delete = commands.add_parser(
        "delete-local",
        help="unregister a model from the local Hugging Face cache",
        description=(
            "Remove modelctl local registration state without deleting shared "
            "Hugging Face cache data or modifying the NAS model."
        ),
        formatter_class=HELP_FORMATTER,
        epilog="""examples:
  modelctl delete-local qwen3-8b-vllm
  modelctl delete-local model-q4 --cache-dir /srv/huggingface/hub

Cache snapshots, refs, and blobs are retained. Use hf cache rm or hf cache prune
explicitly when shared cache data should be removed.""",
    )
    delete.add_argument("name")
    _add_local_root(delete)

    store_delete = commands.add_parser(
        "delete",
        help="delete a model from the managed root store (NAS)",
        description=(
            "Permanently remove a model from the managed model store: its "
            "active reference, journals, eligible staging data, and "
            "now-unreferenced published objects. Dry-run is the default. "
            "This never touches the local Hugging Face cache; use delete-local "
            "to unregister a local cache model instead."
        ),
        formatter_class=HELP_FORMATTER,
        epilog="""examples:
  modelctl delete qwen3-8b-vllm --root /mnt/nas/llm-models
  modelctl delete qwen3-8b-vllm --apply
  modelctl delete qwen3-8b-vllm --apply --yes

With --apply, modelctl prints what will be deleted from the MANAGED ROOT
STORE on the NAS and requires typing 'yes' unless --yes is supplied (use
--yes for scripts and other non-interactive runs). Objects still referenced
by another active model and live staging data are retained.""",
    )
    store_delete.add_argument("name", metavar="NAME")
    store_delete.add_argument(
        "--apply", action="store_true", help="delete the model (default: dry-run)"
    )
    store_delete.add_argument(
        "--yes",
        action="store_true",
        help="skip the typed confirmation (required when stdin is not interactive)",
    )
    store_delete.add_argument("--json", action="store_true", help="emit JSON")
    _add_root(store_delete)

    manifest = commands.add_parser(
        "manifest",
        aliases=["generate-manifest"],
        help="generate a manifest from a Hugging Face model id or URL",
        description=(
            "Query Hugging Face, infer the model format and runtime, and write a "
            "YAML manifest without downloading model weights."
        ),
        formatter_class=HELP_FORMATTER,
        epilog="""examples:
  modelctl manifest Qwen/Qwen3-8B --root /mnt/nas/llm-models
  modelctl manifest https://huggingface.co/Qwen/Qwen3-8B --name qwen3-8b-vllm
  modelctl manifest OWNER/MODEL-GGUF --quantization Q4_K_M
  modelctl manifest OWNER/MODEL --revision v2 --output ./model.yaml

The default output is ROOT/manifests/NAME.yaml. Existing files are not
replaced unless --force is supplied.""",
    )
    manifest.add_argument("source", help="owner/model or huggingface.co model URL")
    manifest.add_argument("--name", help="local model name (default: repository name)")
    manifest.add_argument(
        "--revision", help="branch, tag, or commit (default: URL revision or main)"
    )
    manifest.add_argument(
        "--quantization",
        help="GGUF quantization substring, for example Q4_K_M",
    )
    manifest.add_argument(
        "--runtime",
        choices=["auto", "vllm", "llama.cpp"],
        default="auto",
        help="runtime override (default: auto)",
    )
    manifest.add_argument(
        "--mmproj",
        help="multimodal projector filename or substring when selection is ambiguous",
    )
    manifest.add_argument(
        "--mtp",
        help="MTP draft model filename or substring when selection is ambiguous",
    )
    manifest.add_argument("--output", type=Path, help="output path")
    manifest.add_argument("--force", action="store_true", help="replace an existing manifest")
    _add_root(manifest)

    download = commands.add_parser(
        "download",
        help="generate requirements and download a Hugging Face model",
        description=(
            "Generate the manifest and perform the complete atomic download, "
            "validation, publication, and activation workflow."
        ),
        formatter_class=HELP_FORMATTER,
        epilog="""examples:
  modelctl download Qwen/Qwen3-8B --root /mnt/nas/llm-models
  modelctl download https://huggingface.co/Qwen/Qwen3-8B --name qwen3-8b-vllm
  modelctl download OWNER/MODEL-GGUF --quantization Q4_K_M
  MODELCTL_ROOT=/mnt/nas/llm-models modelctl download OWNER/MODEL --revision v2

Standard weights select vLLM automatically. GGUF-only repositories select
llama.cpp. If multiple GGUF quantizations exist, --quantization is required.""",
    )
    download.add_argument("source", help="owner/model or huggingface.co model URL")
    download.add_argument("--name", help="local model name (default: repository name)")
    download.add_argument("--revision", help="branch, tag, or commit")
    download.add_argument(
        "--quantization",
        help="GGUF quantization substring, for example Q4_K_M",
    )
    download.add_argument(
        "--runtime",
        choices=["auto", "vllm", "llama.cpp"],
        default="auto",
        help="runtime override (default: auto)",
    )
    download.add_argument(
        "--mmproj",
        help="multimodal projector filename or substring when selection is ambiguous",
    )
    download.add_argument(
        "--mtp",
        help="MTP draft model filename or substring when selection is ambiguous",
    )
    download.add_argument(
        "--force",
        action="store_true",
        help="replace a generated manifest with different settings",
    )
    _add_root(download)

    queue = commands.add_parser(
        "queue",
        help="download models from a YAML queue",
        description=(
            "Preflight and download a YAML list of Hugging Face models, with "
            "configurable concurrency. No transfer starts unless every queue "
            "entry and the model store pass validation."
        ),
        formatter_class=HELP_FORMATTER,
        epilog="""queue file (downloads.yaml):
  downloads:
    - source: Qwen/Qwen3-8B
      name: qwen3-8b-vllm
    - source: org/model-GGUF
      name: model-q4
      quantization: Q4_K_M
      runtime: llama.cpp

examples:
  # Sequential (the default):
  modelctl queue downloads.yaml --root /mnt/nas/llm-models

  # Download at most two models at once:
  modelctl queue downloads.yaml --jobs 2 --root /mnt/nas/llm-models

The top-level 'downloads' mapping is required. Each entry requires 'source'.
Optional fields are name, revision, quantization, runtime (auto, vllm, or
llama.cpp), mmproj, mtp, and force (true or false).

Before any transfer, preflight validates the YAML schema, unique effective model
names, existing manifest compatibility, every Hugging Face source and
quantization, and atomic symlink support at the model root. CIFS stores normally
need the 'mfsymlinks' mount option. The queue continues after transfer failures
and exits nonzero if any entry fails.
There is no fixed jobs limit; start with 2 or 3 to avoid contention.""",
    )
    queue.add_argument(
        "file",
        type=Path,
        help="YAML file containing a downloads list",
    )
    queue.add_argument(
        "--jobs",
        type=_positive_int,
        default=1,
        metavar="N",
        help="maximum concurrent downloads (default: 1)",
    )
    _add_root(queue)

    cards = commands.add_parser(
        "sync-cards",
        help="fetch model cards and create local run instructions",
        description=(
            "Backfill sidecar cards for active NAS models. The complete Hugging "
            "Face README is preserved and RUN.md contains verified local modelctl "
            "commands plus relevant upstream usage sections."
        ),
        formatter_class=HELP_FORMATTER,
        epilog="""examples:
  modelctl sync-cards --root /mnt/nas/llm-models
  modelctl sync-cards qwen3-8b model-q4 --root /mnt/nas/llm-models
  modelctl sync-cards qwen3-8b --force --root /mnt/nas/llm-models

Without names, every active model under ROOT/active is inspected. Cards are
published atomically under ROOT/cards/NAME without modifying model objects.
Hugging Face content is fetched at the exact commit stored in object metadata.""",
    )
    cards.add_argument(
        "names",
        nargs="*",
        metavar="NAME",
        help="active model names (default: all active models)",
    )
    cards.add_argument(
        "--force",
        action="store_true",
        help="rebuild cards that already match the active commit",
    )
    _add_root(cards)

    update = commands.add_parser(
        "update",
        help="resolve, download, validate, publish, and activate a model",
        description=(
            "Update a named model using an existing YAML or JSON manifest. The "
            "active reference changes only after validation and publication."
        ),
        formatter_class=HELP_FORMATTER,
        epilog="""examples:
  modelctl update qwen3-8b-vllm --root /mnt/nas/llm-models
  modelctl update model-q4 --manifest ./model-q4.yaml --root /mnt/nas/llm-models
  MODELCTL_ROOT=/mnt/nas/llm-models modelctl update qwen3-8b-vllm

Without --manifest, modelctl reads ROOT/manifests/NAME.yaml, .yml, or .json.""",
    )
    update.add_argument("name")
    update.add_argument("--manifest", type=Path, help="manifest YAML or JSON path")
    _add_root(update)

    path = commands.add_parser(
        "path",
        help="print the resolved active model entrypoint",
        description=(
            "Validate the active object and print its resolved directory or model "
            "file path. No labels or additional text are written to stdout."
        ),
        formatter_class=HELP_FORMATTER,
        epilog="""examples:
  modelctl path qwen3-8b-vllm --root /mnt/nas/llm-models
  MODEL_PATH=$(modelctl path qwen3-8b-vllm --root /mnt/nas/llm-models)
  MODELCTL_ROOT=/mnt/nas/llm-models modelctl path model-q4""",
    )
    path.add_argument("name")
    path.add_argument(
        "--local", action="store_true",
        help="resolve from the local Hugging Face cache",
    )
    path.add_argument("--cache-dir", metavar="PATH")
    _add_root(path)

    serve = commands.add_parser(
        "serve-command",
        help="print a shell-escaped server starter command",
        description=(
            "Print a shell-escaped vLLM or llama.cpp starter command for the "
            "active model. This command never starts the server."
        ),
        formatter_class=HELP_FORMATTER,
        epilog="""examples:
  modelctl serve-command qwen3-8b-vllm --root /mnt/nas/llm-models
  modelctl serve-command model-q4 --root /mnt/nas/llm-models
  sh -c "$(modelctl serve-command qwen3-8b-vllm)"

Review a generated command before evaluating or executing it.""",
    )
    serve.add_argument("name")
    serve.add_argument(
        "--local", action="store_true",
        help="resolve from the local Hugging Face cache",
    )
    serve.add_argument("--cache-dir", metavar="PATH")
    _add_root(serve)

    sync = commands.add_parser(
        "sync-local",
        aliases=["sync"],
        help="copy an active NAS object into the local Hugging Face cache",
        description=(
            "Validate the active NAS object, rsync selected files into staging, "
            "show aggregate transfer progress and speed, publish HF blobs and "
            "a commit snapshot, then register it locally."
        ),
        formatter_class=HELP_FORMATTER,
        epilog="""examples:
  modelctl sync-local unsloth/DeepSeek-V4-Flash-0731 \\
    --source-root /mnt/nas/llm-models --cache-dir ~/.cache/huggingface/hub
  modelctl sync-local qwen3-8b-vllm
  modelctl sync model-q4 --from-root /mnt/nas/llm-models --cache-dir /srv/huggingface/hub

Pass an active model name or its Hugging Face repository id. A repository id
must identify exactly one active model. Selected repository files are published
as HF blobs and snapshot symlinks. Partial selections remain partial snapshots.
Transfer status includes bytes, completion, speed, and ETA; post-transfer ETag
validation and publication phases are also reported. The inference service is
not restarted.""",
    )
    sync.add_argument(
        "name", metavar="MODEL_OR_REPO",
        help="active model name or unique Hugging Face repository id",
    )
    sync.add_argument(
        "--source-root", "--from-root", metavar="PATH", dest="source_root",
        help="NAS source root (default: configured NAS root)",
    )
    sync.add_argument("--rsync", default="rsync", help="rsync executable")
    _add_local_root(sync)

    push = commands.add_parser(
        "push",
        aliases=["sync-remote"],
        help="copy a locally cached model into another host's HF cache over ssh",
        description=(
            "Copy a model from the local Hugging Face cache into another "
            "host's Hugging Face cache over ssh. The model must already be "
            "registered in the local cache (for example via modelctl "
            "sync-local). The command auto-discovers the remote modelctl "
            "binary, preflights the remote cache directory, rsyncs only the "
            "registered snapshot's files, and has receive-cache validate and "
            "atomically publish blobs, a snapshot, refs, and a local "
            "registration on the destination."
        ),
        formatter_class=HELP_FORMATTER,
        epilog="""examples:
  # Register a model in the local cache once, then fan it out over the fabric
  modelctl sync-local incoai/GLM-5.3-Flash-DFlash2 --source-root /mnt/nas/llm-models
  modelctl push incoai/GLM-5.3-Flash-DFlash2 --host 10.100.24.1

  # Minimal: the remote cache directory defaults to this host's cache path
  modelctl push qwen3-8b-vllm --host node-b

  # Custom ssh identity and explicit remote cache directory
  modelctl push model-q4 --host node-b \\
    --port 2222 --identity ~/.ssh/connectx \\
    --cache-dir /srv/huggingface/hub \\
    --remote-modelctl /opt/modelctl/bin/modelctl

  # Parallel rsync streams for many-file models on a fast fabric
  modelctl push glm-5.3-flash-exl3-q4 --host node-b --jobs 4

  # Both CX7 links: distribute streams across both fabric interfaces
  modelctl push glm-5.3-flash-exl3-q4 \\
    --host 10.100.24.1 --host 10.100.25.1 --jobs 8

  # Alias
  modelctl sync-remote qwen3-8b-vllm --host node-b

How it works:
  1. Resolve an active model name (or a unique Hugging Face repository id) in
     the local Hugging Face cache; the model must already be registered there
     (modelctl sync-local or a prior push).
  2. Discover the remote modelctl binary, then run receive-cache --probe over
     ssh for a JSON handshake: protocol version and canonical cache path.
  3. Preflight the remote cache directory (create it and prove it is
     writable), then rsync only the registered snapshot's files plus metadata
     over 'ssh -o Compression=no' into deterministic remote staging.
  4. Run receive-cache over ssh: the remote re-derives the staging path from
     the transferred metadata, validates every file against its retained
     Hugging Face ETag, and atomically publishes blobs, a commit snapshot,
     refs, and a local registration.

ssh and rsync are required on this host. The remote needs modelctl (a plain
'uv tool install' is enough; push auto-discovers ~/.local/bin/modelctl,
/usr/local/bin/modelctl, or /usr/bin/modelctl, or use --remote-modelctl) and
the rsync package. The remote cache directory defaults to the same path this
host uses, so two identical nodes need only --host. Point --host at a fabric
interface (for example the ConnectX-7 IP) when the hostname resolves to a
slower path. Interrupted transfers remain resumable in remote staging and a
rerun resumes them. --jobs N runs up to N concurrent rsync streams; repeating
--host distributes those streams round-robin over several fabric interfaces
of the same destination. When a node exposes multiple links (for example two
ConnectX-7 interfaces), use one --host per link to double aggregate
throughput.""",
    )
    push.add_argument(
        "name", metavar="MODEL_OR_REPO",
        help="registered cache model name or unique Hugging Face repository id",
    )
    push.add_argument(
        "--host",
        action="append",
        required=True,
        metavar="HOST",
        help=(
            "ssh destination (repeatable): the first runs the handshake and "
            "activation, all hosts share the transfer streams"
        ),
    )
    push.add_argument("--port", type=_positive_int, help="ssh port")
    push.add_argument("--identity", metavar="KEY", help="ssh identity file")
    push.add_argument("--ssh", default="ssh", help="ssh executable (default: ssh)")
    push.add_argument("--rsync", default="rsync", help="rsync executable (default: rsync)")
    push.add_argument(
        "--jobs",
        type=_positive_int,
        default=1,
        metavar="N",
        help="maximum concurrent rsync streams (default: 1; raise for many-file models)",
    )
    push.add_argument(
        "--remote-modelctl", metavar="PATH",
        help="remote modelctl executable (default: auto-discovered)",
    )
    push.add_argument(
        "--cache-dir", metavar="PATH",
        help="remote Hugging Face cache directory (default: this host's default cache path)",
    )

    receive = commands.add_parser(
        "receive-cache",
        help="validate and publish a staged cache transfer on this host",
        description=(
            "The remote half of 'modelctl push'. --probe prints a JSON "
            "handshake; otherwise the staged rsync data is validated and "
            "published as blobs, a snapshot, refs, and a registration record. "
            "Normally invoked over ssh by push."
        ),
        formatter_class=HELP_FORMATTER,
        epilog="""examples:
  # Read-only handshake: prints protocol version and canonical cache path
  modelctl receive-cache --probe qwen3-8b-vllm

  # Commit a staged transfer that rsync left in this host's cache staging
  modelctl receive-cache qwen3-8b-vllm --staging /cache/models--Qwen--Qwen3-8B/.modelctl-staging/COMMIT--SELECTION

How it works:
  --probe prints {"proto": 1, "cache": ...} and mutates nothing.
  Otherwise the staged directory's own metadata is validated first; the
  staging path is re-derived from that metadata and must match --staging
  (misplaced or tampered transfers are rejected). Every selected file is then
  verified against its retained Hugging Face ETag before blobs, a commit
  snapshot, refs, and a modelctl registration are published atomically, with
  the journal closed as READY_FOR_SERVICE_RESTART last.

Normally invoked over ssh by 'modelctl push'; useful standalone for manual
inspection or reviewing a manually rsynced staging directory.""",
    )
    receive.add_argument("name", metavar="NAME")
    receive.add_argument(
        "--cache-dir", metavar="PATH",
        help="Hugging Face cache directory (default: HF default)",
    )
    receive.add_argument(
        "--probe", action="store_true",
        help="print the JSON handshake without mutating anything",
    )
    receive.add_argument(
        "--staging", metavar="PATH",
        help="staged transfer directory placed by rsync (provided by push)",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "config":
        if args.config_command in {"set-root", "set-local-root"}:
            root = Path(args.path).expanduser().absolute()
            local = args.config_command == "set-local-root"
            saved = save_local_root(root) if local else save_root(root)
            print(f"{'local_root' if local else 'root'}: {root}")
            print(f"config: {saved}")
        elif args.config_command == "get-local-root":
            print(_cache_dir(None))
        else:
            print(_root(None))
        return 0

    if args.command == "list":
        if args.cache_dir and not args.local:
            raise ModelctlError("--cache-dir requires --local")
        selected = _cache_dir(args.cache_dir) if args.local else _root(args.root)
        _print_models(selected, json_output=args.json, local=args.local)
        return 0

    if args.command == "delete-local":
        validate_name(args.name)
        removed = delete_cached(_selected_cache(args), args.name)
        print(f"unregistered: {removed} (cache data retained)")
        return 0

    if args.command in {"sync-local", "sync"}:
        result = sync_local(
            _root(args.source_root),
            _selected_cache(args),
            args.name,
            rsync=args.rsync,
            progress=lambda message: print(message, flush=True),
        )
        print(result)
        return 0

    if args.command == "push":
        hosts = args.host
        result = push_model(
            _cache_dir(args.cache_dir),
            args.name,
            host=hosts[0],
            job_hosts=hosts[1:] or None,
            port=args.port,
            identity=args.identity,
            ssh=args.ssh,
            rsync=args.rsync,
            remote_modelctl=args.remote_modelctl,
            jobs=args.jobs,
            progress=lambda message: print(message, flush=True),
        )
        print(result)
        return 0

    if args.command == "receive-cache":
        result = receive_staged_cache(
            _cache_dir(args.cache_dir),
            args.name,
            probe=args.probe,
            staging=Path(args.staging) if args.staging else None,
        )
        print(json.dumps(result, indent=2))
        return 0

    root = _root(args.root)
    if args.command == "catalog":
        if args.catalog_command == "path":
            print(catalog_path(root))
            return 0
        if args.catalog_command == "status":
            status = catalog_status(root)
            if args.json:
                print(json.dumps(status.to_dict(), indent=2))
            else:
                detail = f" ({status.detail})" if status.detail else ""
                print(f"[{status.status}] {status.path}{detail}")
                if status.generation is not None:
                    print(
                        f"generation: {status.generation}; models: {status.models}"
                    )
            return 0 if status.status == "ready" else 1
        _, result = refresh_catalog(root, list_active_models)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            action = "updated" if result.changed else "unchanged"
            print(
                f"catalog: {action}; generation {result.generation}; "
                f"{len(result.models)} model(s); {result.path}"
            )
        return 0

    if args.command == "doctor":
        results = audit_active_references(root)
        if args.json:
            print(json.dumps([item.to_dict() for item in results], indent=2))
        else:
            for item in results:
                target = f" -> {item.object}" if item.object is not None else ""
                detail = f" ({item.detail})" if item.detail else ""
                print(f"[{item.status}] {item.name}{target}{detail}")
            warnings = [item for item in results if item.status == "missing_journal"]
            malformed = [
                item
                for item in results
                if item.status
                not in {"valid", "hidden_foreign_entry", "missing_journal"}
            ]
            healthy = len(results) - len(warnings) - len(malformed)
            print(
                f"doctor: {healthy} healthy/ignored, {len(warnings)} warning(s), "
                f"{len(malformed)} malformed"
            )
        return 1 if any(
            item.status not in {"valid", "hidden_foreign_entry"}
            for item in results
        ) else 0

    if args.command == "repair-active":
        names = list(args.names)
        if not names:
            names = [
                item.name
                for item in audit_active_references(root)
                if item.status == "repairable_directory"
            ]
        results = [
            repair_active_reference(root, name, apply=args.apply) for name in names
        ]
        if args.json:
            print(json.dumps([item.to_dict() for item in results], indent=2))
        else:
            for item in results:
                suffix = f"; quarantine: {item.quarantine}" if item.quarantine else ""
                print(
                    f"[{item.status}] {item.name}: {item.reference} -> "
                    f"{item.object}{suffix}"
                )
        return 0

    if args.command == "cleanup-quarantine":
        paths = cleanup_quarantine(root, args.name, apply=args.apply)
        if args.json:
            print(json.dumps([str(path) for path in paths], indent=2))
        else:
            action = "removed" if args.apply else "would remove"
            for path in paths:
                print(f"[{action}] {path}")
            print(f"cleanup: {len(paths)} quarantine path(s)")
        return 0

    if args.command in {"staging-audit", "objects-audit"}:
        results = (
            audit_staging(root)
            if args.command == "staging-audit"
            else audit_objects(root)
        )
        if args.json:
            print(json.dumps([item.to_dict() for item in results], indent=2))
        else:
            base = root / (
                ".staging" if args.command == "staging-audit" else "models"
            )
            for item in results:
                relative = item.path.relative_to(base)
                name = f"; name: {item.name}" if item.name is not None else ""
                detail = f"; {item.detail}" if item.detail else ""
                print(
                    f"[{item.status}] {relative} "
                    f"({_format_size(item.bytes)}{name}{detail})"
                )
            print(
                f"audit: {len(results)} path(s), "
                f"{_format_size(sum(item.bytes for item in results))}"
            )
        return 0

    if args.command in {"cleanup-staging", "gc-objects"}:
        results = (
            cleanup_staging(root, args.paths, apply=args.apply)
            if args.command == "cleanup-staging"
            else cleanup_objects(root, args.paths, apply=args.apply)
        )
        if args.json:
            print(json.dumps([item.to_dict() for item in results], indent=2))
        else:
            base = root / (
                ".staging" if args.command == "cleanup-staging" else "models"
            )
            action = "removed" if args.apply else "would remove"
            for item in results:
                print(
                    f"[{action}] {item.path.relative_to(base)} "
                    f"({_format_size(item.bytes)})"
                )
            print(
                f"cleanup: {len(results)} path(s), "
                f"{_format_size(sum(item.bytes for item in results))}"
            )
        return 0

    if args.command == "delete":
        plan = delete_model(root, args.name, apply=False)
        if args.apply:
            _confirm_root_delete(root, plan, yes=args.yes)
            result = delete_model(root, args.name, apply=True)
        else:
            result = plan
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            _print_delete_result(root, result, apply=args.apply)
        return 0

    if args.command == "download":
        validate_queue_root(root)
        manifest_path, active = download_from_hf(
            root,
            args.source,
            name=args.name,
            revision=args.revision,
            quantization=args.quantization,
            runtime=args.runtime,
            mmproj=args.mmproj,
            mtp=args.mtp,
            force_manifest=args.force,
        )
        print(f"manifest: {manifest_path}")
        print(f"active: {active}")
        return 0

    if args.command == "sync-cards":
        results = sync_model_cards(root, args.names, force=args.force)
        counts = {status: 0 for status in ("updated", "unchanged", "unavailable", "failed")}
        for result in results:
            counts[result.status] += 1
            detail = str(result.path) if result.path else result.message or ""
            print(f"[{result.status}] {result.name}: {detail}")
        print(
            "cards: "
            f"{counts['updated']} updated, {counts['unchanged']} unchanged, "
            f"{counts['unavailable']} unavailable, {counts['failed']} failed"
        )
        if counts["failed"]:
            raise ModelctlError(f"{counts['failed']} model card operation(s) failed")
        return 0

    if args.command == "queue":
        requests = load_download_queue(args.file)
        concurrency = min(args.jobs, len(requests))
        print(f"preflight: validating store and {len(requests)} queue entries")
        validate_queue_root(root)
        prepared = prepare_download_queue(requests, jobs=args.jobs)
        validate_prepared_manifests(root, prepared)
        for index, item in enumerate(prepared, start=1):
            selection = item.request.quantization or item.document.get(
                "format", "auto"
            )
            print(
                f"[{index}/{len(prepared)}] ready: {item.name} "
                f"<- {item.request.source} ({selection})"
            )
        print(
            f"preflight complete: starting {len(prepared)} download(s), "
            f"up to {concurrency} concurrent"
        )
        results = execute_download_queue(root, prepared, jobs=args.jobs)
        failures = 0
        for index, result in enumerate(results, start=1):
            label = prepared[index - 1].name
            if result.succeeded:
                print(
                    f"[{index}/{len(results)}] complete: "
                    f"{label} -> {result.active_path}"
                )
            else:
                failures += 1
                error = result.error
                print(
                    f"[{index}/{len(results)}] failed: {label}: "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                )
        if failures:
            raise DownloadQueueError(
                f"{failures} of {len(results)} queued download(s) failed"
            )
        return 0

    if args.command in {"manifest", "generate-manifest"}:
        document = generate_manifest_document(
            args.source,
            name=args.name,
            revision=args.revision,
            quantization=args.quantization,
            runtime=args.runtime,
            mmproj=args.mmproj,
            mtp=args.mtp,
        )
        print(
            write_generated_manifest(
                root, document, output=args.output, force=args.force
            )
        )
        return 0

    validate_name(args.name)
    if args.command == "update":
        manifest = load_manifest(root, args.name, args.manifest)
        result = update_model(root, manifest)
        print(result)
    elif args.command == "path":
        if args.local:
            if args.root:
                raise ModelctlError("--root cannot be combined with --local; use --cache-dir")
            print(local_active_entrypoint(_cache_dir(args.cache_dir), args.name))
        else:
            if args.cache_dir:
                raise ModelctlError("--cache-dir requires --local")
            print(active_entrypoint(root, args.name))
    elif args.command == "serve-command":
        if args.local:
            if args.root:
                raise ModelctlError("--root cannot be combined with --local; use --cache-dir")
            print(serve_cached_command(_cache_dir(args.cache_dir), args.name))
        else:
            if args.cache_dir:
                raise ModelctlError("--cache-dir requires --local")
            print(serve_command(root, args.name))
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        print("modelctl: interrupted", file=sys.stderr)
        raise SystemExit(130)
    except ModelctlError as exc:
        print(f"modelctl: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"modelctl: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
