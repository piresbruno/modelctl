from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
from .manifest import load_manifest, validate_name
from .operations import active_entrypoint, serve_command, sync_local, update_model

DEFAULT_ROOT = "/var/lib/llm-models"
HELP_FORMATTER = argparse.RawDescriptionHelpFormatter


def _root(value: str | None) -> Path:
    return Path(value or os.environ.get("MODELCTL_ROOT", DEFAULT_ROOT)).expanduser().absolute()


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        metavar="PATH",
        help=f"model store root (default: $MODELCTL_ROOT or {DEFAULT_ROOT})",
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
  modelctl path qwen3-8b --root /mnt/nas/llm-models
  modelctl serve-command qwen3-8b --root /mnt/nas/llm-models
  modelctl sync-local qwen3-8b --source-root /mnt/nas/llm-models --root /var/lib/llm-models

Use 'modelctl COMMAND --help' for command-specific examples.
Set MODELCTL_ROOT to omit --root from commands.""",
    )
    parser.add_argument("--version", action="version", version="modelctl 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

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
    manifest.add_argument("--revision", help="branch, tag, or commit (default: URL revision or main)")
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
llama.cpp), and force (true or false).

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
    _add_root(serve)

    sync = commands.add_parser(
        "sync-local",
        aliases=["sync"],
        help="copy an active NAS object to a local model store",
        description=(
            "Validate the active NAS object, rsync it into local staging, publish "
            "it atomically, and update the local active reference."
        ),
        formatter_class=HELP_FORMATTER,
        epilog="""examples:
  modelctl sync-local qwen3-8b-vllm \\
    --source-root /mnt/nas/llm-models \\
    --root /var/lib/llm-models
  modelctl sync model-q4 --from-root /mnt/nas/llm-models --root /srv/models

Hugging Face local_dir cache metadata is excluded from local runtime copies.
The inference service is not restarted or reloaded.""",
    )
    sync.add_argument("name")
    sync.add_argument(
        "--source-root", "--from-root", required=True, metavar="PATH", dest="source_root"
    )
    sync.add_argument("--rsync", default="rsync", help="rsync executable")
    _add_root(sync)
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _root(args.root)
    if args.command == "download":
        manifest_path, active = download_from_hf(
            root,
            args.source,
            name=args.name,
            revision=args.revision,
            quantization=args.quantization,
            runtime=args.runtime,
            force_manifest=args.force,
        )
        print(f"manifest: {manifest_path}")
        print(f"active: {active}")
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
        print(active_entrypoint(root, args.name))
    elif args.command == "serve-command":
        print(serve_command(root, args.name))
    elif args.command in {"sync-local", "sync"}:
        result = sync_local(
            _root(args.source_root), root, args.name, rsync=args.rsync
        )
        print(result)
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
