# modelctl

`modelctl` atomically downloads Hugging Face model snapshots to a NAS model
store, activates validated revisions, synchronizes active objects to local
storage with resumable `rsync`, and emits service starter commands without
starting a server.

## Install

Python 3.11 or newer is required. `rsync` is also required when using
`sync-local`. The recommended installation method is an isolated uv tool.

From the project folder:

```bash
cd ~/developer/modelctl
uv tool install .
```

Verify that the installed command works:

```bash
modelctl --version
modelctl --help
```

If the shell cannot find `modelctl`, add uv's executable directory to `PATH`
and restart the shell:

```bash
uv tool update-shell
exec "$SHELL"
```

After changing or updating the source, reinstall it with:

```bash
cd ~/developer/modelctl
uv tool install --force .
```

To run directly from the checkout without installing:

```bash
uv run modelctl --help
uv run modelctl download Qwen/Qwen3-8B --root /mnt/nas/llm-models
```

A regular pip installation is also supported:

```bash
python3 -m pip install .
```

To remove the uv tool installation:

```bash
uv tool uninstall modelctl
```

The `huggingface_hub` dependency installs `hf_xet`; its documented environment
variables can be used to tune transfers.

## Quick start: download directly to the NAS

Mount the NAS, choose the managed store root, and give `download` a Hugging
Face model id or URL:

```bash
export MODELCTL_ROOT=/mnt/nas/llm-models

modelctl download Qwen/Qwen3-8B
# URLs are accepted too:
modelctl download https://huggingface.co/Qwen/Qwen3-8B
```

When running from this source checkout instead of an installed package, use
`uv run modelctl` in place of `modelctl`.

`download` performs the full workflow:

1. Queries Hugging Face and resolves the requested revision.
2. Infers vLLM for standard weights or llama.cpp for a GGUF-only repository.
3. Generates `ROOT/manifests/NAME.yaml`.
4. Estimates and downloads the selected files to resumable staging.
5. Validates and atomically publishes the object.
6. Atomically updates `ROOT/active/NAME` only after publication succeeds.

The default name is the lower-cased Hugging Face repository name. Override it
when desired:

```bash
modelctl download Qwen/Qwen3-8B --name qwen3-8b-vllm
```

If a GGUF repository offers several independent quantizations, `modelctl` will
not guess. Select one explicitly:

```bash
modelctl download OWNER/MODEL-GGUF \
  --name model-q4 \
  --quantization Q4_K_M
```

Verify the result or generate a service command without starting the server:

```bash
modelctl path qwen3-8b-vllm
modelctl serve-command qwen3-8b-vllm
```

## Store layout

```text
ROOT/
  manifests/NAME.yaml
  models/ORG/REPO/COMMIT--SELECTION/   # published objects
  active/NAME                          # atomically replaced symlink
  .staging/ORG/REPO/COMMIT--SELECTION/ # resumable, unpublished data
  state/NAME.json                      # state transition journal
```

Staging and published objects are under one root so publication can use a
same-filesystem rename. The active symlink is changed only after object
validation and publication succeed.

## Manifest

```yaml
name: qwen3-8b-vllm
repo: Qwen/Qwen3-8B
revision: main
format: safetensors
include:
  - "*.json"
  - "*.safetensors"
  - "tokenizer*"
runtime:
  type: vllm
  executable: vllm
  args: ["--tensor-parallel-size", "1"]
```

A GGUF manifest must select one quantization. Split quantizations may use one
pattern that selects every shard; the first shard becomes the entrypoint.

```yaml
name: model-q4
repo: org/model-GGUF
revision: main
format: gguf
include: ["Model-Q4_K_M-*.gguf"]
runtime:
  type: llama.cpp
  executable: llama-server
  args: ["--ctx-size", "8192"]
```

`entrypoint` can select a relative GGUF path explicitly. A vLLM+GGUF profile is
rejected in the version 1 workflow. Runtime arguments may contain `{path}`,
`{object}`, and `{name}` placeholders; when `{path}` or `{object}` is present,
the argument list replaces the runtime's default model arguments.

A manifest file may also contain a `models` mapping keyed by model name.

## Commands

### One-step download from Hugging Face

`download` generates the manifest, resolves the revision, downloads the selected
files, validates and publishes the object, and activates it in one command:

```bash
modelctl download Qwen/Qwen3-8B \
  --name qwen3-8b-vllm \
  --root /mnt/nas/llm-models
```

A full Hugging Face URL works as well:

```bash
modelctl download https://huggingface.co/Qwen/Qwen3-8B \
  --root /mnt/nas/llm-models
```

For a GGUF-only repository, the command automatically uses llama.cpp. If the
repository contains multiple independent quantizations, specify which one to
use rather than allowing `modelctl` to guess:

```bash
modelctl download https://huggingface.co/org/model-GGUF \
  --name model-q4 \
  --quantization Q4_K_M \
  --root /mnt/nas/llm-models
```

The inferred local name is the lower-cased repository name. Optional
`--revision` and `--runtime` arguments override automatic selection. The
manifest is retained at `ROOT/manifests/NAME.yaml`; repeating the same command
reuses it and any valid published object. If an existing manifest has different
settings, the command stops unless `--force` is passed.

### Queue multiple downloads

A queue is a YAML mapping with exactly one top-level field, `downloads`.
`downloads` must be a non-empty list of mappings:

```yaml
# downloads.yaml
downloads:
  - source: Qwen/Qwen3-8B
    name: qwen3-8b-vllm
    revision: main

  # Repeated repositories require a unique name for each quantization.
  - source: https://huggingface.co/org/model-GGUF
    name: model-q4
    quantization: Q4_K_M
    runtime: llama.cpp
    force: false

  - source: https://huggingface.co/org/model-GGUF
    name: model-q8
    quantization: Q8_0
    runtime: llama.cpp

  # Auxiliary MTP and DFlash models also need their own names.
  - source: unsloth/gemma-4-31B-it-GGUF
    name: gemma-4-31b-it-mtp-q8-0
    quantization: mtp-gemma-4-31B-it-Q8_0.gguf
    runtime: llama.cpp

  - source: z-lab/gemma-4-26B-A4B-it-DFlash
    name: gemma-4-26b-a4b-it-dflash
    runtime: vllm
```

A bare YAML list is not accepted. Unknown top-level or entry fields are rejected
to catch spelling mistakes before a transfer starts.

| Field | Required | Meaning |
| --- | --- | --- |
| `source` | yes | Hugging Face `owner/model` id or model URL |
| `name` | no | Local model name; defaults to the repository name |
| `revision` | no | Branch, tag, or commit; defaults to `main` |
| `quantization` | no | GGUF quantization substring or filename |
| `runtime` | no | `auto`, `vllm`, or `llama.cpp`; defaults to `auto` |
| `force` | no | Boolean; replace a generated manifest with different settings |

Every queue run performs a complete preflight before starting any model
transfer:

1. Validates the YAML structure, field names, field types, and allowed values.
2. Resolves each effective local model name and rejects duplicates. Give every
   quantization, MTP draft, and DFlash draft a unique `name`.
3. Queries Hugging Face for every entry and validates its source, revision,
   runtime, selected files, and GGUF quantization.
4. Rejects existing manifests with conflicting settings unless that entry sets
   `force: true`.
5. Prepares the model root and verifies that it supports the atomic symlinks
   required by `ROOT/active/NAME`.
6. Prints the fully validated download plan.

If any preflight check fails, the command exits before writing manifests or
downloading model data. CIFS mounts commonly require the `mfsymlinks` mount
option; without it, the root validation fails with an actionable error rather
than failing after a large download.

Run the queue sequentially (the default):

```bash
modelctl queue downloads.yaml --root /mnt/nas/llm-models
```

Download at most two models concurrently:

```bash
modelctl queue downloads.yaml \
  --jobs 2 \
  --root /mnt/nas/llm-models
```

When running from a source checkout, use:

```bash
uv run --project ~/developer/modelctl \
  modelctl queue ~/developer/downloads.yaml --jobs 2 --root /mnt/nas/llm-models
```

Execution behavior after preflight:

- `--jobs` must be at least 1 and defaults to 1.
- The effective concurrency is the smaller of `--jobs` and the queue length.
- There is no fixed concurrency ceiling. Start with 2; NAS throughput, disk
  space, network bandwidth, and Hugging Face throttling determine whether a
  larger value helps.
- The queue continues after a transfer fails, reports every result, and exits
  unsuccessfully if any entry failed.
- Interrupted entries retain resumable staging data. Rerunning the queue reuses
  valid published objects and resumes incomplete downloads.

### Generate a manifest from Hugging Face

Generate a manifest directly from a Hugging Face model id or URL:

```bash
modelctl manifest Qwen/Qwen3-8B \
  --name qwen3-8b-vllm \
  --root /mnt/nas/llm-models

modelctl manifest https://huggingface.co/org/model-GGUF \
  --name model-q4 \
  --quantization Q4_K_M \
  --root /mnt/nas/llm-models
```

When running from a source checkout rather than an installed package, prefix
commands with `uv run`, for example `uv run modelctl manifest ...`.

The generator queries Hugging Face for the selected revision and inspects the
repository filenames. Standard safetensors/bin weights select vLLM; a
GGUF-only repository selects llama.cpp. When multiple GGUF quantizations are
available, generation stops and requires `--quantization`. For sharded GGUFs,
every shard is selected and shard `00001` is used as the entrypoint.

The generated file is written to `ROOT/manifests/NAME.yaml`. The default name
is the lower-cased repository name; use `--name` to override it. Existing
manifests are preserved unless `--force` is passed.

Useful options:

```text
--name NAME                 Set the local model name
--revision REVISION         Select a branch, tag, or commit
--quantization Q4_K_M       Select one GGUF quantization
--runtime auto|vllm|llama.cpp
--output PATH               Write somewhere other than ROOT/manifests
--force                     Replace an existing manifest
```

A complete generate-and-update workflow is:

```bash
export MODELCTL_ROOT=/mnt/nas/llm-models
modelctl manifest Qwen/Qwen3-8B --name qwen3-8b-vllm
modelctl update qwen3-8b-vllm
modelctl path qwen3-8b-vllm
modelctl serve-command qwen3-8b-vllm
```

Download, validate, publish, and activate:

```bash
modelctl update qwen3-8b-vllm --root /mnt/nas/llm-models
```

The update resolves `revision` with `HfApi.model_info`, estimates the selected
snapshot with `snapshot_download(dry_run=True)`, downloads with `local_dir` and
`allow_patterns`, validates expected files, renames the staging directory, and
finally atomically changes `active/NAME`. Interrupted downloads remain in the
same staging directory for Hugging Face metadata-assisted recovery.

Print only the resolved active entrypoint:

```bash
modelctl path qwen3-8b-vllm --root /var/lib/llm-models
```

Print a shell-escaped command without executing it:

```bash
modelctl serve-command qwen3-8b-vllm --root /var/lib/llm-models
```

Synchronize the current NAS object to local storage:

```bash
modelctl sync-local qwen3-8b-vllm \
  --source-root /mnt/nas/llm-models \
  --root /var/lib/llm-models
```

The local copy uses `rsync --archive --delete --partial`, excludes
`.cache/huggingface`, validates in local staging, publishes with rename, and
changes the local reference last. It does not restart or reload an inference
service.

`MODELCTL_ROOT` can provide the default root for all commands.

## Moving existing models to the NAS

### Existing modelctl store

If the models are already managed by `modelctl`, migrate the entire store, not
just `models/`. The manifests, object metadata, active references, and state
files belong together.

First stop any `modelctl update` or `sync-local` operation. An inference server
may continue running from the old store while the copy occurs, but do not
remove the old store until the server has been reconfigured and restarted.

```bash
sudo mkdir -p /mnt/nas/llm-models
sudo chown "$USER":"$USER" /mnt/nas/llm-models

rsync -aH --info=progress2 \
  --exclude='/.locks/' \
  /var/lib/llm-models/ \
  /mnt/nas/llm-models/
```

The trailing slashes are significant. `rsync -a` preserves the relative
`active/NAME` symlinks. Do not add `--delete` for the initial migration unless
you intentionally want destination-only data removed.

Select the NAS store and verify every active model:

```bash
export MODELCTL_ROOT=/mnt/nas/llm-models

for reference in "$MODELCTL_ROOT"/active/*; do
  [ -e "$reference" ] || continue
  modelctl path "$(basename "$reference")"
done
```

Generate updated service commands, change service configuration if it still
uses the old root, and restart the affected inference services:

```bash
modelctl serve-command qwen3-8b-vllm
```

Keep the old store until the services have started successfully from the NAS.

### Models not yet managed by modelctl

For a model downloaded from a known Hugging Face repository, use the one-step
command to create and validate a managed NAS object:

```bash
export MODELCTL_ROOT=/mnt/nas/llm-models
modelctl download OWNER/REPOSITORY --name my-model
```

For GGUF repositories with several variants, add `--quantization`:

```bash
modelctl download OWNER/REPOSITORY-GGUF \
  --name my-model-q4 \
  --quantization Q4_K_M
```

Do not copy arbitrary files directly into `ROOT/models/`. Published object
paths contain commit and selection identities and require `.modelctl.json`
metadata produced during validation. A standalone local directory or GGUF file
that is not recoverable from a known Hugging Face repository cannot currently
be adopted as a managed object; it can be archived elsewhere on the NAS but
will not work with `modelctl path`, `update`, or `serve-command`.

## CLI help

The top-level and every subcommand provide descriptions, option details, and
copyable examples:

```bash
modelctl --help
modelctl download --help
modelctl queue --help
modelctl manifest --help
modelctl update --help
modelctl path --help
modelctl serve-command --help
modelctl sync-local --help
```

From a source checkout, prefix these commands with `uv run`.

## Versioning

`modelctl` follows semantic versioning. While the project is in the `0.x`
series, new functionality increments the minor version and compatible fixes
increment the patch version. The queue and full preflight workflow were added
in `0.2.0`.

`src/modelctl/__init__.py` is the single version source. Hatch reads it when
building the package, and `modelctl --version` imports the same value so package
metadata and CLI output remain synchronized.

## Testing

```bash
uv run --extra test pytest
```
