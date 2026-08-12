# Central Cluster Model Management Plan

## Goal

Manage model placement for a cluster of approximately 2–6 inference nodes from one central `modelctl` installation.

The central server must be able to:

- List models active on the NAS.
- Show which models are present on each node.
- Synchronize a selected NAS model to a selected node.
- Deactivate and remove a model from a node on demand.
- Keep model data transfers direct from the NAS to the destination node instead of passing model bytes through the central server.

Each node has constrained local storage, so the NAS catalog is not synchronized 1:1 to every node.

## Recommended Architecture

Use `modelctl` as a centralized control plane and SSH-triggered `rsync` as the data plane.

```text
NAS ───────────────────────────────► node-local disk
                 model data

Central modelctl ───── SSH ────────► node
                    control only
```

The central server has:

- One `modelctl` installation.
- Read-only access to the NAS model store.
- Read/write management mounts for each node's local model store, used for inventory, metadata validation, locking, and atomic activation.
- SSH access to each node, used to start node-local `rsync` and deletion operations.

Each node has:

- The NAS mounted read-only.
- A writable node-local model store.
- `rsync`, SSH, and standard POSIX utilities.
- No full `modelctl` installation.

This is a suitable simplified solution for 2–6 nodes. It avoids deploying and upgrading the application on every node while preventing the central server from becoming the model-transfer bottleneck.

## Filesystem Layout

### Central server

```text
/mnt/modelctl/nas/                 # NAS modelctl store, read-only
/mnt/modelctl/nodes/
  node-01/                         # node-01 local store, read/write
  node-02/
  node-03/
```

### Each node

```text
/mnt/nas/llm-models/               # NAS modelctl store, read-only
/var/lib/llm-models/               # node-local model store
```

The inference workload reads `/var/lib/llm-models` directly from the node-local disk. It does not access model data through the central server.

## Cluster Configuration

Add a cluster configuration file:

```yaml
# /etc/modelctl/cluster.yaml
nas:
  management_root: /mnt/modelctl/nas
  node_root: /mnt/nas/llm-models

nodes:
  node-01:
    transport: ssh-rsync
    ssh_host: node-01.internal
    ssh_user: modelctl
    management_root: /mnt/modelctl/nodes/node-01
    runtime_root: /var/lib/llm-models
    store_id: 8c51d1ea-0000-0000-0000-000000000001
    reserve_free: 50GiB

  node-02:
    transport: ssh-rsync
    ssh_host: node-02.internal
    ssh_user: modelctl
    management_root: /mnt/modelctl/nodes/node-02
    runtime_root: /var/lib/llm-models
    store_id: a313c47b-0000-0000-0000-000000000002
    reserve_free: 50GiB
```

Path meanings:

- `management_root`: node store as seen by central `modelctl`.
- `runtime_root`: the same store as seen by software running on the node.
- `node_root`: NAS path as seen from each node.
- `management_root` under `nas`: NAS path as seen centrally.

Generated run instructions must use `runtime_root`, never the central management mount.

## Minimal Command Set

### NAS and cluster inventory

```bash
modelctl cluster models --config /etc/modelctl/cluster.yaml
modelctl cluster models --config /etc/modelctl/cluster.yaml --json

modelctl cluster inventory --config /etc/modelctl/cluster.yaml
modelctl cluster inventory node-01 --config /etc/modelctl/cluster.yaml
modelctl cluster inventory --config /etc/modelctl/cluster.yaml --json
```

Example matrix:

```text
MODEL          NAS      node-01   node-02   node-03
qwen3-8b       active   current   absent    current
model-q4       active   absent    current   absent
llama-70b      active   stale     absent    current
```

Inventory states:

- `absent`: active on NAS but not installed locally.
- `current`: local object matches the NAS commit and selection.
- `stale`: installed locally, but NAS points to a newer object.
- `local-only`: installed locally but no longer active on NAS.
- `deactivated`: object exists locally without an active reference.
- `syncing`: an active synchronization operation exists.
- `pending-removal`: object was moved to trash and deletion is incomplete.
- `invalid`: reference, metadata, or expected files failed validation.
- `offline`: node management mount or identity validation failed.

Inventory records should include:

- Model name.
- Hugging Face repository, revision, and commit.
- Format, runtime, and entrypoint.
- Expected model size from `.modelctl.json`.
- NAS object identity.
- Node-local object identity and path.
- Placement status.
- Reclaimable bytes.

### Synchronize a model

```bash
modelctl cluster sync node-01 qwen3-8b
modelctl cluster sync node-01 qwen3-8b --dry-run
```

### Deactivate a model

```bash
modelctl cluster deactivate node-01 qwen3-8b
```

Deactivation removes the node-local active reference but retains the object, preventing new workloads from resolving it while allowing an operator to drain an existing workload.

### Remove model data

```bash
modelctl cluster remove node-01 qwen3-8b --dry-run
modelctl cluster remove node-01 qwen3-8b --confirm-drained
```

### Prune unreferenced data

```bash
modelctl cluster prune node-01 --dry-run
modelctl cluster prune node-01
```

## Direct NAS-to-Node Synchronization

For:

```bash
modelctl cluster sync node-01 qwen3-8b
```

central `modelctl` performs the following workflow:

1. Load and validate the cluster configuration.
2. Validate the central NAS mount and resolve the NAS active model object.
3. Read and validate the model's `.modelctl.json` metadata.
4. Validate the destination node mount and node identity marker.
5. Compare NAS and node identities; return `current` without copying if they match.
6. Calculate required bytes from expected-file metadata.
7. Check node free space, existing staging bytes, reservations, and configured free-space margin.
8. Acquire the node/model operation lock.
9. Create or reuse the node staging directory through the management mount.
10. Derive the equivalent NAS source and staging destination paths as seen on the node.
11. Execute `rsync` on the node through SSH.
12. Validate the completed staging object through the central management mount.
13. Atomically rename staging into the node's `models/` directory.
14. Atomically update the node-local `active/NAME` symlink.
15. Record the operation in the audit/state journal.
16. Report previous unreferenced objects as reclaimable without deleting them automatically.

The remote transfer resembles:

```bash
ssh node-01.internal -- \
  rsync --archive --delete --partial \
    --exclude=/.cache/huggingface/ \
    /mnt/nas/llm-models/models/ORG/REPO/OBJECT/ \
    /var/lib/llm-models/.staging/ORG/REPO/OBJECT/
```

The implementation must construct and escape remote arguments safely. It must not interpolate arbitrary user-provided paths into a shell command.

## Transport Backends

Define a transport interface so orchestration is independent from copying.

### `ssh-rsync` — recommended default

Runs `rsync` on the destination node. Model bytes travel directly from NAS to node-local storage.

### `mounted-rsync` — fallback

Runs `rsync` centrally between the NAS mount and node management mount. This is simpler but routes model bytes through the central server.

Both backends must produce the same staged object and use the same validation and atomic publication workflow.

## Node Identity and Mount Safety

A missing node mount is dangerous: the central process could otherwise write into an empty local mount-point directory.

Initialize every node store with:

```text
/var/lib/llm-models/.modelctl-node.json
```

Example:

```json
{
  "schema": 1,
  "node": "node-01",
  "store_id": "8c51d1ea-0000-0000-0000-000000000001",
  "runtime_root": "/var/lib/llm-models"
}
```

Before any write or deletion, verify:

- The management path is an active mount point.
- The identity marker exists and parses correctly.
- Node name and `store_id` match the configuration.
- The management and runtime path mapping is valid.
- Symlink creation works.
- File locking works.
- Staging and final object directories are on the same filesystem.
- An atomic rename probe succeeds.

Fail before transferring or deleting data if any check fails.

## Path Safety

All operation paths must be derived from validated model and node metadata.

Require:

- NAS sources under configured `node_root/models/`.
- Node destinations under configured `runtime_root/.staging/` or `.trash/`.
- Management paths under the configured node `management_root`.
- No absolute repository-relative paths.
- No `..` components.
- No parent-directory symlink traversal.
- Validated model and node names.

Use structured SSH argument construction and shell escaping. Never concatenate an unchecked path into a remote command string.

## Validation Strategy

Central validation may safely use the management mount for:

- Metadata parsing.
- File existence.
- File size.
- Entrypoint and required-artifact checks.
- Active-reference validation.

If SHA-256 validation is introduced, hashing files through the central mount would read model data back through the central server. Run checksums remotely instead:

```bash
ssh node-01 sha256sum /var/lib/llm-models/.staging/.../FILE
```

Batch remote validation into one SSH invocation where possible.

## Capacity Management

Before synchronization, calculate:

```text
required bytes
- valid existing staging bytes
+ safety allowance
```

Compare that with:

```text
filesystem free bytes
- configured reserve_free
- active operation reservations
```

Store reservations under:

```text
NODE_ROOT/state/reservations/OPERATION.json
```

Reservation metadata should include:

- Node and model.
- Commit and selection identity.
- Reserved bytes.
- Existing staging bytes.
- Start time.
- Central controller identity.
- Operation identifier.

For the initial 2–6-node deployment, operations may be serialized per node to simplify capacity correctness. Per-model parallelism can be added later if required.

## Removal Workflow

Deletion must be a two-stage operation.

### Stage 1: Deactivate

Remove `active/NAME` atomically. The object remains present so existing workloads can be drained.

### Stage 2: Purge

After the operator confirms the workload is drained:

1. Acquire the node/model lock.
2. Validate the node identity and target object.
3. Verify no local active reference points to the same object.
4. Atomically rename the object into `.trash/`.
5. Record the transition.
6. Delete the trashed object, preferably through SSH so traversal and deletion execute locally.
7. Preserve interrupted trash operations for a later prune.

Because filesystem access cannot reliably determine whether an inference process still uses a model, destructive removal requires `--confirm-drained`. A separate `--force` may be provided with a strong warning for invalid objects or emergency recovery.

The NAS must never be modified by cluster removal commands.

## Security

Create a dedicated SSH account on every node:

```text
modelctl
```

Restrict it to:

- Read access to the node's NAS mount.
- Write access only to `/var/lib/llm-models`.
- Connections from the central management server.
- No unrestricted `sudo`.

For stronger isolation, configure an SSH forced-command wrapper that only permits:

- `rsync` from the configured NAS root into `.staging`.
- Capacity queries.
- Optional checksums.
- Deletion under `.trash`.

The wrapper can be a small reviewed shell script and does not require installing the full application on the node.

Export only the node model directory to the central server, not the complete node filesystem. Treat the central server as a privileged management system and retain an audit log for every operation.

## Failure and Recovery Behavior

- Interrupted `rsync` leaves resumable staging data.
- Failed validation never changes the active reference.
- Failed publication leaves the previous active model unchanged.
- A node going offline fails the operation without falling back to the underlying mount-point directory.
- Interrupted deletion leaves the object in `.trash` for later cleanup.
- A stale capacity reservation can only be removed after checking its operation owner and age.
- Inventory reports errors per node without preventing healthy nodes from being listed.

## Suggested Code Structure

```text
src/modelctl/cluster.py
src/modelctl/cluster_config.py
src/modelctl/inventory.py
src/modelctl/node_store.py
src/modelctl/transports.py
src/modelctl/removal.py
src/modelctl/capacity.py
```

Responsibilities:

- `cluster.py`: orchestration and command handlers.
- `cluster_config.py`: strict YAML parsing and path mappings.
- `inventory.py`: NAS/node comparison and JSON/text output.
- `node_store.py`: mount identity and filesystem validation.
- `transports.py`: `ssh-rsync` and `mounted-rsync` implementations.
- `removal.py`: deactivate, trash, purge, and prune.
- `capacity.py`: size checks and reservations.

Extend:

- `cli.py` with cluster subcommands.
- `operations.py` with transport-independent synchronization.
- `layout.py` with node markers, reservations, and trash paths.
- `state.py` with cluster sync and removal states.
- `validation.py` with node-store and staged-object validation.

## Testing

Add tests for:

- Strict cluster configuration parsing.
- NAS and node path mapping.
- Matrix inventory and stable JSON output.
- Current, absent, stale, local-only, invalid, and offline states.
- Missing mount detection.
- Incorrect node identity and `store_id`.
- Refusal to write into an unmounted directory.
- SSH argument/path injection attempts.
- Direct `ssh-rsync` command construction.
- Mounted transport fallback.
- Disk-space checks and reservations.
- Resumable staging.
- Atomic publication and activation.
- Active-model preservation after transfer failure.
- Deactivation without deletion.
- Shared-object reference protection.
- Confirmed removal and interrupted trash cleanup.
- NAS immutability during every node operation.
- Runtime paths in generated commands and cards.
- Offline nodes not preventing inventory of healthy nodes.

Add integration tests with temporary stores and a fake SSH runner. A later optional integration suite can exercise real SSH and mounted filesystems.

## Documentation and Versioning

Update:

- `README.md` with centralized cluster setup and operations.
- CLI help and copyable examples.
- Node identity initialization instructions.
- SSH account and mount requirements.
- Drain/deactivate/remove procedure.
- Troubleshooting for offline mounts and failed transfers.
- `CHANGELOG.md`.

This is new functionality. Under the project's semantic-versioning policy, increment the version from `0.4.0` to `0.5.0`.

## Implementation Phases

### Phase 1 — Configuration and read-only inventory

1. Implement strict cluster YAML parsing.
2. Implement node identity markers and mount validation.
3. Implement NAS catalog and node inventory.
4. Add text matrix and JSON output.

### Phase 2 — Direct synchronization

1. Extract a transport-independent synchronization workflow.
2. Implement `ssh-rsync`.
3. Add disk-space preflight.
4. Preserve resumable staging and atomic activation.
5. Implement `--dry-run`.

### Phase 3 — Lifecycle management

1. Implement deactivation.
2. Implement confirmed trash-based removal.
3. Protect shared/referenced objects.
4. Implement prune and interrupted-removal recovery.

### Phase 4 — Hardening

1. Add capacity reservations or serialize operations per node.
2. Add the restricted SSH forced-command wrapper.
3. Add structured audit logs.
4. Benchmark one and two concurrent node transfers.
5. Add mounted transport as a fallback.

## Deferred Features

For the initial 2–6-node deployment, defer:

- A long-running node agent.
- HTTP APIs.
- Automatic placement or eviction.
- Desired-state reconciliation.
- Scheduler-specific integrations.
- High-availability central control.
- Large-scale parallel transfer scheduling.

These can be added later without changing the cluster inventory or transport abstractions.
