# Plan: Detect, prevent, and repair invalid active model references

## Execution status

Implemented in the working tree:

- centralized symlink verification and collision-resistant temporary links;
- activation postcondition checks, explicit failure state, and restoration of a
  previously valid reference;
- direct-download and queue symlink capability probes;
- `doctor`, dry-run-first `repair-active`, and explicit
  `cleanup-quarantine` commands;
- malformed-reference warnings for human-readable `list` output;
- deterministic audit, repair, rollback, cleanup, CLI, and activation tests;
- README and changelog documentation;
- the tested working-tree build installed as the active `modelctl` tool.

Operational recovery completed on `/mnt/nas`:

- all ten regular active directories passed repair dry-run validation;
- the 17.56 GiB Gemma copy was repaired and verified first;
- the remaining nine were repaired afterward;
- all ten active paths are now verified symlinks to their journaled canonical
  objects and appear in `modelctl list`;
- `/mnt/nas/active` now contains 84 symlinks, no regular model directories, and
  the unrelated `.DS_Store` file;
- post-repair doctor output contains no malformed active references. It retains
  six warnings for older valid symlinks whose latest update journals contain no
  successful activation event.

The CIFS/NAS did not retain the renamed large directory trees at their recorded
quarantine child paths, although small-directory rename probes preserve data as
expected. The canonical objects remained valid and the repaired symlinks resolve
to them. Empty per-model quarantine parent directories and repair journals were
retained. The implementation now verifies whether quarantine survives and
reports `repaired-no-quarantine` instead of claiming retention when this NAS
behavior occurs.

## Confirmed condition

`modelctl list` only treats `ROOT/active/NAME` symbolic links as active models.
The current NAS store contains 85 entries under `/mnt/nas/active`:

- 74 valid symbolic links;
- 10 regular directories corresponding to the newest downloads;
- one unrelated hidden file, `.DS_Store`.

The ten regular directories are intentionally excluded by
`list_active_models()` because accepting arbitrary directories would bypass the
invariant that active references must resolve inside `ROOT/models`.

### Duplicate-directory audit

Every regular directory in `/mnt/nas/active` has a corresponding full directory
under `/mnt/nas/models`, selected from the last `ACTIVE_ON_NAS` event in its
state journal.

| Active name | Files | Logical size |
| --- | ---: | ---: |
| `DeepSeek-V4-Flash-0731` | 214 | 155.44 GiB |
| `DeepSeek-V4-Flash-0731-FP8-DSpark` | 244 | 286.32 GiB |
| `DeepSeek-V4-Flash-0731-GGUF-Unsloth-UD-Q4_K_XL` | 22 | 144.44 GiB |
| `DeepSeek-V4-Flash-0731-GGUF-Unsloth-UD-Q8_K_XL` | 22 | 150.75 GiB |
| `DeepSeek-V4-Flash-0731-NVFP4-utarn` | 217 | 163.51 GiB |
| `DeepSeek-V4-Flash-0731-Unsloth` | 220 | 155.44 GiB |
| `Inkling-Small-GGUF-UD-IQ4_XS` | 19 | 118.66 GiB |
| `Inkling-Small-GGUF-UD-Q4_K_XL` | 22 | 152.06 GiB |
| `Inkling-Small-NVFP4` | 64 | 159.04 GiB |
| `gemma-4-31B-it-qat-q4_0-gguf` | 11 | 17.56 GiB |
| **Total** |  | **1,503.21 GiB / 1.468 TiB** |

Checks completed for all ten pairs:

- the journaled canonical path resolves strictly inside `/mnt/nas/models`;
- both the active directory and canonical object pass `validate_object()` for
  the expected model name;
- `.modelctl.json` is identical in both locations;
- complete tree inventories match by relative path, entry type, file size, and
  symlink target;
- sampled content matches for every readable file: complete hashes for files up
  to 1 MiB and beginning/middle/end samples for larger files;
- one Hugging Face cache tree file per pair is unreadable through both paths,
  but has matching stat metadata and is not a selected model artifact;
- zero corresponding files share an inode, while known-good active symlinks do
  expose the canonical inode. The active directories must therefore be treated
  as independent directory copies, not links.

The duplicate active trees represent about 1.61 TB decimal (1.468 TiB) of
logical data. Physical usage could differ if the NAS uses block-level cloning,
but modelctl must not assume that.

## Root-cause assessment

1. `list_active_models()` in `src/modelctl/operations.py` deliberately skips
   non-symlinks. This is the correct safety behavior and explains why the ten
   models do not appear in `modelctl list`.
2. `update_model()` calls `atomic_symlink()` and immediately journals
   `ACTIVE_ON_NAS`; it does not verify the published reference after
   `os.replace()`.
3. The repository has no intentional code path that copies a model object into
   `ROOT/active`. The duplicate directories cannot be attributed to designed
   modelctl behavior.
4. The CIFS mount currently includes `mfsymlinks`. Direct and modelctl atomic
   symlink probes to affected canonical objects currently succeed, so the
   original failure is not reproducible now.
5. The directory timestamps align closely with publication/activation, but the
   available journals record state transitions rather than filesystem syscall
   results. A transient CIFS/NAS condition, a different client, or an external
   operation remains possible. NAS/SMB audit logs are required to identify the
   originating writer conclusively.
6. The actionable modelctl defect is the false-positive activation: it can
   record `ACTIVE_ON_NAS` without proving that `active/NAME` is the expected
   symlink.
7. The installed command is modelctl 0.9.4 while the repository is 0.9.5.
   Upgrading alone does not repair or prevent this condition.

## Revised implementation plan

### 1. Centralize active-reference validation

- Add a focused helper that uses `lstat()`/`is_symlink()`, reads the link text,
  resolves it strictly, and proves that it points to the exact expected
  immutable object inside `ROOT/models`.
- Reuse this helper in activation, listing, path resolution, deletion, serving,
  synchronization, auditing, and repair so the safety rules cannot diverge.
- Keep hidden/foreign entries excluded, but distinguish them from malformed
  model references in diagnostics.

### 2. Harden activation and rollback

- Validate the active-reference destination before publication. Refuse to
  overwrite a regular file or directory through the normal update path.
- Create the temporary link with a collision-resistant name and verify the
  temporary link before `os.replace()`.
- Verify the final link immediately after `os.replace()` and before fsync and
  the `ACTIVE_ON_NAS` transition.
- Preserve the previous valid link target before replacement. If final
  verification fails, attempt to restore the previous valid reference and
  record an explicit activation-failure state. Never recursively delete an
  unexpected directory during rollback.
- Add a state such as `FAILED_TO_ACTIVATE` containing the expected object,
  observed reference type, rollback result, and actionable error text.
- Run the root symlink capability probe for direct `download` operations as
  well as queues. The queue-only preflight is insufficient, and the final
  per-activation check remains authoritative after long downloads.

### 3. Add store audit and user-visible diagnostics

- Add a domain-level audit that classifies each active entry as:
  valid symlink, hidden/foreign entry, broken symlink, outside-store symlink,
  wrong-object symlink, regular directory, invalid object, missing journal, or
  journal/object mismatch.
- Expose it through `modelctl doctor` with human-readable and `--json` output.
- Keep `modelctl list` safe and backward compatible, but print a concise stderr
  warning in human mode when non-hidden malformed references are skipped, with
  guidance to run `modelctl doctor`. Keep JSON stdout machine-readable.
- Include filesystem type and relevant mount capabilities in doctor output
  when available, without making `/proc` parsing a correctness dependency.

### 4. Add a dedicated conservative repair command

Use a dedicated `modelctl repair-active` command rather than hiding mutations
behind `doctor`.

- Default to dry-run; require an explicit `--apply` for filesystem changes.
- Resolve the intended object using independent evidence:
  1. the latest successful journal event;
  2. active-directory metadata;
  3. the canonical object metadata/path under `ROOT/models`.
- Offer repair only when the evidence agrees, the canonical object passes
  `validate_object()` for the expected name and commit, and the active directory
  independently passes validation with identical metadata.
- Acquire the per-model lock and re-run all checks after locking.
- Rename the regular active directory to a same-filesystem quarantine area,
  then publish and verify the symlink. A regular non-empty directory cannot be
  portably replaced atomically by a symlink, so document the brief repair gap.
- If publication or verification fails, restore the quarantined directory and
  report whether rollback succeeded.
- Never delete the quarantined tree as part of repair. Add a separate cleanup
  operation that requires a successful post-repair audit and explicit user
  confirmation.
- Journal audit, quarantine, publication, verification, rollback, and cleanup
  stages.

### 5. Test failure and recovery paths

Add deterministic tests covering:

- the symlink helper rejecting regular directories, broken links, wrong-object
  links, and links outside `ROOT/models`;
- `atomic_symlink()` returning without publishing the requested symlink;
- activation verification failing before `ACTIVE_ON_NAS` is recorded;
- restoration of the previous valid active reference after publication failure;
- direct downloads performing the filesystem capability check;
- audit classification for every supported condition;
- repair dry-run making no changes;
- successful quarantine and repair;
- rollback when repair cannot publish or verify the symlink;
- refusal to repair when journal, metadata, name, commit, or canonical object
  evidence differs;
- preservation of quarantined data until explicit cleanup;
- human warnings and JSON output remaining well formed.

Include a failure-path test proving that an invalid reference can never become
active and that a previously active revision remains usable.

### 6. Operational recovery for this NAS

After implementation and tests:

1. Stop modelctl updates and consumers using the ten malformed active paths.
2. Save the doctor report and take a NAS snapshot/backup of manifests, state,
   object metadata, and active-reference directories.
3. Upgrade the installed CLI to the fixed release.
4. Validate all ten canonical objects and run `repair-active` in dry-run mode.
5. Repair `gemma-4-31B-it-qat-q4_0-gguf` first because it is the smallest
   affected copy (17.56 GiB).
6. Verify `modelctl doctor`, `modelctl list`, `modelctl path`, and the generated
   serve command, then test its consumer.
7. Repair the remaining nine models and verify every link resolves to the
   journaled object under `/mnt/nas/models`.
8. Keep quarantined copies through an observation period. Remove them only with
   the explicit cleanup workflow after consumers are confirmed healthy.
9. Review NAS/SMB audit logs around the journal timestamps to identify the
   writer or filesystem behavior that created the copies.

## Acceptance criteria

- A download cannot reach `ACTIVE_ON_NAS` unless `active/NAME` is a verified
  symlink to the exact expected object under `models/`.
- A failed activation preserves or restores the previously valid reference.
- `modelctl doctor` reports all ten malformed references without following them
  unsafely.
- Repair is dry-run by default, rollback-safe, and never deletes the only known
  valid copy.
- All ten repaired models appear in `modelctl list` and resolve to their
  validated canonical objects.
- Existing valid symlinks, `.DS_Store`, and unrelated hidden entries remain
  untouched.
