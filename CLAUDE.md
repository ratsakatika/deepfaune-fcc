# Notes for AI agents working on this repository

This fork adds FCC's batch tooling (`dfrun.py`, `deepfaune_batch.py`) around
the official DeepFaune v1.4.1 engine. Field machines self-update by pulling
`main` at launch, so **main must always be releasable**.

## Versioning (do this on every change)

`TOOL_VERSION` in `dfrun.py` is the user-visible version shown in the banner.
**Increment it in the same commit as any change merged to main**, using the
owner's scheme: the DEFAULT is the THIRD digit (x.x.1) - fixes, refinements,
dashboard/HTML changes, anything short of a new capability in the batch tool
itself. The SECOND digit (x.1.0) is reserved for a genuinely new capability
the owner would announce, and the FIRST digit (2.0.0) for complete overhauls.
When unsure, bump the third digit. Never merge behaviour changes that leave
the version untouched - users then report "1.x.y" for two different programs,
which has already caused confusion.

Note: versions 1.6.0-1.10.0 over-counted a run of small updates as second-digit
bumps; the owner reset the line to 1.5.4 on 2026-08-31. That reset was
deliberate - do not "correct" the version upward past history, just keep
incrementing from the current value.

## Ground rules

- Run `python -m pytest tests/ -q` before pushing; keep it green. The tests
  run without torch or model weights - keep new logic testable that way
  (pure helper functions, injected fakes).
- `predictTools.py`, `detectTools.py`, `classifTools.py` are the upstream
  engine, shared with the official GUI (`deepfauneGUI.py`). Change them only
  additively: no signature changes to existing methods, no behaviour changes
  to existing code paths.
- The archive drive is mounted read-only and must never be written to. All
  outputs go to the out-dir (default `~/df_out`); shard CSVs there are the
  source of truth and their existence is the resume mechanism.
- Shard CSV schemas may differ between tool versions; anything that merges
  or reads them must match columns by NAME (see `merge_csvs`,
  `shard_header_union`), never by position.
- The target machine is CPU-only with 16 GiB RAM and a history of OOM kills:
  keep the worker's memory guards (`--min-avail-gib`, oom_score_adj,
  telemetry) intact, and prefer stdlib-only code in the batch tools' import
  path (torch is imported lazily, only on the real run path).
- British English in user-facing text; no em dashes.
