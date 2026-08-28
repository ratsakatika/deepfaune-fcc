# Notes for AI agents working on this repository

This fork adds FCC's batch tooling (`dfrun.py`, `deepfaune_batch.py`) around
the official DeepFaune v1.4.1 engine. Field machines self-update by pulling
`main` at launch, so **main must always be releasable**.

## Versioning (do this on every change)

`TOOL_VERSION` in `dfrun.py` is the user-visible version shown in the banner.
**Increment it in the same commit as any change merged to main**, using the
owner's scheme: bump the THIRD digit (x.x.1) for small updates and fixes,
the SECOND digit (x.1.0) for medium features, and the FIRST digit (2.0.0)
only for complete overhauls. Never merge behaviour changes that leave the
version untouched - users then report "1.x.y" for two different programs,
which has already caused confusion.

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
