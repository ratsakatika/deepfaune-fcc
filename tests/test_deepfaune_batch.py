"""Unit tests for the model-free logic in deepfaune_batch.

These cover directory sharding, junk filtering, the atomic temp-then-rename CSV
write, the skip-if-CSV-exists resume behaviour and partition selection. None of
them need torch or the model weights, so they run anywhere.
"""

import csv
import json
import os
import re
import sys

import pytest

import deepfaune_batch as dfb


def _args(**overrides):
    """Build an args namespace with defaults, then apply overrides."""
    args = dfb.build_arg_parser().parse_args(["--out-dir", "/tmp/unused"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# ---------------------------------------------------------------------------
# junk and extension filtering
# ---------------------------------------------------------------------------
def test_is_junk():
    assert dfb.is_junk("._photo.jpg")
    assert dfb.is_junk(".DS_Store")
    assert not dfb.is_junk("photo.jpg")
    assert not dfb.is_junk("DS_Store.jpg")


def test_is_image_file():
    for good in ["a.jpg", "a.JPG", "a.jpeg", "a.png", "a.bmp", "a.tif", "a.tiff", "a.gif"]:
        assert dfb.is_image_file(good)
    for bad in ["a.mp4", "a.mov", "a.txt", "a.json", "noext", "a.JPG.bak"]:
        assert not dfb.is_image_file(bad)


def test_list_images_in_dir_sorts_and_filters(tmp_path):
    names = ["b.JPG", "a.jpeg", "._a.jpeg", ".DS_Store", "clip.mp4", "notes.txt"]
    paths = dfb.list_images_in_dir(str(tmp_path), names)
    assert [os.path.basename(p) for p in paths] == ["a.jpeg", "b.JPG"]
    assert all(os.path.dirname(p) == str(tmp_path) for p in paths)


# ---------------------------------------------------------------------------
# sharding by leaf directory
# ---------------------------------------------------------------------------
def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_find_shards(tmp_path):
    root = tmp_path
    # Root directly holds an image (so root itself is a leaf shard), plus junk,
    # a non-image and a video that must all be ignored.
    _touch(root / "a.jpg")
    _touch(root / "._a.jpg")
    _touch(root / ".DS_Store")
    _touch(root / "notes.txt")
    _touch(root / "clip.mp4")
    # A site with two images and a deeper camera directory.
    _touch(root / "siteA" / "img2.jpeg")
    _touch(root / "siteA" / "img1.JPG")
    _touch(root / "siteA" / "cam1" / "p1.png")
    # siteB has no direct images, only a camera subdirectory with images.
    _touch(root / "siteB" / "cam2" / "q2.bmp")
    _touch(root / "siteB" / "cam2" / "q1.tif")
    # An entirely empty directory is not a shard.
    (root / "empty").mkdir()

    shards = dfb.find_shards(str(root))
    rel = {
        os.path.relpath(d, str(root)): [os.path.basename(p) for p in imgs]
        for d, imgs in shards
    }
    assert set(rel) == {
        ".",
        "siteA",
        os.path.join("siteA", "cam1"),
        os.path.join("siteB", "cam2"),
    }
    assert rel["."] == ["a.jpg"]
    assert rel["siteA"] == ["img1.JPG", "img2.jpeg"]  # sorted, junk dropped
    assert rel[os.path.join("siteA", "cam1")] == ["p1.png"]
    assert rel[os.path.join("siteB", "cam2")] == ["q1.tif", "q2.bmp"]


def test_find_shards_is_sorted_and_deterministic(tmp_path):
    for d in ["z", "a", "m"]:
        _touch(tmp_path / d / "img.jpg")
    first = dfb.find_shards(str(tmp_path))
    second = dfb.find_shards(str(tmp_path))
    assert first == second
    assert [os.path.basename(d) for d, _ in first] == ["a", "m", "z"]


def test_find_shards_skips_dirs_without_direct_images(tmp_path):
    _touch(tmp_path / "parent" / "child" / "img.jpg")
    shards = dfb.find_shards(str(tmp_path))
    rels = {os.path.relpath(d, str(tmp_path)) for d, _ in shards}
    assert rels == {os.path.join("parent", "child")}


# ---------------------------------------------------------------------------
# partition selection
# ---------------------------------------------------------------------------
def test_select_partition_single_returns_all():
    shards = [("d0", []), ("d1", []), ("d2", [])]
    assert dfb.select_partition(shards, 1, 0) == shards


def test_select_partition_disjoint_and_complete():
    shards = [(f"d{i}", [f"d{i}/x.jpg"]) for i in range(10)]
    n = 3
    collected = []
    for p in range(n):
        collected.extend(dfb.select_partition(shards, n, p))
    # Every shard appears exactly once across all partitions.
    assert sorted(collected) == sorted(shards)
    assert len(collected) == len(shards)


def test_select_partition_balanced():
    shards = [(f"d{i}", []) for i in range(9)]
    sizes = [len(dfb.select_partition(shards, 3, p)) for p in range(3)]
    assert sizes == [3, 3, 3]


# ---------------------------------------------------------------------------
# CSV naming
# ---------------------------------------------------------------------------
def test_shard_csv_name_stable_and_unique(tmp_path):
    root = str(tmp_path)
    a = os.path.join(root, "siteA", "cam1")
    b = os.path.join(root, "siteB", "cam1")
    name_a1 = dfb.shard_csv_name(root, a)
    name_a2 = dfb.shard_csv_name(root, a)
    name_b = dfb.shard_csv_name(root, b)
    assert name_a1 == name_a2  # deterministic across calls
    assert name_a1 != name_b  # different shards get different names
    assert name_a1.endswith(".csv")
    assert re.match(r"^[A-Za-z0-9._-]+\.csv$", name_a1)


def test_shard_csv_name_for_root_itself(tmp_path):
    root = str(tmp_path)
    name = dfb.shard_csv_name(root, root)
    assert name.endswith(".csv")
    assert re.match(r"^[A-Za-z0-9._-]+\.csv$", name)


# ---------------------------------------------------------------------------
# atomic CSV write
# ---------------------------------------------------------------------------
def test_atomic_write_csv_success(tmp_path):
    target = tmp_path / "sub" / "out.csv"  # parent created on the fly
    rows = [["a.jpg", "2024:01:01 00:00:00", 1, "cat", 0.9, "cat", 0.88, 1, 0]]
    dfb.atomic_write_csv(str(target), dfb.CSV_HEADER, rows)

    assert target.exists()
    with open(target, newline="", encoding="utf-8") as handle:
        got = list(csv.reader(handle))
    assert got[0] == dfb.CSV_HEADER
    assert got[1][0] == "a.jpg"
    assert got[1][3] == "cat"
    # No temporary file left behind.
    leftovers = [p.name for p in target.parent.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_atomic_write_csv_failure_preserves_original(tmp_path, monkeypatch):
    target = tmp_path / "out.csv"
    target.write_text("original,data\n", encoding="utf-8")

    def boom(src, dst):
        raise OSError("simulated disk failure during rename")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        dfb.atomic_write_csv(str(target), dfb.CSV_HEADER, [["x"] * len(dfb.CSV_HEADER)])

    # The original is untouched and no temp file is left behind.
    assert target.read_text(encoding="utf-8") == "original,data\n"
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "out.csv"]
    assert leftovers == []


# ---------------------------------------------------------------------------
# resume behaviour (skip shards whose CSV already exists)
# ---------------------------------------------------------------------------
def test_plan_shards_skips_existing_and_rescan_includes_all(tmp_path):
    root = tmp_path / "src"
    _touch(root / "camA" / "1.jpg")
    _touch(root / "camB" / "1.jpg")
    out = tmp_path / "out"
    out.mkdir()

    shards = dfb.find_shards(str(root))
    assert len(shards) == 2

    # Mark the first shard as already done by creating its CSV.
    first_dir = shards[0][0]
    done_name = dfb.shard_csv_name(str(root), first_dir)
    (out / done_name).write_text("done\n", encoding="utf-8")

    plan = dfb.plan_shards(shards, str(out), str(root), rescan=False)
    planned_dirs = {d for d, _, _ in plan}
    assert first_dir not in planned_dirs
    assert len(plan) == 1

    # With rescan, the finished shard is included again.
    plan_all = dfb.plan_shards(shards, str(out), str(root), rescan=True)
    assert len(plan_all) == 2


def test_plan_shards_returns_csv_paths_in_out_dir(tmp_path):
    root = tmp_path / "src"
    _touch(root / "cam" / "1.jpg")
    out = tmp_path / "out"
    shards = dfb.find_shards(str(root))
    plan = dfb.plan_shards(shards, str(out), str(root), rescan=False)
    assert len(plan) == 1
    _dir, _imgs, csv_path = plan[0]
    assert os.path.dirname(csv_path) == os.path.abspath(str(out))
    assert csv_path.endswith(".csv")


# ---------------------------------------------------------------------------
# misc helpers
# ---------------------------------------------------------------------------
def test_format_duration():
    assert dfb.format_duration(0) == "0:00:00"
    assert dfb.format_duration(65) == "0:01:05"
    assert dfb.format_duration(3661) == "1:01:01"
    assert dfb.format_duration(-5) == "0:00:00"  # clamped


def test_progress_true_overall_and_summary():
    # Archive of 100 images, 25 classified in earlier sessions.
    progress = dfb.Progress(archive_total=100, done_start=25)
    assert progress.true_done == 25
    progress.complete_shard(25)  # 25 more this session
    assert progress.true_done == 50
    assert progress.true_pct() == 50.0
    summary = progress.summary()
    assert "50/100" in summary
    assert "ETA" in summary


def test_progress_true_pct_capped_at_100():
    # A stale over-count (more classified than the recomputed archive total)
    # must not report above 100 per cent.
    progress = dfb.Progress(archive_total=10, done_start=12)
    assert progress.true_pct() == 100.0


def test_rate_tracker_smooths_over_window():
    tracker = dfb.RateTracker(window_seconds=100)
    tracker.update(0, now=0.0)
    tracker.update(100, now=10.0)  # 100 images in 10 s
    assert abs(tracker.rate() - 10.0) < 1e-9
    # A single sample cannot give a rate.
    assert dfb.RateTracker().rate() == 0.0


# ---------------------------------------------------------------------------
# shard-pattern and system-folder helpers
# ---------------------------------------------------------------------------
def test_is_shard_csv():
    assert dfb.is_shard_csv("siteA__cam1__0a1b2c3d.csv")
    assert dfb.is_shard_csv("/path/to/x__deadbeef.csv")
    assert not dfb.is_shard_csv("master.csv")
    assert not dfb.is_shard_csv("summary.csv")
    assert not dfb.is_shard_csv("shard__0a1b2c3d.csv.tmp")
    assert not dfb.is_shard_csv("x__short.csv")  # not 8 hex chars


def test_is_system_dir():
    assert dfb.is_system_dir("$RECYCLE.BIN")
    assert dfb.is_system_dir("System Volume Information")
    assert dfb.is_system_dir("FOUND.000")
    assert not dfb.is_system_dir("Camera Trap Monitoring")


def test_find_shards_excludes_system_dirs(tmp_path):
    _touch(tmp_path / "$RECYCLE.BIN" / "junk.jpg")
    _touch(tmp_path / "System Volume Information" / "junk.jpg")
    _touch(tmp_path / "FOUND.000" / "junk.jpg")
    _touch(tmp_path / "cam1" / "real.jpg")
    rels = {os.path.relpath(d, str(tmp_path)) for d, _ in dfb.find_shards(str(tmp_path))}
    assert rels == {"cam1"}


def test_count_classified_images(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    _write_shard_csv(
        out / "a__0a1b2c3d.csv",
        [["/a/1.jpg", "d", 1, "wolf", 0.9, "wolf", 0.9, 1, 0]],
    )
    _write_shard_csv(
        out / "b__1a2b3c4d.csv",
        [["/b/1.jpg", "d", 1, "lynx", 0.8, "lynx", 0.8, 1, 0],
         ["/b/2.jpg", "d", 1, "empty", 1.0, "empty", 1.0, 0, 0]],
    )
    # Non-shard files must not be counted.
    (out / "master.csv").write_text("filename\nx\n", encoding="utf-8")
    assert dfb.count_classified_images(str(out)) == 3


def test_validate_common_threads():
    assert dfb.validate_common(_args(threads=0)) is not None
    assert dfb.validate_common(_args(threads=1)) is None


def test_default_parameters():
    # Threshold 0.5 (matches the first long run and the demo) and maxlag 10.
    args = dfb.build_arg_parser().parse_args(["--out-dir", "/tmp/unused"])
    assert args.threshold == 0.5
    assert args.maxlag == 10
    assert args.threads == 4


def test_dry_run_and_merge_mutually_exclusive(tmp_path, capsys):
    rc = dfb.main(["--out-dir", str(tmp_path), "--dry-run", "--merge"])
    assert rc == 2
    assert "mutually exclusive" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------
def test_validate_common_accepts_defaults():
    assert dfb.validate_common(_args()) is None


def test_validate_common_rejects_batch_size_zero_and_negative():
    err = dfb.validate_common(_args(batch_size=0))
    assert err is not None and "batch-size" in err
    assert dfb.validate_common(_args(batch_size=-3)) is not None


def test_validate_common_threshold_bounds():
    assert dfb.validate_common(_args(threshold=0)) is not None
    assert dfb.validate_common(_args(threshold=1.5)) is not None
    assert dfb.validate_common(_args(threshold=-0.1)) is not None
    assert dfb.validate_common(_args(threshold=1.0)) is None
    assert dfb.validate_common(_args(threshold=0.5)) is None


def test_validate_common_maxlag_heartbeat_mergeevery():
    assert dfb.validate_common(_args(maxlag=-1)) is not None
    assert dfb.validate_common(_args(maxlag=0)) is None
    assert dfb.validate_common(_args(heartbeat_secs=-1)) is not None
    assert dfb.validate_common(_args(merge_every=-1)) is not None


def test_validate_common_max_images():
    assert dfb.validate_common(_args(max_images=0)) is not None
    assert dfb.validate_common(_args(max_images=None)) is None
    assert dfb.validate_common(_args(max_images=10)) is None


def test_validate_common_partition_range():
    assert dfb.validate_common(_args(num_partitions=1, partition=1)) is not None
    assert dfb.validate_common(_args(num_partitions=0)) is not None
    assert dfb.validate_common(_args(num_partitions=2, partition=0)) is None


def test_batch_size_zero_rejected_before_torch_import(tmp_path, capsys, monkeypatch):
    # Start from a clean slate so the check is independent of any other test that
    # may have placed a torch (real or stubbed) into sys.modules; monkeypatch
    # restores the previous state afterwards.
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    rc = dfb.main(
        [
            "--root", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--software-dir", str(tmp_path),
            "--batch-size", "0",
        ]
    )
    assert rc == 2
    # The critical guarantee: the run bailed out before importing the engine.
    assert "torch" not in sys.modules
    assert "batch-size" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# path containment (the commonpath fix)
# ---------------------------------------------------------------------------
def test_is_within():
    assert dfb.is_within("/a/b/c", "/a/b")
    assert dfb.is_within("/a/b", "/a/b")  # equal counts as within
    assert not dfb.is_within("/a/bc", "/a/b")  # sibling sharing a prefix
    assert not dfb.is_within("/x", "/a/b")
    # The edge case a string-prefix check gets wrong: root being "/".
    assert dfb.is_within("/anything", "/")


# ---------------------------------------------------------------------------
# master CSV merge
# ---------------------------------------------------------------------------
def _write_shard_csv(path, data_rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(dfb.CSV_HEADER)
        writer.writerows(data_rows)


def test_merge_csvs_header_once_quoting_and_ignores_decoys(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    _write_shard_csv(
        out / "shardA__0a1b2c3d.csv",
        [["/a/1.jpg", "d", 1, "wolf, grey", 0.9, "wolf", 0.9, 1, 0]],
    )
    _write_shard_csv(
        out / "shardB__1a2b3c4d.csv",
        [["/b/2.jpg", "d", 1, "lynx", 0.8, "lynx", 0.8, 1, 0]],
    )
    # Decoys that must be ignored: temp file, JSON sidecar, and a non-shard CSV
    # (master/summary files do not match the shard-name pattern).
    (out / "partial.csv.tmp").write_text("garbage\n", encoding="utf-8")
    (out / "run_metadata.p0.json").write_text("{}", encoding="utf-8")
    (out / "summary.csv").write_text("not,a,shard\n", encoding="utf-8")

    merge_out = out / "master.csv"
    n_files, n_rows = dfb.merge_csvs(str(out), str(merge_out))
    assert (n_files, n_rows) == (2, 2)

    with open(merge_out, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == dfb.CSV_HEADER  # header written exactly once
    assert len(rows) == 3  # header + two data rows
    species = {r[3] for r in rows[1:]}
    assert "wolf, grey" in species  # comma survived quoting as one field
    assert "lynx" in species
    # The merge's own temp file was renamed away (the decoy .tmp is left alone).
    assert not [p for p in out.iterdir() if p.name.startswith("master.csv.")]


def test_merge_csvs_excludes_master_and_is_idempotent(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    _write_shard_csv(
        out / "shardA__0a1b2c3d.csv",
        [["/a/1.jpg", "d", 1, "wolf", 0.9, "wolf", 0.9, 1, 0]],
    )
    merge_out = out / "master.csv"
    dfb.merge_csvs(str(out), str(merge_out))
    # Re-running must not fold the master back into itself.
    n_files, n_rows = dfb.merge_csvs(str(out), str(merge_out))
    assert (n_files, n_rows) == (1, 1)
    with open(merge_out, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 2  # header + one data row


def test_iter_shard_csvs_filters_and_excludes(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "a__0a1b2c3d.csv").write_text("x", encoding="utf-8")
    (out / "b__1a2b3c4d.csv").write_text("x", encoding="utf-8")
    (out / "c__2a3b4c5d.csv.tmp").write_text("x", encoding="utf-8")  # temp, ignored
    (out / "master.csv").write_text("x", encoding="utf-8")  # non-shard, ignored
    (out / "meta.json").write_text("x", encoding="utf-8")  # sidecar, ignored
    got = [
        os.path.basename(p)
        for p in dfb.iter_shard_csvs(str(out), exclude=[str(out / "b__1a2b3c4d.csv")])
    ]
    assert got == ["a__0a1b2c3d.csv"]


# ---------------------------------------------------------------------------
# provenance sidecar
# ---------------------------------------------------------------------------
def test_write_run_metadata(tmp_path):
    sw = tmp_path / "sw"
    sw.mkdir()
    (sw / "ChangeLog.txt").write_text(
        "2025-11-07  Author\n\n    Version 1.4.1\n", encoding="utf-8"
    )
    (sw / "model.pt").write_bytes(b"x" * 10)
    out = tmp_path / "out"
    out.mkdir()
    args = _args(detector="DFbsMDS", birds=True, threshold=0.6, partition=0)

    path, meta = dfb.write_run_metadata(str(out), str(sw), "/some/root", args)
    assert os.path.basename(path) == "run_metadata.p0.json"

    with open(path, encoding="utf-8") as handle:
        on_disk = json.load(handle)
    assert on_disk["deepfaune_version"] == "1.4.1"
    assert on_disk["weights"]["model.pt"] == 10
    assert on_disk["root"] == "/some/root"
    assert on_disk["detector"] == "DFbsMDS"
    assert on_disk["birds"] is True
    assert on_disk["threshold"] == 0.6
    assert "hostname" in on_disk
    assert on_disk["utc_start_time"].endswith("+00:00")
    assert "orchestrator_git_commit" in on_disk  # a sha or None


def test_read_deepfaune_version_missing_changelog(tmp_path):
    assert dfb.read_deepfaune_version(str(tmp_path)) is None


def test_metadata_one_file_per_partition(tmp_path):
    sw = tmp_path / "sw"
    sw.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    p0, _ = dfb.write_run_metadata(
        str(out), str(sw), "/r", _args(partition=0, num_partitions=2)
    )
    p1, _ = dfb.write_run_metadata(
        str(out), str(sw), "/r", _args(partition=1, num_partitions=2)
    )
    assert os.path.basename(p0) == "run_metadata.p0.json"
    assert os.path.basename(p1) == "run_metadata.p1.json"
    assert p0 != p1
