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
    # Threshold 0.5 and maxlag 20 match the first long run (and the demo).
    args = dfb.build_arg_parser().parse_args(["--out-dir", "/tmp/unused"])
    assert args.threshold == 0.5
    assert args.maxlag == 20
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
    # Header written exactly once, with the derived sequence_id appended.
    assert rows[0] == dfb.CSV_HEADER + [dfb.SEQUENCE_ID_COLUMN]
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


# The shard header written by versions before the raw-score columns existed.
OLD_CSV_HEADER = [
    "filename", "date", "seqnum", "prediction_seq", "score_seq",
    "prediction_image", "score_image", "animal_count", "human_count",
]


def test_merge_csvs_maps_mixed_schemas_by_column_name(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    # An old shard (pre raw-score columns) alongside a new full-schema shard.
    new_header = dfb.full_csv_header(["wolf", "lynx"], ["corvid"])
    with open(out / "old__0a1b2c3d.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(OLD_CSV_HEADER)
        writer.writerow(["/a/1.jpg", "d", 1, "wolf", 0.9, "wolf", 0.9, 1, 0])
    with open(out / "new__1a2b3c4d.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(new_header)
        writer.writerow(
            ["/b/2.jpg", "d", 1, "lynx", "lynx", 0.8, "yes", "lynx", "lynx",
             0.8, 1, 0, "0.9000", "0.0000", "0.0000", "0.1000", "0.9000", ""]
        )

    merge_out = out / "master.csv"
    n_files, n_rows = dfb.merge_csvs(str(out), str(merge_out))
    assert (n_files, n_rows) == (2, 2)
    with open(merge_out, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    # The widest header supplies the column order (old is a subset of new),
    # with the derived sequence_id appended.
    assert rows[0] == new_header + [dfb.SEQUENCE_ID_COLUMN]
    by_file = {r[0]: dict(zip(rows[0], r)) for r in rows[1:]}
    old = by_file["/a/1.jpg"]
    assert old["prediction_seq"] == "wolf"  # re-mapped by name, not position
    assert old["score_seq"] == "0.9"
    assert old["top1_seq"] == ""  # absent in the old schema: left blank
    assert old["score_wolf"] == ""
    new = by_file["/b/2.jpg"]
    assert new["above_threshold"] == "yes"
    assert new["score_lynx"] == "0.9000"
    assert new["det_conf_animal"] == "0.9000"


def test_shard_header_union_falls_back_to_base_header(tmp_path):
    out = tmp_path / "out"
    out.mkdir()  # no shards at all
    assert dfb.shard_header_union(str(out)) == dfb.CSV_HEADER


def test_full_csv_header_sanitises_class_names():
    header = dfb.full_csv_header(["wild boar", "red deer"], ["corvid"])
    assert header == dfb.CSV_HEADER + [
        "score_wild_boar", "score_red_deer", "birdscore_corvid",
    ]
    # Without the bird head there are no birdscore columns.
    assert dfb.full_csv_header(["cat"]) == dfb.CSV_HEADER + ["score_cat"]


# ---------------------------------------------------------------------------
# impossible-species exclusion (--exclude-classes)
# ---------------------------------------------------------------------------
def test_parse_excluded_classes():
    # Case-insensitive, whitespace-tolerant, dupes dropped, engine order kept.
    names, err = dfb.parse_excluded_classes(" Marmot , ibex,  WILD  BOAR ,ibex")
    assert err is None
    assert names == ["ibex", "marmot", "wild boar"]  # engine index order
    assert dfb.parse_excluded_classes("") == ([], None)
    assert dfb.parse_excluded_classes("  ,  ") == ([], None)
    names, err = dfb.parse_excluded_classes("ibex,dragon,unicorn")
    assert names is None
    assert "dragon" in err and "unicorn" in err and "choose from" in err


def test_validate_common_rejects_unknown_excluded_class():
    args = _args(exclude_classes="wolf,dragon")
    err = dfb.validate_common(args)
    assert err is not None and "--exclude-classes" in err and "dragon" in err
    assert dfb.validate_common(_args(exclude_classes="wolf, lynx")) is None


def test_run_metadata_records_excluded_classes(tmp_path):
    sw = tmp_path / "sw"
    sw.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    args = _args(exclude_classes="marmot,ibex", partition=0)
    _path, meta = dfb.write_run_metadata(str(out), str(sw), "/root", args)
    assert meta["excluded_classes"] == ["ibex", "marmot"]


def test_animal_classes_en_matches_engine():
    """The stdlib copy must track classifTools exactly (skips without torch)."""
    classifTools = pytest.importorskip("classifTools")
    assert dfb.ANIMAL_CLASSES_EN == list(classifTools.txt_animalclasses["en"])


# ---------------------------------------------------------------------------
# per-image CSV rows (build_rows against a stub predictor; no torch needed)
# ---------------------------------------------------------------------------
class _StubPredictor:
    """Minimal stand-in exposing the getters build_rows reads.

    Image 0: a confident wolf. Image 1: below-threshold (prediction rewritten
    to "undefined" but top1 kept). Image 2: empty (no animal crop, NaN scores).
    """

    def __init__(self):
        nan = float("nan")
        self._files = ["/a/1.jpg", "/a/2.jpg", "/a/3.jpg"]
        self._seq = (["wolf", "undefined", "empty"],
                     [0.97, 0.41, 1.0], None, [1, 1, 0])
        self._top1 = ["wolf", "lynx", "empty"]
        self._img = (["wolf", "undefined", "empty"], [0.95, 0.4, 1.0],
                     ["wolf", "lynx", "empty"])
        self._detconf = [[0.91, 0.0, 0.0], [0.52, 0.0, 0.0], [0.0, 0.0, 0.0]]
        self._scores = [[0.95, 0.05], [0.4, 0.6], [nan, nan]]
        self._bird = [[0.7, 0.3], [0.5, 0.5], [nan, nan]]

    def getPredictions(self):
        return self._seq

    def getPredictedTop1(self):
        return self._top1

    def getPredictionsBaseAll(self):
        return self._img

    def getDetectionConfs(self):
        return self._detconf

    def getClassScores(self):
        return self._scores, self._bird

    def getDates(self):
        return ["d1", "d2", "d3"]

    def getSeqnums(self):
        return [1, 2, 3]

    def getFilenames(self):
        return self._files

    def getHumanCount(self):
        return [0, 0, 0]


def test_build_rows_full_schema():
    rows = dfb.build_rows(_StubPredictor())
    header = dfb.full_csv_header(["wolf", "lynx"], ["corvid", "raptor"])
    assert all(len(r) == len(header) for r in rows)
    r0, r1, r2 = (dict(zip(header, r)) for r in rows)
    # Above threshold: prediction equals top1.
    assert (r0["prediction_seq"], r0["top1_seq"], r0["above_threshold"]) == \
        ("wolf", "wolf", "yes")
    assert r0["score_wolf"] == "0.9500"
    assert r0["det_conf_animal"] == "0.9100"
    # Below threshold: label preserved in top1, flagged "no", scores kept.
    assert (r1["prediction_seq"], r1["top1_seq"], r1["above_threshold"]) == \
        ("undefined", "lynx", "no")
    assert r1["score_lynx"] == "0.6000"
    assert r1["birdscore_corvid"] == "0.5000"
    # Empty image: no crop, so class-score cells are blank, det confs zero.
    assert (r2["above_threshold"], r2["score_wolf"], r2["birdscore_raptor"]) == \
        ("yes", "", "")
    assert r2["det_conf_animal"] == "0.0000"


def test_build_rows_without_bird_head():
    stub = _StubPredictor()
    stub._bird = None
    stub.getClassScores = lambda: (stub._scores, None)
    rows = dfb.build_rows(stub)
    header = dfb.full_csv_header(["wolf", "lynx"])
    assert all(len(r) == len(header) for r in rows)


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


# ---------------------------------------------------------------------------
# self-protection guards and the telemetry flight recorder
# ---------------------------------------------------------------------------
def test_parse_meminfo():
    text = "MemTotal:  16000000 kB\nMemAvailable:  1048576 kB\nShmem: 2048 kB\nbadline\n"
    info = dfb.parse_meminfo(text)
    assert info["MemAvailable"] == 1048576 * 1024
    assert info["Shmem"] == 2048 * 1024


def test_memory_pressure():
    low, avail = dfb.memory_pressure({"MemAvailable": 1 * dfb.GIB}, 1.5)
    assert low and abs(avail - 1.0) < 1e-6
    low, _ = dfb.memory_pressure({"MemAvailable": 4 * dfb.GIB}, 1.5)
    assert not low
    # Unknown meminfo must never trigger a protective stop.
    assert dfb.memory_pressure({}, 1.5) == (False, None)


def test_shard_reads_failing():
    # Mass failure: many failures AND most of the shard failing.
    assert dfb.shard_reads_failing(60, 80)
    # A sprinkling of corrupt files in a big shard: keep going.
    assert not dfb.shard_reads_failing(30, 5000)
    # A few failures at the very start: below the absolute floor.
    assert not dfb.shard_reads_failing(8, 10)
    assert not dfb.shard_reads_failing(0, 0)


def test_telemetry_sample_and_append(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    sample = dfb.telemetry_sample("running", 123, 4.567, str(out), str(root))
    assert sample["state"] == "running"
    assert sample["done"] == 123
    assert sample["rate_img_s"] == 4.57
    assert sample["root_ok"] == 1
    assert set(sample) == set(dfb.TELEMETRY_FIELDS)
    # A vanished root is recorded as 0 - the drive-disconnect fingerprint.
    gone = dfb.telemetry_sample("running", 1, 0.0, str(out), str(root / "nope"))
    assert gone["root_ok"] == 0

    dfb.append_telemetry(str(out), 0, sample)
    dfb.append_telemetry(str(out), 0, gone)
    with open(out / "telemetry.p0.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2  # header written once
    assert rows[1]["root_ok"] == "0"


def test_append_telemetry_rotates(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(dfb, "TELEMETRY_MAX_BYTES", 100)
    sample = {k: "x" for k in dfb.TELEMETRY_FIELDS}
    for _ in range(6):
        dfb.append_telemetry(str(out), 0, sample)
    assert (out / "telemetry.p0.csv.old").exists()
    # The fresh generation starts with a header again.
    first = (out / "telemetry.p0.csv").read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith("time,")


# ---------------------------------------------------------------------------
# pidfile single-instance guard (service vs manual runs)
# ---------------------------------------------------------------------------
def test_acquire_pidfile(tmp_path):
    out = str(tmp_path)
    # No pidfile yet: claim it.
    assert dfb.acquire_pidfile(out, pid=111, alive_check=lambda p: False)
    assert (tmp_path / "dfrun.worker.pid").read_text().strip() == "111"
    # A live worker holds it: refuse.
    assert not dfb.acquire_pidfile(out, pid=222, alive_check=lambda p: True)
    assert (tmp_path / "dfrun.worker.pid").read_text().strip() == "111"
    # The holder is dead: claim it over the stale entry.
    assert dfb.acquire_pidfile(out, pid=333, alive_check=lambda p: False)
    assert (tmp_path / "dfrun.worker.pid").read_text().strip() == "333"
    # Re-claiming our own pidfile is fine even if "alive".
    assert dfb.acquire_pidfile(out, pid=333, alive_check=lambda p: True)


def test_guard_defaults_keep_going_philosophy():
    args = _args()
    assert args.min_avail_gib == 0      # never stop for memory by default
    assert args.min_disk_gib == 0.5     # stop only when writes would fail anyway


# ---------------------------------------------------------------------------
# --resume-last: restore the current run's own settings from its metadata
# ---------------------------------------------------------------------------
def test_apply_resume_metadata_restores_everything():
    args = _args()
    metadata = {
        "detector": "DFbsMDS", "threshold": 0.8, "maxlag": 60, "birds": False,
        "lang": "en", "batch_size": 8, "threads": 3, "merge_every": 600,
        "heartbeat_secs": 30, "root": "/media/rim/My Book1",
        "excluded_classes": ["genet", "ibex"],
    }
    applied = dfb.apply_resume_metadata(args, metadata)
    assert args.detector == "DFbsMDS"
    assert args.threshold == 0.8
    assert args.maxlag == 60
    assert args.birds is False
    assert args.threads == 3
    assert args.root == "/media/rim/My Book1"
    assert args.exclude_classes == "genet,ibex"
    assert ("detector", "DFbsMDS") in applied


def test_apply_resume_metadata_tolerates_old_schema():
    # Metadata from an older version lacks threads/merge_every: keep defaults.
    args = _args()
    default_threads = args.threads
    dfb.apply_resume_metadata(args, {"detector": "MDS"})
    assert args.detector == "MDS"
    assert args.threads == default_threads


def test_resume_last_without_metadata_exits_cleanly(tmp_path):
    # Exit 0, not an error: a supervisor must not restart-loop on "nothing
    # to resume".
    rc = dfb.main(["--out-dir", str(tmp_path), "--resume-last"])
    assert rc == 0


def test_run_metadata_records_resume_fields(tmp_path):
    sw = tmp_path / "sw"
    sw.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    args = _args(threads=3, merge_every=600, heartbeat_secs=30)
    _path, meta = dfb.write_run_metadata(str(out), str(sw), "/root", args)
    # Every field a resume needs is recorded.
    for field in dfb.RESUME_FIELDS:
        if field in ("root",):
            continue
        assert field in meta, field
    assert meta["threads"] == 3
    assert meta["heartbeat_secs"] == 30


# ---------------------------------------------------------------------------
# derived columns: globally unique sequence_id and re-rooted paths
# ---------------------------------------------------------------------------
def test_shard_relpath_from_row_recovers_the_prefix(tmp_path):
    # A shard CSV name embeds sha1(relpath)[:8], so the archive-relative path
    # can be recovered from an absolute path under ANY mount name.
    root = "/media/rim/My Book1"
    leaf = root + "/Camera Trap Monitoring/SiteA/101_BTCF"
    name = dfb.shard_csv_name(root, leaf)
    rel = "Camera Trap Monitoring/SiteA/101_BTCF"
    assert dfb.shard_relpath_from_row(name, leaf) == rel
    # The same shard recorded under a different mount name resolves identically.
    assert dfb.shard_relpath_from_row(
        name, "/media/rim/My Book2/Camera Trap Monitoring/SiteA/101_BTCF"
    ) == rel
    # An unrelated directory has no matching split: leave the path alone.
    assert dfb.shard_relpath_from_row(name, "/somewhere/else") is None
    assert dfb.shard_relpath_from_row("nothash.csv", leaf) is None


def test_sequence_id_and_rerooted_path():
    assert dfb.sequence_id("0a1b2c3d", "471") == "0a1b2c3d-471"
    assert dfb.sequence_id("0a1b2c3d", "") == ""      # no seqnum recorded
    assert dfb.sequence_id(None, "471") == ""         # not a shard CSV
    assert dfb.rerooted_path(
        "/media/rim/My Book2/CTM/SiteA/IMG_1.JPG", "/media/rim/My Book1", "CTM/SiteA"
    ) == "/media/rim/My Book1/CTM/SiteA/IMG_1.JPG"
    # Missing pieces leave the path untouched.
    assert dfb.rerooted_path("/x/1.jpg", None, "rel") == "/x/1.jpg"
    assert dfb.rerooted_path("/x/1.jpg", "/root", None) == "/x/1.jpg"


def test_merge_adds_unique_sequence_ids_and_normalises_paths(tmp_path):
    """The real defect: two folders each numbering their sequences from 1, and
    the same archive recorded under two mount names after a remount."""
    out = tmp_path / "out"
    out.mkdir()
    root1 = "/media/rim/My Book1"
    root2 = "/media/rim/My Book2"
    leaf_a = "CTM/SiteA/100_BTCF"
    leaf_b = "CTM/SiteB/101_BTCF"
    name_a = dfb.shard_csv_name(root1, f"{root1}/{leaf_a}")
    name_b = dfb.shard_csv_name(root1, f"{root1}/{leaf_b}")
    # Shard A was classified before the remount, shard B after it.
    _write_shard_csv(out / name_a, [
        [f"{root1}/{leaf_a}/1.jpg", "d", 1, "wolf", "wolf", 0.9, "yes",
         "wolf", "wolf", 0.9, 1, 0, "0.9", "0", "0"],
        [f"{root1}/{leaf_a}/2.jpg", "d", 2, "lynx", "lynx", 0.8, "yes",
         "lynx", "lynx", 0.8, 1, 0, "0.8", "0", "0"],
    ])
    _write_shard_csv(out / name_b, [
        [f"{root2}/{leaf_b}/9.jpg", "d", 1, "bear", "bear", 0.95, "yes",
         "bear", "bear", 0.95, 1, 0, "0.9", "0", "0"],
    ])

    merge_out = out / "master.csv"
    dfb.merge_csvs(str(out), str(merge_out), canonical_root=root1)
    with open(merge_out, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    # Every path now sits under one root, despite the mixed mount names.
    assert all(r["filename"].startswith(root1 + "/") for r in rows)
    assert f"{root1}/{leaf_b}/9.jpg" in [r["filename"] for r in rows]
    # seqnum collides across shards; sequence_id does not.
    assert [r["seqnum"] for r in rows].count("1") == 2
    ids = [r[dfb.SEQUENCE_ID_COLUMN] for r in rows]
    assert len(set(ids)) == 3
    # The id ties a row to its shard, and to its sequence within that shard.
    assert ids[0].endswith("-1") and ids[1].endswith("-2")
    assert ids[0].split("-")[0] == ids[1].split("-")[0]      # same shard
    assert ids[2].split("-")[0] != ids[0].split("-")[0]      # different shard


def test_merge_without_canonical_root_leaves_paths_alone(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    _write_shard_csv(out / "a__0a1b2c3d.csv",
                     [["/media/x/My Book9/CTM/1.jpg", "d", 7, "wolf", "wolf",
                       0.9, "yes", "wolf", "wolf", 0.9, 1, 0, "0.9", "0", "0"]])
    merge_out = out / "master.csv"
    dfb.merge_csvs(str(out), str(merge_out))
    with open(merge_out, newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["filename"] == "/media/x/My Book9/CTM/1.jpg"   # untouched
    assert row[dfb.SEQUENCE_ID_COLUMN] == "0a1b2c3d-7"        # still derived


# ---------------------------------------------------------------------------
# detector benchmark (pure parts; the timing itself needs torch)
# ---------------------------------------------------------------------------
import benchmark_detectors as bench  # noqa: E402


def test_human_duration():
    assert bench.human_duration(45) == "45 sec"
    assert bench.human_duration(600) == "10 min"
    assert bench.human_duration(9 * 3600) == "9.0 hours"
    assert bench.human_duration(4.2 * 86400) == "4.2 days"
    assert bench.human_duration(None) == "unknown"
    assert bench.human_duration(-1) == "unknown"


def test_extrapolate():
    assert bench.extrapolate(5.0, 1_845_397) == pytest.approx(369079.4, rel=1e-4)
    assert bench.extrapolate(0, 100) is None
    assert bench.extrapolate(5.0, 0) is None


def test_sample_from_master_is_representative(tmp_path):
    master = tmp_path / "master.csv"
    with open(master, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(dfb.CSV_HEADER)
        for i in range(1000):
            row = ["x"] * len(dfb.CSV_HEADER)
            row[0] = f"/archive/cam{i % 7}/IMG_{i:04d}.JPG"
            w.writerow(row)
    picked, seen = bench.sample_from_master(str(master), 50, seed=3)
    assert seen == 1000
    assert len(picked) == 50
    assert len(set(picked)) == 50                      # no duplicates
    assert all(p.startswith("/archive/") for p in picked)
    # Deterministic for a given seed, so a benchmark can be repeated exactly.
    assert bench.sample_from_master(str(master), 50, seed=3)[0] == picked
    assert bench.sample_from_master(str(master), 50, seed=4)[0] != picked


def test_format_table_reports_relative_cost():
    rows = [
        {"detector": "DFbsMDS", "rate": 5.0, "load_s": 12.0, "animal_frac": 0.20},
        {"detector": "DFMDS", "rate": 2.5, "load_s": 13.0, "animal_frac": 0.24},
        {"detector": "MDR", "error": "out of memory"},
    ]
    out = bench.format_table(rows, 1_845_397, baseline="DFbsMDS")
    assert "DFbsMDS" in out and "DFMDS" in out
    assert "2.00x" in out          # DFMDS is twice the cost of the baseline
    assert "4.3 days" in out       # 1,845,397 / 5 img/s
    assert "8.5 days" in out       # 1,845,397 / 2.5 img/s
    assert "failed" in out and "out of memory" in out
