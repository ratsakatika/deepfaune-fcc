"""Unit tests for the model-free logic in deepfaune_batch.

These cover directory sharding, junk filtering, the atomic temp-then-rename CSV
write, the skip-if-CSV-exists resume behaviour and partition selection. None of
them need torch or the model weights, so they run anywhere.
"""

import csv
import os
import re

import pytest

import deepfaune_batch as dfb


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


def test_progress_summary_runs():
    progress = dfb.Progress(100)
    progress.add(25)
    summary = progress.summary()
    assert "25/100" in summary
    assert "ETA" in summary
