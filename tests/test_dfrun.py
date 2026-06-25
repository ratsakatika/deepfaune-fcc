"""Unit tests for dfrun's pre-flight and pure logic.

These cover source detection, the read-only check, the swap check, done and
remaining counting with the cache, the self-update decision (with mocked git
calls, never the network), and the spreadsheet output writer. The interactive
prompts, the rich readout, psutil sampling and the detached launch are not
exercised here.
"""

import csv
import os

import pytest

import dfrun


@pytest.fixture(autouse=True)
def _clear_updated_env(monkeypatch):
    # The update path sets DFRUN_UPDATED in the real environment; make sure no
    # test starts with it set.
    monkeypatch.delenv("DFRUN_UPDATED", raising=False)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def _write_shard(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(dfrun.dfb.CSV_HEADER)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# source detection and read-only check (Stage 2)
# ---------------------------------------------------------------------------
def test_looks_like_archive_marker_dir(tmp_path):
    (tmp_path / "Camera Trap Monitoring").mkdir()
    assert dfrun.looks_like_archive(str(tmp_path))


def test_looks_like_archive_wtm_far(tmp_path):
    (tmp_path / "WTM_FAR_001").mkdir()
    assert dfrun.looks_like_archive(str(tmp_path))


def test_looks_like_archive_negative(tmp_path):
    (tmp_path / "random").mkdir()
    assert not dfrun.looks_like_archive(str(tmp_path))


def test_is_read_only():
    assert dfrun.is_read_only("ro,relatime,nosuid")
    assert not dfrun.is_read_only("rw,relatime")
    assert not dfrun.is_read_only("")


# ---------------------------------------------------------------------------
# swap (A7, Stage 4)
# ---------------------------------------------------------------------------
def test_needs_swap():
    assert dfrun.needs_swap(2 * 1024 ** 3)
    assert not dfrun.needs_swap(16 * 1024 ** 3)
    assert not dfrun.needs_swap(32 * 1024 ** 3)


def test_create_swapfile_runs_steps_and_handles_failure():
    calls = []
    ok = dfrun.create_swapfile(8, path="/swapfile", runner=lambda cmd: calls.append(cmd) or True)
    assert ok
    assert any("mkswap" in cmd for cmd in calls)
    assert any("swapon" in cmd for cmd in calls)
    # A failing step aborts and returns False.
    assert not dfrun.create_swapfile(8, runner=lambda cmd: False)


# ---------------------------------------------------------------------------
# work assessment and cache (Stage 3)
# ---------------------------------------------------------------------------
def test_get_total_count_caches_and_invalidates(tmp_path):
    source = tmp_path / "drive"
    _touch(source / "camA" / "1.jpg")
    _touch(source / "camB" / "1.jpg")
    _touch(source / "camB" / "2.jpg")
    out = tmp_path / "out"
    out.mkdir()

    total, shards, cached = dfrun.get_total_count(str(out), str(source), "UUID1", rescan=False)
    assert (total, shards, cached) == (3, 2, False)
    # Second call hits the cache.
    total, shards, cached = dfrun.get_total_count(str(out), str(source), "UUID1", rescan=False)
    assert (total, shards, cached) == (3, 2, True)
    # A different drive UUID invalidates the cache.
    _, _, cached = dfrun.get_total_count(str(out), str(source), "UUID2", rescan=False)
    assert cached is False
    # --rescan forces a fresh count even with a valid cache.
    _, _, cached = dfrun.get_total_count(str(out), str(source), "UUID2", rescan=True)
    assert cached is False


def test_assess_work_counts_done_and_remaining(tmp_path):
    source = tmp_path / "drive"
    _touch(source / "camA" / "1.jpg")
    _touch(source / "camA" / "2.jpg")
    _touch(source / "camB" / "1.jpg")
    out = tmp_path / "out"
    out.mkdir()
    # One shard already classified (two rows).
    _write_shard(
        out / "done__0a1b2c3d.csv",
        [["/x/1.jpg", "d", 1, "wolf", 0.9, "wolf", 0.9, 1, 0],
         ["/x/2.jpg", "d", 1, "empty", 1.0, "empty", 1.0, 0, 0]],
    )
    done, remaining, total, shards, pct, _cached = dfrun.assess_work(
        str(out), str(source), "UUID1", rescan=False
    )
    assert done == 2
    assert total == 3
    assert remaining == 1
    assert round(pct, 1) == 66.7


# ---------------------------------------------------------------------------
# self-update decision (Stage 0) - the four required scenarios
# ---------------------------------------------------------------------------
def test_decide_update_scenarios():
    assert dfrun.decide_update(True, False, False, True, True, 3)[0] == "update"
    assert dfrun.decide_update(True, True, False, True, True, 3)[0] == "refuse"   # dirty
    assert dfrun.decide_update(True, False, True, True, True, 3)[0] == "refuse"   # worker
    assert dfrun.decide_update(False, False, False, True, True, 3)[0] == "skip"   # unreachable
    assert dfrun.decide_update(True, False, False, False, True, 3)[0] == "refuse"  # wrong branch
    assert dfrun.decide_update(True, False, False, True, False, 3)[0] == "refuse"  # diverged
    assert dfrun.decide_update(True, False, False, True, True, 0)[0] == "current"  # up to date


def _mock_git(monkeypatch, *, fetch, dirty, branch, ff, behind, pull_ok=True):
    monkeypatch.setattr(dfrun, "load_config", lambda: {})
    monkeypatch.setattr(dfrun, "git_fetch", lambda *a, **k: fetch)
    monkeypatch.setattr(dfrun, "git_is_dirty", lambda repo: dirty)
    monkeypatch.setattr(dfrun, "git_current_branch", lambda repo: branch)
    monkeypatch.setattr(dfrun, "git_rev", lambda repo, ref: "newhash" if "origin" in ref else "oldhash")
    monkeypatch.setattr(dfrun, "git_ff_possible", lambda repo, a, b: ff)
    monkeypatch.setattr(dfrun, "git_behind_count", lambda repo, a, b: behind)
    monkeypatch.setattr(dfrun, "git_incoming_subjects", lambda repo, a, b: ["abc fix"])

    class _Result:
        returncode = 0 if pull_ok else 1
        stderr = "" if pull_ok else "boom"

    monkeypatch.setattr(dfrun, "run_git", lambda repo, args, timeout=60: _Result())


def test_self_update_applies_when_behind(monkeypatch, tmp_path):
    _mock_git(monkeypatch, fetch=True, dirty=False, branch=dfrun.UPDATE_BRANCH, ff=True, behind=2)
    execv_calls = []
    monkeypatch.setattr(dfrun.os, "execv", lambda exe, argv: execv_calls.append((exe, argv)))
    args = dfrun.build_arg_parser().parse_args([])
    result = dfrun.self_update_check(args, str(tmp_path), worker_is_running=False, prompt=lambda m: "y")
    assert result is True            # it re-exec'd
    assert execv_calls               # os.execv was called


def test_self_update_refuses_when_dirty(monkeypatch, tmp_path):
    _mock_git(monkeypatch, fetch=True, dirty=True, branch=dfrun.UPDATE_BRANCH, ff=True, behind=2)
    monkeypatch.setattr(dfrun.os, "execv", lambda *a: pytest.fail("must not update"))
    args = dfrun.build_arg_parser().parse_args([])
    assert dfrun.self_update_check(args, str(tmp_path), worker_is_running=False, prompt=lambda m: "y") is False


def test_self_update_refuses_when_worker_running(monkeypatch, tmp_path):
    _mock_git(monkeypatch, fetch=True, dirty=False, branch=dfrun.UPDATE_BRANCH, ff=True, behind=2)
    monkeypatch.setattr(dfrun.os, "execv", lambda *a: pytest.fail("must not update"))
    args = dfrun.build_arg_parser().parse_args([])
    assert dfrun.self_update_check(args, str(tmp_path), worker_is_running=True, prompt=lambda m: "y") is False


def test_self_update_continues_when_unreachable(monkeypatch, tmp_path, capsys):
    _mock_git(monkeypatch, fetch=False, dirty=False, branch=dfrun.UPDATE_BRANCH, ff=True, behind=0)
    monkeypatch.setattr(dfrun.os, "execv", lambda *a: pytest.fail("must not update"))
    args = dfrun.build_arg_parser().parse_args([])
    assert dfrun.self_update_check(args, str(tmp_path), worker_is_running=False, prompt=lambda m: "y") is False
    assert "remote unreachable" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# single instance / launch helpers (A8, A10)
# ---------------------------------------------------------------------------
def test_pidfile_round_trip_and_worker_running(tmp_path):
    out = tmp_path
    assert dfrun.read_pidfile(str(out)) is None
    dfrun.write_pidfile(str(out), 999_999)  # almost certainly not alive
    assert dfrun.read_pidfile(str(out)) == 999_999
    assert not dfrun.worker_running(str(out))
    # This very process is alive but is not a deepfaune_batch worker.
    dfrun.write_pidfile(str(out), os.getpid())
    assert not dfrun.worker_running(str(out))


def test_build_worker_command_includes_flags():
    params = {
        "detector": "DFbsMDS", "birds": True, "threshold": 0.8, "maxlag": 10,
        "batch_size": 8, "threads": 4, "merge_every": 600,
    }
    cmd = dfrun.build_worker_command("/sw", "/src", "/out", params)
    assert "--birds" in cmd
    assert "deepfaune_batch.py" in " ".join(cmd)
    assert "--detector" in cmd and "DFbsMDS" in cmd
    # birds off must drop the flag.
    params["birds"] = False
    assert "--birds" not in dfrun.build_worker_command("/sw", "/src", "/out", params)


# ---------------------------------------------------------------------------
# statistics and completion outputs (Stage 6, Stage 7, A11)
# ---------------------------------------------------------------------------
def test_station_from_path():
    assert dfrun.station_from_path("/media/My Book/SiteA/Cam1/IMG.JPG") == "SiteA/Cam1"
    assert dfrun.station_from_path("IMG.JPG") == "(unknown)"


def test_is_excluded_label():
    for label in ["empty", "human", "vehicle", "undefined", "bird undefined"]:
        assert dfrun.is_excluded_label(label)
    for label in ["wolf", "lynx", "bird passerine"]:
        assert not dfrun.is_excluded_label(label)


def test_stats_accumulator_incremental(tmp_path):
    out = tmp_path
    _write_shard(
        out / "a__0a1b2c3d.csv",
        [["/s/SiteA/Cam1/1.jpg", "2024:01:01 00:00:00", 1, "wolf", 0.9, "wolf", 0.9, 1, 0],
         ["/s/SiteA/Cam1/2.jpg", "2024:01:01 00:00:05", 1, "empty", 1.0, "empty", 1.0, 0, 0],
         ["/s/SiteA/Cam1/3.jpg", "1970:01:01 00:00:00", 1, "human", 1.0, "human", 1.0, 0, 1]],
    )
    stats = dfrun.StatsAccumulator()
    stats.consume_new(str(out))
    assert stats.total == 3
    assert stats.empty == 1
    assert stats.unset_clock == 1
    assert dict(stats.species) == {"wolf": 1}  # empty/human excluded
    assert round(stats.blank_pct(), 1) == 33.3
    # Re-consuming does not double count.
    stats.consume_new(str(out))
    assert stats.total == 3


def test_write_wildlife_and_summary(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    desktop = tmp_path / "Desktop"
    _write_shard(
        out / "a__0a1b2c3d.csv",
        [["/s/SiteA/Cam1/1.jpg", "d", 1, "wolf", 0.9, "wolf", 0.9, 1, 0],
         ["/s/SiteA/Cam1/2.jpg", "d", 1, "empty", 1.0, "empty", 1.0, 0, 0],
         ["/s/SiteB/Cam2/3.jpg", "d", 1, "lynx", 0.8, "lynx", 0.8, 1, 0],
         ["/s/SiteB/Cam2/4.jpg", "d", 1, "human", 1.0, "human", 1.0, 0, 1]],
    )
    rows, capped, n_species, n_stations = dfrun.write_wildlife_and_summary(str(out), str(desktop))
    assert rows == 2          # wolf and lynx only (empty, human excluded)
    assert not capped
    assert n_species == 2
    assert n_stations == 2

    with open(desktop / dfrun.DESKTOP_WILDLIFE, newline="", encoding="utf-8") as handle:
        wildlife = list(csv.reader(handle))
    assert wildlife[0] == dfrun.dfb.CSV_HEADER + ["station"]
    labels = {row[3] for row in wildlife[1:]}
    assert labels == {"wolf", "lynx"}

    with open(desktop / dfrun.DESKTOP_SUMMARY, newline="", encoding="utf-8") as handle:
        summary = list(csv.reader(handle))
    assert summary[0] == ["group", "name", "count"]
    groups = {row[0] for row in summary[1:]}
    assert groups == {"species", "station"}


def test_absolute_eta():
    from datetime import datetime
    when = datetime(2026, 6, 27, 2, 14, 0)
    assert dfrun.absolute_eta(3600, now=when) == "finishes Sat 27 Jun, 03:14"
    assert dfrun.absolute_eta(0) == "finishes: unknown"


# ---------------------------------------------------------------------------
# dashboard integration
# ---------------------------------------------------------------------------
def test_ensure_desktop_master_regenerates(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    monkeypatch.setattr(dfrun, "DESKTOP_DIR", str(desktop))
    _write_shard(
        out / "a__0a1b2c3d.csv",
        [["/x/1.jpg", "d", 1, "wolf", 0.9, "wolf", 0.9, 1, 0]],
    )
    master = dfrun.ensure_desktop_master(str(out))
    assert master == str(desktop / dfrun.DESKTOP_MASTER)
    assert os.path.exists(master)
    with open(master, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == dfrun.dfb.CSV_HEADER
    assert any(r[3] == "wolf" for r in rows[1:])


def test_dashboard_command_builds_expected_invocation(tmp_path, monkeypatch):
    software = tmp_path / "sw"
    software.mkdir()
    (software / dfrun.DASHBOARD_BUILDER).write_text("# builder\n", encoding="utf-8")
    (software / dfrun.PROTOCOL_NAME).write_text("xlsx", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    monkeypatch.setattr(dfrun, "DESKTOP_DIR", str(desktop))
    _write_shard(
        out / "a__0a1b2c3d.csv",
        [["/x/1.jpg", "d", 1, "wolf", 0.9, "wolf", 0.9, 1, 0]],
    )
    cmd, out_html = dfrun.dashboard_command(str(software), str(out))
    assert cmd is not None
    assert cmd[1] == str(software / dfrun.DASHBOARD_BUILDER)
    assert cmd[2] == str(desktop / dfrun.DESKTOP_MASTER)   # detections = Desktop master
    assert cmd[3] == str(software / dfrun.PROTOCOL_NAME)
    assert cmd[4] == str(desktop / dfrun.DESKTOP_DASHBOARD)
    assert out_html == str(desktop / dfrun.DESKTOP_DASHBOARD)


def test_dashboard_command_missing_builder(tmp_path, monkeypatch):
    monkeypatch.setattr(dfrun, "DESKTOP_DIR", str(tmp_path / "Desktop"))
    out = tmp_path / "out"
    out.mkdir()
    _write_shard(out / "a__0a1b2c3d.csv", [["/x/1.jpg", "d", 1, "wolf", 0.9, "wolf", 0.9, 1, 0]])
    cmd, out_html = dfrun.dashboard_command(str(tmp_path / "no_software"), str(out))
    assert cmd is None and out_html is None


def test_build_dashboard_missing_builder_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(dfrun, "DESKTOP_DIR", str(tmp_path / "Desktop"))
    assert dfrun.build_dashboard(str(tmp_path / "no_software"), str(tmp_path)) is None


def test_maybe_rebuild_dashboard_guards(tmp_path, monkeypatch):
    # No change in mtime: no build, returns the same handle.
    monkeypatch.setattr(dfrun, "spawn_dashboard", lambda *a: pytest.fail("should not build"))
    assert dfrun._maybe_rebuild_dashboard("/sw", str(tmp_path), 100.0, 100.0, None) is None
    # No software dir: no build.
    assert dfrun._maybe_rebuild_dashboard(None, str(tmp_path), 200.0, 100.0, None) is None
    # Low memory: no build.
    monkeypatch.setattr(dfrun, "enough_memory_for_dashboard", lambda: False)
    assert dfrun._maybe_rebuild_dashboard("/sw", str(tmp_path), 200.0, 100.0, None) is None
