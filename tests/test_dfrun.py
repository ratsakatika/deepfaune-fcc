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


GIB = 1024 ** 3
_MISSING = lambda path: (False, 0, False)  # noqa: E731  (no swapfile on disk)
_PLENTY = lambda path: 100 * GIB  # noqa: E731  (lots of free disk)


def test_create_swapfile_runs_steps_and_handles_failure(monkeypatch):
    monkeypatch.setattr(dfrun, "fstab_has_swapfile", lambda path: False)
    calls = []
    ok = dfrun.create_swapfile(
        8, path="/swapfile", runner=lambda cmd: calls.append(cmd) or True,
        status=_MISSING, free_bytes=_PLENTY,
    )
    assert ok
    assert any("fallocate" in cmd for cmd in calls)
    assert any("mkswap" in cmd for cmd in calls)
    assert any("swapon" in cmd for cmd in calls)
    # Persisted so a reboot does not orphan the file as dead disk weight.
    assert any("/etc/fstab" in " ".join(cmd) for cmd in calls)
    # A failing step aborts and returns False.
    assert not dfrun.create_swapfile(
        8, runner=lambda cmd: False, status=_MISSING, free_bytes=_PLENTY
    )


def test_create_swapfile_refuses_when_disk_would_fill(monkeypatch):
    monkeypatch.setattr(dfrun, "fstab_has_swapfile", lambda path: False)
    calls = []
    # 12 GiB wanted, 13 GiB free: would leave under the 5 GiB safety margin.
    ok = dfrun.create_swapfile(
        12, runner=lambda cmd: calls.append(cmd) or True,
        status=_MISSING, free_bytes=lambda path: 13 * GIB,
    )
    assert not ok
    assert calls == []  # refused before touching the system


def test_create_swapfile_reuses_inactive_file(monkeypatch):
    monkeypatch.setattr(dfrun, "fstab_has_swapfile", lambda path: False)
    calls = []
    # A 16 GiB file from a previous boot, currently inactive; want 12 GiB.
    ok = dfrun.create_swapfile(
        12, runner=lambda cmd: calls.append(cmd) or True,
        status=lambda path: (True, 16 * GIB, False),
        free_bytes=lambda path: 1 * GIB,  # nearly full disk: reuse needs none
    )
    assert ok
    assert not any("fallocate" in cmd for cmd in calls)  # no new allocation
    assert any("swapon" in cmd for cmd in calls)


def test_create_swapfile_already_active_only_persists(monkeypatch):
    monkeypatch.setattr(dfrun, "fstab_has_swapfile", lambda path: True)
    calls = []
    ok = dfrun.create_swapfile(
        8, runner=lambda cmd: calls.append(cmd) or True,
        status=lambda path: (True, 16 * GIB, True), free_bytes=_PLENTY,
    )
    assert ok
    assert calls == []  # active, big enough and persisted: nothing to do


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
    # Up to date wins over every refusal: no "update available" lie while a
    # worker runs (the bug this ordering fixes), nor for dirty/wrong-branch.
    assert dfrun.decide_update(True, False, True, True, True, 0)[0] == "current"   # worker + current
    assert dfrun.decide_update(True, True, False, False, True, 0)[0] == "current"  # dirty, off-branch + current
    # Diverged with behind=0 is NOT "current": remote history was rewritten.
    assert dfrun.decide_update(True, False, False, True, False, 0)[0] == "refuse"


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


def test_self_update_applies_then_exits(monkeypatch, tmp_path, capsys):
    # A successful update must EXIT so the user relaunches on the new code,
    # never continue this process (old code in memory, new checkout on disk).
    _mock_git(monkeypatch, fetch=True, dirty=False, branch=dfrun.UPDATE_BRANCH, ff=True, behind=2)
    args = dfrun.build_arg_parser().parse_args([])
    with pytest.raises(SystemExit) as exc:
        dfrun.self_update_check(args, str(tmp_path), worker_is_running=False, prompt=lambda m: "y")
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Updated" in out and "run dfrun again" in out


def test_self_update_refuses_when_dirty(monkeypatch, tmp_path):
    _mock_git(monkeypatch, fetch=True, dirty=True, branch=dfrun.UPDATE_BRANCH, ff=True, behind=2)
    args = dfrun.build_arg_parser().parse_args([])
    assert dfrun.self_update_check(args, str(tmp_path), worker_is_running=False, prompt=lambda m: "y") is False


def test_self_update_refuses_when_worker_running(monkeypatch, tmp_path):
    _mock_git(monkeypatch, fetch=True, dirty=False, branch=dfrun.UPDATE_BRANCH, ff=True, behind=2)
    args = dfrun.build_arg_parser().parse_args([])
    assert dfrun.self_update_check(args, str(tmp_path), worker_is_running=True, prompt=lambda m: "y") is False


def test_self_update_continues_when_unreachable(monkeypatch, tmp_path, capsys):
    _mock_git(monkeypatch, fetch=False, dirty=False, branch=dfrun.UPDATE_BRANCH, ff=True, behind=0)
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
    # excluded classes pass through as one comma-separated flag value.
    assert "--exclude-classes" not in dfrun.build_worker_command("/sw", "/src", "/out", params)
    params["exclude_classes"] = ["ibex", "marmot"]
    cmd = dfrun.build_worker_command("/sw", "/src", "/out", params)
    assert cmd[cmd.index("--exclude-classes") + 1] == "ibex,marmot"


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
    assert wildlife[0] == (
        dfrun.dfb.CSV_HEADER + [dfrun.dfb.SEQUENCE_ID_COLUMN, "station"]
    )
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
# desktop master
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
    assert rows[0] == dfrun.dfb.CSV_HEADER + [dfrun.dfb.SEQUENCE_ID_COLUMN]
    assert any(r[3] == "wolf" for r in rows[1:])


def test_parse_detector():
    assert dfrun.parse_detector("dfbsmds") == ("DFbsMDS", True)
    assert dfrun.parse_detector("DF") == ("DF", True)
    assert dfrun.parse_detector("nope") == (None, False)


def test_parse_onoff():
    assert dfrun.parse_onoff("on")[0] is True
    assert dfrun.parse_onoff("OFF")[0] is False
    assert dfrun.parse_onoff("maybe") == (None, False)


def test_parse_threshold():
    assert dfrun.parse_threshold("0.5") == (0.5, True)
    assert dfrun.parse_threshold("1") == (1.0, True)
    assert dfrun.parse_threshold("0") == (None, False)
    assert dfrun.parse_threshold("1.5") == (None, False)
    assert dfrun.parse_threshold("x") == (None, False)


def test_parse_int_helpers():
    assert dfrun.parse_nonneg_int("0") == (0, True)
    assert dfrun.parse_nonneg_int("-1") == (None, False)
    assert dfrun.parse_pos_int("1") == (1, True)
    assert dfrun.parse_pos_int("0") == (None, False)
    assert dfrun.parse_pos_int("x") == (None, False)


def test_prompt_setting_keep_change_invalid(monkeypatch):
    monkeypatch.setattr(dfrun, "_default_prompt", lambda m: "")     # Enter keeps
    assert dfrun.prompt_setting("e", "L", 4, dfrun.parse_pos_int) == 4
    monkeypatch.setattr(dfrun, "_default_prompt", lambda m: "8")    # valid change
    assert dfrun.prompt_setting("e", "L", 4, dfrun.parse_pos_int) == 8
    monkeypatch.setattr(dfrun, "_default_prompt", lambda m: "abc")  # invalid keeps
    assert dfrun.prompt_setting("e", "L", 4, dfrun.parse_pos_int) == 4


def test_finish_outputs_removes_legacy_desktop_files(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    out.mkdir()
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    monkeypatch.setattr(dfrun, "DESKTOP_DIR", str(desktop))
    _write_shard(
        out / "a__0a1b2c3d.csv",
        [["/x/1.jpg", "d", 1, "wolf", 0.9, "wolf", 0.9, 1, 0]],
    )
    for name in dfrun.DESKTOP_LEGACY:
        (desktop / name).write_text("stale", encoding="utf-8")
    dfrun.finish_outputs(str(out), software_dir=None)
    for name in dfrun.DESKTOP_LEGACY:
        assert not (desktop / name).exists()
    assert (desktop / dfrun.DESKTOP_MASTER).exists()
    assert "superseded" in capsys.readouterr().out


def _resume_meta(tmp_path):
    import json
    meta = {"detector": "DFbsMDS", "birds": False, "threshold": 0.8, "maxlag": 60,
            "threads": 3, "excluded_classes": ["ibex"],
            "utc_start_time": "2026-08-31T14:47:00"}
    (tmp_path / "run_metadata.p0.json").write_text(json.dumps(meta), encoding="utf-8")


def test_stage_configure_resume_is_one_keypress(tmp_path, monkeypatch):
    _resume_meta(tmp_path)
    prompts = []
    monkeypatch.setattr(dfrun, "_default_prompt", lambda m: (prompts.append(m), "")[1])
    args = dfrun.build_arg_parser().parse_args([])
    params = dfrun.stage_configure(args, str(tmp_path), argv=[])
    assert params is not None
    assert params["detector"] == "DFbsMDS" and params["threads"] == 3
    assert params["exclude_classes"] == ["ibex"]
    # One prompt only: the reuse confirmation, no per-setting questions.
    assert len(prompts) == 1 and "Reuse" in prompts[0]


def test_stage_configure_resume_n_opens_adjustment(tmp_path, monkeypatch):
    _resume_meta(tmp_path)
    answers = iter(["n", "", "", "", "", "", "4", "", ""])   # adjust: change threads to 4
    prompts = []
    monkeypatch.setattr(dfrun, "_default_prompt", lambda m: (prompts.append(m), next(answers))[1])
    args = dfrun.build_arg_parser().parse_args([])
    params = dfrun.stage_configure(args, str(tmp_path), argv=[])
    assert params is not None
    assert params["threads"] == 4                 # the adjusted knob
    assert params["detector"] == "DFbsMDS"        # everything else kept
    assert any("Proceed with these settings?" in m for m in prompts)


def test_stage_configure_fresh_run_still_prompts(tmp_path, monkeypatch):
    prompts = []
    monkeypatch.setattr(dfrun, "_default_prompt", lambda m: (prompts.append(m), "")[1])
    args = dfrun.build_arg_parser().parse_args([])
    params = dfrun.stage_configure(args, str(tmp_path), argv=[])
    assert params is not None
    assert not any("Reuse" in m for m in prompts)
    assert any("Detector" in m for m in prompts)


# ---------------------------------------------------------------------------
# --diagnose verdicts (pure logic over status + flight recorder)
# ---------------------------------------------------------------------------
def _sample(**overrides):
    base = {
        "time": "2026-08-28T10:00:00+00:00", "state": "running", "done": "1000",
        "rate_img_s": "5.0", "mem_available_gib": "8.5", "shmem_gib": "1.0",
        "swap_used_gib": "0.1", "worker_rss_gib": "3.2", "disk_free_gib": "9.0",
        "load1": "3.1", "root_ok": "1",
    }
    base.update(overrides)
    return base


def test_diagnose_verdict_scenarios():
    running = {"state": "running"}
    late = 1e12  # a boot long after any sample
    assert dfrun.diagnose_verdict(True, running, _sample(), None).startswith("healthy")
    assert dfrun.diagnose_verdict(False, {"state": "finished"}, None, None).startswith("finished")
    verdict = dfrun.diagnose_verdict(False, {"state": "stopped", "reason": "low-memory: x"}, None, None)
    assert "low-memory" in verdict
    # Dead-while-"running" cases, distinguished by the last sample:
    assert "drive" in dfrun.diagnose_verdict(False, running, _sample(root_ok="0"), None)
    assert "disk" in dfrun.diagnose_verdict(False, running, _sample(disk_free_gib="0.3"), None)
    assert "memory" in dfrun.diagnose_verdict(False, running, _sample(mem_available_gib="0.4"), None)
    assert "reboot or power loss" in dfrun.diagnose_verdict(False, running, _sample(), late)
    # Healthy sample, no reboot: software fault.
    assert "software" in dfrun.diagnose_verdict(False, running, _sample(), 0)
    # No telemetry at all.
    assert "no telemetry" in dfrun.diagnose_verdict(False, running, None, None)


def test_read_telemetry_tail(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    assert dfrun.read_telemetry_tail(str(out)) == []
    for i in range(12):
        dfrun.dfb.append_telemetry(str(out), 0, _sample(done=str(i)))
    tail = dfrun.read_telemetry_tail(str(out), n=5)
    assert len(tail) == 5
    assert tail[-1]["done"] == "11"


# ---------------------------------------------------------------------------
# orphaned swapfile detection and the auto-resume service unit
# ---------------------------------------------------------------------------
def test_parse_proc_swaps():
    text = ("Filename                Type        Size    Used    Priority\n"
            "/swap.img               file        4194300 0       -1\n"
            "/swapfile               file        16777212 0      -1\n")
    assert dfrun.parse_proc_swaps(text) == ["/swap.img", "/swapfile"]
    assert dfrun.parse_proc_swaps("Filename Type\n") == []


def test_orphaned_swapfiles_detection(tmp_path):
    big = tmp_path / "swapfile2"
    big.write_bytes(b"\0" * 2048)
    small = tmp_path / "swaptiny"
    small.write_bytes(b"\0" * 10)
    other = tmp_path / "notswap.img"
    other.write_bytes(b"\0" * 2048)
    candidates = dfrun.list_swapfile_candidates(str(tmp_path), min_bytes=1024)
    assert candidates == [(str(big), 2048)]  # small and non-swap names ignored
    # Active swap is not an orphan; inactive is.
    assert dfrun.orphaned_swapfiles(candidates, active=[str(big)]) == []
    assert dfrun.orphaned_swapfiles(candidates, active=["/swap.img"]) == [(str(big), 2048)]


def test_render_service_unit():
    unit = dfrun.render_service_unit("rim", "/home/rim/sw", "/home/rim/df_out")
    assert "User=rim" in unit
    assert "Restart=on-failure" in unit                # violent deaths only:
    # a clean stop (dfrun --stop, systemctl stop, finished run, protective
    # stop) exits 0 and stays stopped - the user is never locked into a run.
    assert dfrun.MOUNT_SCRIPT_PATH in unit             # drive mounted before start
    assert "--resume-last" in unit                     # current run's OWN settings
    assert "--write-pidfile" in unit                   # single-instance with manual runs
    # No settings are baked into the unit; they come from run metadata.
    assert "--detector" not in unit
    assert "--threshold" not in unit

    script = dfrun.render_mount_script("/media/rim/My Book1", "/home/rim/df_out")
    assert dfrun.EXPECTED_DRIVE_UUID in script         # waits for the right drive
    assert "mount -o ro" in script                     # read-only, always
    assert "run_metadata.p0.json" in script            # root follows the current run
    assert "'/media/rim/My Book1'" in script           # install-time fallback
    assert script.startswith("#!/bin/bash")


# ---------------------------------------------------------------------------
# reusing a partially complete run's settings
# ---------------------------------------------------------------------------
def test_explicit_cli_settings():
    given = dfrun.explicit_cli_settings(
        ["--threads", "3", "--no-birds", "--yes"]
    )
    assert given == {"threads", "birds"}
    # Species exclusion is no longer a classification setting, so no flag for it.
    assert dfrun.explicit_cli_settings(["--exclude-classes=wolf"]) == set()
    assert dfrun.explicit_cli_settings([]) == set()
    assert dfrun.explicit_cli_settings(None) == set()
    assert "detector" in dfrun.explicit_cli_settings(["--detector", "MDS"])


def test_apply_previous_settings_carries_the_run_forward():
    # The real incident: a restart fell back to parser defaults and silently
    # classified the rest of the archive with a different exclusion list.
    params = {
        "detector": "DF", "birds": True, "threshold": 0.5, "maxlag": 20,
        "batch_size": 8, "threads": 4, "merge_every": 600,
        "exclude_classes": [],
    }
    previous = {
        "detector": "DFbsMDS", "birds": False, "threshold": 0.8, "maxlag": 60,
        "batch_size": 8, "threads": 3, "merge_every": 600,
        "excluded_classes": ["genet", "ibex", "porcupine", "reindeer", "wolverine"],
    }
    taken = dfrun.apply_previous_settings(params, previous)
    assert params["detector"] == "DFbsMDS"
    assert params["threshold"] == 0.8
    assert params["maxlag"] == 60
    assert params["birds"] is False
    assert params["threads"] == 3
    assert params["exclude_classes"] == ["genet", "ibex", "porcupine", "reindeer", "wolverine"]
    assert set(taken) == {"detector", "threshold", "maxlag", "birds", "threads",
                          "exclude_classes"}


def test_apply_previous_settings_respects_explicit_flags():
    params = {"detector": "DF", "birds": True, "threshold": 0.5, "maxlag": 20,
              "batch_size": 8, "threads": 2, "merge_every": 600,
              "exclude_classes": ["wolf"]}
    previous = {"detector": "DFbsMDS", "threshold": 0.8, "threads": 3,
                "excluded_classes": ["ibex"]}
    taken = dfrun.apply_previous_settings(
        params, previous, explicit={"threads", "exclude_classes"}
    )   # exclude_classes stays honoured as an "explicit" key for old callers
    assert params["detector"] == "DFbsMDS"   # not given: resumed
    assert params["threads"] == 2            # given on the command line: kept
    assert params["exclude_classes"] == ["wolf"]
    assert "threads" not in taken and "exclude_classes" not in taken


def test_apply_previous_settings_without_previous_run():
    params = {"detector": "DF", "threads": 4, "exclude_classes": []}
    assert dfrun.apply_previous_settings(params, None) == []
    assert params["detector"] == "DF"
