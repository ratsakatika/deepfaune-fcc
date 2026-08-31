#!/usr/bin/env python3
# Copyright (c) 2025 Fundatia Conservation Carpathia batch tooling.
#
# dfrun: a friendly command-line front end for the DeepFaune batch orchestrator
# (deepfaune_batch.py). It does NOT reimplement the classification engine; it
# wraps the orchestrator as a detached subprocess and adds the operator
# ergonomics learned from a long real run: a self-update check, robust drive
# detection, a work assessment with a true overall percentage, a confirmation
# screen, a detached launch that survives an SSH or editor disconnection, a
# single-instance guard, a live readout, and easy spreadsheet-friendly outputs
# on the Desktop.
#
# British English throughout; no em dashes. Only the standard library and
# deepfaune_batch are imported at module load; rich and psutil are imported
# lazily where used so the pure logic stays cheap and testable.

import argparse
import csv
import getpass
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timedelta

# Make deepfaune_batch importable when run as a script from the repo.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import deepfaune_batch as dfb  # noqa: E402  (local, stdlib-only at import)


####################################################################################
### CONSTANTS (edit these in one place)
####################################################################################
TOOL_NAME = "dfrun"
# Bump on every change merged to main: third digit for small changes, second
# for medium features, first for complete overhauls. See CLAUDE.md
# ("Versioning") - the self-updater shows this to users, so a stale number
# makes different code report the same version.
TOOL_VERSION = "1.9.0"
GITHUB_URL = "https://github.com/ratsakatika/deepfaune-fcc"
# Single configurable contact string, as required.
CONTACT = (
    "Questions: Tom Ratsakatika trr26@cam.ac.uk / ratsakatika@gmail.com "
    "or open a GitHub issue"
)

# Self-update: pull only from this remote and branch, fast-forward only.
UPDATE_REMOTE = "origin"
UPDATE_BRANCH = "main"

# Drive identification.
EXPECTED_DRIVE_UUID = "7A31-2026"
ARCHIVE_MARKERS = ("Camera Trap Monitoring",)
WTM_FAR_RE = re.compile(r"WTM_FAR", re.IGNORECASE)
SEARCH_GLOBS = ("/media/{user}/*", "/mnt/*")

# Memory: offer a swapfile if total swap is below this.
MIN_TOTAL_SWAP_GIB = 16

# Run parameter defaults. Threshold 0.5 and maxlag 20 match the first long run
# (and the demo script), so a continued run stays consistent with the shards
# already classified. (The official GUI uses 0.8/10.)
DEFAULTS = {
    "detector": "DF",
    "birds": True,
    "threshold": 0.5,
    "maxlag": 20,
    "batch_size": 8,
    "threads": 4,
    "merge_every": 600,
}

DEFAULT_OUT_DIR = os.path.expanduser("~/df_out")
DEFAULT_SOFTWARE_DIR = _HERE
DESKTOP_DIR = os.path.expanduser("~/Desktop")
CONFIG_PATH = os.path.expanduser("~/.config/dfrun/config.json")

# Filenames inside the output directory.
PIDFILE_NAME = "dfrun.worker.pid"
CONSOLE_LOG_NAME = "dfrun.console.log"
TOTAL_CACHE_NAME = ".dfrun_total_cache.json"
MASTER_NAME = "master.csv"

# Desktop output names (grouped by a common prefix).
DESKTOP_MASTER = "deepfaune_master.csv"
DESKTOP_WILDLIFE = "deepfaune_wildlife.csv"
DESKTOP_SUMMARY = "deepfaune_summary.csv"
DESKTOP_DASHBOARD = "deepfaune_dashboard.html"
DESKTOP_GUIDE = "How_to_use_dfrun.html"
DESKTOP_WORKBENCH = "dashboard_and_filter.html"

# The dashboard builder (in the software dir) and the field protocol it reads.
DASHBOARD_BUILDER = "build_dashboard.py"
PROTOCOL_NAME = "FieldProtocols_WTM_FAR_23.xlsx"
# Skip a live dashboard rebuild when available memory is below this, so the
# build cannot trigger the out-of-memory killer mid-run. The build at the end,
# after the worker has exited and freed memory, always runs.
MIN_DASHBOARD_MEM_GIB = 4

# Excel's worksheet row limit (including the header row).
EXCEL_ROW_LIMIT = 1_048_576

# Labels excluded from "wildlife" and species tallies (English).
NON_ANIMAL_LABELS = {"empty", "human", "vehicle", "undefined"}
STATION_LEVELS = 2  # trailing path components used as a station key


####################################################################################
### SMALL JSON / CONFIG HELPERS
####################################################################################
def load_json(path, default=None):
    """Load JSON from path, returning default on any error."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def save_json(path, data):
    """Write JSON atomically; best-effort (errors are swallowed)."""
    directory = os.path.dirname(path) or "."
    try:
        os.makedirs(directory, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        pass


def load_config():
    return load_json(CONFIG_PATH, default={}) or {}


def save_config(config):
    save_json(CONFIG_PATH, config)


####################################################################################
### SELF-UPDATE (Stage 0): check and offer, fast-forward only, fail open
####################################################################################
def run_git(repo_dir, args, timeout=15):
    """Run a git command in repo_dir, returning the CompletedProcess or None."""
    try:
        return subprocess.run(
            ["git", "-C", repo_dir] + args,
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def git_fetch(repo_dir, remote=UPDATE_REMOTE, branch=UPDATE_BRANCH, timeout=5):
    """Time-limited fetch. True on success, False on timeout/error/no network."""
    result = run_git(repo_dir, ["fetch", remote, branch], timeout=timeout)
    return bool(result and result.returncode == 0)


def git_is_dirty(repo_dir):
    """True if the working tree has uncommitted or untracked changes."""
    result = run_git(repo_dir, ["status", "--porcelain"])
    if result is None or result.returncode != 0:
        return True  # cannot tell: treat as dirty and refuse to update
    return bool(result.stdout.strip())


def git_current_branch(repo_dir):
    result = run_git(repo_dir, ["rev-parse", "--abbrev-ref", "HEAD"])
    if result is None or result.returncode != 0:
        return None
    return result.stdout.strip()


def git_rev(repo_dir, ref):
    result = run_git(repo_dir, ["rev-parse", ref])
    if result is None or result.returncode != 0:
        return None
    return result.stdout.strip()


def git_ff_possible(repo_dir, local_ref, remote_ref):
    """True if local_ref is an ancestor of remote_ref (a fast-forward)."""
    result = run_git(
        repo_dir, ["merge-base", "--is-ancestor", local_ref, remote_ref]
    )
    return bool(result and result.returncode == 0)


def git_behind_count(repo_dir, local_ref, remote_ref):
    result = run_git(repo_dir, ["rev-list", "--count", f"{local_ref}..{remote_ref}"])
    if result is None or result.returncode != 0:
        return 0
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def git_incoming_subjects(repo_dir, local_ref, remote_ref):
    result = run_git(
        repo_dir, ["log", "--oneline", f"{local_ref}..{remote_ref}"]
    )
    if result is None or result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def decide_update(fetch_ok, dirty, worker_running, on_branch, ff_possible, behind):
    """Pure decision for the self-update check.

    Returns (action, reason) where action is one of:
      "skip"      remote unreachable; continue quietly,
      "refuse"    a precondition failed; continue, printing the reason,
      "current"   already up to date,
      "update"    behind and fast-forwardable; offer to update.
    """
    if not fetch_ok:
        return "skip", "remote unreachable"
    # Being up to date is decided BEFORE any refusal: a refusal message reads
    # "update available but not applied", which must never be printed when
    # there is no update (e.g. attaching to a live worker while current).
    if behind <= 0 and ff_possible:
        return "current", "already up to date"
    if worker_running:
        return "refuse", "a classification worker is running; not changing code under a live run"
    if dirty:
        return "refuse", "working tree has local changes; commit or restore first"
    if not on_branch:
        return "refuse", f"not on the {UPDATE_BRANCH} branch"
    if not ff_possible:
        return "refuse", "local branch has diverged; fast-forward not possible"
    return "update", f"{behind} commits behind"


def self_update_check(args, repo_dir, worker_is_running, prompt=None):
    """Stage 0: check for a newer version and offer a fast-forward update.

    Fails open: any problem prints a short line and returns without disrupting
    the run. Returns True only if it has re-exec'd (it will not return then).
    """
    prompt = prompt or _default_prompt
    if os.environ.get("DFRUN_UPDATED") == "1":
        return False  # already re-exec'd once this launch; avoid loops
    config = load_config()
    preference = args.update_check or config.get("update_check") or "auto"
    if preference == "never":
        print("Update check disabled (update-check=never).")
        return False
    if args.no_update:
        print("Update check skipped (--no-update).")
        return False

    fetch_ok = git_fetch(repo_dir)
    local = git_rev(repo_dir, "HEAD")
    remote = git_rev(repo_dir, f"{UPDATE_REMOTE}/{UPDATE_BRANCH}")
    on_branch = git_current_branch(repo_dir) == UPDATE_BRANCH
    ff = bool(local and remote and git_ff_possible(repo_dir, "HEAD", remote))
    behind = git_behind_count(repo_dir, "HEAD", f"{UPDATE_REMOTE}/{UPDATE_BRANCH}") if fetch_ok else 0
    dirty = git_is_dirty(repo_dir)

    action, reason = decide_update(
        fetch_ok, dirty, worker_is_running, on_branch, ff, behind
    )
    if action == "skip":
        print("Update check skipped: remote unreachable.")
        return False
    if action == "refuse":
        print(f"Update available but not applied: {reason}.")
        return False
    if action == "current":
        print("Up to date.")
        return False

    # action == "update"
    print(f"A newer version is available ({behind} commits):")
    for subject in git_incoming_subjects(repo_dir, "HEAD", remote):
        print(f"  {subject}")
    answer = prompt(f"Update now? [y/N] ")
    if answer.strip().lower() not in ("y", "yes"):
        print("Update declined; continuing on the current version.")
        return False

    old_hash = local
    result = run_git(repo_dir, ["pull", "--ff-only", UPDATE_REMOTE, UPDATE_BRANCH], timeout=60)
    if result is None or result.returncode != 0:
        msg = (result.stderr.strip() if result else "git pull failed")
        print(f"Update failed, keeping the current version: {msg}")
        return False
    new_hash = git_rev(repo_dir, "HEAD")
    print(f"Updated {old_hash[:10]} -> {new_hash[:10]}.")
    print(f"To roll back: git reset --hard {old_hash}")
    # Re-exec so the new code takes effect, passing through the original args.
    os.environ["DFRUN_UPDATED"] = "1"
    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:])
    return True  # not reached


def _default_prompt(message):
    try:
        return input(message)
    except EOFError:
        return ""


####################################################################################
### DRIVE DETECTION (Stage 2)
####################################################################################
def looks_like_archive(path):
    """True if path looks like the camera-trap archive root.

    It is the archive if it directly contains a marker directory (for example
    "Camera Trap Monitoring") or any directory whose name matches the WTM_FAR
    pattern.
    """
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    for name in entries:
        if name in ARCHIVE_MARKERS or WTM_FAR_RE.search(name):
            if os.path.isdir(os.path.join(path, name)):
                return True
    return False


def candidate_mount_dirs(user=None):
    """Expand the search globs to existing directories to inspect."""
    import glob
    user = user or os.environ.get("USER") or ""
    dirs = []
    for pattern in SEARCH_GLOBS:
        for path in glob.glob(pattern.format(user=user)):
            if os.path.isdir(path):
                dirs.append(path)
    return sorted(set(dirs))


def find_archive_candidates(user=None):
    """Return mounted directories that look like the camera archive."""
    return [d for d in candidate_mount_dirs(user) if looks_like_archive(d)]


def path_device_uuid(path):
    """Best-effort exFAT/filesystem UUID for the device backing path.

    Resolves the mount source via findmnt, then maps it to a UUID via the
    /dev/disk/by-uuid symlinks. Returns None if it cannot be determined.
    """
    try:
        result = subprocess.run(
            ["findmnt", "-no", "SOURCE", "--target", path],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    source = result.stdout.strip()
    if not source:
        return None
    by_uuid = "/dev/disk/by-uuid"
    try:
        for uuid in os.listdir(by_uuid):
            if os.path.realpath(os.path.join(by_uuid, uuid)) == os.path.realpath(source):
                return uuid
    except OSError:
        return None
    return None


def mount_options(path):
    """Return the mount option string for the filesystem holding path, or ''."""
    try:
        result = subprocess.run(
            ["findmnt", "-no", "OPTIONS", "--target", path],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def is_read_only(options):
    """True if a mount option string indicates a read-only mount."""
    tokens = [tok.strip() for tok in options.split(",")]
    return "ro" in tokens


####################################################################################
### WORK ASSESSMENT (Stage 3) AND CACHE
####################################################################################
def cache_path(out_dir):
    return os.path.join(out_dir, TOTAL_CACHE_NAME)


def load_total_cache(out_dir, source, drive_uuid):
    """Return a cached (total, shards) if valid for this drive and root, else None.

    The cache records the drive UUID and root and is invalidated if either
    differs, so a changed drive cannot silently reuse a stale count.
    """
    data = load_json(cache_path(out_dir))
    if not data:
        return None
    if data.get("uuid") != drive_uuid or data.get("root") != os.path.abspath(source):
        return None
    if "total" not in data or "shards" not in data:
        return None
    return data["total"], data["shards"]


def save_total_cache(out_dir, source, drive_uuid, total, shards):
    save_json(cache_path(out_dir), {
        "uuid": drive_uuid,
        "root": os.path.abspath(source),
        "total": total,
        "shards": shards,
    })


def count_total_images(source):
    """Total images and shard count on the drive (the expensive scan)."""
    shards = dfb.find_shards(source)
    total = sum(len(imgs) for _, imgs in shards)
    return total, len(shards)


def get_total_count(out_dir, source, drive_uuid, rescan):
    """Return (total, shards, from_cache), using the cache unless rescan is set."""
    if not rescan:
        cached = load_total_cache(out_dir, source, drive_uuid)
        if cached is not None:
            return cached[0], cached[1], True
    total, shards = count_total_images(source)
    save_total_cache(out_dir, source, drive_uuid, total, shards)
    return total, shards, False


def assess_work(out_dir, source, drive_uuid, rescan):
    """Return (done, remaining, total, shards, pct, from_cache)."""
    done = dfb.count_classified_images(out_dir)
    total, shards, from_cache = get_total_count(out_dir, source, drive_uuid, rescan)
    remaining = max(0, total - done)
    pct = (100.0 * done / total) if total else 0.0
    return done, remaining, total, shards, pct, from_cache


####################################################################################
### MEMORY / SWAP (A7, Stage 4)
####################################################################################
def gib(n_bytes):
    return n_bytes / (1024 ** 3)


def needs_swap(total_swap_bytes, minimum_gib=MIN_TOTAL_SWAP_GIB):
    """True if total swap is below the recommended minimum."""
    return gib(total_swap_bytes) < minimum_gib


def current_swap_total_bytes():
    try:
        import psutil
    except ImportError:
        return 0
    return psutil.swap_memory().total


def parse_proc_swaps(text):
    """Paths of active swap files/partitions from /proc/swaps content."""
    paths = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if parts:
            paths.append(parts[0])
    return paths


def active_swap_paths():
    try:
        with open("/proc/swaps", encoding="utf-8") as handle:
            return parse_proc_swaps(handle.read())
    except OSError:
        return []


def list_swapfile_candidates(directory="/", min_bytes=dfb.GIB):
    """Large swap-looking regular files directly in directory: [(path, size)].

    Nothing on this box legitimately creates multi-GiB files at the root of
    the disk except swap files, so anything matching here that is not active
    swap is orphaned dead weight (the incident that once filled the disk).
    """
    found = []
    try:
        names = os.listdir(directory)
    except OSError:
        return found
    for name in sorted(names):
        if not name.startswith("swap"):
            continue
        path = os.path.join(directory, name)
        try:
            if not os.path.isfile(path):
                continue
            size = os.path.getsize(path)
        except OSError:
            continue
        if size >= min_bytes:
            found.append((path, size))
    return found


def orphaned_swapfiles(candidates=None, active=None):
    """Swap-looking files that are NOT currently in use as swap."""
    if candidates is None:
        candidates = list_swapfile_candidates()
    if active is None:
        active = active_swap_paths()
    active_set = {os.path.abspath(p) for p in active}
    return [(p, s) for p, s in candidates if os.path.abspath(p) not in active_set]


def reclaim_orphaned_swapfiles(prompt_fn=None):
    """Report orphaned swapfiles and offer to remove them (sudo). Best-effort.

    Interactive only: with no prompt function the orphans are reported with
    the removal command, never deleted silently.
    """
    orphans = orphaned_swapfiles()
    for path, size in orphans:
        print(
            f"NOTE: {path} uses {size / dfb.GIB:.0f} GiB of disk but is not "
            "active swap (orphaned; likely left by a reboot)."
        )
        if prompt_fn is None:
            print(f"  Reclaim the space with: sudo rm {path}")
            continue
        reply = prompt_fn(f"  Remove {path} now to reclaim the space (sudo)? [y/N] ")
        if reply.strip().lower() in ("y", "yes"):
            try:
                if subprocess.run(["sudo", "rm", path]).returncode == 0:
                    print(f"  Removed; {size / dfb.GIB:.0f} GiB reclaimed.")
                else:
                    print("  Removal failed; continuing.")
            except (OSError, subprocess.SubprocessError):
                print("  Removal failed; continuing.")
    return orphans


def swapfile_status(path="/swapfile"):
    """Return (exists, size_bytes, active) for a swapfile path.

    Active means listed in /proc/swaps. A swapfile created on a previous boot
    but never added to /etc/fstab shows up here as exists-but-inactive: full
    size on disk, doing nothing.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return False, 0, False
    active = False
    try:
        with open("/proc/swaps", encoding="utf-8") as handle:
            active = any(
                line.split()[:1] == [path] for line in list(handle)[1:]
            )
    except OSError:
        pass
    return True, size, active


def free_disk_bytes(path="/"):
    """Free bytes available to non-root users on path's filesystem."""
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def fstab_has_swapfile(path="/swapfile"):
    """True if /etc/fstab already lists the swapfile (so it survives reboots)."""
    try:
        with open("/etc/fstab", encoding="utf-8") as handle:
            return any(
                line.split()[:1] == [path]
                for line in handle
                if not line.lstrip().startswith("#")
            )
    except OSError:
        return False


# Never let swapfile creation leave the disk with less free space than this:
# a full root filesystem stops the classifier (and everything else) dead.
MIN_FREE_AFTER_SWAP_GIB = 5


def create_swapfile(size_gib, path="/swapfile", runner=None,
                    status=None, free_bytes=None):
    """Create (or re-enable) a swapfile via sudo and persist it in /etc/fstab.

    An existing file at path is reused: after a reboot an unpersisted swapfile
    is inactive but still full-size on disk, so re-enabling it costs nothing
    while recreating it blindly wastes gigabytes. Creation is refused when the
    disk would be left with less than MIN_FREE_AFTER_SWAP_GIB free. The fstab
    entry is appended (once) so the swapfile keeps working after reboots
    instead of becoming dead weight. Returns True when the swapfile ends up
    active; any failure returns False so the caller continues without swap.
    """
    runner = runner or (lambda cmd: subprocess.run(cmd).returncode == 0)
    exists, size, active = (status or swapfile_status)(path)
    wanted = size_gib * 1024 ** 3
    steps = []
    if active and size >= wanted:
        print(f"Swapfile {path} is already active ({gib(size):.0f} GiB).")
    else:
        if active:  # active but too small: rebuild it at the new size
            steps.append(["sudo", "swapoff", path])
        grow = wanted - (size if exists else 0)
        if grow > 0:
            free = (free_bytes or free_disk_bytes)(os.path.dirname(path) or "/")
            if free < grow + MIN_FREE_AFTER_SWAP_GIB * 1024 ** 3:
                print(
                    f"Not enough disk for a {size_gib} GiB swapfile: only "
                    f"{gib(free):.1f} GiB free and at least "
                    f"{MIN_FREE_AFTER_SWAP_GIB} GiB must stay free. Skipping "
                    "swap creation; free some space and try again."
                )
                return False
        if exists and size >= wanted:
            print(f"Reusing the existing {gib(size):.0f} GiB swapfile at {path}.")
        else:
            steps.append(["sudo", "fallocate", "-l", f"{size_gib}G", path])
        steps += [
            ["sudo", "chmod", "600", path],
            ["sudo", "mkswap", path],
            ["sudo", "swapon", path],
        ]
    if not fstab_has_swapfile(path):
        steps.append([
            "sudo", "sh", "-c",
            f"echo '{path} none swap sw 0 0' >> /etc/fstab",
        ])
    for cmd in steps:
        try:
            if not runner(cmd):
                return False
        except (OSError, subprocess.SubprocessError):
            return False
    return True


####################################################################################
### SINGLE INSTANCE (A10) AND DETACHED LAUNCH (A8)
####################################################################################
def pidfile_path(out_dir):
    return os.path.join(out_dir, PIDFILE_NAME)


def read_pidfile(out_dir):
    try:
        with open(pidfile_path(out_dir), encoding="utf-8") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def write_pidfile(out_dir, pid):
    save_text(pidfile_path(out_dir), f"{pid}\n")


def save_text(path, text):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp, path)


def process_is_worker(pid):
    """True if pid is a live deepfaune_batch worker (not just any process).

    Relies on the command line rather than the process name, so counting worker
    threads in htop cannot raise a false duplicate-run alarm.
    """
    if pid is None:
        return False
    try:
        import psutil
        try:
            proc = psutil.Process(pid)
            return any("deepfaune_batch" in part for part in proc.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    except ImportError:
        pass
    # Fallback without psutil: read /proc/<pid>/cmdline.
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            return b"deepfaune_batch" in handle.read()
    except OSError:
        return False


def worker_running(out_dir):
    """True if the worker recorded in the pidfile is alive."""
    return process_is_worker(read_pidfile(out_dir))


def build_worker_command(software_dir, source, out_dir, params):
    cmd = [
        sys.executable,
        os.path.join(software_dir, "deepfaune_batch.py"),
        "--software-dir", software_dir,
        "--root", source,
        "--out-dir", out_dir,
        "--detector", params["detector"],
        "--threshold", str(params["threshold"]),
        "--maxlag", str(params["maxlag"]),
        "--batch-size", str(params["batch_size"]),
        "--threads", str(params["threads"]),
        "--heartbeat-secs", "30",
        "--merge-every", str(params["merge_every"]),
    ]
    if params.get("birds", True):
        cmd.append("--birds")
    if params.get("exclude_classes"):
        cmd += ["--exclude-classes", ",".join(params["exclude_classes"])]
    return cmd


def launch_detached(software_dir, source, out_dir, params):
    """Launch the worker fully detached from the terminal and editor session.

    start_new_session=True calls setsid, so the worker lives in its own session
    and is not killed when the SSH or VS Code connection drops. Its stdio is
    redirected to a console log, and the orchestrator keeps writing its own
    detailed log unchanged.
    """
    os.makedirs(out_dir, exist_ok=True)
    cmd = build_worker_command(software_dir, source, out_dir, params)
    console = open(os.path.join(out_dir, CONSOLE_LOG_NAME), "ab")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=console,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=software_dir,
    )
    write_pidfile(out_dir, proc.pid)
    return proc.pid


####################################################################################
### STATISTICS FROM SHARD CSVs (Stage 6, A11)
####################################################################################
def station_from_path(path, levels=STATION_LEVELS):
    """Station key: the last `levels` directory components of an image path."""
    directory = os.path.dirname(str(path)).replace("\\", "/")
    parts = [p for p in directory.split("/") if p]
    if not parts:
        return "(unknown)"
    return "/".join(parts[-levels:])


def is_excluded_label(label):
    low = label.strip().lower()
    if low in NON_ANIMAL_LABELS:
        return True
    return low.endswith("undefined")  # for example "bird undefined"


class StatsAccumulator:
    """Incrementally tallies species, blanks and stations from shard CSVs.

    Each shard CSV is written once and never modified, so counting it once is
    correct. Already-counted files are tracked so the master is never re-read in
    full. Non-survey artefacts are handled: unset-clock (1970) timestamps are
    flagged, and excluded labels are kept out of the species tally.
    """

    def __init__(self):
        self.counted = set()
        self.species = Counter()
        self.by_station = Counter()
        self.total = 0
        self.empty = 0
        self.unset_clock = 0

    def consume_new(self, out_dir):
        for path in dfb.iter_shard_csvs(out_dir):
            name = os.path.basename(path)
            if name in self.counted:
                continue
            try:
                self._consume_file(path)
            except OSError:
                continue  # mid-rotation or unreadable: retry next tick
            self.counted.add(name)

    def _consume_file(self, path):
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                self.total += 1
                label = (row.get("prediction_seq") or "").strip()
                low = label.lower()
                if (row.get("date") or "").startswith("1970"):
                    self.unset_clock += 1
                if low == "empty":
                    self.empty += 1
                elif label and not is_excluded_label(label):
                    self.species[label] += 1
                    self.by_station[station_from_path(row.get("filename", ""))] += 1

    def top_species(self, n=5):
        return self.species.most_common(n)

    def blank_pct(self):
        return (100.0 * self.empty / self.total) if self.total else 0.0


####################################################################################
### COMPLETION OUTPUTS (Stage 7)
####################################################################################
def copy_file(src, dst):
    """Copy src to dst atomically (temp then rename). Best-effort."""
    try:
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        tmp = dst + ".tmp"
        with open(src, "rb") as fin, open(tmp, "wb") as fout:
            while True:
                chunk = fin.read(1024 * 1024)
                if not chunk:
                    break
                fout.write(chunk)
        os.replace(tmp, dst)
        return True
    except OSError:
        return False


def recorded_archive_root(out_dir):
    """The archive root the current run recorded, or None.

    Used as the canonical root for path normalisation, so outputs rebuilt at
    any time agree with wherever the run itself read the drive.
    """
    meta = load_json(os.path.join(out_dir, "run_metadata.p0.json")) or {}
    return meta.get("root")


def write_wildlife_and_summary(out_dir, desktop_dir, canonical_root=None):
    """Write the spreadsheet-friendly wildlife and summary CSVs to the Desktop.

    Reads the shard CSVs (the source of truth) once. The wildlife file holds
    only actual animals (excludes empty, human and vehicle) and is capped at
    Excel's row limit. The summary file holds per-species and per-station counts.
    Rows carry the same derived columns as the master CSV: a globally unique
    sequence_id (seqnum alone restarts in every folder) and, when
    canonical_root is given, image paths re-rooted so a drive mounted under a
    different name still yields one consistent set of paths.
    Returns (wildlife_rows, capped, species_count, station_count).
    """
    os.makedirs(desktop_dir, exist_ok=True)
    species = Counter()
    stations = Counter()
    wildlife_path = os.path.join(desktop_dir, DESKTOP_WILDLIFE)
    capped = False
    written = 0
    tmp = wildlife_path + ".tmp"
    # Union of the shard headers, so per-class score columns (and any columns
    # from older shards) all appear; missing cells are left blank per row.
    header = dfb.shard_header_union(out_dir)
    if dfb.SEQUENCE_ID_COLUMN not in header:
        header = header + [dfb.SEQUENCE_ID_COLUMN]
    with open(tmp, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(header + ["station"])
        for path in dfb.iter_shard_csvs(out_dir):
            try:
                handle = open(path, newline="", encoding="utf-8")
            except OSError:
                continue
            with handle:
                reader = csv.DictReader(handle)
                shard_hash = dfb.shard_hash_from_name(path)
                rel = None  # resolved from the first row of this shard
                resolved = False
                for row in reader:
                    if not resolved:
                        resolved = True
                        if canonical_root:
                            rel = dfb.shard_relpath_from_row(
                                path, os.path.dirname(row.get("filename") or "")
                            )
                    filename = row.get("filename", "")
                    if rel is not None:
                        filename = dfb.rerooted_path(filename, canonical_root, rel)
                        row["filename"] = filename
                    row[dfb.SEQUENCE_ID_COLUMN] = dfb.sequence_id(
                        shard_hash, row.get("seqnum", "")
                    )
                    label = (row.get("prediction_seq") or "").strip()
                    low = label.lower()
                    if low in {"empty", "human", "vehicle"} or not label:
                        continue
                    station = station_from_path(filename)
                    species[label] += 1
                    stations[station] += 1
                    if written < EXCEL_ROW_LIMIT - 1:
                        writer.writerow(
                            [row.get(col, "") for col in header] + [station]
                        )
                        written += 1
                    else:
                        capped = True
    os.replace(tmp, wildlife_path)

    summary_path = os.path.join(desktop_dir, DESKTOP_SUMMARY)
    tmp = summary_path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["group", "name", "count"])
        for name, count in species.most_common():
            writer.writerow(["species", name, count])
        for name, count in stations.most_common():
            writer.writerow(["station", name, count])
    os.replace(tmp, summary_path)
    return written, capped, len(species), len(stations)


####################################################################################
### DASHBOARD (wraps build_dashboard.py; never reimplements it)
####################################################################################
def open_in_browser(path):
    """Open a file in the default browser, detached. Best-effort."""
    try:
        subprocess.Popen(
            ["xdg-open", path],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
        )
    except OSError:
        pass


def ensure_desktop_master(out_dir):
    """Ensure the master CSV is on the Desktop, regenerating it if missing.

    Returns the Desktop master path, or None if it could not be produced. The
    dashboard's detections are this Desktop master, as requested.
    """
    desktop_master = os.path.join(DESKTOP_DIR, DESKTOP_MASTER)
    if not os.path.exists(desktop_master):
        src = os.path.join(out_dir, MASTER_NAME)
        try:
            if not os.path.exists(src):
                dfb.merge_csvs(out_dir, src, canonical_root=recorded_archive_root(out_dir))
            copy_file(src, desktop_master)
        except OSError:
            return None
    return desktop_master if os.path.exists(desktop_master) else None


def dashboard_command(software_dir, out_dir):
    """Return (cmd, out_html) to render the dashboard, or (None, None).

    Detections come from the Desktop master (regenerated if missing); the field
    protocol in software_dir, if present, adds the camera map (the builder omits
    the map gracefully when it is absent).
    """
    builder = os.path.join(software_dir, DASHBOARD_BUILDER)
    if not os.path.exists(builder):
        return None, None
    desktop_master = ensure_desktop_master(out_dir)
    if not desktop_master:
        return None, None
    out_html = os.path.join(DESKTOP_DIR, DESKTOP_DASHBOARD)
    protocol = os.path.join(software_dir, PROTOCOL_NAME)
    return [sys.executable, builder, desktop_master, protocol, out_html], out_html


def dashboard_env(out_dir):
    """Environment for the builder, caching its web assets under out_dir (not the
    repo, so the self-update dirty-tree check is never tripped)."""
    env = dict(os.environ)
    env["DASHBOARD_ASSET_DIR"] = os.path.join(out_dir, "dashboard_assets")
    return env


def build_dashboard(software_dir, out_dir, open_after=False):
    """Build the dashboard HTML on the Desktop (blocking). Best-effort.

    Runs the builder in a subprocess so its memory is released and a failure
    cannot disrupt the run. Returns the HTML path or None.
    """
    if not software_dir:
        return None  # no builder location known; the CSVs are still written
    cmd, out_html = dashboard_command(software_dir, out_dir)
    if not cmd:
        return None
    try:
        result = subprocess.run(
            cmd, cwd=software_dir, env=dashboard_env(out_dir),
            capture_output=True, text=True, timeout=3600,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"Dashboard build skipped: {exc}")
        return None
    if result.returncode != 0:
        print("Dashboard build failed (continuing):")
        for line in (result.stderr or result.stdout or "").strip().splitlines()[-5:]:
            print(f"  {line}")
        return None
    if open_after:
        open_in_browser(out_html)
    return out_html


def spawn_dashboard(software_dir, out_dir):
    """Start a dashboard build in the background (non-blocking). Returns Popen or None."""
    cmd, _out_html = dashboard_command(software_dir, out_dir)
    if not cmd:
        return None
    try:
        log = open(os.path.join(out_dir, "dashboard.log"), "ab")
        return subprocess.Popen(
            cmd, cwd=software_dir, env=dashboard_env(out_dir),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError:
        return None


def enough_memory_for_dashboard(min_gib=MIN_DASHBOARD_MEM_GIB):
    """True if there is enough free RAM to build the dashboard mid-run safely."""
    try:
        import psutil
        return gib(psutil.virtual_memory().available) >= min_gib
    except Exception:
        return True  # cannot tell; allow


####################################################################################
### TIME / FORMAT HELPERS
####################################################################################
def absolute_eta(eta_seconds, now=None):
    """Format an absolute finish time, for example 'finishes Fri 27 Jun, 03:14'."""
    if not eta_seconds or eta_seconds <= 0:
        return "finishes: unknown"
    now = now or datetime.now()
    finish = now + timedelta(seconds=eta_seconds)
    return "finishes " + finish.strftime("%a %d %b, %H:%M")


def status_path(out_dir, partition=0):
    return os.path.join(out_dir, f"status.p{partition}.json")


def read_status(out_dir, partition=0):
    return load_json(status_path(out_dir, partition))


####################################################################################
### AUTO-RESUME SERVICE (--install-service): systemd supervises the worker
####################################################################################
SERVICE_NAME = "deepfaune-worker.service"
SERVICE_PATH = f"/etc/systemd/system/{SERVICE_NAME}"
MOUNT_SCRIPT_PATH = "/usr/local/bin/deepfaune-mount.sh"


def render_mount_script(fallback_source, out_dir, uuid=EXPECTED_DRIVE_UUID):
    """The root-run pre-start script: wait for the drive, mount it read-only.

    The mountpoint is read from the current run's own metadata at start time
    (so the service follows wherever the run was actually launched), falling
    back to the path captured at install time. A separate script rather than
    an inline ExecStartPre because systemd does not understand shell quoting
    of paths with spaces ("My Book1").
    """
    meta = shlex.quote(os.path.join(out_dir, "run_metadata.p0.json"))
    fallback = shlex.quote(fallback_source)
    return f"""#!/bin/bash
# Auto-generated by dfrun --install-service. Waits up to 5 minutes for the
# archive drive (UUID {uuid}) and mounts it read-only at the current run's
# recorded root. Exits 0 either way; the worker reports a missing root.
ROOT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["root"])' {meta} 2>/dev/null)
[ -n "$ROOT" ] || ROOT={fallback}
for _ in $(seq 60); do
    blkid -U {uuid} >/dev/null 2>&1 && break
    sleep 5
done
mkdir -p "$ROOT"
mountpoint -q "$ROOT" || mount -o ro UUID={uuid} "$ROOT" || true
exit 0
"""


def render_service_unit(user, software_dir, out_dir):
    """Render the systemd unit that resumes the worker after violent deaths.

    No settings are baked in: ExecStart uses --resume-last, so every start
    re-reads run_metadata.p0.json and continues the CURRENT run exactly as
    it was launched. Restart=on-failure means only violent deaths (OOM kill,
    crash) trigger a restart: a finished run, a protective stop, a SIGTERM
    from dfrun --stop and systemctl stop all exit cleanly and STAY stopped,
    so a user is never locked into a run they chose to end. The unit is not
    enabled at boot for the same reason.
    """
    python = shlex.quote(sys.executable)
    batch = shlex.quote(os.path.join(software_dir, "deepfaune_batch.py"))
    return f"""[Unit]
Description=DeepFaune camera-trap classifier worker (crash auto-resume)
After=local-fs.target

[Service]
Type=simple
User={user}
ExecStartPre=+{MOUNT_SCRIPT_PATH}
ExecStart={python} {batch} --software-dir {shlex.quote(software_dir)} --out-dir {shlex.quote(out_dir)} --resume-last --write-pidfile
Restart=on-failure
RestartSec=120
Nice=10

[Install]
WantedBy=multi-user.target
"""


def service_installed():
    return os.path.exists(SERVICE_PATH)


def install_service(args, out_dir, software_dir):
    """Write and load the crash-resume unit (needs sudo). Not boot-enabled."""
    meta = load_json(os.path.join(out_dir, "run_metadata.p0.json")) or {}
    fallback = meta.get("root")
    if not fallback:
        found = stage_find_source(args)
        if not found:
            print("Cannot determine the archive path: run a classification "
                  "once first, or attach the drive.")
            return 1
        fallback = found[0]
    unit = render_service_unit(getpass.getuser(), software_dir, out_dir)
    script = render_mount_script(fallback, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tmp_unit = os.path.join(out_dir, "deepfaune-worker.service.tmp")
    tmp_script = os.path.join(out_dir, "deepfaune-mount.sh.tmp")
    with open(tmp_unit, "w", encoding="utf-8") as handle:
        handle.write(unit)
    with open(tmp_script, "w", encoding="utf-8") as handle:
        handle.write(script)
    print(f"Installing {SERVICE_NAME} (crash auto-resume; no boot autostart)...")
    steps = [
        ["sudo", "cp", tmp_script, MOUNT_SCRIPT_PATH],
        ["sudo", "chmod", "755", MOUNT_SCRIPT_PATH],
        ["sudo", "cp", tmp_unit, SERVICE_PATH],
        ["sudo", "systemctl", "daemon-reload"],
    ]
    for cmd in steps:
        try:
            if subprocess.run(cmd).returncode != 0:
                print(f"Failed at: {' '.join(cmd)}")
                return 1
        except (OSError, subprocess.SubprocessError) as err:
            print(f"Failed at: {' '.join(cmd)} ({err})")
            return 1
    for tmp in (tmp_unit, tmp_script):
        try:
            os.unlink(tmp)
        except OSError:
            pass
    print(
        "Installed. From now on dfrun launches runs under systemd, which\n"
        "restarts the worker two minutes after an OOM kill or crash - and\n"
        "ONLY then: a finished run, dfrun --stop, or systemctl stop all stay\n"
        "stopped, and nothing autostarts at boot (after a power cut, open\n"
        "Run DeepFaune / run dfrun to continue).\n"
        f"  status:  systemctl status {SERVICE_NAME}\n"
        f"  stop:    dfrun --stop\n"
        f"  remove:  dfrun --uninstall-service"
    )
    return 0


def launch_via_service(software_dir, source, out_dir, params):
    """Start the run under systemd with the settings just confirmed.

    The chosen settings are written to run_metadata.p0.json FIRST (via the
    orchestrator's own writer, so the schema always matches), then the
    service starts and its --resume-last reads them back. Returns True when
    the service started.
    """
    cmd = build_worker_command(software_dir, source, out_dir, params)
    worker_args = dfb.build_arg_parser().parse_args(cmd[2:])
    dfb.write_run_metadata(out_dir, software_dir, source, worker_args)
    try:
        return subprocess.run(
            ["sudo", "systemctl", "restart", SERVICE_NAME]
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def stop_run(out_dir):
    """Intentionally stop the current run - and make it STAY stopped."""
    if service_installed():
        try:
            subprocess.run(["sudo", "systemctl", "stop", SERVICE_NAME])
        except (OSError, subprocess.SubprocessError):
            pass
    pid = read_pidfile(out_dir)
    if pid and process_is_worker(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Asked worker {pid} to stop at a clean boundary "
                  "(the shard in progress is redone on the next launch).")
        except OSError as err:
            print(f"Could not signal worker {pid}: {err}")
            return 1
    else:
        print("No running worker found.")
    print("Stopped intentionally: nothing will restart it until you relaunch.")
    return 0


def uninstall_service():
    """Disable and remove the auto-resume unit (needs sudo)."""
    for cmd in (
        ["sudo", "systemctl", "disable", "--now", SERVICE_NAME],
        ["sudo", "rm", "-f", SERVICE_PATH, MOUNT_SCRIPT_PATH],
        ["sudo", "systemctl", "daemon-reload"],
    ):
        try:
            subprocess.run(cmd)
        except (OSError, subprocess.SubprocessError):
            pass
    print(f"{SERVICE_NAME} removed (any running worker was stopped).")
    return 0


####################################################################################
### DIAGNOSTICS (--diagnose): read the flight recorder and explain the state
####################################################################################
def boot_time_unix():
    """When this machine booted, in unix seconds; None if unknowable."""
    try:
        with open("/proc/uptime", encoding="utf-8") as handle:
            return time.time() - float(handle.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def read_telemetry_tail(out_dir, n=8, partition=0):
    """The newest n flight-recorder samples, as dicts (oldest first)."""
    path = os.path.join(out_dir, f"telemetry.p{partition}.csv")
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return []
    return rows[-n:]


def diagnose_verdict(worker_alive, status, last_sample, boot_unix):
    """One-sentence explanation of the run's state or its last death. Pure.

    status is the parsed status dict (or None); last_sample the newest
    telemetry row as strings (or None); boot_unix the machine's boot time.
    """
    if worker_alive:
        return "healthy: a worker is running now"
    state = (status or {}).get("state")
    reason = (status or {}).get("reason")
    if state == "finished":
        return "finished: the last run completed normally"
    if state == "stopped":
        if reason:
            return f"stopped itself deliberately - {reason}"
        return "stopped cleanly (operator signal or watchdog)"
    if state != "running":
        return "no run recorded in this output directory yet"
    # Status says "running" but no process exists: it died without a goodbye.
    if last_sample:
        def num(key):
            try:
                return float(last_sample.get(key) or "nan")
            except (TypeError, ValueError):
                return float("nan")
        if last_sample.get("root_ok") == "0":
            return ("crashed: the archive drive had disconnected "
                    "(root_ok=0 in the last telemetry sample)")
        if num("disk_free_gib") < 1.0:
            return (f"crashed: the output disk was nearly full "
                    f"({num('disk_free_gib'):.1f} GiB free at the last sample)")
        if num("mem_available_gib") < 1.0:
            return (f"crashed: memory exhaustion - only "
                    f"{num('mem_available_gib'):.1f} GiB available at the last "
                    "sample (OOM kill likely)")
        try:
            sample_unix = datetime.fromisoformat(last_sample["time"]).timestamp()
            if boot_unix is not None and boot_unix > sample_unix:
                return ("killed by a reboot or power loss: the machine booted "
                        "after the last healthy sample")
        except (KeyError, ValueError):
            pass
        return ("crashed while system readings looked healthy - likely a "
                "software fault; check dfrun.console.log for a traceback")
    return ("crashed with no telemetry recorded (run predates the flight "
            "recorder?) - check dfrun.console.log and the journal")


def run_diagnose(out_dir):
    """Print a text diagnosis: settings, status, flight recorder, verdict."""
    print(f"Diagnostics for {out_dir}")
    meta = load_json(os.path.join(out_dir, "run_metadata.p0.json")) or {}
    if meta:
        excluded = ", ".join(meta.get("excluded_classes") or []) or "none"
        print(
            f"Run settings: detector={meta.get('detector')} "
            f"threshold={meta.get('threshold')} maxlag={meta.get('maxlag')} "
            f"birds={meta.get('birds')} lang={meta.get('lang')} "
            f"excluded=[{excluded}]"
        )
        commit = str(meta.get("orchestrator_git_commit") or "")[:10]
        print(f"Launched: {meta.get('utc_start_time')} | code version {commit}")
    alive = worker_running(out_dir)
    status = read_status(out_dir)
    if status:
        age = int(time.time() - (status.get("updated_unix") or 0))
        line = (
            f"Status: {status.get('state')} | "
            f"{status.get('true_done', 0):,} done ({status.get('true_pct')}%) | "
            f"last update {age}s ago"
        )
        if status.get("reason"):
            line += f" | reason: {status['reason']}"
        print(line)
    tail = read_telemetry_tail(out_dir)
    if tail:
        print("Flight recorder (newest last):")
        header = ["time", "state", "avail", "shmem", "swap", "rss", "disk", "load", "root"]
        keys = ["time", "state", "mem_available_gib", "shmem_gib",
                "swap_used_gib", "worker_rss_gib", "disk_free_gib", "load1", "root_ok"]
        print("  " + "  ".join(f"{h:>6}" if h != "time" else f"{h:20}" for h in header))
        for row in tail:
            cells = []
            for head, key in zip(header, keys):
                value = str(row.get(key, ""))
                cells.append(f"{value:20.20}" if head == "time" else f"{value:>6.6}")
            print("  " + "  ".join(cells))
    else:
        print("Flight recorder: no telemetry file yet (run predates it, or no run).")
    print("Verdict:", diagnose_verdict(alive, status, tail[-1] if tail else None,
                                       boot_time_unix()))
    print(f"Deeper: tail -30 {os.path.join(out_dir, CONSOLE_LOG_NAME)}")
    return 0


####################################################################################
### STAGES
####################################################################################
def stage_banner():
    line = "=" * 70
    print(line)
    print(f" {TOOL_NAME} v{TOOL_VERSION}  -  DeepFaune camera-trap batch runner")
    print(f" {GITHUB_URL}")
    print(f" {CONTACT}")
    print(line)


def stage_find_source(args):
    """Stage 2: find and confirm the source drive. Returns (source, uuid) or None."""
    candidates = []
    if args.source:
        candidates = [args.source]
    else:
        candidates = find_archive_candidates()
    chosen = None
    if candidates:
        print("Found candidate archive location(s):")
        for path in candidates:
            uuid = path_device_uuid(path)
            tag = " (UUID matches)" if uuid == EXPECTED_DRIVE_UUID else ""
            print(f"  {path}{tag}")
        chosen = candidates[0]
    else:
        print("No archive auto-detected under /media or /mnt.")
    if not args.yes:
        reply = _default_prompt(
            f"Use [{chosen or 'enter a path'}]? Press Enter to accept or type a path: "
        ).strip()
        if reply:
            chosen = reply
    if not chosen or not os.path.isdir(chosen):
        print(f"Source not usable: {chosen!r}")
        return None
    options = mount_options(chosen)
    if not is_read_only(options):
        print(
            f"WARNING: {chosen} is not mounted read-only (options: {options or 'unknown'}). "
            "Remount read-only to protect the archive, for example:\n"
            f"  sudo mount -o remount,ro \"$(findmnt -no TARGET --target '{chosen}')\""
        )
        if not args.yes and _default_prompt("Continue anyway? [y/N] ").strip().lower() not in ("y", "yes"):
            return None
    uuid = path_device_uuid(chosen)
    if uuid and uuid != EXPECTED_DRIVE_UUID:
        print(f"Note: drive UUID {uuid} differs from the expected {EXPECTED_DRIVE_UUID}.")
    return chosen, uuid


def stage_assess(out_dir, source, uuid, rescan):
    """Stage 3: report done/remaining/total with the true percentage."""
    print("Assessing work (this can take a while the first time)...")
    done, remaining, total, shards, pct, from_cache = assess_work(
        out_dir, source, uuid, rescan
    )
    src = "cached" if from_cache else "freshly counted"
    print(f"  Already classified: {done:,}")
    print(f"  Remaining:          {remaining:,}")
    print(f"  Total on drive:     {total:,} in {shards:,} shards ({src})")
    print(f"  Done:               {pct:.1f}%")
    return done, remaining, total


# --- interactive setting prompts -------------------------------------------
def parse_detector(reply):
    for choice in dfb.DETECTOR_CHOICES:
        if choice.lower() == reply.strip().lower():
            return choice, True
    return None, False


def parse_onoff(reply):
    r = reply.strip().lower()
    if r in ("on", "yes", "y", "true", "1"):
        return True, True
    if r in ("off", "no", "n", "false", "0"):
        return False, True
    return None, False


def parse_threshold(reply):
    try:
        value = float(reply)
    except ValueError:
        return None, False
    return (value, True) if 0 < value <= 1 else (None, False)


def parse_nonneg_int(reply):
    try:
        value = int(reply)
    except ValueError:
        return None, False
    return (value, True) if value >= 0 else (None, False)


def parse_pos_int(reply):
    try:
        value = int(reply)
    except ValueError:
        return None, False
    return (value, True) if value >= 1 else (None, False)


# Command-line options that override a resumed run's recorded settings. Used
# to tell "the user asked for this value" apart from "this is the parser
# default", so resuming never silently changes how a run is classified.
SETTING_FLAGS = {
    "detector": ("--detector",),
    "threshold": ("--threshold",),
    "maxlag": ("--maxlag",),
    "batch_size": ("--batch-size",),
    "threads": ("--threads",),
    "merge_every": ("--merge-every",),
    "birds": ("--birds", "--no-birds"),
}


def explicit_cli_settings(argv, flags=SETTING_FLAGS):
    """Names of settings the user gave explicitly on the command line."""
    argv = list(argv or [])
    given = set()
    for name, options in flags.items():
        for option in options:
            if any(arg == option or arg.startswith(option + "=") for arg in argv):
                given.add(name)
                break
    return given


# Metadata field -> params key, for settings a resume should carry forward.
RESUMED_SETTINGS = {
    "detector": "detector",
    "threshold": "threshold",
    "maxlag": "maxlag",
    "birds": "birds",
    "batch_size": "batch_size",
    "threads": "threads",
    "merge_every": "merge_every",
}


def apply_previous_settings(params, previous, explicit=()):
    """Overlay a partially complete run's settings onto params.

    A run in progress owns its settings: continuing it must not silently
    reclassify the remainder with different ones (the drift that produced a
    mixed exclusion list mid-archive). Anything named explicitly on the
    command line still wins. Returns the list of settings taken from the
    previous run, for reporting.
    """
    if not previous:
        return []
    taken = []
    for field, key in RESUMED_SETTINGS.items():
        if key in explicit or field not in previous or previous[field] is None:
            continue
        if params.get(key) != previous[field]:
            params[key] = previous[field]
            taken.append(key)
    if "exclude_classes" not in explicit and previous.get("excluded_classes") is not None:
        recorded = list(previous["excluded_classes"])
        if sorted(recorded) != sorted(params.get("exclude_classes") or []):
            params["exclude_classes"] = recorded
            taken.append("exclude_classes")
    return taken


def prompt_setting(explain, label, current, parser, shown=None):
    """Print a one-line explanation, then prompt. Return the new value or current.

    Enter (blank) keeps the current value; an unparseable entry keeps it too,
    with a short note.
    """
    display = shown if shown is not None else current
    print(explain)
    reply = _default_prompt(f"  {label} [{display}]: ").strip()
    if not reply:
        return current
    value, ok = parser(reply)
    if not ok:
        print(f"  Not valid; keeping {display}.")
        return current
    return value


def _print_params(params):
    for key in ("detector", "birds", "threshold", "maxlag", "batch_size", "threads", "merge_every"):
        print(f"  {key:12s} {params[key]}")
    excluded = params.get("exclude_classes") or []
    print(f"  {'exclude':12s} {', '.join(excluded) if excluded else 'none'}")


def stage_configure(args, out_dir, argv=None, source=None):
    """Stage 4: show parameters and let the user confirm or change them.

    When the out-dir already holds a partially complete run, that run's own
    settings become the suggested values, so continuing it cannot silently
    reclassify the remainder differently. Anything given explicitly on the
    command line still wins, and the prompts can still change anything.
    """
    params = {
        "detector": args.detector,
        "birds": args.birds,
        "threshold": args.threshold,
        "maxlag": args.maxlag,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "merge_every": args.merge_every,
        # Never set for a new run: species exclusion belongs to filtering, where
        # it is reversible, not to classification, where it is permanent. A run
        # started before this changed keeps its own list (below), so resuming it
        # stays consistent with the shards it has already written.
        "exclude_classes": [],
    }
    # Any recorded run in this out-dir owns its settings: continuing it (or
    # adding new photographs to it later) must classify consistently with the
    # shards already written. No line counting here - assess_work has already
    # walked the shard CSVs and this only needs to know a run exists.
    previous = load_json(os.path.join(out_dir, "run_metadata.p0.json"))
    if previous:
        explicit = explicit_cli_settings(
            sys.argv[1:] if argv is None else argv
        )
        taken = apply_previous_settings(params, previous, explicit)
        started = str(previous.get("utc_start_time") or "")[:16].replace("T", " ")
        print(
            f"Continuing the run already in this folder (started {started} UTC); "
            "its settings are the suggested values below."
        )
        if taken:
            print("  Reusing from that run: " + ", ".join(sorted(taken)))
        if params["exclude_classes"]:
            print(
                "  NOTE: that run excluded species during classification "
                f"({', '.join(params['exclude_classes'])}). Continuing with the "
                "same list so its results stay consistent. New runs no longer "
                "do this: exclude species in dashboard_and_filter.html instead, "
                "where the choice can be undone."
            )
        if explicit:
            print("  Overridden on the command line: " + ", ".join(sorted(explicit)))
        prev_root = previous.get("root")
        if source and prev_root and os.path.abspath(prev_root) != os.path.abspath(source):
            print(
                f"NOTE: that run read the archive at {prev_root}; it is now at "
                f"{source}. The drive has been mounted under a different name. "
                "Image paths are normalised to the current mount when the "
                "master CSV is rebuilt, so the outputs stay consistent."
            )
    print("Run parameters:")
    _print_params(params)
    if args.yes:
        return params
    print("\nAdjust the settings. Press Enter to keep each suggested value.")
    params["detector"] = prompt_setting(
        "Detector: the model that finds animals before they are named. DF is "
        "fastest (default); DFbsMDS is more thorough; DFMDS is most thorough but "
        "slow; MDR is too slow without a graphics card.",
        f"Detector ({'/'.join(dfb.DETECTOR_CHOICES)})", params["detector"], parse_detector)
    params["birds"] = prompt_setting(
        "Bird sub-groups: sort birds into groups (corvid, raptor, passerine and "
        "so on) instead of just 'bird'.",
        "Bird sub-groups (on/off)", params["birds"], parse_onoff,
        shown=("on" if params["birds"] else "off"))
    params["threshold"] = prompt_setting(
        "Confidence threshold: how sure the model must be to name a species. "
        "Higher means fewer wrong names but more animals left 'undefined'; lower "
        "names more but with more mistakes.",
        "Threshold (0 to 1)", params["threshold"], parse_threshold)
    params["maxlag"] = prompt_setting(
        "Sequence gap: photos at one camera within this many seconds count as a "
        "single detection, so a burst of frames is counted once.",
        "Sequence gap (seconds)", params["maxlag"], parse_nonneg_int)
    params["batch_size"] = prompt_setting(
        "Batch size: how many image crops are classified at once. Affects speed "
        "and memory only, not the results.",
        "Batch size", params["batch_size"], parse_pos_int)
    params["threads"] = prompt_setting(
        "Processor cores: how much of the processor to use. More is faster but "
        "uses more memory and runs hotter.",
        "Processor cores", params["threads"], parse_pos_int)
    params["merge_every"] = prompt_setting(
        "Update interval: how often the master CSV and dashboard refresh during "
        "the run (seconds). Does not affect the results.",
        "Update interval (seconds)", params["merge_every"], parse_nonneg_int)
    print("\nFinal settings:")
    _print_params(params)
    if _default_prompt("Proceed with these settings? [Y/n] ").strip().lower() in ("n", "no"):
        print("Aborted by user before launch.")
        return None
    return params


def stage_swap(args):
    """Stage 4 (swap): reclaim orphaned swapfiles, then check total swap."""
    # The only thing that has ever filled this disk is orphaned swapfiles;
    # catch them at launch rather than stopping a run days later.
    reclaim_orphaned_swapfiles(prompt_fn=None if args.yes else _default_prompt)
    total = current_swap_total_bytes()
    print(f"Swap total: {gib(total):.1f} GiB")
    # Non-interactive override by flag (sizes the swapfile up or down).
    if args.swap_gib is not None:
        if args.swap_gib <= 0:
            print("Skipping swapfile creation (--swap-gib <= 0).")
            return
        print(f"Creating a {args.swap_gib} GiB swapfile (requires sudo)...")
        if create_swapfile(args.swap_gib):
            print("Swapfile active and persisted in /etc/fstab.")
        else:
            print("Swapfile creation did not complete; continuing without it.")
        return
    if not needs_swap(total):
        return
    print(
        f"Swap is below the recommended {MIN_TOTAL_SWAP_GIB} GiB. A swapfile guards "
        "against the out-of-memory killer on this 16 GiB box."
    )
    if args.yes:
        return  # non-interactive: do not touch system swap without consent
    reply = _default_prompt(
        "Create a swapfile now? Enter size in GiB (for example 16), or Enter to skip: "
    ).strip()
    if not reply:
        return
    try:
        size = int(reply)
    except ValueError:
        print("Not a number; skipping swap creation.")
        return
    print(f"Creating a {size} GiB swapfile (requires sudo)...")
    if create_swapfile(size):
        print("Swapfile active and persisted in /etc/fstab.")
    else:
        print("Swapfile creation did not complete; continuing without it.")


####################################################################################
### LIVE READOUT (Stage 6)
####################################################################################
def system_info():
    """CPU load and temperature, available RAM and swap used. Best-effort."""
    info = {"cpu_pct": None, "cpu_temp": None, "mem_avail": None, "swap_used": None}
    try:
        import psutil
    except ImportError:
        return info
    try:
        info["cpu_pct"] = psutil.cpu_percent(interval=None)
        info["mem_avail"] = psutil.virtual_memory().available  # MemAvailable
        info["swap_used"] = psutil.swap_memory().used
        temps = psutil.sensors_temperatures() or {}
        for key in ("coretemp", "k10temp", "acpitz"):
            if key in temps and temps[key]:
                info["cpu_temp"] = temps[key][0].current
                break
    except Exception:
        pass
    return info


def render_readout(out_dir, stats, status, sysinfo):
    """Build the rich panel for the live readout."""
    from rich.table import Table
    from rich.panel import Panel

    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="bold")
    table.add_column()

    status = status or {}
    total = status.get("archive_total")
    done = status.get("true_done")
    pct = status.get("true_pct")
    if total:
        table.add_row("Classified", f"{done:,} / {total:,}  ({pct:.1f}%)")
    else:
        table.add_row("Classified", "starting...")
    table.add_row("Blanks", f"{stats.blank_pct():.1f}% empty")
    top = stats.top_species()
    if top:
        table.add_row("Top species", ", ".join(f"{name} ({n})" for name, n in top))
    else:
        table.add_row("Top species", "none yet")
    table.add_row("ETA", absolute_eta(status.get("eta_seconds")))
    table.add_row("Rate", f"{status.get('rate_img_per_s', 0):.2f} img/s")
    table.add_row("Current shard", status.get("current_shard") or "-")
    if status.get("skipped_unreadable"):
        table.add_row("Skipped (unreadable)", str(status["skipped_unreadable"]))
    if stats.unset_clock:
        table.add_row("Unset-clock (1970)", str(stats.unset_clock))

    cpu = sysinfo.get("cpu_pct")
    temp = sysinfo.get("cpu_temp")
    cpu_str = f"{cpu:.0f}%" if cpu is not None else "n/a"
    temp_str = f"{temp:.0f} C" if temp is not None else "n/a"
    table.add_row("CPU", f"{cpu_str}  /  {temp_str}")
    mem = sysinfo.get("mem_avail")
    swp = sysinfo.get("swap_used")
    mem_str = f"{gib(mem):.1f} GiB avail" if mem is not None else "n/a"
    swp_str = f"{gib(swp):.1f} GiB used" if swp is not None else "n/a"
    table.add_row("Memory", f"RAM {mem_str}  /  swap {swp_str}")

    start = status.get("start_time_unix")
    if start:
        elapsed = dfb.format_duration(time.time() - start)
        table.add_row("Elapsed", elapsed)
    state = status.get("state", "?")
    return Panel(table, title=f"{TOOL_NAME}: {state}", border_style="green")


def live_readout(out_dir, software_dir=None, refresh_secs=3):
    """Stage 6: render the live panel until the worker exits or the user detaches.

    Closing the readout (Ctrl-C) does not stop the worker; re-running the tool
    reattaches. Whenever master.csv changes it is copied to the Desktop and the
    dashboard is rebuilt in the background (memory permitting). Returns the final
    status dict.
    """
    try:
        from rich.live import Live
    except ImportError:
        return _plain_readout(out_dir, software_dir, refresh_secs)
    stats = StatsAccumulator()
    last_master_mtime = None
    dash_proc = None
    final_status = None
    try:
        with Live(refresh_per_second=4, screen=False) as live:
            while True:
                stats.consume_new(out_dir)
                status = read_status(out_dir) or {}
                final_status = status or final_status
                live.update(render_readout(out_dir, stats, status, system_info()))
                new_mtime = _refresh_desktop_master(out_dir, last_master_mtime)
                dash_proc = _maybe_rebuild_dashboard(
                    software_dir, out_dir, new_mtime, last_master_mtime, dash_proc
                )
                last_master_mtime = new_mtime
                state = status.get("state")
                if state in ("finished", "stopped") and not worker_running(out_dir):
                    break
                if status and not worker_running(out_dir) and state not in ("finished", "stopped"):
                    # Worker vanished without a clean final status.
                    break
                time.sleep(refresh_secs)
    except KeyboardInterrupt:
        print("\nDetached from the readout; the worker keeps running.")
        print(f"Re-run {TOOL_NAME} to reattach.")
    return final_status


def _plain_readout(out_dir, software_dir=None, refresh_secs=3):
    """Fallback readout without rich: periodic one-line status."""
    stats = StatsAccumulator()
    last_master_mtime = None
    dash_proc = None
    final_status = None
    try:
        while True:
            stats.consume_new(out_dir)
            status = read_status(out_dir) or {}
            final_status = status or final_status
            if status.get("archive_total"):
                print(
                    f"{status.get('true_done', 0):,}/{status['archive_total']:,} "
                    f"({status.get('true_pct', 0):.1f}%) | "
                    f"blanks {stats.blank_pct():.1f}% | {absolute_eta(status.get('eta_seconds'))}"
                )
            new_mtime = _refresh_desktop_master(out_dir, last_master_mtime)
            dash_proc = _maybe_rebuild_dashboard(
                software_dir, out_dir, new_mtime, last_master_mtime, dash_proc
            )
            last_master_mtime = new_mtime
            if status.get("state") in ("finished", "stopped") and not worker_running(out_dir):
                break
            time.sleep(refresh_secs)
    except KeyboardInterrupt:
        print(f"\nDetached; worker keeps running. Re-run {TOOL_NAME} to reattach.")
    return final_status


def _refresh_desktop_master(out_dir, last_mtime):
    master = os.path.join(out_dir, MASTER_NAME)
    try:
        mtime = os.path.getmtime(master)
    except OSError:
        return last_mtime
    if mtime != last_mtime:
        copy_file(master, os.path.join(DESKTOP_DIR, DESKTOP_MASTER))
    return mtime


def _maybe_rebuild_dashboard(software_dir, out_dir, new_mtime, last_mtime, dash_proc):
    """Rebuild the dashboard in the background when the master has changed.

    Skips if no software dir, if a previous build is still running, or if free
    memory is low. Returns the (possibly new) background process handle.
    """
    if not software_dir or new_mtime == last_mtime:
        return dash_proc
    if dash_proc is not None and dash_proc.poll() is None:
        return dash_proc  # a build is still in progress; do not overlap
    if not enough_memory_for_dashboard():
        return dash_proc  # too little free RAM right now; the next change retries
    return spawn_dashboard(software_dir, out_dir) or dash_proc


####################################################################################
### MAIN
####################################################################################
def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description="Friendly front end for the DeepFaune batch orchestrator.",
    )
    parser.add_argument("--source", help="Path to the camera archive root (default: auto-detect).")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help=f"Output directory (default: {DEFAULT_OUT_DIR}).")
    parser.add_argument("--software-dir", default=DEFAULT_SOFTWARE_DIR, help="DeepFaune source and weights directory.")
    parser.add_argument("--detector", default=DEFAULTS["detector"], choices=dfb.DETECTOR_CHOICES)
    parser.add_argument("--threshold", type=float, default=DEFAULTS["threshold"])
    parser.add_argument("--maxlag", type=int, default=DEFAULTS["maxlag"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--threads", type=int, default=DEFAULTS["threads"])
    parser.add_argument("--swap-gib", type=int, default=None,
                        help="Create a swapfile of this many GiB before launch (needs sudo).")
    parser.add_argument("--merge-every", type=int, default=DEFAULTS["merge_every"])
    birds = parser.add_mutually_exclusive_group()
    birds.add_argument("--birds", dest="birds", action="store_true", default=DEFAULTS["birds"])
    birds.add_argument("--no-birds", dest="birds", action="store_false")
    parser.add_argument("--rescan", action="store_true", help="Force a fresh total count, ignoring the cache.")
    parser.add_argument("--yes", action="store_true", help="Accept defaults without prompting (non-interactive).")
    parser.add_argument("--attach", action="store_true", help="Attach to the live readout of a running worker and exit.")
    parser.add_argument("--dashboard", action="store_true", help="Build and open the dashboard from the current results, then exit.")
    parser.add_argument("--diagnose", action="store_true", help="Print run settings, the flight recorder tail, and a crash/health verdict, then exit.")
    parser.add_argument("--outputs", action="store_true", help="Rebuild master.csv and every Desktop output from the shard CSVs now (adds sequence_id and normalises paths), then exit. Use --source to set the archive root for the paths.")
    parser.add_argument("--stop", action="store_true", help="Intentionally stop the current run (clean boundary; nothing restarts it until you relaunch).")
    parser.add_argument("--install-service", action="store_true", help="Install a systemd service that restarts the worker after an OOM kill or crash, resuming the current run's own settings. Never restarts an intentionally stopped or finished run, and does not autostart at boot (needs sudo).")
    parser.add_argument("--uninstall-service", action="store_true", help="Disable and remove the auto-resume service (needs sudo).")
    parser.add_argument("--no-update", action="store_true", help="Skip the self-update check this launch.")
    parser.add_argument("--update-check", choices=("auto", "never"), default=None,
                        help="Persist the update-check preference (auto or never; default: auto).")
    parser.add_argument("--watchdog", action="store_true",
                        help="Auto-resume the worker if it exits with work remaining (bounded retries).")
    parser.add_argument("--max-retries", type=int, default=3, help="Watchdog retry cap (default: 3).")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    out_dir = os.path.abspath(args.out_dir)
    software_dir = os.path.abspath(args.software_dir)

    # Persist the update-check preference if the user set it explicitly.
    if args.update_check is not None:
        config = load_config()
        config["update_check"] = args.update_check
        save_config(config)

    # Diagnose mode: read-only, no update check, no drive needed.
    if args.diagnose:
        return run_diagnose(out_dir)

    # Rebuild every output from the shard CSVs, without touching the archive.
    if args.outputs:
        if worker_running(out_dir):
            print("A worker is running; its next merge rebuilds master.csv anyway.")
            print("Stop it first (dfrun --stop) if you want the Desktop files now.")
            return 1
        finish_outputs(out_dir, software_dir, force_merge=True,
                       canonical_root=args.source)
        return 0

    stage_banner()

    if args.stop:
        return stop_run(out_dir)
    if args.uninstall_service:
        return uninstall_service()
    if args.install_service:
        return install_service(args, out_dir, software_dir)

    # Stage 0: self-update (before any work). Never under a live worker.
    self_update_check(args, _HERE, worker_running(out_dir))

    # Dashboard-only mode: build (and open) the dashboard from current results.
    if args.dashboard:
        print("Building the dashboard from the current results...")
        html = build_dashboard(software_dir, out_dir, open_after=True)
        if html:
            print(f"Dashboard: {html}")
            return 0
        print("Could not build the dashboard (is there any output yet?).")
        return 1

    # Attach mode: just show the readout of a running worker.
    if args.attach:
        if not worker_running(out_dir):
            print("No running worker to attach to.")
            return 1
        final = live_readout(out_dir, software_dir=software_dir)
        if final and final.get("state") == "finished" and not worker_running(out_dir):
            print("Run finished.")
            finish_outputs(out_dir, software_dir)
        return 0

    # Single instance (A10): if a worker is already running, reattach.
    if worker_running(out_dir):
        print("A worker is already running; attaching to its live readout.")
        final = live_readout(out_dir, software_dir=software_dir)
        if final and final.get("state") == "finished" and not worker_running(out_dir):
            print("Run finished.")
            finish_outputs(out_dir, software_dir)
        return 0

    found = stage_find_source(args)
    if not found:
        return 1
    source, uuid = found

    stage_assess(out_dir, source, uuid, args.rescan)
    params = stage_configure(args, out_dir, argv=argv, source=source)
    if params is None:
        return 1
    stage_swap(args)

    print("Launching the classifier, detached...")
    supervise(software_dir, source, out_dir, params, args)
    return 0


def supervise(software_dir, source, out_dir, params, args):
    """Launch the worker, show the readout, and optionally auto-resume.

    With --watchdog, if the worker exits with work remaining while the drive is
    still present, the run is resumed up to --max-retries times, each logged. A
    user detaching (the worker stays alive) is never treated as a failure.
    """
    retries = 0
    while True:
        if service_installed():
            # The service starts the worker with these settings (written to
            # the run metadata first) and systemd owns crash-resume.
            if launch_via_service(software_dir, source, out_dir, params):
                print(f"Launched under systemd ({SERVICE_NAME}): crashes "
                      "auto-resume; dfrun --stop stops it for good.")
            else:
                print("Service start failed; launching directly instead.")
                launch_detached(software_dir, source, out_dir, params)
        else:
            pid = launch_detached(software_dir, source, out_dir, params)
            print(f"Worker PID {pid}.")
        print(
            "Detailed log: "
            f"{os.path.join(out_dir, 'deepfaune_batch.p0.log')}"
        )
        print("You can close this window or disconnect (if connected remotely); the run continues.")
        final = live_readout(out_dir, software_dir=software_dir)
        if worker_running(out_dir):
            print("Detached; the worker is still running. Re-run dfrun to reattach.")
            return
        state = (final or {}).get("state")
        if state == "finished":
            print("Run finished.")
            finish_outputs(out_dir, software_dir)
            return
        if (args.watchdog and not service_installed() and retries < args.max_retries
                and os.path.isdir(source) and looks_like_archive(source)):
            done = dfb.count_classified_images(out_dir)
            total = (final or {}).get("archive_total") or 0
            if total and done < total:
                retries += 1
                print(
                    f"Watchdog: worker stopped with work remaining; resuming "
                    f"(attempt {retries}/{args.max_retries})..."
                )
                continue
        print("Worker stopped early. Re-run the tool to resume from where it left off.")
        finish_outputs(out_dir, software_dir)  # build outputs/dashboard from what is done
        return


def finish_outputs(out_dir, software_dir, force_merge=False, canonical_root=None):
    """Stage 7: write the Desktop outputs and build (and open) the dashboard.

    force_merge rebuilds master.csv even when one exists, which is what
    --outputs needs after shard CSVs have been added, removed or re-derived
    (the merge is where sequence_id and path normalisation are applied).
    canonical_root overrides the archive root recorded by the last run, for
    when the drive has since been remounted elsewhere.
    """
    root = canonical_root or recorded_archive_root(out_dir)
    print("Writing the spreadsheet-friendly outputs to the Desktop...")
    master = os.path.join(out_dir, MASTER_NAME)
    if force_merge or not os.path.exists(master):
        n_files, n_rows = dfb.merge_csvs(out_dir, master, canonical_root=root)
        print(f"  master.csv: {n_files:,} shards, {n_rows:,} rows"
              + (f", paths under {root}" if root else ""))
    copy_file(master, os.path.join(DESKTOP_DIR, DESKTOP_MASTER))
    rows, capped, n_species, n_stations = write_wildlife_and_summary(
        out_dir, DESKTOP_DIR, canonical_root=root
    )
    print(f"  {DESKTOP_MASTER}: every image (too large for Excel; open in R)")
    print(f"  {DESKTOP_WILDLIFE}: {rows:,} wildlife rows" + (" (capped to Excel's limit)" if capped else ""))
    print(f"  {DESKTOP_SUMMARY}: {n_species} species, {n_stations} stations")
    print("Building the dashboard (this can take a few minutes on the full archive)...")
    html = build_dashboard(software_dir, out_dir, open_after=True)
    if html:
        print(f"  {DESKTOP_DASHBOARD}: opened in your browser")
    else:
        print("  dashboard not built (see dashboard.log); the CSVs are still available")
    print("Open the wildlife and summary files in Excel, the master in R, and the dashboard in a browser.")
    workbench = os.path.join(DESKTOP_DIR, DESKTOP_WORKBENCH)
    if os.path.exists(workbench):
        print(f"To explore and filter by your own thresholds and species, open "
              f"{DESKTOP_WORKBENCH} and load {DESKTOP_MASTER}.")


if __name__ == "__main__":
    sys.exit(main())
