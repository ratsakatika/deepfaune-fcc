#!/usr/bin/env python3
# Copyright (c) 2025 Fundatia Conservation Carpathia batch tooling.
#
# A resumable, CPU-only batch orchestrator that drives the official DeepFaune
# v1.4.1 PredictorImage engine over a large camera-trap archive. It does NOT
# reimplement the engine: it shards the work by leaf directory, builds the
# detector and classifier once per process, injects those singletons into each
# lightweight PredictorImage, and writes one CSV per shard atomically so that an
# interrupted run can be restarted and will skip the shards already finished.
#
# Design notes (see HANDOVER.md section 6 for the full specification):
#   - Sharding by leaf directory bounds memory (one folder of images in flight at
#     a time) and preserves EXIF burst sequences (a deployment's burst lives in
#     one folder; grouping across folders would corrupt sequence aggregation).
#   - The 1.2 GB classifier is built once and reused for every shard.
#   - The per-shard CSVs are the source of truth. The optional master CSV is only
#     ever a derived snapshot, regenerated (never appended) from the shard CSVs.
#   - Outputs (CSVs, logs, metadata) go to local disk only; the source archive is
#     read-only and must never be written to.
#   - CPU only: CUDA is disabled by setting CUDA_VISIBLE_DEVICES="" before torch
#     is imported, and the engine is additionally asked for device="cpu".
#
# Only the standard library is imported at module load. torch and the DeepFaune
# engine are imported lazily inside run(), so --dry-run, --merge and the unit
# tests stay cheap and never need torch or the model weights.

import argparse
import csv
import glob
import hashlib
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("deepfaune_batch")

# Photo extensions classified by this run. Videos (.mp4 and similar) are
# deliberately excluded; they need PredictorVideo, not PredictorImage.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}

# Forced inference device for the whole run. This box has no usable GPU.
DEVICE = "cpu"

# Detector names understood by detectTools.Detector.
DETECTOR_CHOICES = ["DF", "MDS", "DFbsMDS", "DFMDS", "MDR"]

# Languages supported by the engine's class dictionaries (no Romanian: map
# English labels to Romanian downstream if FCC needs it).
LANG_CHOICES = ["fr", "en", "it", "de", "es"]

CSV_HEADER = [
    "filename",
    "date",
    "seqnum",
    "prediction_seq",
    "score_seq",
    "prediction_image",
    "score_image",
    "animal_count",
    "human_count",
]

# A shard CSV is named "<sanitised-relpath>__<8 hex>.csv" by shard_csv_name.
# This pattern selects only shard CSVs, so the merge and the counters ignore
# master.csv, the summary files and any Desktop copies.
SHARD_CSV_RE = re.compile(r"__[0-9a-f]{8}\.csv$")

# Non-survey directories that appear on the exFAT drive and must never be
# classified or counted: the Windows recycle bin, volume metadata, and the
# exFAT repair artefacts left behind by a filesystem check.
SYSTEM_DIR_NAMES = {"$RECYCLE.BIN", "System Volume Information"}


def is_shard_csv(name):
    """True if name looks like a per-shard CSV produced by shard_csv_name."""
    return bool(SHARD_CSV_RE.search(os.path.basename(name)))


def is_system_dir(name):
    """True for OS or filesystem directories that are not survey content."""
    return name in SYSTEM_DIR_NAMES or name.startswith("FOUND.")


####################################################################################
### PURE LOGIC (no torch, unit-tested)
####################################################################################
def is_junk(name):
    """True for macOS metadata files we must never feed to the engine.

    Catches AppleDouble sidecars (._something) and Finder's .DS_Store, including
    ._image.jpg files that would otherwise pass the extension filter.
    """
    return name.startswith("._") or name == ".DS_Store"


def is_image_file(name):
    """True if the filename has a still-image extension we classify."""
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def list_images_in_dir(dirpath, filenames):
    """Return the sorted absolute image paths directly inside one directory.

    Junk files are dropped; videos and other non-images are ignored.
    """
    images = sorted(
        fn for fn in filenames if not is_junk(fn) and is_image_file(fn)
    )
    return [str(Path(dirpath) / fn) for fn in images]


def find_shards(root):
    """Enumerate shards under root.

    A shard is one leaf directory (a directory that directly contains at least
    one image) paired with the sorted list of image paths it holds directly.
    Directories with no images of their own are not shards even if their
    subdirectories contain images. The returned list is sorted by directory
    path so that every process computes the same order, which is what makes
    --partition selection consistent across processes.
    """
    root = os.path.abspath(root)
    shards = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Prune non-survey directories in place so we never descend into them,
        # and keep the traversal deterministic.
        dirnames[:] = sorted(d for d in dirnames if not is_system_dir(d))
        images = list_images_in_dir(dirpath, filenames)
        if images:
            shards.append((dirpath, images))
    shards.sort(key=lambda item: item[0])
    return shards


def select_partition(shards, num_partitions, partition):
    """Return the disjoint subset of shards handled by this partition.

    With num_partitions == 1 every shard is returned. Otherwise shard i goes to
    partition (i % num_partitions); across all partitions this covers every
    shard exactly once.
    """
    if num_partitions <= 1:
        return list(shards)
    return [s for i, s in enumerate(shards) if i % num_partitions == partition]


def shard_csv_name(root, leaf_dir):
    """Deterministic, collision-resistant CSV filename for a shard.

    The name is built from the shard's path relative to root, so it is stable
    across runs (this is what lets a restart detect an already-finished shard).
    A short hash of the relative path is appended to guarantee uniqueness even
    if two different paths sanitise to the same readable prefix.
    """
    root = os.path.abspath(root)
    leaf_dir = os.path.abspath(leaf_dir)
    rel = os.path.relpath(leaf_dir, root)
    if rel == ".":
        rel = os.path.basename(root) or "root"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", rel.replace(os.sep, "__"))
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8]
    return f"{safe}__{digest}.csv"


def plan_shards(shards, out_dir, root, rescan):
    """Filter shards down to those still needing work (resume behaviour).

    Returns a list of (leaf_dir, image_paths, csv_path). A shard whose final CSV
    already exists is skipped unless rescan is set, in which case it is
    reprocessed and the CSV overwritten.
    """
    out_dir = os.path.abspath(out_dir)
    plan = []
    for leaf_dir, image_paths in shards:
        csv_path = os.path.join(out_dir, shard_csv_name(root, leaf_dir))
        if os.path.exists(csv_path) and not rescan:
            continue
        plan.append((leaf_dir, image_paths, csv_path))
    return plan


def is_within(child, parent):
    """True if child equals parent or sits inside it.

    Uses commonpath rather than a string prefix so edge cases such as parent
    being the filesystem root ("/") do not misfire (root + os.sep would be "//").
    """
    child = os.path.abspath(child)
    parent = os.path.abspath(parent)
    try:
        return os.path.commonpath([child, parent]) == parent
    except ValueError:
        # Raised when the paths cannot be compared (different drives on Windows).
        return False


def atomic_write_csv(path, header, rows):
    """Write a CSV to a temp file in the same directory, then atomically rename.

    A run that is killed mid-write therefore never leaves a partial final CSV:
    either the rename completed (a good CSV exists) or it did not (the shard is
    redone on resume). The temp file is removed on any failure.
    """
    path = os.fspath(path)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=directory, prefix=os.path.basename(path) + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)  # atomic within the same filesystem
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def iter_shard_csvs(out_dir, exclude=()):
    """Yield the per-shard CSV paths in out_dir, sorted.

    Globs "*.csv" and keeps only files matching the shard-name pattern, so the
    ".tmp" temp files, the JSON metadata sidecar, master.csv and the summary
    files are all ignored. Paths in exclude (such as the master itself) are
    skipped too, so regenerating the master never reads its own output.
    """
    exclude_abs = {os.path.abspath(p) for p in exclude}
    for path in sorted(glob.glob(os.path.join(out_dir, "*.csv"))):
        if not is_shard_csv(path):
            continue
        if os.path.abspath(path) in exclude_abs:
            continue
        yield path


def merge_csvs(out_dir, merge_out, header=CSV_HEADER):
    """Stream-concatenate the per-shard CSVs in out_dir into merge_out.

    The header is written once; each source file's own header row is dropped and
    its data rows are copied through csv.writer so quoting stays correct. Files
    are streamed, not loaded whole. The write is atomic (temp then rename) and
    merge_out is excluded from the inputs, so this is an idempotent rebuild that
    never appends. Returns (n_files, n_rows).
    """
    out_dir = os.path.abspath(out_dir)
    merge_out = os.path.abspath(merge_out)
    directory = os.path.dirname(merge_out) or "."
    os.makedirs(directory, exist_ok=True)
    sources = list(iter_shard_csvs(out_dir, exclude=[merge_out]))
    fd, tmp = tempfile.mkstemp(
        dir=directory, prefix=os.path.basename(merge_out) + ".", suffix=".tmp"
    )
    n_files = 0
    n_rows = 0
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)
            writer.writerow(header)
            for src in sources:
                with open(src, newline="", encoding="utf-8") as handle:
                    reader = csv.reader(handle)
                    try:
                        next(reader)  # drop this file's header row
                    except StopIteration:
                        continue  # empty file
                    n_files += 1
                    for row in reader:
                        writer.writerow(row)
                        n_rows += 1
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, merge_out)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return n_files, n_rows


def count_classified_images(out_dir):
    """Count images already classified across all existing shard CSVs.

    Each shard CSV holds one row per image plus a header, so the count is the
    sum of (lines - 1) over the shard CSVs. Line counting is used for speed; an
    embedded newline in a quoted field would be a negligible over-count and is
    not worth a slower csv parse over the whole output directory. This is the
    basis for the true overall percentage that survives restarts.
    """
    total = 0
    for path in iter_shard_csvs(out_dir):
        try:
            with open(path, "rb") as handle:
                lines = sum(1 for _ in handle)
        except OSError:
            continue
        total += max(0, lines - 1)
    return total


def format_duration(seconds):
    """Format a number of seconds as H:MM:SS for human-readable ETAs."""
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


class RateTracker:
    """Smoothed throughput from a rolling time window of cumulative counts.

    Sampling the running image count over a recent window (rather than dividing
    total-by-elapsed) makes the rate update continuously between shard
    completions instead of reading near zero mid-shard and jumping on "Finished".
    """

    def __init__(self, window_seconds=300):
        self.window = window_seconds
        self.samples = deque()  # (monotonic_time, cumulative_count)

    def update(self, count, now=None):
        now = time.monotonic() if now is None else now
        self.samples.append((now, count))
        # Keep at least two samples; drop those older than the window.
        while len(self.samples) > 2 and now - self.samples[0][0] > self.window:
            self.samples.popleft()

    def rate(self):
        if len(self.samples) < 2:
            return 0.0
        t0, c0 = self.samples[0]
        t1, c1 = self.samples[-1]
        dt = t1 - t0
        return (c1 - c0) / dt if dt > 0 else 0.0


class Progress:
    """Tracks true overall progress across all sessions plus a smoothed rate.

    The true figure is images classified across every session (counted from the
    existing shard CSVs at start, plus shards completed this session) divided by
    the whole-archive total, so the percentage does not understate real progress
    after a restart. The rate and ETA come from a rolling window so they update
    continuously, and the ETA is for finishing the whole archive.
    """

    def __init__(self, archive_total, done_start, window_seconds=300):
        self.archive_total = archive_total          # images on the drive
        self.done_start = done_start                # classified before this session
        self.completed_session = 0                  # persisted this session
        self.current_shard_done = 0                 # in flight, not yet persisted
        self.start = time.monotonic()
        self.rate_tracker = RateTracker(window_seconds)
        self.rate_tracker.update(0.0, self.start)

    @property
    def true_done(self):
        return self.done_start + self.completed_session

    @property
    def processed_session(self):
        return self.completed_session + self.current_shard_done

    def note_progress(self, current_shard_done):
        """Record in-flight progress within the current shard (for the rate)."""
        self.current_shard_done = current_shard_done
        self.rate_tracker.update(self.processed_session)

    def complete_shard(self, n):
        """Record a shard whose CSV has been written (persisted progress)."""
        self.completed_session += n
        self.current_shard_done = 0
        self.rate_tracker.update(self.processed_session)

    def true_pct(self):
        if not self.archive_total:
            return 100.0
        return min(100.0, 100.0 * self.true_done / self.archive_total)

    def rate(self):
        return self.rate_tracker.rate()

    def eta_seconds(self):
        rate = self.rate()
        remaining = max(0, self.archive_total - self.true_done)
        return remaining / rate if rate > 0 else 0.0

    def summary(self):
        return (
            f"{self.true_done}/{self.archive_total} images "
            f"({self.true_pct():.1f}%) | {self.rate():.2f} img/s | "
            f"ETA {format_duration(self.eta_seconds())}"
        )


class Stopper:
    """Records a SIGINT/SIGTERM request so loops can stop at a clean boundary."""

    def __init__(self):
        self.stop = False

    def handle(self, signum, frame):
        if self.stop:
            # A second signal means the operator is impatient: leave now.
            LOG.warning("Second signal %s received; exiting immediately", signum)
            raise SystemExit(130)
        self.stop = True
        LOG.warning(
            "Signal %s received; will stop at the next clean boundary "
            "(current shard is abandoned and redone on resume)",
            signum,
        )


####################################################################################
### PROVENANCE (stdlib only; records what produced the outputs)
####################################################################################
def read_deepfaune_version(software_dir):
    """Best-effort DeepFaune version string from ChangeLog.txt, or None."""
    path = os.path.join(software_dir, "ChangeLog.txt")
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = re.search(r"Version\s+([0-9][0-9A-Za-z.\-]*)", line)
                if match:
                    return match.group(1)
    except OSError:
        return None
    return None


def collect_weight_info(software_dir):
    """Map each weight filename (*.pt) in software_dir to its size in bytes."""
    info = {}
    for path in sorted(glob.glob(os.path.join(software_dir, "*.pt"))):
        try:
            info[os.path.basename(path)] = os.path.getsize(path)
        except OSError:
            info[os.path.basename(path)] = None
    return info


def get_git_commit():
    """This orchestrator's own git commit, or None if not available."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        result = subprocess.run(
            ["git", "-C", here, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def build_run_metadata(out_dir, software_dir, root, args):
    """Assemble the provenance dictionary written at the start of a run."""
    return {
        "detector": args.detector,
        "threshold": args.threshold,
        "maxlag": args.maxlag,
        "birds": args.birds,
        "lang": args.lang,
        "batch_size": args.batch_size,
        "deepfaune_version": read_deepfaune_version(software_dir),
        "weights": collect_weight_info(software_dir),
        "root": root,
        "out_dir": out_dir,
        "hostname": socket.gethostname(),
        "utc_start_time": datetime.now(timezone.utc).isoformat(),
        "orchestrator_git_commit": get_git_commit(),
        "partition": args.partition,
        "num_partitions": args.num_partitions,
    }


def write_run_metadata(out_dir, software_dir, root, args):
    """Write run_metadata.p<partition>.json atomically (one file per partition).

    Returns (path, metadata).
    """
    metadata = build_run_metadata(out_dir, software_dir, root, args)
    path = os.path.join(out_dir, f"run_metadata.p{args.partition}.json")
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(
        dir=directory, prefix=os.path.basename(path) + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path, metadata


def write_status(out_dir, partition, status):
    """Write status.p<partition>.json atomically for the live readout to poll.

    One file per partition so processes do not race. Best-effort: a failed
    status write must never interrupt classification, so errors are swallowed.
    """
    path = os.path.join(out_dir, f"status.p{partition}.json")
    directory = os.path.dirname(path) or "."
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(
            dir=directory, prefix=os.path.basename(path) + ".", suffix=".tmp"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(status, handle)
        os.replace(tmp, path)
    except OSError:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


####################################################################################
### ENGINE GLUE (imports torch indirectly; only used on the real run path)
####################################################################################
def install_shared_models(predict_tools, classif_tools, detect_tools,
                          birds, detector_name, device):
    """Build the detector and classifier once and inject them into the engine.

    PredictorImage normally constructs a fresh Detector and Classifier per
    instance, which would reload the 1.2 GB classifier for every shard. We build
    them once here and rebind the module-level names the engine looks up, so
    every subsequently constructed PredictorImage reuses these singletons.

    The detector has no isinstance check against it, so a simple factory closure
    is enough. The classifier is different: the engine runs
    isinstance(self.classifier, ClassifierWithBirds) to decide whether to apply
    the bird head, so that name must stay bound to a real type. We therefore use
    a one-instance-per-process subclass: its __new__ returns the single shared
    instance and its __init__ loads weights only the first time, while isinstance
    keeps working because the shared instance really is of that type.

    Returns (classifier, detector).
    """
    detector = detect_tools.Detector(name=detector_name, device=device)

    def detector_factory(name=None, device=None):
        return detector

    predict_tools.Detector = detector_factory

    base_cls = (
        classif_tools.ClassifierWithBirds if birds else classif_tools.Classifier
    )

    class SharedClassifier(base_cls):
        _instance = None

        def __new__(cls, *args, **kwargs):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

        def __init__(self, *args, **kwargs):
            if getattr(self, "_df_loaded", False):
                return
            super().__init__(*args, **kwargs)
            self._df_loaded = True

    classifier = SharedClassifier(device)  # loads weights now, once
    if birds:
        # ClassifierWithBirds is both constructed and used in isinstance checks.
        predict_tools.ClassifierWithBirds = SharedClassifier
    else:
        # Only the plain Classifier is constructed; leave ClassifierWithBirds as
        # the real class so isinstance(self.classifier, ClassifierWithBirds)
        # remains correctly False.
        predict_tools.Classifier = SharedClassifier
    return classifier, detector


def build_rows(predictor):
    """Read a finished predictor and assemble the per-image CSV rows."""
    seq_class, seq_score, _boxes, count = predictor.getPredictions()
    img_class, img_score, _boxes2, _count2 = predictor.getPredictionsBase()
    dates = predictor.getDates()
    seqnums = predictor.getSeqnums()
    filenames = predictor.getFilenames()
    humancount = predictor.getHumanCount()
    rows = []
    for k in range(len(filenames)):
        rows.append(
            [
                filenames[k],
                dates[k],
                int(seqnums[k]),
                seq_class[k],
                seq_score[k],
                img_class[k],
                img_score[k],
                int(count[k]),
                int(humancount[k]),
            ]
        )
    return rows


def process_shard(predict_tools, image_paths, threshold, maxlag, lang, birds,
                  batch_size, detector_name, stopper, heartbeat_secs, progress,
                  shard_label, report=None):
    """Run the engine over one shard.

    Returns the list of CSV rows, or None if a stop was requested mid-shard (in
    which case nothing is written and the shard is redone on resume). The rate
    tracker is fed every batch so throughput and ETA update continuously, and
    the optional report callback refreshes the status file on each heartbeat.
    """
    predictor = predict_tools.PredictorImage(
        image_paths,
        threshold,
        maxlag,
        lang,
        birds,
        BATCH_SIZE=batch_size,
        detectorname=detector_name,
        device=DEVICE,
    )
    total = len(image_paths)
    predictor.resetBatch()
    last_hb = time.monotonic()
    while True:
        _batch, k1, k2, _k1seq, _k2seq = predictor.nextBatch()
        if k1 >= total:  # sentinel return: every batch has been processed
            break
        if stopper.stop:
            LOG.warning(
                "  abandoning shard %s at image %d/%d (redo on resume)",
                shard_label, min(k2, total), total,
            )
            return None
        # Feed the smoothed rate every batch so the readout never reads zero.
        progress.note_progress(min(k2, total))
        now = time.monotonic()
        if heartbeat_secs and (now - last_hb) >= heartbeat_secs:
            LOG.info(
                "  %s: %d/%d images in shard | overall %s",
                shard_label, min(k2, total), total, progress.summary(),
            )
            if report is not None:
                report(shard_label, "running")
            last_hb = now
    return build_rows(predictor)


####################################################################################
### RUN AND DRY-RUN
####################################################################################
def setup_logging(out_dir, partition):
    """Log to stdout (captured under tmux) and to a file on local disk."""
    logfile = os.path.join(out_dir, f"deepfaune_batch.p{partition}.log")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    LOG.addHandler(stream)
    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setFormatter(fmt)
    LOG.addHandler(file_handler)
    return logfile


def validate_common(args):
    """Validate the numeric arguments present in every mode.

    Returns an error string or None. The critical check is --batch-size: a value
    of 0 leaves the engine's batch pointers with k1 == k2, so process_shard never
    reaches its sentinel and the run hangs; a negative value crashes at tensor
    allocation. Path checks (root, software-dir) are mode-specific and done in
    the respective entry points.
    """
    if args.num_partitions < 1:
        return f"--num-partitions must be >= 1 (got {args.num_partitions})"
    if args.partition < 0 or args.partition >= args.num_partitions:
        return (
            f"--partition must be in [0, {args.num_partitions}) "
            f"(got {args.partition})"
        )
    if args.batch_size < 1:
        return f"--batch-size must be >= 1 (got {args.batch_size})"
    if args.threads < 1:
        return f"--threads must be >= 1 (got {args.threads})"
    if args.maxlag < 0:
        return f"--maxlag must be >= 0 (got {args.maxlag})"
    if not 0 < args.threshold <= 1:
        return f"--threshold must satisfy 0 < t <= 1 (got {args.threshold})"
    if args.heartbeat_secs < 0:
        return f"--heartbeat-secs must be >= 0 (got {args.heartbeat_secs})"
    if args.merge_every < 0:
        return f"--merge-every must be >= 0 (got {args.merge_every})"
    if args.max_images is not None and args.max_images < 1:
        return f"--max-images must be >= 1 if given (got {args.max_images})"
    return None


def dry_run(args):
    """Build the shard manifest and print counts and the partition split.

    Deliberately imports nothing heavy: no torch, no model weights. This lets the
    operator sanity-check file enumeration on the box before committing to a run.
    """
    err = validate_common(args)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if not args.root:
        print("error: --root is required for --dry-run", file=sys.stderr)
        return 2
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"error: root is not a directory: {root}", file=sys.stderr)
        return 2
    shards = find_shards(root)
    total_images = sum(len(imgs) for _, imgs in shards)
    print(f"Dry run (no models loaded, torch not imported)")
    print(f"Root: {root}")
    print(f"Leaf-directory shards with images: {len(shards)}")
    print(f"Total images: {total_images}")
    print(f"Partitions: {args.num_partitions}")
    for p in range(args.num_partitions):
        sel = select_partition(shards, args.num_partitions, p)
        imgs = sum(len(i) for _, i in sel)
        marker = "  <- selected" if p == args.partition else ""
        print(f"  partition {p}: {len(sel)} shards, {imgs} images{marker}")
    out_dir = os.path.abspath(args.out_dir)
    sel = select_partition(shards, args.num_partitions, args.partition)
    plan = plan_shards(sel, out_dir, root, args.rescan)
    remaining_images = sum(len(imgs) for _, imgs, _ in plan)
    print(
        f"Selected partition {args.partition}: {len(sel)} shards; "
        f"{len(plan)} shards / {remaining_images} images remaining after "
        f"resume skip (out-dir={out_dir})"
    )
    return 0


def merge_mode(args):
    """Standalone master rebuild: concatenate the per-shard CSVs, then exit.

    Like --dry-run, this imports nothing heavy (no torch, no models). The master
    is always rebuilt from scratch, so it stays a faithful snapshot of whatever
    shard CSVs currently exist in --out-dir.
    """
    err = validate_common(args)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isdir(out_dir):
        print(f"error: out-dir is not a directory: {out_dir}", file=sys.stderr)
        return 2
    merge_out = (
        os.path.abspath(args.merge_out)
        if args.merge_out
        else os.path.join(out_dir, "master.csv")
    )
    n_files, n_rows = merge_csvs(out_dir, merge_out)
    print(f"Merged {n_files} shard CSVs ({n_rows} rows) -> {merge_out}")
    return 0


def run(args):
    """Execute the batch: enumerate, skip finished shards, classify the rest."""
    err = validate_common(args)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if not args.root:
        print("error: --root is required for a run", file=sys.stderr)
        return 2
    if not args.software_dir:
        print("error: --software-dir is required for a run", file=sys.stderr)
        return 2
    root = os.path.abspath(args.root)
    out_dir = os.path.abspath(args.out_dir)
    software_dir = os.path.abspath(args.software_dir)

    if not os.path.isdir(root):
        print(f"error: root is not a directory: {root}", file=sys.stderr)
        return 2
    if not os.path.isdir(software_dir):
        print(f"error: --software-dir not found: {software_dir}", file=sys.stderr)
        return 2
    # Refuse to write next to the (read-only) source archive. commonpath handles
    # edge cases (such as root being "/") that a string prefix check would not.
    if is_within(out_dir, root):
        print(
            f"error: refusing to write outputs inside the source tree "
            f"(out-dir={out_dir} is under root={root})",
            file=sys.stderr,
        )
        return 2

    os.makedirs(out_dir, exist_ok=True)
    setup_logging(out_dir, args.partition)

    # CPU only: hide every GPU from torch so CUDA cannot be selected even by
    # accident. This must run before torch is imported.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Thread limits must be set before torch is imported to bind the native
    # pools. Capping threads also bounds PyTorch's shared-memory footprint,
    # which contributed to the earlier out-of-memory kill. setdefault respects
    # any value the operator already exported.
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))

    if args.detector == "MDR":
        LOG.warning("Detector MDR (redwood, 1280 px) is slow on CPU; expect a long run")

    LOG.info("DeepFaune batch starting")
    LOG.info("root=%s", root)
    LOG.info("out_dir=%s", out_dir)
    LOG.info("software_dir=%s", software_dir)
    LOG.info(
        "detector=%s birds=%s threshold=%s maxlag=%s lang=%s batch_size=%s",
        args.detector, args.birds, args.threshold, args.maxlag, args.lang,
        args.batch_size,
    )
    LOG.info(
        "partition %d of %d | rescan=%s",
        args.partition, args.num_partitions, args.rescan,
    )

    # Provenance sidecar, one file per partition so processes do not race.
    meta_path, _meta = write_run_metadata(out_dir, software_dir, root, args)
    LOG.info("Wrote run metadata: %s", os.path.basename(meta_path))

    # The engine and its weights live in software_dir; import from there.
    sys.path.insert(0, software_dir)
    import torch  # noqa: E402  (lazy by design)
    import predictTools  # noqa: E402
    import classifTools  # noqa: E402
    import detectTools  # noqa: E402

    torch.set_num_threads(args.threads)
    LOG.info(
        "torch %s | cuda available: %s | intra-op threads: %d",
        torch.__version__, torch.cuda.is_available(), torch.get_num_threads(),
    )
    if torch.cuda.is_available():
        LOG.warning("CUDA reported available; this run forces device=cpu anyway")
    device = torch.device(DEVICE)

    LOG.info("Loading models once for this process...")
    t0 = time.monotonic()
    install_shared_models(
        predictTools, classifTools, detectTools,
        birds=args.birds, detector_name=args.detector, device=device,
    )
    LOG.info("Models loaded in %.1fs", time.monotonic() - t0)

    # Record images the engine cannot read to a sidecar so they can be audited,
    # and keep a running count rather than letting a bad file abort the run.
    unreadable_path = os.path.join(out_dir, "unreadable_images.txt")
    skipped = {"count": 0, "fh": None}

    def record_unreadable(filename, reason):
        skipped["count"] += 1
        if skipped["fh"] is None:  # open lazily so a clean run leaves no file
            skipped["fh"] = open(unreadable_path, "a", encoding="utf-8")
        skipped["fh"].write(
            f"{datetime.now(timezone.utc).isoformat()}\t{reason}\t{filename}\n"
        )
        skipped["fh"].flush()

    detectTools.skipped_image_hook = record_unreadable

    shards = find_shards(root)
    selected = select_partition(shards, args.num_partitions, args.partition)
    plan = plan_shards(selected, out_dir, root, args.rescan)
    archive_total = sum(len(imgs) for _, imgs in shards)
    plan_images = sum(len(imgs) for _, imgs, _ in plan)
    done_start = count_classified_images(out_dir)
    LOG.info(
        "Archive total: %d images | already classified (all sessions): %d (%.1f%%)",
        archive_total, done_start,
        (100.0 * done_start / archive_total) if archive_total else 100.0,
    )
    LOG.info(
        "Shards to process this session after resume skip: %d (%d images)",
        len(plan), plan_images,
    )

    # The master is a derived snapshot. Only partition 0 writes it, so a
    # multi-process run does not race on it; during such a run the master
    # reflects only the shards finished so far (by any partition), so run
    # "--merge" once all partitions have finished for the definitive file.
    merge_out = (
        os.path.abspath(args.merge_out)
        if args.merge_out
        else os.path.join(out_dir, "master.csv")
    )
    do_merge = args.merge_every > 0 and args.partition == 0
    if args.merge_every > 0 and args.partition != 0:
        LOG.info(
            "--merge-every ignored on partition %d; only partition 0 writes the master",
            args.partition,
        )

    progress = Progress(archive_total, done_start)
    start_unix = time.time()

    def report(current_shard, state):
        write_status(out_dir, args.partition, {
            "pid": os.getpid(),
            "state": state,
            "start_time_unix": start_unix,
            "updated_unix": time.time(),
            "current_shard": current_shard,
            "archive_total": progress.archive_total,
            "true_done": progress.true_done,
            "true_pct": round(progress.true_pct(), 2),
            "rate_img_per_s": round(progress.rate(), 3),
            "eta_seconds": int(progress.eta_seconds()),
            "session_processed": progress.processed_session,
            "skipped_unreadable": skipped["count"],
            "detector": args.detector,
            "threshold": args.threshold,
            "maxlag": args.maxlag,
            "birds": args.birds,
        })

    if not plan:
        LOG.info("Nothing to do; all shards in this partition already have CSVs")
        if do_merge:
            n_files, n_rows = merge_csvs(out_dir, merge_out)
            LOG.info("Master rebuilt: %d shards, %d rows -> %s", n_files, n_rows, merge_out)
        report(None, "finished")
        return 0

    stopper = Stopper()
    signal.signal(signal.SIGINT, stopper.handle)
    signal.signal(signal.SIGTERM, stopper.handle)

    report(None, "running")
    processed = 0
    last_merge = time.monotonic()
    for leaf_dir, image_paths, csv_path in plan:
        if stopper.stop:
            LOG.warning("Stopping before next shard as requested")
            break
        if args.max_images is not None and progress.completed_session >= args.max_images:
            LOG.info(
                "Reached --max-images %d (%d images this session); stopping at shard boundary",
                args.max_images, progress.completed_session,
            )
            break
        shard_label = os.path.relpath(leaf_dir, root)
        LOG.info(
            "Shard %s: %d images -> %s",
            shard_label, len(image_paths), os.path.basename(csv_path),
        )
        report(shard_label, "running")
        rows = process_shard(
            predictTools, image_paths,
            threshold=args.threshold, maxlag=args.maxlag, lang=args.lang,
            birds=args.birds, batch_size=args.batch_size,
            detector_name=args.detector, stopper=stopper,
            heartbeat_secs=args.heartbeat_secs, progress=progress,
            shard_label=shard_label, report=report,
        )
        if rows is None:  # aborted mid-shard by a signal
            break
        atomic_write_csv(csv_path, CSV_HEADER, rows)
        progress.complete_shard(len(image_paths))
        processed += 1
        LOG.info(
            "Finished %s | %s | skipped unreadable so far: %d",
            shard_label, progress.summary(), skipped["count"],
        )
        report(shard_label, "running")
        if do_merge and (time.monotonic() - last_merge) >= args.merge_every:
            n_files, n_rows = merge_csvs(out_dir, merge_out)
            LOG.info(
                "Master refreshed: %d shards, %d rows -> %s",
                n_files, n_rows, os.path.basename(merge_out),
            )
            last_merge = time.monotonic()

    if do_merge:
        n_files, n_rows = merge_csvs(out_dir, merge_out)
        LOG.info(
            "Master (final this session): %d shards, %d rows -> %s",
            n_files, n_rows, merge_out,
        )

    if skipped["fh"] is not None:
        skipped["fh"].close()
    final_state = "stopped" if stopper.stop else "finished"
    report(None, final_state)
    LOG.info(
        "Run ended (%s): %d shards written this session | %s | "
        "skipped unreadable: %d",
        final_state, processed, progress.summary(), skipped["count"],
    )
    return 0


####################################################################################
### CLI
####################################################################################
def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="deepfaune_batch.py",
        description=(
            "Resumable CPU-only batch classifier driving the official DeepFaune "
            "v1.4.1 PredictorImage engine, one CSV per leaf directory."
        ),
    )
    parser.add_argument(
        "--software-dir",
        help="Directory of the DeepFaune v1.4.1 source and weights (added to "
             "sys.path; its predictTools is imported). Required for a run.",
    )
    parser.add_argument(
        "--root",
        help="Top of the image tree to classify (read-only source archive). "
             "Required for a run and for --dry-run.",
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Directory on local disk for the per-shard CSVs, master, logs and "
             "metadata.",
    )
    parser.add_argument(
        "--detector", default="DF", choices=DETECTOR_CHOICES,
        help="Detector model (default: DF, the lightest).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification confidence threshold (default: 0.5, matching the "
             "demo script and the first long run).",
    )
    parser.add_argument(
        "--maxlag", type=int, default=20,
        help="Seconds between photos to count as one EXIF burst (default: 20, "
             "matching the first long run).",
    )
    parser.add_argument(
        "--lang", default="en", choices=LANG_CHOICES,
        help="Label language (default: en; no Romanian, map downstream).",
    )
    parser.add_argument(
        "--birds", action="store_true",
        help="Enable the 8-way bird sub-classifier head.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Classifier batch size (default: 8).",
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="Intra-op CPU threads (default: 4). Must be >= 1; capping this also "
             "bounds memory use.",
    )
    parser.add_argument(
        "--num-partitions", type=int, default=1,
        help="Split shards across this many cooperating processes (default: 1).",
    )
    parser.add_argument(
        "--partition", type=int, default=0,
        help="Which partition this process handles, in [0, num-partitions).",
    )
    parser.add_argument(
        "--heartbeat-secs", type=int, default=60,
        help="Seconds between in-shard progress lines (default: 60; 0 disables).",
    )
    parser.add_argument(
        "--rescan", action="store_true",
        help="Reprocess every shard even if its CSV exists (overwrite).",
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Stop the run at the next shard boundary after this many images "
             "(for benchmarks and time-boxed trials).",
    )
    parser.add_argument(
        "--merge-every", type=int, default=0,
        help="During a run, regenerate the master CSV every N seconds and once "
             "at the end. Only partition 0 writes it. 0 disables (default).",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Standalone: rebuild the master CSV from the per-shard CSVs in "
             "--out-dir, then exit. No models loaded.",
    )
    parser.add_argument(
        "--merge-out", default=None,
        help="Master CSV path (default: <out-dir>/master.csv).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build the shard manifest and print counts without loading models.",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    # Reject contradictory mode combinations: --dry-run and --merge are each
    # standalone terminal modes and cannot be requested together.
    if args.dry_run and args.merge:
        print(
            "error: --dry-run and --merge are mutually exclusive",
            file=sys.stderr,
        )
        return 2
    if args.dry_run:
        return dry_run(args)
    if args.merge:
        return merge_mode(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
