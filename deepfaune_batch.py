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
#   - Outputs (CSVs and the log) go to local disk only; the source archive is
#     read-only and must never be written to.
#   - CPU only: device is forced to "cpu" and no CUDA path is ever taken.
#
# Only the standard library is imported at module load. torch and the DeepFaune
# engine are imported lazily inside run(), so --dry-run and the unit tests stay
# cheap and never need the model weights.

import argparse
import csv
import hashlib
import logging
import os
import re
import signal
import sys
import tempfile
import time
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
        dirnames.sort()  # deterministic, depth-first traversal
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


def format_duration(seconds):
    """Format a number of seconds as H:MM:SS for human-readable ETAs."""
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


class Progress:
    """Tracks images done against a target to report throughput and an ETA."""

    def __init__(self, total_images):
        self.total = total_images
        self.done = 0
        self.start = time.monotonic()

    def add(self, n):
        self.done += n

    def summary(self):
        elapsed = time.monotonic() - self.start
        rate = self.done / elapsed if elapsed > 0 and self.done > 0 else 0.0
        remaining = max(0, self.total - self.done)
        eta = remaining / rate if rate > 0 else 0.0
        pct = (100.0 * self.done / self.total) if self.total else 100.0
        return (
            f"{self.done}/{self.total} images ({pct:.1f}%) | "
            f"{rate:.2f} img/s | ETA {format_duration(eta)}"
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
                  shard_label):
    """Run the engine over one shard.

    Returns the list of CSV rows, or None if a stop was requested mid-shard (in
    which case nothing is written and the shard is redone on resume).
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
        batch, k1, k2, k1seq, k2seq = predictor.nextBatch()
        if k1 >= total:  # sentinel return: every batch has been processed
            break
        if stopper.stop:
            LOG.warning(
                "  abandoning shard %s at image %d/%d (redo on resume)",
                shard_label, min(k2, total), total,
            )
            return None
        now = time.monotonic()
        if heartbeat_secs and (now - last_hb) >= heartbeat_secs:
            LOG.info(
                "  %s: %d/%d images in shard | overall %s",
                shard_label, min(k2, total), total, progress.summary(),
            )
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
    """Validate the arguments shared by run and dry-run. Returns an error string
    or None."""
    if not os.path.isdir(args.root):
        return f"root is not a directory: {args.root}"
    if args.num_partitions < 1:
        return f"--num-partitions must be >= 1 (got {args.num_partitions})"
    if args.partition < 0 or args.partition >= args.num_partitions:
        return (
            f"--partition must be in [0, {args.num_partitions}) "
            f"(got {args.partition})"
        )
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
    root = os.path.abspath(args.root)
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


def run(args):
    """Execute the batch: enumerate, skip finished shards, classify the rest."""
    err = validate_common(args)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    root = os.path.abspath(args.root)
    out_dir = os.path.abspath(args.out_dir)
    software_dir = os.path.abspath(args.software_dir)

    # Refuse to write next to the (read-only) source archive.
    if out_dir == root or out_dir.startswith(root + os.sep):
        print(
            f"error: refusing to write outputs inside the source tree "
            f"(out-dir={out_dir} is under root={root})",
            file=sys.stderr,
        )
        return 2
    if not os.path.isdir(software_dir):
        print(f"error: --software-dir not found: {software_dir}", file=sys.stderr)
        return 2

    os.makedirs(out_dir, exist_ok=True)
    setup_logging(out_dir, args.partition)

    # Thread limits must be set before torch is imported to bind the native
    # pools; setdefault respects any value the operator already exported.
    if args.threads and args.threads > 0:
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

    # The engine and its weights live in software_dir; import from there.
    sys.path.insert(0, software_dir)
    import torch  # noqa: E402  (lazy by design)
    import predictTools  # noqa: E402
    import classifTools  # noqa: E402
    import detectTools  # noqa: E402

    if args.threads and args.threads > 0:
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

    shards = find_shards(root)
    selected = select_partition(shards, args.num_partitions, args.partition)
    plan = plan_shards(selected, out_dir, root, args.rescan)
    selected_images = sum(len(imgs) for _, imgs in selected)
    plan_images = sum(len(imgs) for _, imgs, _ in plan)
    LOG.info(
        "Shards in this partition: %d (%d images)", len(selected), selected_images
    )
    LOG.info(
        "Shards to process after resume skip: %d (%d images)",
        len(plan), plan_images,
    )
    if not plan:
        LOG.info("Nothing to do; all shards in this partition already have CSVs")
        return 0

    stopper = Stopper()
    signal.signal(signal.SIGINT, stopper.handle)
    signal.signal(signal.SIGTERM, stopper.handle)

    progress = Progress(plan_images)
    processed = 0
    for leaf_dir, image_paths, csv_path in plan:
        if stopper.stop:
            LOG.warning("Stopping before next shard as requested")
            break
        shard_label = os.path.relpath(leaf_dir, root)
        LOG.info(
            "Shard %s: %d images -> %s",
            shard_label, len(image_paths), os.path.basename(csv_path),
        )
        rows = process_shard(
            predictTools, image_paths,
            threshold=args.threshold, maxlag=args.maxlag, lang=args.lang,
            birds=args.birds, batch_size=args.batch_size,
            detector_name=args.detector, stopper=stopper,
            heartbeat_secs=args.heartbeat_secs, progress=progress,
            shard_label=shard_label,
        )
        if rows is None:  # aborted mid-shard by a signal
            break
        atomic_write_csv(csv_path, CSV_HEADER, rows)
        progress.add(len(image_paths))
        processed += 1
        LOG.info("Finished %s | %s", shard_label, progress.summary())

    LOG.info(
        "Run ended: %d shards written this session | %s",
        processed, progress.summary(),
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
        "--software-dir", required=True,
        help="Directory of the DeepFaune v1.4.1 source and weights (added to "
             "sys.path; its predictTools is imported).",
    )
    parser.add_argument(
        "--root", required=True,
        help="Top of the image tree to classify (read-only source archive).",
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Directory on local disk for the per-shard CSVs and the log.",
    )
    parser.add_argument(
        "--detector", default="DF", choices=DETECTOR_CHOICES,
        help="Detector model (default: DF, the lightest).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification confidence threshold (default: 0.5).",
    )
    parser.add_argument(
        "--maxlag", type=int, default=20,
        help="Seconds between photos to count as one EXIF burst (default: 20).",
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
        "--threads", type=int, default=0,
        help="Intra-op CPU threads; 0 leaves the torch/library default.",
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
        "--dry-run", action="store_true",
        help="Build the shard manifest and print counts without loading models.",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.dry_run:
        return dry_run(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
