#!/usr/bin/env python3
# Copyright (c) 2025 Fundatia Conservation Carpathia batch tooling.
#
# benchmark_detectors.py: measure how fast each detector option classifies real
# images from the archive, and extrapolate to a full pass.
#
# Detector choice is the one setting that cannot be changed retrospectively (a
# threshold or a species exclusion can be re-derived from the recorded scores;
# a box the detector never found cannot). Choosing it is therefore a real
# commitment, and this puts numbers under that decision.
#
# Each detector is timed in its own subprocess so model loading, memory and
# any accumulated state cannot leak between measurements. Only the
# classification itself is timed; model loading is reported separately.
#
# British English throughout; no em dashes.

import argparse
import csv
import glob
import json
import os
import random
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import deepfaune_batch as dfb  # noqa: E402  (local, stdlib-only at import)


####################################################################################
### PURE HELPERS (unit-tested; no torch)
####################################################################################
def human_duration(seconds):
    """A coarse, readable duration: '4.2 days', '9.5 hours', '12 min'."""
    if seconds is None or seconds != seconds or seconds < 0:
        return "unknown"
    if seconds < 90:
        return f"{seconds:.0f} sec"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86400:.1f} days"


def extrapolate(rate_img_s, total_images):
    """Seconds to process total_images at rate_img_s, or None if unusable."""
    if not rate_img_s or rate_img_s <= 0 or not total_images:
        return None
    return total_images / rate_img_s


def sample_from_master(master_path, n, seed=0):
    """Reservoir-sample n image paths from a master CSV.

    Sampling from the master rather than walking the drive keeps the mix of
    empty and animal images representative, which matters: DFbsMDS only runs
    its second detector on images where the first found nothing, so a sample
    skewed towards animals would flatter it.
    """
    rng = random.Random(seed)
    picked, seen = [], 0
    with open(master_path, newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            path = row.get("filename") or ""
            if not path:
                continue
            seen += 1
            if len(picked) < n:
                picked.append(path)
            else:
                j = rng.randrange(seen)
                if j < n:
                    picked[j] = path
    return picked, seen


def sample_from_archive(root, n, seed=0, max_shards=400):
    """Fallback sampler: take images from shards found by walking the archive."""
    rng = random.Random(seed)
    shards = dfb.find_shards(root)
    if not shards:
        return [], 0
    rng.shuffle(shards)
    picked, seen = [], 0
    for _leaf, images in shards[:max_shards]:
        for path in images:
            seen += 1
            if len(picked) < n:
                picked.append(path)
            elif rng.randrange(seen) < n:
                picked[rng.randrange(n)] = path
        if len(picked) >= n and seen > n * 8:
            break
    return picked, seen


def format_table(results, total_images, baseline=None):
    """Render the comparison table. results: list of dicts from run_one."""
    lines = []
    head = (f"{'detector':<10}{'img/s':>9}{'vs base':>9}{'load':>8}"
            f"{'animals':>9}{'full pass':>14}")
    lines.append(head)
    lines.append("-" * len(head))
    base_rate = None
    for r in results:
        if baseline and r["detector"] == baseline and r.get("rate"):
            base_rate = r["rate"]
    for r in results:
        if r.get("error"):
            lines.append(f"{r['detector']:<10}{'failed':>9}   {r['error'][:44]}")
            continue
        rate = r["rate"]
        rel = f"{base_rate / rate:.2f}x" if base_rate and rate else "-"
        total = extrapolate(rate, total_images)
        lines.append(
            f"{r['detector']:<10}{rate:>9.2f}{rel:>9}{r['load_s']:>7.0f}s"
            f"{r['animal_frac'] * 100:>8.0f}%{human_duration(total):>14}"
        )
    return "\n".join(lines)


####################################################################################
### THE MEASUREMENT (imports torch; runs in a subprocess, one detector per run)
####################################################################################
def run_one(args):
    """Benchmark a single detector over the given images. Prints JSON."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))

    with open(args.image_list, encoding="utf-8") as handle:
        images = [ln.rstrip("\n") for ln in handle if ln.strip()]

    sys.path.insert(0, args.software_dir)
    import torch  # noqa: E402
    import predictTools  # noqa: E402
    import classifTools  # noqa: E402
    import detectTools  # noqa: E402

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    t0 = time.monotonic()
    dfb.install_shared_models(
        predictTools, classifTools, detectTools,
        birds=args.birds, detector_name=args.detector, device=device,
    )
    load_s = time.monotonic() - t0

    class _Stopper:
        stop = False

    class _Progress:
        completed_session = 0

        def note_progress(self, k):
            pass

        def summary(self):
            return ""

    t0 = time.monotonic()
    rows = dfb.process_shard(
        predictTools, images,
        threshold=args.threshold, maxlag=args.maxlag, lang="en",
        birds=args.birds, batch_size=args.batch_size,
        detector_name=args.detector, stopper=_Stopper(),
        heartbeat_secs=0, progress=_Progress(), shard_label="benchmark",
    )
    elapsed = time.monotonic() - t0

    animals = 0
    if rows:
        try:
            det_col = dfb.CSV_HEADER.index("det_conf_animal")
            animals = sum(1 for r in rows if float(r[det_col] or 0) > 0)
        except (ValueError, IndexError):
            animals = 0
    n = len(images)
    print("BENCHMARK_JSON " + json.dumps({
        "detector": args.detector,
        "images": n,
        "elapsed_s": round(elapsed, 2),
        "rate": round(n / elapsed, 4) if elapsed > 0 else 0,
        "load_s": round(load_s, 1),
        "animal_frac": round(animals / n, 4) if n else 0,
    }))
    return 0


####################################################################################
### DRIVER
####################################################################################
def worker_is_running(out_dir):
    """True if a classification worker holds the pidfile (its CPU would skew this)."""
    try:
        with open(os.path.join(out_dir, "dfrun.worker.pid"), encoding="utf-8") as handle:
            pid = int(handle.read().strip())
    except (OSError, ValueError):
        return False
    return dfb.pid_is_worker(pid)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="benchmark_detectors.py",
        description="Time each detector on real archive images and extrapolate "
                    "to a full pass. Detector choice cannot be changed "
                    "retrospectively, so measure before committing.",
    )
    parser.add_argument("--software-dir", default=_HERE,
                        help="DeepFaune source and weights directory.")
    parser.add_argument("--root", help="Archive root (only needed without a master CSV).")
    parser.add_argument("--out-dir", default=os.path.expanduser("~/df_out"),
                        help="Output directory, read for the master CSV and pidfile.")
    parser.add_argument("--images", type=int, default=150,
                        help="Images per detector (default: 150).")
    parser.add_argument("--detectors", default="DF,DFbsMDS,DFMDS,MDR",
                        help="Comma-separated detectors to time, in order.")
    parser.add_argument("--baseline", default="DFbsMDS",
                        help="Detector the 'vs base' column compares against.")
    parser.add_argument("--total-images", type=int, default=0,
                        help="Archive size for the extrapolation (default: from "
                             "the total cache or the master CSV).")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--maxlag", type=int, default=60)
    parser.add_argument("--birds", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="Benchmark even while a worker is running (skews results).")
    # Internal: one detector, one subprocess.
    parser.add_argument("--one", dest="detector", help=argparse.SUPPRESS)
    parser.add_argument("--image-list", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.detector:
        return run_one(args)

    out_dir = os.path.abspath(args.out_dir)
    if worker_is_running(out_dir) and not args.force:
        print("A classification worker is running; its CPU use would skew every\n"
              "measurement. Stop it first with 'dfrun --stop', or pass --force.",
              file=sys.stderr)
        return 2

    # Sample images: prefer the master CSV, which reflects the real mix of
    # empty and animal images across the archive.
    master = os.path.join(out_dir, "master.csv")
    if os.path.exists(master):
        images, seen = sample_from_master(master, args.images, args.seed)
        source = f"master.csv ({seen:,} rows)"
    elif args.root:
        images, seen = sample_from_archive(args.root, args.images, args.seed)
        source = f"archive walk ({seen:,} images seen)"
    else:
        print("error: no master.csv in --out-dir; pass --root to sample the drive",
              file=sys.stderr)
        return 2

    images = [p for p in images if os.path.exists(p)]
    if len(images) < 10:
        print(f"error: only {len(images)} sampled images are readable. Is the "
              "drive mounted where the master CSV recorded it?", file=sys.stderr)
        return 2

    total = args.total_images
    if not total:
        cache = os.path.join(out_dir, ".dfrun_total_cache.json")
        try:
            with open(cache, encoding="utf-8") as handle:
                total = int(json.load(handle).get("total") or 0)
        except (OSError, ValueError, TypeError):
            total = 0
    if not total:
        total = seen

    listing = os.path.join(out_dir, ".benchmark_images.txt")
    with open(listing, "w", encoding="utf-8") as handle:
        handle.write("\n".join(images) + "\n")

    print(f"Benchmarking {len(images)} images sampled from {source}")
    print(f"Extrapolating to {total:,} images · {args.threads} threads · "
          f"batch {args.batch_size}\n")

    results = []
    for name in [d.strip() for d in args.detectors.split(",") if d.strip()]:
        if name not in dfb.DETECTOR_CHOICES:
            print(f"skipping unknown detector {name}")
            continue
        print(f"  {name}: loading models and timing…", end="", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__),
               "--one", name, "--image-list", listing,
               "--software-dir", os.path.abspath(args.software_dir),
               "--threads", str(args.threads), "--batch-size", str(args.batch_size),
               "--threshold", str(args.threshold), "--maxlag", str(args.maxlag)]
        if args.birds:
            cmd.append("--birds")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
        except (OSError, subprocess.SubprocessError) as err:
            results.append({"detector": name, "error": str(err)})
            print(" failed")
            continue
        line = next((ln for ln in proc.stdout.splitlines()
                     if ln.startswith("BENCHMARK_JSON ")), None)
        if not line:
            tail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
            results.append({"detector": name, "error": tail})
            print(f" failed ({tail[:60]})")
            continue
        r = json.loads(line[len("BENCHMARK_JSON "):])
        results.append(r)
        print(f" {r['rate']:.2f} img/s")

    print("\n" + format_table(results, total, baseline=args.baseline))
    print("\n'animals' is the share of sampled images where the detector found an\n"
          "animal box: the recall difference you are paying the extra time for.")
    try:
        os.unlink(listing)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
