#!/usr/bin/env python3
"""
deepfaune_batch.py - resumable, CPU-only orchestrator around the OFFICIAL DeepFaune
PredictorImage engine (predictTools.py from the deepfaune/software repository),
for classifying very large camera-trap archives (target: ~1.8M images on a
read-only external drive).

It does NOT reimplement detection or classification. It imports the published
engine and drives it, adding only the layer that engine lacks for archive-scale work:
  1. Shards work by leaf directory, which keeps each PredictorImage instance small
     and preserves EXIF-timestamp sequences (a deployment's burst lives in one folder).
  2. Writes one CSV per shard, written to a temp name and atomically renamed on
     success; a shard whose CSV already exists is skipped. Crash-safe and resumable.
  3. Loads the classifier (~1.2 GB) and detector ONCE per process and injects them
     into each lightweight PredictorImage, so weights are not reloaded every shard.
  4. Filters macOS AppleDouble (._*) and .DS_Store files the engine would otherwise try.
  5. Detached-run friendly: heartbeat with throughput + ETA; clean stop on SIGINT/SIGTERM.
  6. Trivial multi-process parallelism via --num-partitions/--partition (each process
     handles a disjoint subset of shards; crash-isolated by the per-shard CSV scheme).

PREREQUISITES
  * Unzip deepfaune/software (the folder containing predictTools.py) and place the
    downloaded weights INSIDE it (the engine resolves weights relative to its own path):
        deepfaune-vit_large_patch14_dinov2.lvd142m.v4.pt        (classifier, required)
        deepfaune-yolov8s_960.pt                                 (DF detector)
        md_v1000.0.0-sorrel.pt        (only for MDS / DFbsMDS / DFMDS)
        md_v1000.0.0-redwood.pt       (only for MDR; GPU recommended, avoid on CPU)
        deepfaune-vit_large_patch14_dinov2.lvd142m.v4-bird_head.pt   (only with --birds)
    Obtain weights + checksums from the official documentation and verify them.
  * Install per the DeepFaune docs, with CPU torch:
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        pip install ultralytics yolov5 timm pandas dill hachoir openpyxl setuptools==81
    Confirm: python -c "import torch; print(torch.cuda.is_available())"  -> False

USAGE (single process)
    python deepfaune_batch.py \
        --software-dir /path/to/software-master \
        --root "/media/rim/My Book" \
        --out-dir ~/df_out \
        --detector DFbsMDS --threshold 0.5 --maxlag 20 --batch-size 8

Outputs: per-shard CSVs under <out-dir>/csv/. Concatenate when finished:
    { read -r h < "$(ls ~/df_out/csv/*.csv | head -1)"; echo "$h";
      tail -q -n +2 ~/df_out/csv/*.csv; } > ~/df_out/all_results.csv
"""

import os
import sys
import csv
import time
import signal
import logging
import argparse
import hashlib

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}
DETECTOR_MAP = {"DF": "DF", "MDS": "MDS", "DFbsMDS": "DFbsMDS", "DFMDS": "DFMDS", "MDR": "MDR"}

_STOP = False


def _handle_signal(signum, frame):
    global _STOP
    _STOP = True
    logging.warning("Signal %s received; finishing the current shard then exiting.", signum)


def shard_id(rel_dir):
    """Stable, filesystem-safe identifier for a shard directory."""
    slug = rel_dir.replace(os.sep, "__").replace(" ", "_")
    if len(slug) > 120:  # keep filenames sane; disambiguate with a short hash
        slug = slug[:100] + "_" + hashlib.sha1(rel_dir.encode()).hexdigest()[:10]
    return slug or "_root_"


def list_images_in_dir(dirpath):
    """Non-recursive: image files directly in this directory, junk filtered, sorted."""
    out = []
    try:
        with os.scandir(dirpath) as it:
            for e in it:
                if not e.is_file():
                    continue
                name = e.name
                if name.startswith("._") or name == ".DS_Store":
                    continue
                if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                    out.append(e.path)
    except OSError as err:
        logging.error("Cannot scan %s: %s", dirpath, err)
    return sorted(out)


def build_or_load_shards(root, out_dir, rescan):
    """Return list of (rel_dir, n_images). Cached to <out-dir>/shards.tsv."""
    cache = os.path.join(out_dir, "shards.tsv")
    if os.path.exists(cache) and not rescan:
        shards = []
        with open(cache, newline="") as f:
            for rel, n in csv.reader(f, delimiter="\t"):
                shards.append((rel, int(n)))
        logging.info("Loaded %d shards from cache.", len(shards))
        return shards

    logging.info("Walking %s to build the shard manifest (one-time)...", root)
    shards, t0 = [], time.time()
    for dirpath, _dirs, files in os.walk(root):
        n = 0
        for name in files:
            if name.startswith("._") or name == ".DS_Store":
                continue
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                n += 1
        if n:
            shards.append((os.path.relpath(dirpath, root), n))
        if time.time() - t0 > 30:
            logging.info("  ...scanned, %d shards so far", len(shards))
            t0 = time.time()
    shards.sort()
    tmp = cache + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        for rel, n in shards:
            w.writerow([rel, n])
    os.replace(tmp, cache)
    logging.info("Manifest: %d shards, %d images.", len(shards), sum(n for _, n in shards))
    return shards


def patch_engine_to_reuse_models(predictTools, device, birds, detector_name):
    """Build classifier + detector once and make every PredictorImage reuse them."""
    if birds:
        clf = predictTools.ClassifierWithBirds(device)
    else:
        clf = predictTools.Classifier(device)
    det = predictTools.Detector(name=detector_name, device=device)

    predictTools.Classifier = lambda *a, **k: clf
    predictTools.ClassifierWithBirds = lambda *a, **k: clf
    predictTools.Detector = lambda *a, **k: det
    return clf, det


def process_shard(predictTools, files, threshold, maxlag, lang, birds, batch_size, detector_name):
    """Run the official engine on one shard; return a list of row dicts."""
    predictor = predictTools.PredictorImage(
        files, threshold, maxlag, lang, birds,
        BATCH_SIZE=batch_size, detectorname=detector_name, device="cpu",
    )
    predictor.allBatch()

    filenames = predictor.getFilenames()
    dates = predictor.getDates()
    seqnums = predictor.getSeqnums()
    cls_seq, score_seq, _boxes_seq, count = predictor.getPredictions()
    cls_base, score_base, _boxes_base, _count_base = predictor.getPredictionsBase()
    humancount = predictor.getHumanCount()

    rows = []
    for i in range(len(filenames)):
        rows.append({
            "filename": filenames[i],
            "date": dates[i],
            "seqnum": seqnums[i],
            "prediction_seq": cls_seq[i],
            "score_seq": score_seq[i],
            "prediction_image": cls_base[i],
            "score_image": score_base[i],
            "animal_count": count[i],
            "human_count": humancount[i],
        })
    return rows


def write_shard_csv(path, rows):
    fields = ["filename", "date", "seqnum", "prediction_seq", "score_seq",
              "prediction_image", "score_image", "animal_count", "human_count"]
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)  # atomic: a complete CSV is the done-marker


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--software-dir", required=True, help="Folder containing predictTools.py + weights")
    ap.add_argument("--root", required=True, help="Archive root (read-only mount)")
    ap.add_argument("--out-dir", required=True, help="Output dir on local SSD")
    ap.add_argument("--detector", choices=list(DETECTOR_MAP), default="DFbsMDS")
    ap.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    ap.add_argument("--maxlag", type=int, default=20, help="Max seconds between images in a sequence")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--birds", action="store_true", help="Enable optional bird sub-classification")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = leave default)")
    ap.add_argument("--num-partitions", type=int, default=1)
    ap.add_argument("--partition", type=int, default=0, help="This process handles shards where idx %% N == partition")
    ap.add_argument("--rescan", action="store_true", help="Rebuild the shard manifest")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "csv"), exist_ok=True)
    log_path = args.log or os.path.join(args.out_dir, f"run_p{args.partition}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | p" + str(args.partition) + " | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Thread env must be set before torch is imported (via predictTools).
    if args.threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))

    sys.path.insert(0, os.path.abspath(args.software_dir))
    import torch
    import predictTools  # the official engine
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    detector_name = DETECTOR_MAP[args.detector]
    logging.info("Loading models once (detector=%s, birds=%s, device=cpu)...", detector_name, args.birds)
    patch_engine_to_reuse_models(predictTools, torch.device("cpu"), args.birds, detector_name)

    shards = build_or_load_shards(args.root, args.out_dir, args.rescan)
    assigned = [(i, rel, n) for i, (rel, n) in enumerate(shards)
                if i % args.num_partitions == args.partition]
    total_imgs = sum(n for _, _, n in assigned)
    csv_dir = os.path.join(args.out_dir, "csv")
    failed_log = os.path.join(args.out_dir, f"failed_p{args.partition}.tsv")

    logging.info("This process owns %d shards (%d images).", len(assigned), total_imgs)
    done_imgs, done_shards, t0, last = 0, 0, time.time(), time.time()

    for idx, rel, n in assigned:
        if _STOP:
            break
        out_csv = os.path.join(csv_dir, shard_id(rel) + ".csv")
        if os.path.exists(out_csv):
            done_imgs += n
            done_shards += 1
            continue

        files = list_images_in_dir(os.path.join(args.root, rel))
        if not files:
            write_shard_csv(out_csv, [])  # mark empty shard done
            done_shards += 1
            continue

        try:
            rows = process_shard(predictTools, files, args.threshold, args.maxlag,
                                 args.lang, args.birds, args.batch_size, detector_name)
            write_shard_csv(out_csv, rows)
        except Exception as e:
            logging.exception("Shard FAILED (left for retry): %s", rel)
            with open(failed_log, "a") as f:
                f.write(f"{rel}\t{type(e).__name__}: {e}\n")
            continue  # no CSV written -> retried next run

        done_imgs += len(files)
        done_shards += 1
        now = time.time()
        if now - last >= 30:
            rate = done_imgs / (now - t0) if now > t0 else 0.0
            remaining = max(0, total_imgs - done_imgs)
            eta_h = (remaining / rate / 3600) if rate > 0 else float("inf")
            logging.info("Shards %d/%d | images %d/%d | %.2f img/s | ETA %.1f h | last: %s",
                         done_shards, len(assigned), done_imgs, total_imgs, rate, eta_h, rel)
            last = now

    logging.info("Stopped. Shards done this process: %d/%d, images: %d/%d.",
                 done_shards, len(assigned), done_imgs, total_imgs)


if __name__ == "__main__":
    main()
