# DeepFaune batch classification: handover to Claude Code

## 1. Mission
Classify ~1.8 million camera-trap photographs for Fundatia Conservation Carpathia (FCC,
Romania) using the official DeepFaune v1.4.1 implementation, on a CPU-only PC, reading
images from a read-only external drive and writing results to the local solid-state
drive (SSD). The job is a multi-day batch that must be resumable, run detached, and
never write to the source drive. The programme is lynx-focused, but every image should
be classified.

Your first job: install dependencies in a virtual environment, then author the batch
orchestrator `deepfaune_batch.py`, then run a smoke test on the bundled `testdata/`.
See section 7.

## 2. Target machine (verified from diagnostics)
- Lenovo ThinkCentre neo 50s Gen 3; Ubuntu 24.04.4 LTS; kernel 6.17.0-35; Python 3.12.
- CPU: Intel Core i9-12900, 16 cores / 24 threads.
- RAM: ~16 GiB (the binding constraint). Swap 4 GiB.
- GPU: none usable (`nvidia-smi` absent). Inference is CPU-only.
- Storage: NVMe 1.9 TB; Linux root `/` is ext4 ~54 GB (writable, holds /home). External
  WD My Book 10.9 TB exFAT at `/media/rim/My Book`, 3.6 TB used; the images live here;
  mounted READ-ONLY.
- Dual-boot with Windows. Do not reboot during this work; there is no out-of-band
  console (boot order is set to Ubuntu as a fallback only).

CRITICAL for code: this box is CPU-only. Do NOT write CUDA paths or assume a GPU. The
high-end GPU workstation in the user's saved preferences is a different machine and does
not apply here. Use the CPU build of PyTorch and CPU parallelism (threads, optionally
multiple processes).

## 3. Safety constraints (do not violate)
1. Read images only from `/media/rim/My Book` (read-only). Never write there. Verify with
   `findmnt` before a run; the read-only state is not persistent across a reboot.
2. Write all outputs (CSVs, logs) to the SSD under /home. Results are small.
3. Weights must never enter git (large; gitignored).
4. Before the full run, confirm with the user that an independent OFFLINE backup of the
   3.6 TB exists. The read-only mount guards against accidental writes, not against a
   compromised host.

## 4. DeepFaune install state
- Software (v1.4.1, stable) is unpacked at: `/home/rim/deepfaune-src-1.4.1-08112025`
  This path is the `--software-dir`.
- Tarball SHA256 recorded: 2281655059d1bf1ef3b058af9234c8afdd64d34fed1f5dd82a7584eca032c4af
  (self-recorded baseline; no official checksum was located).
- All five weights are present in that folder (gitignored):
  - deepfaune-vit_large_patch14_dinov2.lvd142m.v4.pt (classifier, 1.2 GB)
  - deepfaune-vit_large_patch14_dinov2.lvd142m.v4-bird_head.pt (bird sub-classifier, 8.2 MB)
  - deepfaune-yolov8s_960.pt (DF detector, 22 MB)
  - md_v1000.0.0-sorrel.pt (MegaDetector sorrel, 960 px, 19 MB)
  - md_v1000.0.0-redwood.pt (MegaDetector redwood, 1280 px, 269 MB)
- Python dependencies are NOT yet installed. git is installed. Create a virtual environment.

Install (CPU torch first, to avoid pulling CUDA wheels):
```
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0
python3 -m venv ~/df-venv && source ~/df-venv/bin/activate && pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics yolov5 timm pandas dill hachoir openpyxl setuptools==81
python -c "import torch; print('cuda:', torch.cuda.is_available())"   # must print False
```
`setuptools==81` is required (yolov5 breaks with newer setuptools). The graphical-user-
interface dependencies (PySimpleGUI, tkinter) are not needed for headless batch work.

## 5. How the engine works (verified by reading the source in this repo)
Entry point: `from predictTools import PredictorImage`.
Signature: `PredictorImage(filenames, threshold, maxlag, LANG, birdclassification,
BATCH_SIZE=8, detectorname=..., device=None)`. Pass `device="cpu"`.
- Two stages: a detector runs per image and labels animal/human/vehicle/empty; the
  classifier (timm `vit_large_patch14_dinov2.lvd142m`, DINOv2 ViT-L, crop size 182,
  ImageNet normalisation) runs batched over the animal crops. 38 animal classes plus
  human, vehicle, empty. With `birdclassification=True`, an 8-way bird head
  (1024 -> 2048 -> 8) further splits crops classed as "bird", reusing the same embedding,
  so the extra compute is negligible.
- Detector options (string `detectorname`): "DF" (yolov8s at 960, lightest), "MDS"
  (sorrel at 960), "DFbsMDS" (run DF, fall back to sorrel only when DF finds nothing),
  "DFMDS" (ensemble of both on every image), "MDR" (redwood at 1280, GPU recommended,
  avoid on CPU).
- Sequences: `findSequences(maxlag)` groups images into bursts by Exchangeable-image-file-
  format (EXIF) timestamp within `maxlag` seconds and averages logits across each burst.
  `getPredictions()` returns sequence-level results; `getPredictionsBase()` returns per
  image. Also `getDates()`, `getSeqnums()`, `getFilenames()`, `getHumanCount()`.
- Batch loop: `allBatch()` runs everything; read results afterwards. NOTE: it pre-allocates
  result arrays sized to the whole file list and writes nothing until the end, so it must
  be driven per shard, not over all 1.8M images at once.
- Robustness: `Detector.bestBoxDetection` catches all per-image exceptions and returns
  "empty", so a corrupt file does not stop a run.
- Defaults: classification threshold 0.5; maxlag 20 s. Use LANG 'en'. The class dictionary
  has no Romanian ('ro') option (languages are fr/en/it/de/es/no/se/pt); passing 'ro'
  would error. Map English labels to Romanian afterwards if FCC needs that.

## 6. Orchestrator specification (what to build)
A resumable wrapper that drives PredictorImage. Requirements:
1. Shard by leaf directory (the directory directly containing images). This bounds memory
   and preserves sequences: a deployment's burst lives in one folder, and grouping across
   folders would corrupt the sequence aggregation.
2. One CSV per shard, written to a temp name then atomically renamed on success. Skip any
   shard whose final CSV already exists. This gives resumable, restart-safe runs.
3. Build the classifier and detector ONCE per process and inject them into each lightweight
   PredictorImage, so the 1.2 GB classifier is not reloaded per shard. Technique: after
   `import predictTools`, construct `Classifier`/`ClassifierWithBirds` and `Detector` once,
   then rebind `predictTools.Classifier`, `predictTools.ClassifierWithBirds` and
   `predictTools.Detector` to return those singletons. Confirm the constructor signatures
   against the source in the repo.
4. Filter macOS junk: skip names starting with `._` and `.DS_Store`. Keep image extensions
   (.jpg .jpeg .png .bmp .tif .tiff .gif). Exclude videos (.mp4 and similar) from the photo run.
5. Detached-run friendly: heartbeat logging with images-done, throughput and an estimated
   time of arrival (ETA); clean commit on SIGINT/SIGTERM.
6. Optional parallelism via `--num-partitions N --partition i` (each process handles a
   disjoint subset of shards, crash-isolated by the per-shard CSV). DEFAULT to 1. The source
   drive is a universal-serial-bus (USB) spinning disk, so concurrent processes may seek-
   thrash and run slower; benchmark 1 vs 2 vs 3 while watching `htop` and `iostat -x 5`.
7. All outputs to the SSD; never to the read-only drive.

Suggested flags: `--software-dir --root --out-dir --detector --threshold --maxlag --lang
--birds --batch-size --threads --num-partitions --partition --rescan`.
Suggested CSV columns: filename, date, seqnum, prediction_seq, score_seq,
prediction_image, score_image, animal_count, human_count. With --birds the predictions may
be bird subgroups.
A prior 281-line version of this orchestrator existed and was removed from the repo so you
can author it cleanly; the points above are its specification.

## 7. First tasks, in order
1. Create the venv and install dependencies (section 4). Confirm `torch.cuda.is_available()`
   is False.
2. Author `deepfaune_batch.py` per section 6. Commit it.
3. Smoke test on the bundled labelled samples, bird head on:
   - Run the engine over `/home/rim/deepfaune-src-1.4.1-08112025/testdata/` with `--birds`.
   - Confirm weights load and labels are sensible (cat1 -> cat, hedgehog -> hedgehog,
     empty2 -> empty), and `bird1.JPG` returns a bird subgroup (anseriform/corvid/passerine
     and so on) rather than the bare "bird" label.
4. Benchmark the detector on a representative real subtree (a few thousand images under
   `/media/rim/My Book`): measure images/sec and an ETA for 1.8M. Compare `--detector DF`
   (start here) with `--detector DFbsMDS`. Spot-check recall on known images. All five
   weights are present, so any detector works.
5. Choose the root and run the full job under tmux. It skips completed shards, so it is
   stop/restart-safe.

## 8. Decisions locked in
- Use the official DeepFaune v1.4.1 implementation, not the user's older one-image-at-a-time
  camera-traps wrapper.
- Bird head ENABLED (`--birds`) for this run.
- Detector: start with `DF`; benchmark `DFbsMDS`. Avoid `MDR` on CPU.
- Shard by leaf directory; one CSV per shard; resumable.
- Labels in English ('en').

## 9. Repo and workflow
- Private GitHub repo: `ratsakatika/deepfaune-fcc`, branch `main`. Contains the DeepFaune
  v1.4.1 source and `testdata/`; weights are gitignored; no orchestrator yet.
- `.gitignore` already excludes *.pt/*.pth/*.onnx, archives, __pycache__, *.log, df_out/,
  outputs/, config.local.*, .env, virtual environments.
- Weight-staging gate before any commit: `git add -A`, then confirm
  `git diff --cached --name-only | grep -E '\.(pt|pth|onnx)$'` returns nothing. A weight in
  a commit is hard to remove and GitHub rejects files over 100 MB.
- Workflow loop: author in Claude Code, commit and push, then on the box `git pull` and
  review the diff before running. The box runs only reviewed code from this repo.

## 10. Open threads (not blocking the run)
- Confirm an offline backup of the 3.6 TB exists.
- A read-only audit of cron jobs, systemd timers, services and accounts is still pending
  (the network has a history of compromise); separate from this work.
- The GitHub token is a fine-grained personal access token scoped to deepfaune-fcc
  (Contents read and write); revoke it when the project ends. If `credential.helper store`
  was used, the token is in plaintext in ~/.git-credentials.

## 11. Coding preferences for this work
- CPU-only: no CUDA. Use the CPU torch build; parallelise with CPU threads
  (`torch.set_num_threads`, `OMP_NUM_THREADS`) and optionally multiple processes.
- Be explicit about input/output, batching and memory: data is on a slow USB drive and RAM
  is ~16 GB; stream rather than load everything, keep batches and worker counts modest,
  watch resident memory.
- Prefer well-maintained libraries; the stack is fixed by DeepFaune (torch, ultralytics,
  yolov5, timm, opencv, pandas).
- British English; precise and concise; no em dashes.

## Reference
Rigoudy, N., Dussert, G., Benyoub, A. et al. (2023) 'The DeepFaune initiative: a
collaborative effort towards the automatic identification of European fauna in camera trap
images', European Journal of Wildlife Research, 69(6), 113.
https://doi.org/10.1007/s10344-023-01742-7
