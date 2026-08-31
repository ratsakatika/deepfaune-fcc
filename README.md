#  WELCOME TO DEEPFAUNE SOFTWARE REPOSITORY


<img src="icons/logoINEE.png" width="50%" align=center>
<br>


---
# FCC CAMERA-TRAP BATCH TOOLING
---

This fork adds a resumable, CPU-only pipeline for classifying the Fundatia
Conservation Carpathia camera-trap archive with the official DeepFaune v1.4.1
engine, plus a friendly front end. Two commands:

- `dfrun` - the friendly front end: it finds the drive, assesses the work,
  confirms the settings, launches the classifier detached, and shows a live
  readout.
- `deepfaune_batch.py` - the orchestrator that `dfrun` drives. It can also be
  run directly.

## User guide (everyday use)

A step-by-step guide is on the desktop as **How_to_use_dfrun.html**; the same
guide is here: [docs/USER_GUIDE.html](docs/USER_GUIDE.html). In short:

1. Connect the "My Book" hard drive.
2. Open the **Run DeepFaune** icon on the desktop. The first time only,
   right-click it and choose "Allow Launching".
3. Confirm the settings at the prompts: press Enter to accept a value, or type a
   new value and press Enter.
4. Processing runs on the computer. You may close the window or disconnect (if connected remotely); it
   continues. Open the icon again at any time to view progress.
5. When it finishes, the dashboard opens in a web browser and the result files
   are on the desktop. If it stops, open the icon again to resume.

In normal use you will not reprocess the whole archive. When you add new
camera-trap photographs to the hard drive, run the tool again: it processes only
the new folders and updates the results and the dashboard. That incremental
update is the main day-to-day use.

## Quick start

1. Install the DeepFaune dependencies in a virtual environment (see the
   DeepFaune documentation below), then add the front-end extras:
   `pip install rich psutil plotly openpyxl`.
2. Put the tool on PATH and add a Desktop shortcut: `make install`
   (or `bash install.sh`). The first time you use the Desktop icon, right-click
   it and choose "Allow Launching" so GNOME trusts it.
3. Plug in the archive drive (mounted read-only) and run `dfrun`.
4. Follow the stages: it checks for a newer version, finds the drive, reports
   how much is already done, asks you to confirm the settings, then launches.
5. Watch the live readout. You can close the window or disconnect (if connected
   remotely); the run keeps going. Re-run `dfrun` to reattach to it.
6. When it finishes, a dashboard opens in your browser and the
   spreadsheet-friendly outputs are on your Desktop.

To resume after any stop, just run `dfrun` again; finished shards are skipped.

## Parameters and defaults

| Setting | Default | Notes |
| --- | --- | --- |
| detector | DF | DF, MDS, DFbsMDS, DFMDS or MDR. DF is the lightest; avoid MDR on CPU. |
| birds | on | The 8-way bird sub-classifier head. |
| threshold | 0.5 | Matches the demo script and the first long run (the GUI uses 0.8). |
| maxlag | 20 s | Matches the demo script and the first long run (the GUI uses 10). |
| batch-size | 8 | Must be at least 1. |
| threads | 4 | Must be at least 1. Capping this also bounds memory use. |
| merge-every | 600 s | Rebuild master.csv on this interval; 0 disables it. |

What each setting does:

- **detector** - the first stage that finds the animal, human or vehicle boxes in
  each photo (a separate model then names the species, and that part is the same
  whichever detector you choose). Two base detectors and two ways of combining
  them, roughly fastest/lowest recall to slowest/highest
  (`DF` ~ `MDS` < `DFbsMDS` < `DFMDS` << `MDR`):
  - **DF** (default) - DeepFaune's own detector, a YOLOv8s model trained on
    European camera-trap images, run at 960 px. The lightest and fastest.
  - **MDS** - MegaDetector "sorrel" (MegaDetector v1000, the widely used
    general-purpose detector from Microsoft / Dan Morris) at 960 px. Similar speed
    to DF but trained differently, so it finds some things DF misses.
  - **MDR** - MegaDetector "redwood" (v1000) at 1280 px. Potentially the best at
    finding animals, but much slower and intended for a machine with a graphics
    card; avoid it on this CPU-only box.
  - **DFbsMDS** ("DF, backstop MDS") - runs DF first, and only tries MDS on photos
    where DF found nothing. A cheap way to recover a few animals DF alone misses.
  - **DFMDS** ("DF + MDS ensemble") - runs both detectors on every photo and
    merges their boxes. The highest recall of the 960 px options, but the slowest.

  For this box: DF is the default and what the first long run used; DFbsMDS is the
  sensible "a bit more thorough" option; DFMDS only if you can spare the time; MDR
  not on CPU.
- **birds** - when on, crops classed as "bird" are split into eight groups
  (corvid, raptor, passerine and so on); when off they stay as "bird".

Note there is no species-exclusion setting here. Excluding species that cannot
occur in the survey area is a filtering decision, not a classification one: the
raw `score_*` columns are recorded for every class regardless, so the same
exclusion applied in `dashboard_and_filter.html` gives an identical result and
can be undone, while excluding during classification is permanent and splits
the archive if it is ever changed mid-run. `deepfaune_batch.py` keeps an
`--exclude-classes` flag only so a run started before this changed can be
resumed consistently.
- **threshold** - the classification confidence cut-off. A prediction scoring
  below it is recorded as "undefined". Higher (for example 0.8) means fewer false
  positives (fewer wrong species labels) but more false negatives (more real
  animals left undefined); lower means more photos get a species name, but more
  of those names are wrong. 0.5 is a balanced middle.
- **maxlag** - the time gap, in seconds, within which consecutive photos in the
  same folder are grouped into one detection event (a sequence), so a burst of
  frames is counted once. A larger gap merges more frames into a single event
  (less double-counting of a lingering animal, but two separate visits close in
  time may merge); a smaller gap splits them into more events.
- **batch-size** - how many animal crops are classified at once. This affects
  speed and memory only, not the results; larger is slightly faster but uses more
  memory.
- **threads** - how many processor cores are used. More is faster but uses more
  memory and runs hotter; capping it also bounds memory use.
- **merge-every** - how often, in seconds, the master CSV and dashboard are
  rebuilt during a run. This affects how often you see progress, not the results.

Note on consistency: the first long run used threshold 0.5 and maxlag 20, and
the tool now defaults to exactly that, so a continued run stays consistent with
the shards already classified (no mixed dataset). The official GUI uses 0.8/10.
`dfrun` warns if a previous run used different values, and `--rescan` re-does
everything uniformly.

## Outputs

Working files stay in the output directory (default `~/df_out`):

- one CSV per leaf directory (the source of truth), named `<path>__<hash>.csv`;
- `master.csv`, a regenerated snapshot of all shard CSVs;
- `run_metadata.p0.json` (provenance), `status.p0.json` (live status),
  `deepfaune_batch.p0.log` (detailed log), and `unreadable_images.txt` if any
  files could not be read.

On the Desktop, for easy download:

- `deepfaune_master.csv` (every image; too large for Excel, open it in R);
- `deepfaune_wildlife.csv` (animals only; within the spreadsheet row limit);
- `deepfaune_summary.csv` (per-species and per-station counts);
- `deepfaune_dashboard.html` (interactive charts and a camera map; opens in a browser).

### Result columns

Each shard CSV (and therefore the master and wildlife files) has one row per
image with the full raw model outputs, so the classification threshold can be
changed retrospectively without re-running anything:

- `filename`, `date`, `seqnum` - the image, its EXIF date, and the sequence
  (detection event) it was grouped into.
- `prediction_seq`, `top1_seq`, `score_seq`, `above_threshold` - the
  sequence-level prediction the pipeline reports. Below the threshold the
  engine rewrites `prediction_seq` to "undefined", but `top1_seq` keeps the
  best-scoring label and `score_seq` its score, and `above_threshold`
  (yes/no) says whether the threshold was passed - so no information is lost.
- `prediction_image`, `top1_image`, `score_image` - the same, per single image
  (no sequence averaging).
- `animal_count`, `human_count` - counts of boxes found by the detector.
- `det_conf_animal`, `det_conf_human`, `det_conf_vehicle` - the detector's raw
  confidence: the best box confidence per category (0 = no box found).
- `score_<class>` (one column per animal class) - the classifier's raw softmax
  score over all animal classes for this image, recorded whatever the
  threshold. Blank when the detector found no animal (the classifier never
  ran: there was no crop to classify).
- `birdscore_<class>` (one column per bird group, when birds is on) - the bird
  sub-classifier's softmax; a conditional "which bird" distribution, only
  meaningful when the animal is a bird.

- `sequence_id` - a globally unique key for the detection event. **`seqnum`
  alone is not unique**: the engine numbers sequences per folder, restarting
  at 1 in every one, so seqnum 1 recurs about once per folder across the
  archive. Group by `sequence_id` (or by folder plus `seqnum`) when counting
  events, never by `seqnum` on its own. The dashboard already does this.

`sequence_id` and the image paths are derived when the master and Desktop
CSVs are built rather than in the shards, so they apply uniformly to every
row, including shards written by older versions. Paths are re-rooted onto the
run's own archive root, so a drive that remounts under a different name (My
Book1 vs My Book2) still yields one consistent set of paths.

To re-threshold after the fact: take rows where the max of the `score_*`
columns (or `score_seq`) clears your new cut-off, using `top1_seq` as the
label. Shards written by older versions of the tool lack the raw-score
columns; in the merged master those cells are simply blank. Re-run those
folders with `deepfaune_batch.py --rescan` if you need their raw scores too.

## Dashboard

`deepfaune_dashboard.html` is built from the Desktop master CSV by
`build_dashboard.py`, with the camera map placed from
`FieldProtocols_WTM_FAR_23.xlsx` in the software directory (the map is omitted if
the protocol is absent). It opens automatically when a run finishes, and is
rebuilt in the background each time the master is updated, so you can open it
early to watch the results grow. To rebuild it on demand without a run, use
`dfrun --dashboard`.

## Diagnostics and self-protection

The worker keeps a flight recorder: one line every ~30 seconds in
`telemetry.p0.csv` (memory, swap, shared memory, disk, load, worker size,
whether the archive drive is reachable), fsynced so the final lines survive
any crash. `dfrun --diagnose` prints the run settings, the newest samples and
a one-line verdict on what a dead run died of (memory exhaustion, full disk,
drive disconnect, reboot/power loss, or software fault).

The run philosophy is to keep going: with swap sized like RAM the worker
slows down under memory pressure rather than stopping, and nothing halts
out of caution. The protections are shaped around that, each born from a
real incident: the worker volunteers itself to the kernel's out-of-memory
killer (in a true emergency the resumable worker dies, not the desktop or
systemd - whose deaths once orphaned 13 GiB of shared memory), the launch
checks for and offers to remove orphaned swapfiles (the only thing that has
ever filled this disk), the worker stops only when the output disk is
genuinely about to run out (`--min-disk-gib`, default 0.5), and a shard is
abandoned unwritten if the archive drive vanishes or reads fail en masse -
so a disconnect can never be recorded as thousands of false "empty" images
(that guard cannot fire while the drive is present). An optional
`--min-avail-gib` pause-under-pressure guard exists but is off by default.
Protective stops record their reason in `status.p0.json` and the log.

Rebuilding outputs on demand: `dfrun --outputs` regenerates `master.csv` and
every Desktop file from the shard CSVs without reclassifying anything. Use it
after adding or removing shard CSVs, or to apply derived columns (such as
`sequence_id`) to results produced by an earlier version. Add
`--source /path/to/archive` to set the root the image paths are normalised
onto, when the drive has since been remounted elsewhere.

## Dashboard and result filter

`dashboard_and_filter.html` on the Desktop opens in a browser and reads the
master CSV locally - nothing is uploaded. It carries the full survey dashboard,
computed live from whatever filters are set: the header KPIs, the camera-grid
map, the daily-rhythm ring, the through-the-seasons stacked chart (weekly
timeline or by-month, with a click-through from ecological group to species),
the what-the-cameras-saw species bars (events or photos, guild legend
toggles) and the busiest-cameras bars, followed by the events table.

The survey statistics (thesis paragraph and KPI tiles) sit just below the
load step and recompute from the loaded CSV on every Generate; the official
survey / all images scope is a filter option alongside the thresholds. When
served from the out-dir (for example `python3 -m http.server 8000` next to
the master file) it loads `deepfaune_master.csv` automatically. The 2023
field protocol's camera coordinates are embedded in the page, so the
camera-grid map works out of the box; a `FieldProtocols_WTM_FAR_23.xlsx`
next to the page, or one loaded via the map panel's picker, replaces the
embedded set. Because the `score_*` columns
hold the complete distribution over every animal class, it re-derives
predictions rather than just filtering the ones already written: set a global
or per-species confidence threshold (typed or by slider), require a minimum
sequence length in seconds, untick species that cannot occur so their
detections fall to the next-best species, and switch scope between the
official survey (Camera Trap Monitoring/Collections) and all images. The
events table lists events oldest first, shows the top three species with
scores to three decimal places (naming the top choice even when it scored
below the threshold, shown in red rather than hidden behind "undefined"),
and, when the image files are reachable from where the page is opened,
clicking a row previews its photographs (a silent probe decides this, so
nothing ever renders broken). "Keep below-threshold events" sits with the
threshold control, since those are animal detections rather than non-animal
categories. Events whose camera clock was wrong
(1970 dates) are counted in station and map totals but left off the time
panels, exactly as in the original dashboard. The ring saves as PNG or SVG,
the seasonal chart and both bar panels save as PNG, and the species summary
saves as a CSV. Two filtered exports are offered: an events CSV (one row per
event, with `top2`/`top3` columns) and an images CSV that re-reads the master
and keeps exactly its columns, writing only the rows of surviving events with
the sequence-level verdict columns refreshed to the re-derived result.
The seasonal chart and the map load their libraries (Plotly, Leaflet) from
public CDNs, so those two panels need an internet connection the first time;
everything else works offline. Events are grouped by `sequence_id`.

## Auto-resume service

`dfrun --install-service` installs a systemd unit, `deepfaune-worker.service`,
that restarts the worker two minutes after a violent death - an out-of-memory
kill or a crash - and ONLY then. It never resurrects a run that ended any
other way: a finished run, a protective stop, `dfrun --stop` and
`sudo systemctl stop deepfaune-worker` all stay stopped, and nothing
autostarts at boot (after a power cut, open Run DeepFaune to continue). So
the user is never locked into a run they meant to end.

No settings are baked in at install time: each start reads
`run_metadata.p0.json` and resumes the CURRENT run exactly as it was
launched (`--resume-last`), mounting the archive drive read-only at the
run's own recorded root first. With the service installed, `dfrun` launches
runs under systemd automatically (writing the confirmed settings to the
metadata before starting) and still attaches to the live readout as before;
the shared pidfile means the service and manual runs never double-classify.
`dfrun --uninstall-service` removes it.

## Resume and safety

- Settings follow the run: when the out-dir already holds a run, that run's
  own settings (detector, threshold, maxlag, birds, exclusions, threads)
  become the suggested values, so continuing it cannot silently reclassify
  the remainder differently. Anything given explicitly on the command line
  still wins, and the prompts can still change anything.
- Resumable: each shard's CSV is written atomically, and a shard that already
  has a CSV is skipped, so stopping and restarting is safe.
- Incremental: because finished folders are skipped, adding new photographs to
  the drive and running again classifies only the new folders and refreshes the
  outputs and dashboard. The full archive is not reprocessed.
- CPU only: CUDA is disabled before torch is imported and the device is forced
  to cpu.
- The source drive is read-only and is never written to; all outputs go to
  local disk. `dfrun` checks the read-only mount and the drive UUID first.
- A single-instance lock prevents two workers, and the worker is launched
  detached so it survives an SSH or editor disconnection.

## Running the orchestrator directly

`python deepfaune_batch.py --software-dir <dir> --root <archive> --out-dir
~/df_out --birds`. Use `--dry-run` to enumerate without loading models,
`--merge` to rebuild master.csv, and `--help` for all flags.

## Tests

`make test` (or `python -m pytest tests/ -q`).

---
# NEWS
---
## October 2025
Release v1.4 is available

* New categories 'golden jackal', 'raccoon dog', 'porcupine' and 'muskrat'.
* Bird classification into 'anseriform', 'columbiform', 'corvid', 'galliform', 'passerine', 'piciform', 'raptor', 'otherbird'  is possible (optional).
* New possibility to choose between detectors.
* Device choice is now possible.

Supported categories/species : BADGER, BEAR, BEAVER, BIRD, BISON, CAT, CHAMOIS, COW, DOG, EQUID, FALLOW DEER, FOX, GENET, GOAT, GOLDEN JACKAL, HEDGEHOG, IBEX, LAGOMORPH, LYNX, MARMOT, MICROMAMMAL, MOOSE, MOUFLON, MUSKRAT, MUSTELID, NUTRIA, OTTER, PORCUPINE, RACCOON, RACCOON DOG, RED DEER, REINDEER, ROE DEER, SHEEP, SQUIRREL, WILD BOAR, WOLF, WOLVERINE +  HUMAN + VEHICULE + EMPTY

+ (option) ANSERIFORM, COLUMBIFORM, CORVID, GALLIFORM, PASSERINE, PICIFORM, RAPTOR, OTHERBIRD


## February 2025
Release v1.3 is available.

* New categories 'bison', 'moose', 'reindeer' and 'wolverine'  (in french 'bison', 'elan', 'renne' and 'glouton').
* Even more efficient classification model, still based on vit_large_patch14_dinov2 architecture.
* New possibility to choose between our yolov8s at resolution 960 for detection and MegaDetector (Microsoft) yolov10x at resolution 640.
* Use of more icons instead of text in software design.
* Animal counts and human counts are managed independently, and displayed in the interface.
* Column 'HumanPresence' replaced by 'HumanCount'.

Supported categories/species : BADGER, BEAR, BEAVER, BIRD, BISON, CAT, CHAMOIS/ISARD, COW, DOG, EQUID, FALLOW DEER, FOX, GENET, GOAT, HEDGEHOG, IBEX, LAGOMORPH, LYNX, MARMOT, MICROMAMMAL, MOUFLON, MOOSE, MUSTELID, NUTRIA, OTTER, RACCOON, RED DEER, REINDEER, ROE DEER, SHEEP, SQUIRREL, WILD BOAR, WOLF, WOLVERINE + HUMAN + VEHICULE + EMPTY


## October 2024
Release v1.2 is available.  

Supported categories/species : BADGER, BEAR, BEAVER, BIRD, CAT, CHAMOIS/ISARD, COW, DOG, EQUID, FALLOW DEER, FOX, GENET, GOAT, HEDGEHOG, IBEX, LAGOMORPH, LYNX, MARMOT, MICROMAMMAL, MOUFLON, MUSTELID, NUTRIA, OTTER, RACCOON, RED DEER, ROE DEER, SHEEP, SQUIRREL, WILD BOAR, WOLF + HUMAN + VEHICULE + EMPTY 

---
# DOCUMENTATION & INSTALLATION PROCEDURE
---

Please refer to the online documentation at [https://deepfaune.pages.math.cnrs.fr/software/](https://deepfaune.pages.math.cnrs.fr/software/)

---
# LICENSE
---

All of the source code to this product is available under the [CeCILL](http://www.cecill.info), compatible with [GNU GPL](http://www.gnu.org/licenses/gpl-3.0.html).

Our model parameters ('deepfaune-*.pt' files) are available under the [Creative Commons Attribution-ShareAlike 4.0 International Public License](https://creativecommons.org/licenses/by-sa/4.0/).
They cannot be used without citing and referencing the name 'DeepFaune'.

Know your rights.

---
# TEAM & CONTACT
---

The DeepFaune software is developped by the Deepfaune team at CNRS. For more information about the project, please visit [https://www.deepfaune.cnrs.fr](https://www.deepfaune.cnrs.fr)

For any question, bug or feedback, feel free to send an email to [Vincent Miele](https://vmiele.gitlab.io/) <!--or use the Gitlab Service Desk-->

---
# REFERENCES
---

[Rig23] Rigoudy, N., Dussert G., the DeepFaune consortium, Spataro, B., Miele, V. & Chamaillé-Jammes, S. (2023) *The DeepFaune initiative: a collaborative effort towards the automatic identification of the European fauna in camera-trap images.* [European Journal of Wildlife Research](https://link.springer.com/article/10.1007/s10344-023-01742-7)

[Dus24] Dussert, G., Chamaillé-Jammes, S. Dray, S. &  Miele, V. (2024) *Being confident in confidence scores: calibration in deep learning models for camera trap image sequences.* [Remote Sensing in Ecology and Conservation](https://zslpublications.onlinelibrary.wiley.com/doi/10.1002/rse2.412)

[Dus25] Dussert, G., Dray, S., Chamaillé-Jammes, S. &  Miele, V. (2025) *Paying Attention to Other Animal Detections Improves Camera Trap Classification Models.* [biorxiv](https://www.biorxiv.org/content/10.1101/2025.07.15.664849.full.pdf)