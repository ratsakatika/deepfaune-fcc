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
2. Open the **dfrun (DeepFaune)** icon on the desktop. The first time only,
   right-click it and choose "Allow Launching".
3. Confirm the settings at the prompts: press Enter to accept a value, or type a
   new value and press Enter.
4. Processing runs on the computer. You may close the window or disconnect; it
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
5. Watch the live readout. You can close the window or disconnect; the run
   keeps going. Re-run `dfrun` to reattach to it.
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

## Dashboard

`deepfaune_dashboard.html` is built from the Desktop master CSV by
`build_dashboard.py`, with the camera map placed from
`FieldProtocols_WTM_FAR_23.xlsx` in the software directory (the map is omitted if
the protocol is absent). It opens automatically when a run finishes, and is
rebuilt in the background each time the master is updated, so you can open it
early to watch the results grow. To rebuild it on demand without a run, use
`dfrun --dashboard`.

## Resume and safety

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