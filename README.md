# lightweight-tracker

CPU-only Python aerial object tracker (NumPy + OpenCV). No GPU, no deep learning, no external model weights.

You draw a box on frame 1. The tracker holds lock using Kalman prediction, a pluggable correlation matcher, and a confidence-gated state machine that governs when to update the appearance model and when to widen the search.

---

## Where things stand

**v1** — NCC template matcher, 4-state machine, reacquisition, metrics, and video output. Works on stable targets but drifts on fast maneuvers.

**v2** — MOSSE adaptive correlation filter is now the default backend, with CSRT and KCF available through `CvTrackerMatcher` behind the same `IMatcher` interface. The tracker supports manual ROI, matcher selection, online appearance adaptation, confidence-gated updates, and annotated video output. MOSSE performs best on fast small targets; the hardest failure mode remains a 90° turn plus smoke occlusion.

**v2.1 (implemented)** — the runtime is working on local macOS; the CLI supports `--matcher`, `--roi`, `--multi`, and video export; and the codebase includes a `tracker.yaml` schema in `TrackerConfig` even though runtime YAML parsing is not yet wired in. The project has reverted the cone/fan/360 directional search idea and restored the plain MOSSE + re-detector + Kalman baseline. The multi-target tracker remains intact, and the first orchestration step is complete: the multi-target detector now uses parallelized heat-map tiling as a pure speed optimization.

**v2.2** — running the tracker with no video path opens a native file-picker dialog (macOS via `osascript`, Linux via `zenity`, `tkinter` fallback) so you can choose a video from your machine, then draw the target box on frame 1.

**v2.3 (current)** — the whole project is now **pure Python**. The C++17/CMake implementation was ported module-for-module (same state machine, thresholds, MOSSE filter, rotation recovery, Kalman model and multi-target tracker), so there is nothing to compile. The last C++ version is preserved in git history (`git log -- src/`).

**v3 (next)** — keep the baseline as MOSSE + re-detector + Kalman for single-target, supported by the multi-target path. Next planned work is scale-aware detection for tiny targets and selective full-sweep gating for the multi-target detector, so the system only runs the expensive heat-map sweep when a track is uncertain or lost.

---

## Architecture

```
VideoSource
    |
    v
TrackerEngine (main loop)
    |
    +-- KalmanBoxTracker   predict / correct
    +-- SearchRegion       build the ROI for this frame
    +-- IMatcher           pluggable appearance model
    |       MosseMatcher       (MOSSE adaptive filter, default)
    |       CvTrackerMatcher   (CSRT or KCF via cv::Tracker)
    |       TemplateMatcher    (NCC, v1 baseline)
    +-- ConfidenceEvaluator    fuse NCC + PSR -> [0,1] score
    +-- ReacquisitionManager   widen ROI, restore pristine template
    |
    v
Visualizer + TrackerMetrics
```

State machine: **TRACKING → UNCERTAIN → LOST → REACQUIRING → TRACKING**

Confidence thresholds: strong `≥ 0.70` (update model), weak `0.45–0.70` (freeze model), lost `< 0.45` (widen search, restore template).

Kalman state vector: `[cx, cy, vx, vy, w, h]`. Measurement: `[cx, cy, w, h]`.

---

## Install

Requires Python ≥ 3.9. No compiler or CMake needed.

```bash
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt                       # numpy + opencv-python
```

Works with OpenCV 4.x and 5.x. If `cv2.TrackerCSRT_create` / `cv2.TrackerKCF_create` are missing (rare on some OpenCV 4.x builds), install `opencv-contrib-python` instead.

---

## Run

```bash
# no arguments: opens a file picker to choose a video, then select the target on frame 1
python -m tracker

# MOSSE (default) — manual ROI selection on frame 1
python -m tracker data/trial1.MP4

# preset the ROI to skip the selection dialog (useful for fair A/B)
python -m tracker data/trial2.MP4 --matcher mosse --roi 10,1081,120,287 --no-rotation

# compare all three backends on the same clip with the same box
python -m tracker data/trial1.MP4 --matcher mosse --roi 812,140,46,58 --output outputs/videos/mosse.mp4
python -m tracker data/trial1.MP4 --matcher csrt  --roi 812,140,46,58 --output outputs/videos/csrt.mp4
python -m tracker data/trial1.MP4 --matcher kcf   --roi 812,140,46,58 --output outputs/videos/kcf.mp4

# multi-target mode for look-alike objects
python -m tracker data/trial1.MP4 --matcher mosse --multi --roi 812,140,46,58 --output outputs/videos/multi.mp4

# NCC baseline (v1 matcher, slowest to drift but weakest on fast targets)
python -m tracker data/trial1.MP4 --matcher ncc

# webcam
python -m tracker --camera 0

# headless (no GUI window, still writes video)
python -m tracker data/trial2.MP4 --no-display --output outputs/videos/redetectorbike.mp4
```

On frame 1, drag a box around the target, then press **Enter** or **Space** to confirm. Press `q` or **Esc** to stop early. Metrics print to the terminal at the end.

---

## CLI reference

| Flag | Default | Notes |
|---|---|---|
| `--matcher <name>` | `mosse` | `mosse` · `csrt` · `kcf` · `ncc` |
| `--roi x,y,w,h` | (manual) | Preset box; required for fair backend comparisons |
| `--learn <f>` | `0.05` | Template learn rate. MOSSE works best around `0.125` |
| `--anchor <f>` | `0` | MOSSE pull-back toward the frame-1 filter (0 = off) |
| `--prior <f>` | `0` | Velocity-oriented motion prior width (0 = off) |
| `--no-rotation` | — | Disable MOSSE log-polar rotation recovery |
| `--multi` | — | Multi-target mode: keep identities apart by motion |
| `--no-adapt` | — | Freeze the appearance model (diagnostic) |
| `--output <path>` | `outputs/videos/tracked.mp4` | Annotated video path |
| `--no-display` | — | Headless mode |
| `--camera <i>` | — | Use webcam index `i` instead of a file |

---

## Matchers

### MOSSE (`--matcher mosse`)
Adaptive correlation filter (Bolme et al., 2010). The filter is learned in the Fourier domain and updated online — adapts to appearance change, handles slight orientation shifts. This is the current best backend. PSR is the native quality signal and feeds directly into the confidence evaluator.

Key parameters:
- `sigma` — width of the desired Gaussian response (default `2.0`)
- `initWarps` — number of random affine warps used to bootstrap the filter (default `8`)
- `anchor` — per-update pull-back toward the initial filter; helps against catastrophic drift but slows adaptation (default `0`, i.e. off)
- `prior` — biases the response map in the direction of predicted motion (default `0`)

### CSRT (`--matcher csrt`)
OpenCV's discriminative correlation filter with channel and spatial reliability. Heavier than MOSSE, generally more accurate on slow/large targets, loses lock faster on sudden small-target maneuvers in this footage.

### KCF (`--matcher kcf`)
Kernelized Correlation Filter. Faster than CSRT, similar accuracy profile in practice. Both CSRT and KCF are wrapped by `CvTrackerMatcher`, which adapts the `cv::Tracker` interface to `IMatcher`.

### NCC (`--matcher ncc`)
Normalized cross-correlation, the v1 baseline. Rigid template, no online update unless `adapt` is true. Good for sanity-checking the pipeline; loses lock earliest on real footage.

---

## Adding a new matcher

1. Create `tracker/yours.py` with a class that subclasses `IMatcher` (in [tracker/matchers.py](tracker/matchers.py)).
2. Implement `set_template`, `update_template`, `use_original_template`, `match` and `has_template`. `set_motion_hint` and `set_reacquiring` are optional.
3. Wire it into `tracker/cli.py` (add the name to `--matcher` choices and an `elif` branch that constructs it).

Nothing else changes. The engine, state machine, confidence evaluator, and visualizer are all matcher-agnostic.

---

## What to look at next (v3 targets)

- **Ground-truth metrics** — load a ground-truth `.txt` (MOT format) alongside the video and compute per-frame IoU and center error. The `TrackerMetrics` class already tracks frame count, lost-frame count, and runtime; extend it here.
- **Benchmark table** — run all four matchers on all three clips in one script, collect the metrics, and print a comparison table. This will make the tracker comparisons quantitative instead of only visual.
- **`tracker.yaml` parsing** — The `TrackerConfig` dataclass in [tracker/types.py](tracker/types.py) already defines every tunable field. Wire it to the YAML file at [configs/tracker.yaml](configs/tracker.yaml) using `cv2.FileStorage` or PyYAML so runtime parameters can be loaded from disk.
- **Multi-template memory** — store the N best patches seen during TRACKING and fall back to them during reacquisition instead of relying only on the pristine frame-1 crop.

---

## File map

```
tracker/
  types.py        — TrackStatus, MatchResult, Evaluation, TrackState, TrackerConfig
  matchers.py     — IMatcher interface, NCC TemplateMatcher, CSRT/KCF CvTrackerMatcher
  mosse.py        — MOSSE adaptive filter + log-polar RotationEstimator
  kalman.py       — Kalman predict / correct
  components.py   — ConfidenceEvaluator, search region, ReacquisitionManager, metrics, Visualizer
  engine.py       — VideoSource + TrackerEngine (single-target main loop)
  multi.py        — MultiTargetTracker (motion-track manager) + run loop
  cli.py          — argument parsing, file picker; entry point
configs/tracker.yaml — config file (fields defined, parsing not yet wired)
requirements.txt
data/                — test clips (not committed; git-ignored)
outputs/videos/      — written here by default
```

---

## Feedback & collaboration

Contributions, ideas, and feedback on future versions are welcome. This is an open project and collaboration is encouraged.

- **Give feedback:** open an issue with your thoughts on the v3 targets above (scale-aware detection, selective full-sweep gating, ground-truth metrics, benchmarking) or anything else you'd like to see.
- **Branch and build on it:** fork the repo, create a feature branch (`git checkout -b feature/your-idea`), and open a pull request. New matchers are especially easy to add; see [Adding a new matcher](#adding-a-new-matcher).
- **Share results:** if you run the tracker on your own footage, I'd love to hear where it holds lock and where it fails.
- **Collaborate:** if you want to work on a larger piece together, get in touch via a GitHub issue or discussion.

---

## Acknowledgements

This project was developed in collaboration with **Megh Lad**.

The original collaborative repository is available at:
https://github.com/Meghlad/lightweight_tracker

This repository is maintained under my GitHub account to document my own contributions while preserving the complete collaborative Git history.

## License

See [LICENSE](LICENSE).
