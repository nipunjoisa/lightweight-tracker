# lightweight-tracker

CPU-only C++17 aerial object tracker. No GPU, no deep learning, no external model weights.

You draw a box on frame 1. The tracker holds lock using Kalman prediction, a pluggable correlation matcher, and a confidence-gated state machine that governs when to update the appearance model and when to widen the search.

---

## Where things stand

**v1** — NCC template matcher, 4-state machine, reacquisition, metrics, and video output. Works on stable targets but drifts on fast maneuvers.

**v2 (current)** — MOSSE adaptive correlation filter is now the default backend, with CSRT and KCF available through `CvTrackerMatcher` behind the same `IMatcher` interface. The tracker supports manual ROI, matcher selection, online appearance adaptation, confidence-gated updates, and annotated video output. MOSSE performs best on fast small targets; the hardest failure mode remains a 90° turn plus smoke occlusion.

**v2.1 (implemented)** — the runtime is working on local macOS; the CLI supports `--matcher`, `--roi`, `--multi`, and video export; and the codebase includes a `tracker.yaml` schema in `include/Types.hpp` even though runtime YAML parsing is not yet wired in. The project has reverted the cone/fan/360 directional search idea and restored the plain MOSSE + re-detector + Kalman baseline. The multi-target tracker remains intact, and the first orchestration step is complete: the multi-target detector now uses parallelized heat-map tiling as a pure speed optimization.

**v2.2** — running `./build/tracker` with no arguments now opens a native file-picker dialog (macOS via `osascript`, Linux via `zenity`) so you can choose a video from your machine, then draw the target box on frame 1.

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

## Build

Requires: CMake ≥ 3.16, a C++17 compiler, OpenCV 4.x.

```bash
# macOS
brew install cmake opencv

# configure + build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# binary is at build/tracker
```

If `find_package(OpenCV)` fails on Apple Silicon:

```bash
cmake -S . -B build -DOpenCV_DIR=$(brew --prefix opencv)/lib/cmake/opencv4
```

### Build troubleshooting

- **OpenCV 5.x** (current Homebrew default): `cv::getRotationMatrix2D` moved to `opencv2/geometry.hpp`. Either add `#include <opencv2/geometry.hpp>` to `src/MosseMatcher.cpp`, or configure with `-DCMAKE_CXX_FLAGS="-include opencv2/geometry.hpp"`. Point CMake at the right config dir: `-DOpenCV_DIR=$(brew --prefix opencv)/lib/cmake/opencv5`.
- **macOS: "tapi error: malformed file" / compiler can't link a test program**: the Command Line Tools linker can't read the newest SDK. Build against an older installed SDK, e.g. `export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk` (see `ls /Library/Developer/CommandLineTools/SDKs`).

---

## Run

```bash
# no arguments: opens a file picker to choose a video, then select the target on frame 1
./build/tracker

# MOSSE (default) — manual ROI selection on frame 1
./build/tracker data/trial1.MP4

# preset the ROI to skip the selection dialog (useful for fair A/B)
./build/tracker data/trial2.MP4 --matcher mosse --roi 10,1081,120,287 --no-rotation

# compare all three backends on the same clip with the same box
./build/tracker data/trial1.MP4 --matcher mosse --roi 812,140,46,58 --output outputs/videos/mosse.mp4
./build/tracker data/trial1.MP4 --matcher csrt  --roi 812,140,46,58 --output outputs/videos/csrt.mp4
./build/tracker data/trial1.MP4 --matcher kcf   --roi 812,140,46,58 --output outputs/videos/kcf.mp4

# multi-target mode for look-alike objects
./build/tracker data/trial1.MP4 --matcher mosse --multi --roi 812,140,46,58 --output outputs/videos/multi.mp4

# NCC baseline (v1 matcher, slowest to drift but weakest on fast targets)
./build/tracker data/trial1.MP4 --matcher ncc

# webcam
./build/tracker --camera 0

# headless (no GUI window, still writes video)
./build/tracker data/trial2.MP4 --no-display --output outputs/videos/redetectorbike.mp4
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

1. Create `include/YourMatcher.hpp` and `src/YourMatcher.cpp`.
2. Inherit from `IMatcher` and implement the five virtual methods (`setTemplate`, `updateTemplate`, `useOriginalTemplate`, `match`, `hasTemplate`). The `setMotionHint` override is optional.
3. Add the `.cpp` to `CMakeLists.txt` under `add_executable(tracker ...)`.
4. Wire it into `main.cpp` with an `else if (matcherName == "yours")` branch.

Nothing else changes. The engine, state machine, confidence evaluator, and visualizer are all matcher-agnostic.

---

## What to look at next (v3 targets)

- **Ground-truth metrics** — load a ground-truth `.txt` (MOT format) alongside the video and compute per-frame IoU and center error. The `TrackerMetrics` class already tracks frame count, lost-frame count, and runtime; extend it here.
- **Benchmark table** — run all four matchers on all three clips in one script, collect the metrics, and print a comparison table. This will make the tracker comparisons quantitative instead of only visual.
- **`tracker.yaml` parsing** — `TrackerConfig` in [include/Types.hpp](include/Types.hpp) already defines every tunable field. Wire it to the YAML file at [configs/tracker.yaml](configs/tracker.yaml) using `cv::FileStorage` so runtime parameters can be loaded from disk.
- **Multi-template memory** — store the N best patches seen during TRACKING and fall back to them during reacquisition instead of relying only on the pristine frame-1 crop.

---

## File map

```
include/
  IMatcher.hpp          — matcher interface (the only contract the engine sees)
  MosseMatcher.hpp      — MOSSE adaptive filter
  CvTrackerMatcher.hpp  — CSRT / KCF wrapper
  TemplateMatcher.hpp   — NCC baseline
  KalmanBoxTracker.hpp  — Kalman predict / correct
  SearchRegion.hpp      — ROI expansion logic
  ConfidenceEvaluator.hpp
  ReacquisitionManager.hpp
  TrackerEngine.hpp
  TrackerMetrics.hpp
  Visualizer.hpp
  VideoSource.hpp
  Types.hpp             — TrackStatus, MatchResult, Evaluation, TrackState, TrackerConfig
  MultiTargetTracker.hpp — multi-target motion-track manager

src/                    — implementations (mirror of include/)
configs/tracker.yaml    — config file (fields defined, parsing not yet wired)
data/                   — test clips (not committed; git-ignored)
outputs/videos/         — written here by default
CMakeLists.txt
```

---
## Acknowledgements

This project was developed in collaboration with **Megh Lad**.

The original collaborative repository is available at:
https://github.com/Meghlad/lightweight_tracker

This repository is maintained under my GitHub account to document my own contributions while preserving the complete collaborative Git history.

## License

See [LICENSE](LICENSE).

cmake --build /Users/meghlad/Downloads/lightweight_tracker/build -j 2>&1 | grep -E "error:|warning:|Built target|Error"
