# lightweight-tracker

CPU-only C++17 aerial object tracker. No GPU, no deep learning, no external model weights.

You draw a box on frame 1. The tracker holds lock using Kalman prediction, a pluggable correlation matcher, and a confidence-gated state machine that governs when to update the appearance model and when to widen the search.

---

## Where things stand

**v1** — NCC template matcher, 4-state machine, reacquisition, metrics, video out. Works but drifts on fast maneuvers.

**v2 (current)** — MOSSE adaptive correlation filter as the default backend. CSRT and KCF wired in via OpenCV behind the same `IMatcher` interface. Benchmarked against all three on real aerial footage; MOSSE wins the fast-small-target regime. The one unsolved case is a 90° turn plus smoke occlusion — that defeated every backend and every tuning combination tried.

**v3 (next)** — ground-truth IoU / center-error metrics, per-tracker benchmark table, `tracker.yaml` config parsing (the struct is already in place, just not wired to the file).

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

---

## Run

```bash
# MOSSE (default) — manual ROI selection on frame 1
./build/tracker data/trial1.MP4

# preset the ROI to skip the selection dialog (useful for fair A/B)
./build/tracker data/trial1.MP4 --matcher mosse --roi 812,140,46,58

# compare all three backends on the same clip with the same box
./build/tracker data/trial1.MP4 --matcher mosse --roi 812,140,46,58 --output outputs/videos/mosse.mp4
./build/tracker data/trial1.MP4 --matcher csrt  --roi 812,140,46,58 --output outputs/videos/csrt.mp4
./build/tracker data/trial1.MP4 --matcher kcf   --roi 812,140,46,58 --output outputs/videos/kcf.mp4

# NCC baseline (v1 matcher, slowest to drift but weakest on fast targets)
./build/tracker data/trial1.MP4 --matcher ncc

# webcam
./build/tracker --camera 0

# headless (no GUI window, still writes video)
./build/tracker data/trial1.MP4 --no-display --output outputs/videos/run.mp4
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
- **Benchmark table** — run all four matchers on all three clips in one script, collect the metrics, print a comparison table. Makes the portfolio result quantitative rather than just visual.
- **`tracker.yaml` parsing** — `TrackerConfig` in [include/Types.hpp](include/Types.hpp) has every tunable field. Wire it to the YAML file at [configs/tracker.yaml](configs/tracker.yaml) using `cv::FileStorage` (already a dependency via OpenCV).
- **Multi-template memory** — store the N best patches seen during TRACKING and fall back to them during reacquisition instead of always the pristine frame-1 crop.

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

src/                    — implementations (mirror of include/)
configs/tracker.yaml    — config file (fields defined, parsing not yet wired)
data/                   — test clips (trial1–3.MP4)
outputs/videos/         — written here by default
CMakeLists.txt
```

---

## License

See [LICENSE](LICENSE).
