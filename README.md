# cpu-object-tracker

CPU-only C++17 aerial object tracker: Kalman prediction + local ROI template
matching + confidence-aware updates + reacquisition. No GPU, no deep learning.

The tracker detects/initializes **once** (you draw the box), then tracks cheaply
inside a small search window using prediction and correlation matching.

## Pipeline

```
predict (Kalman) -> build search ROI -> multi-scale NCC match -> confidence (PSR)
   -> gate (speed/size) -> branch:
        strong  : correct Kalman + slow template update      [TRACKING]
        weak    : correct Kalman, freeze template             [UNCERTAIN]
        low     : prediction only, widen ROI, pristine tmpl   [LOST / REACQUIRING]
```

State vector `[cx, cy, vx, vy, w, h]`, measurement `[cx, cy, w, h]`.
Confidence = `nccWeight * clamp(NCC) + psrWeight * clamp(PSR / psrSaturation)`.
Thresholds: strong `>= 0.70`, weak `0.45..0.70`, lost `< 0.45`.

## Build (macOS)

```bash
# 1. Toolchain + OpenCV
brew install cmake opencv

# 2. Configure + build
cd cpu-object-tracker
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# binary: build/tracker
```

If `find_package(OpenCV)` fails, point CMake at the install:

```bash
cmake -S . -B build -DOpenCV_DIR=$(brew --prefix opencv)/lib/cmake/opencv4
```

## Run

```bash
# track in a video file
./build/tracker data/sample_videos/clip.mp4

# webcam
./build/tracker --camera 0

# custom output, headless
./build/tracker data/sample_videos/clip.mp4 --output outputs/videos/run1.mp4 --no-display
```

A window opens on the first frame: drag a box around the target, press ENTER or
SPACE to confirm. Then tracking runs. Press `q` or ESC to stop early. Metrics
print to the terminal at the end; the annotated video is written to `--output`.

## VS Code

Install the **C/C++** and **CMake Tools** extensions. Open the folder, pick a
kit when prompted, then use the CMake Tools status-bar buttons (Configure /
Build / Run), or just build from the integrated terminal with the commands
above. Pass the video path as a launch argument in `.vscode/launch.json` if you
want to debug.

## Roadmap

- v1 (this): full module architecture, NCC + PSR matching, multi-scale search,
  4-state machine, dual-template reacquisition, metrics, video out.
- v2: MOSSE/KCF backend behind `IMatcher`, multi-template memory, `tracker.yaml`
  parsing (config struct is already in place).
- v3: ground-truth IoU / center-error metrics, benchmark + compare-trackers.
