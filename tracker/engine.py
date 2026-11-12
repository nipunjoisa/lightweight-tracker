"""Video I/O and the single-target tracking loop."""
from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .components import (ConfidenceEvaluator, ReacquisitionManager, TrackerMetrics,
                         Visualizer, compute_search_region, intersect, to_int_rect)
from .kalman import KalmanBoxTracker
from .matchers import IMatcher
from .types import TrackerConfig, TrackState, TrackStatus


class VideoSource:
    def __init__(self) -> None:
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps = 30.0
        self.size: Tuple[int, int] = (0, 0)  # (w, h)

    def open_file(self, path: str) -> bool:
        return self._open(path)

    def open_camera(self, index: int) -> bool:
        return self._open(index)

    def _open(self, src) -> bool:
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            return False
        f = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = f if 1.0 < f < 1000.0 else 30.0  # cameras frequently report 0
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return True

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        return frame if ok and frame is not None and frame.size else None


def open_writer(path: str, fps: float, size: Tuple[int, int]) -> Optional[cv2.VideoWriter]:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not w.isOpened():
        print(f"[engine] warning: could not open writer at {path}")
        return None
    return w


def select_initial_roi(frame: np.ndarray, cfg: TrackerConfig):
    """Preset ROI or interactive selection. Returns (x, y, w, h) or None."""
    bounds = (0, 0, frame.shape[1], frame.shape[0])
    if cfg.init_roi is not None:
        roi = intersect(cfg.init_roi, bounds)
    else:
        win = "Select target  |  drag a box, ENTER/SPACE to confirm, c to cancel"
        roi = tuple(int(v) for v in cv2.selectROI(win, frame, False, False))
        cv2.destroyWindow(win)
    if roi[2] < 4 or roi[3] < 4:
        print("[engine] invalid/empty ROI")
        return None
    x, y, w, h = roi
    print(f"init roi: {x},{y},{w},{h}  (reuse with --roi {x},{y},{w},{h})")
    return roi


class TrackerEngine:
    def __init__(self, cfg: TrackerConfig, matcher: IMatcher) -> None:
        self.cfg = cfg
        self.matcher = matcher
        self.kalman = KalmanBoxTracker()
        self.confidence = ConfidenceEvaluator(cfg)
        self.reacq = ReacquisitionManager(cfg)
        self.metrics = TrackerMetrics()
        self.viz = Visualizer()

    def run(self, source: VideoSource) -> bool:
        cfg = self.cfg
        frame = source.read()
        if frame is None:
            print("[engine] could not read first frame")
            return False
        roi = select_initial_roi(frame, cfg)
        if roi is None:
            return False

        # One-time setup: template + Kalman seed.
        self.matcher.set_template(frame, roi)
        roi_f = tuple(float(v) for v in roi)
        self.kalman.init(roi_f)
        self.reacq.reset()
        writer = open_writer(cfg.output_path, source.fps, source.size)

        believed = roi_f        # current best estimate of the target box
        last_confident = roi_f  # last box we actually measured (re-detect anchor)
        expansion = cfg.search_expansion
        last_was_lost = False

        while True:
            frame = source.read()
            if frame is None:
                break
            t0 = time.perf_counter()

            # 1) Predict where the target should be.
            predicted = self.kalman.predict()
            self.matcher.set_motion_hint(self.kalman.velocity())

            # 2) Build the search ROI. While locked, center on the prediction; once
            #    lost, RE-DETECT around the last position actually measured.
            searching = self.reacq.lost_frames > 0
            self.matcher.set_reacquiring(searching)
            search_center = last_confident if searching else predicted
            roi_rect = compute_search_region(search_center, expansion,
                                             (frame.shape[1], frame.shape[0]))

            # 3) Multi-scale match inside the ROI.
            expected = (max(4, int(round(predicted[2]))), max(4, int(round(predicted[3]))))
            m = self.matcher.match(frame, roi_rect, expected, cfg.scales)

            # 4) Fuse confidence + gating (reachable radius grows with time lost).
            speed_ref = last_confident if searching else predicted
            size_ref = last_confident if searching else believed
            slack = min(8.0, 1.0 + self.reacq.lost_frames) if searching else 1.0
            ev = self.confidence.evaluate(m, speed_ref, size_ref, slack)

            # 5) Accept/reject. While re-detecting demand a STRONG match to re-lock.
            accept = (ev.confidence >= cfg.reacquire_confidence) if searching \
                else (ev.status != TrackStatus.LOST)

            if accept:
                believed = self.kalman.correct(m.box)
                last_confident = m.box
                reported = TrackStatus.TRACKING if ev.confidence >= cfg.strong_threshold \
                    else TrackStatus.UNCERTAIN
                if reported == TrackStatus.TRACKING and cfg.adapt:
                    self.matcher.update_template(frame, to_int_rect(m.box), cfg.template_learn_rate)
            else:
                believed = predicted  # coast on prediction only
                if not last_was_lost:
                    self.matcher.use_original_template()  # revert to pristine on first loss
                reported = TrackStatus.REACQUIRING if last_was_lost else TrackStatus.LOST

            # 6) Reacquisition policy decides NEXT frame's search expansion.
            expansion = self.reacq.on_status(reported)
            last_was_lost = not accept

            frame_ms = (time.perf_counter() - t0) * 1000.0
            fps = 1000.0 / frame_ms if frame_ms > 0 else 0.0

            self.metrics.record(reported, ev.confidence, frame_ms)
            vis = self.viz.draw(frame, TrackState(
                box=believed, predicted=predicted, search_roi=roi_rect,
                confidence=ev.confidence, status=reported,
                lost_frames=self.reacq.lost_frames, fps=fps))
            if writer is not None:
                writer.write(vis)
            if cfg.display:
                cv2.imshow("lightweight-tracker", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

        self.metrics.print_summary(source.fps)
        if cfg.display:
            cv2.destroyAllWindows()
        if writer is not None:
            writer.release()
            print(f"saved output video to: {cfg.output_path}")
        return True
