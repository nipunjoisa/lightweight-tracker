"""Engine building blocks: confidence, search region, reacquisition, metrics, HUD."""
from __future__ import annotations

import math
from typing import List

import cv2
import numpy as np

from .types import (Box, Evaluation, MatchResult, TrackerConfig, TrackState,
                    TrackStatus)


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def center(b: Box):
    return (b[0] + b[2] * 0.5, b[1] + b[3] * 0.5)


def intersect(r, bounds):
    """Intersection of two (x, y, w, h) int rects; width/height may be 0."""
    x0, y0 = max(r[0], bounds[0]), max(r[1], bounds[1])
    x1 = min(r[0] + r[2], bounds[0] + bounds[2])
    y1 = min(r[1] + r[3], bounds[1] + bounds[3])
    return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def to_int_rect(b: Box):
    return (int(round(b[0])), int(round(b[1])), int(round(b[2])), int(round(b[3])))


# --------------------------------------------------------------------------
class ConfidenceEvaluator:
    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg

    def classify(self, c: float) -> TrackStatus:
        if c >= self.cfg.strong_threshold:
            return TrackStatus.TRACKING
        if c >= self.cfg.weak_threshold:
            return TrackStatus.UNCERTAIN
        return TrackStatus.LOST

    def evaluate(self, m: MatchResult, predicted: Box, previous: Box,
                 speed_slack: float = 1.0) -> Evaluation:
        if not m.valid:
            return Evaluation(0.0, TrackStatus.LOST, False)
        c = self.cfg
        conf = c.ncc_weight * clamp01(m.ncc) + c.psr_weight * clamp01(m.psr / c.psr_saturation)
        gates = True

        # Speed gate (slack widens the reachable radius while re-detecting).
        pc, mc = center(predicted), center(m.box)
        disp = math.hypot(mc[0] - pc[0], mc[1] - pc[1])
        if disp > c.max_speed * speed_slack:
            conf *= 0.4
            gates = False

        # Size gate.
        prev_area = max(1.0, previous[2] * previous[3])
        cur_area = max(1.0, m.box[2] * m.box[3])
        ratio = max(cur_area / prev_area, prev_area / cur_area)
        if ratio > c.max_size_ratio:
            conf *= 0.6
            gates = False

        conf = clamp01(conf)
        return Evaluation(conf, self.classify(conf), gates)


# --------------------------------------------------------------------------
def compute_search_region(box: Box, expansion: float, frame_wh) -> tuple:
    cx, cy = center(box)
    w = max(8.0, box[2] * expansion)
    h = max(8.0, box[3] * expansion)
    roi = (int(round(cx - w * 0.5)), int(round(cy - h * 0.5)), int(round(w)), int(round(h)))
    return intersect(roi, (0, 0, frame_wh[0], frame_wh[1]))


# --------------------------------------------------------------------------
class ReacquisitionManager:
    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.lost_frames = 0
        self.expansion = self.cfg.search_expansion

    def on_status(self, status: TrackStatus) -> float:
        """Return NEXT frame's search expansion."""
        if status in (TrackStatus.LOST, TrackStatus.REACQUIRING):
            self.lost_frames += 1
            self.expansion = min(self.cfg.reacq_max_expansion,
                                 self.cfg.search_expansion
                                 + self.cfg.reacq_expansion_step * self.lost_frames)
        else:
            self.lost_frames = 0
            self.expansion = self.cfg.search_expansion
        return self.expansion


# --------------------------------------------------------------------------
class TrackerMetrics:
    def __init__(self) -> None:
        self.total = 0
        self.conf_sum = 0.0
        self.frame_ms: List[float] = []
        self.counts = {s: 0 for s in TrackStatus}
        self.cur_streak = 0
        self.longest_streak = 0
        self.reacquisitions = 0
        self.was_lost = False

    def record(self, status: TrackStatus, confidence: float, frame_ms: float) -> None:
        self.total += 1
        self.conf_sum += confidence
        self.frame_ms.append(frame_ms)
        self.counts[status] += 1
        locked = status in (TrackStatus.TRACKING, TrackStatus.UNCERTAIN)
        if locked:
            if self.was_lost:
                self.reacquisitions += 1
            self.cur_streak += 1
            self.longest_streak = max(self.longest_streak, self.cur_streak)
            self.was_lost = False
        else:
            self.cur_streak = 0
            self.was_lost = True

    def print_summary(self, source_fps: float) -> None:
        if self.total == 0:
            print("\n[metrics] no frames processed")
            return
        ms = np.array(self.frame_ms)
        mean_ms, med_ms = float(ms.mean()), float(np.median(ms))
        mean_fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
        med_fps = 1000.0 / med_ms if med_ms > 0 else 0.0

        def pct(n):
            return 100.0 * n / self.total

        print("\n================ tracker metrics ================")
        print(f"frames processed     : {self.total}")
        print(f"source fps           : {source_fps:.1f}")
        print(f"processing fps (mean): {mean_fps:.1f}")
        print(f"processing fps (med) : {med_fps:.1f}")
        print(f"mean confidence      : {self.conf_sum / self.total:.3f}")
        for s in TrackStatus:
            print(f"{s.value:<21}: {self.counts[s]} ({pct(self.counts[s]):.1f}%)")
        print(f"longest locked streak: {self.longest_streak} frames")
        print(f"reacquisitions       : {self.reacquisitions}")
        print("=================================================")


# --------------------------------------------------------------------------
_STATUS_COLOR = {  # BGR
    TrackStatus.TRACKING: (80, 220, 80),
    TrackStatus.UNCERTAIN: (0, 200, 255),
    TrackStatus.LOST: (60, 60, 230),
    TrackStatus.REACQUIRING: (0, 140, 255),
}


def status_color(s: TrackStatus):
    return _STATUS_COLOR.get(s, (200, 200, 200))


class Visualizer:
    def draw(self, frame: np.ndarray, st: TrackState) -> np.ndarray:
        out = frame.copy()
        col = status_color(st.status)

        def rect(r, color):
            x, y, w, h = to_int_rect(r)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)

        rect(st.search_roi, (120, 120, 120))   # search ROI
        rect(st.predicted, (200, 160, 60))     # raw prediction
        rect(st.box, col)                      # believed target

        cx, cy = center(st.box)
        cv2.circle(out, (int(round(cx)), int(round(cy))), 6, col, -1, cv2.LINE_AA)
        cv2.circle(out, (int(round(cx)), int(round(cy))), 8, (0, 0, 0), 1, cv2.LINE_AA)

        lines = [(f"STATUS: {st.status.value}", col),
                 (f"CONF  : {st.confidence:.2f}", (255, 255, 255)),
                 (f"FPS   : {st.fps:.1f}", (255, 255, 255)),
                 (f"LOST  : {st.lost_frames}", (255, 255, 255))]
        y = 24
        for text, c in lines:
            cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 1, cv2.LINE_AA)
            y += 26
        return out
