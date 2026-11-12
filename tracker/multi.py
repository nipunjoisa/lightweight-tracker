"""Multi-target tracking: one shared MOSSE heat map, one Kalman track per object.

Identity is carried by motion (the Kalman state), NOT appearance -- which is the
whole point for near-identical objects.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .components import (TrackerMetrics, center, clamp01, compute_search_region,
                         intersect, status_color, to_int_rect)
from .engine import VideoSource, open_writer, select_initial_roi
from .kalman import KalmanBoxTracker
from .mosse import MosseMatcher
from .types import Box, TrackerConfig, TrackStatus


@dataclass
class MtTrack:
    id: int
    kf: KalmanBoxTracker
    box: Box
    conf: float = 0.0
    missed: int = 0   # consecutive frames without a match
    hits: int = 0
    primary: bool = False  # the user-initialized target
    status: TrackStatus = TrackStatus.TRACKING


class MultiTargetTracker:
    def __init__(self, cfg: TrackerConfig, appearance: MosseMatcher) -> None:
        self.cfg = cfg
        self.app = appearance
        self.tracks: List[MtTrack] = []
        self.target_size = (0, 0)
        self.next_id = 0
        self.max_tracks = 4
        self.max_missed = 30  # cull a distractor after this

    def init(self, frame: np.ndarray, roi) -> None:
        self.app.set_template(frame, roi)
        self.target_size = (max(8, roi[2]), max(8, roi[3]))
        box = tuple(float(v) for v in roi)
        kf = KalmanBoxTracker()
        kf.init(box)
        self.tracks.append(MtTrack(self.next_id, kf, box, primary=True, hits=1))
        self.next_id += 1

    def primary(self) -> Optional[MtTrack]:
        for t in self.tracks:
            if t.primary:
                return t
        return self.tracks[0] if self.tracks else None

    def _region(self, frame_wh):
        """Union of all predicted boxes, padded, so the heat map covers every track."""
        x0 = min(t.box[0] for t in self.tracks)
        y0 = min(t.box[1] for t in self.tracks)
        x1 = max(t.box[0] + t.box[2] for t in self.tracks)
        y1 = max(t.box[1] + t.box[3] for t in self.tracks)
        pad_x, pad_y = self.target_size[0] * 2.0, self.target_size[1] * 2.0
        r = (int(round(x0 - pad_x)), int(round(y0 - pad_y)),
             int(round((x1 - x0) + 2 * pad_x)), int(round((y1 - y0) + 2 * pad_y)))
        return intersect(r, (0, 0, frame_wh[0], frame_wh[1]))

    def _spawn(self, box: Box) -> None:
        if len(self.tracks) >= self.max_tracks:
            return
        kf = KalmanBoxTracker()
        kf.init(box)
        self.tracks.append(MtTrack(self.next_id, kf, box, status=TrackStatus.REACQUIRING))
        self.next_id += 1

    def step(self, frame: np.ndarray) -> None:
        if not self.tracks:
            return
        cfg = self.cfg
        tw, th = self.target_size

        # 1) Predict every track.
        for t in self.tracks:
            t.box = t.kf.predict()

        # 2) Read the shared heat map -> candidate detections (peaks).
        region = self._region((frame.shape[1], frame.shape[0]))
        dets = self.app.detect(frame, region, len(self.tracks) + 2,
                               cfg.weak_threshold * 12.0)

        # 3) Greedy association by motion: each track (primary first) claims its nearest
        #    unclaimed detection within a speed-scaled gate.
        claim = [-1] * len(dets)
        order = sorted(range(len(self.tracks)), key=lambda i: not self.tracks[i].primary)
        track_det = [-1] * len(self.tracks)
        for ti in order:
            pc = center(self.tracks[ti].box)
            gate = cfg.max_speed * (1.0 + min(7, self.tracks[ti].missed))
            best, best_det = float("inf"), -1
            for di, d in enumerate(dets):
                if claim[di] >= 0:
                    continue
                dc = center(d.box)
                dist = math.hypot(dc[0] - pc[0], dc[1] - pc[1])
                if dist < best and dist <= gate:
                    best, best_det = dist, di
            if best_det >= 0:
                claim[best_det] = ti
                track_det[ti] = best_det

        # 4) Update matched, coast unmatched.
        for ti, t in enumerate(self.tracks):
            if track_det[ti] >= 0:
                d = dets[track_det[ti]]
                cx, cy = center(d.box)
                meas = (cx - tw * 0.5, cy - th * 0.5, float(tw), float(th))
                t.box = t.kf.correct(meas)
                t.conf = clamp01(d.psr / 12.0)
                t.missed = 0
                t.hits += 1
                t.status = TrackStatus.TRACKING if t.conf >= cfg.strong_threshold \
                    else TrackStatus.UNCERTAIN
                # Adapt the shared appearance only from the PRIMARY's confident match,
                # so a distractor can never drag the shared filter off the real target.
                if t.primary and t.status == TrackStatus.TRACKING and cfg.adapt:
                    self.app.update_template(frame, to_int_rect(t.box), cfg.template_learn_rate)
            else:
                t.box = t.kf.state_box()  # coast on prediction
                t.conf = 0.0
                t.missed += 1
                t.status = TrackStatus.REACQUIRING if t.missed > 1 else TrackStatus.LOST

        # 5) Spawn a track for any strong, unclaimed, well-separated detection.
        for di, d in enumerate(dets):
            if claim[di] >= 0 or d.psr < cfg.strong_threshold * 12.0:
                continue
            c = center(d.box)
            if any(math.hypot(center(t.box)[0] - c[0], center(t.box)[1] - c[1]) < min(tw, th)
                   for t in self.tracks):
                continue
            self._spawn((c[0] - tw * 0.5, c[1] - th * 0.5, float(tw), float(th)))

        # 6) Cull stale distractors (never the primary -- it coasts indefinitely).
        self.tracks = [t for t in self.tracks if t.primary or t.missed <= self.max_missed]


def run_multi_target(source: VideoSource, cfg: TrackerConfig) -> int:
    frame = source.read()
    if frame is None:
        print("[multi] cannot read first frame")
        return 3
    roi = select_initial_roi(frame, cfg)
    if roi is None:
        return 3

    mtt = MultiTargetTracker(cfg, MosseMatcher(2.0, 8, anchor=0.0, prior_frac=0.0,
                                               rot_enable=False))
    mtt.init(frame, roi)
    writer = open_writer(cfg.output_path, source.fps, source.size)
    metrics = TrackerMetrics()

    while True:
        frame = source.read()
        if frame is None:
            break
        t0 = time.perf_counter()
        mtt.step(frame)
        frame_ms = (time.perf_counter() - t0) * 1000.0

        vis = frame.copy()
        prim = mtt.primary()
        for t in mtt.tracks:
            col = status_color(t.status)
            x, y, w, h = to_int_rect(t.box)
            cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2 if t.primary else 1, cv2.LINE_AA)
            cx, cy = center(t.box)
            cv2.circle(vis, (int(cx), int(cy)), 5 if t.primary else 3, col, -1, cv2.LINE_AA)
            cv2.putText(vis, f"{'P' if t.primary else 'D'}{t.id}", (x, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        fps = 1000.0 / frame_ms if frame_ms > 0 else 0.0
        hud = (f"MULTI  tracks:{len(mtt.tracks)}  "
               f"{prim.status.value if prim else '-'}  "
               f"conf:{prim.conf if prim else 0.0:.2f}  fps:{fps:.0f}")
        cv2.putText(vis, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if prim:
            metrics.record(prim.status, prim.conf, frame_ms)
        if writer is not None:
            writer.write(vis)
        if cfg.display:
            cv2.imshow("lightweight-tracker [multi]", vis)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break

    metrics.print_summary(source.fps)
    if cfg.display:
        cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
        print(f"saved output video to: {cfg.output_path}")
    return 0
