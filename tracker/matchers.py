"""Matcher interface plus the NCC (v1) and CSRT/KCF (cv2.Tracker) backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .components import clamp01, intersect
from .types import MatchResult


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def compute_psr(response: np.ndarray, peak_xy: Tuple[int, int]) -> float:
    """Peak-to-sidelobe ratio: (peak - mean_sidelobe) / (std_sidelobe + eps)."""
    px, py = peak_xy
    peak_val = float(response[py, px])
    mask = np.ones(response.shape, dtype=bool)
    win = 5  # exclude an 11x11 window around the peak
    mask[max(0, py - win):py + win + 1, max(0, px - win):px + win + 1] = False
    side = response[mask]
    if side.size < 8:
        return 0.0
    return (peak_val - float(side.mean())) / (float(side.std()) + 1e-6)


class IMatcher(ABC):
    """Backend-agnostic matcher contract. The engine only sees this."""

    @abstractmethod
    def set_template(self, frame: np.ndarray, box: Tuple[int, int, int, int]) -> None:
        """Capture the reference template (working + pristine original)."""

    @abstractmethod
    def update_template(self, frame: np.ndarray, box: Tuple[int, int, int, int],
                        alpha: float) -> None:
        """Slowly blend current appearance into the working template."""

    @abstractmethod
    def use_original_template(self) -> None:
        """Restore the working template to the pristine original (on loss)."""

    def set_motion_hint(self, velocity: Tuple[float, float]) -> None:
        """Optional: predicted velocity (px/frame)."""

    def set_reacquiring(self, reacquiring: bool) -> None:
        """Optional: supervisor is in re-detection mode."""

    @abstractmethod
    def match(self, frame: np.ndarray, search_region: Tuple[int, int, int, int],
              expected_size: Tuple[int, int], scales: List[float]) -> MatchResult:
        """Search inside search_region; returned box is in full-frame coords."""

    @abstractmethod
    def has_template(self) -> bool: ...


# --------------------------------------------------------------------------
class TemplateMatcher(IMatcher):
    """Normalized cross-correlation baseline (v1)."""

    def __init__(self) -> None:
        self.original: Optional[np.ndarray] = None
        self.working: Optional[np.ndarray] = None

    def has_template(self) -> bool:
        return self.working is not None

    def set_template(self, frame, box):
        gray = to_gray(frame)
        x, y, w, h = intersect(box, (0, 0, gray.shape[1], gray.shape[0]))
        if w < 4 or h < 4:
            return
        self.original = gray[y:y + h, x:x + w].copy()
        self.working = self.original.copy()

    def use_original_template(self):
        if self.original is not None:
            self.working = self.original.copy()

    def update_template(self, frame, box, alpha):
        if self.working is None:
            return
        gray = to_gray(frame)
        x, y, w, h = intersect(box, (0, 0, gray.shape[1], gray.shape[0]))
        if w < 4 or h < 4:
            return
        patch = cv2.resize(gray[y:y + h, x:x + w], (self.working.shape[1], self.working.shape[0]))
        self.working = cv2.addWeighted(self.working, 1.0 - alpha, patch, alpha, 0.0)

    def match(self, frame, search_region, expected_size, scales):
        out = MatchResult()
        if self.working is None:
            return out
        gray = to_gray(frame)
        rx, ry, rw, rh = intersect(search_region, (0, 0, gray.shape[1], gray.shape[0]))
        if rw < 8 or rh < 8:
            return out
        roi_img = gray[ry:ry + rh, rx:rx + rw]

        best_val, best_loc, best_size, best_resp = -1e9, None, None, None
        for s in scales:
            tw = max(4, int(round(expected_size[0] * s)))
            th = max(4, int(round(expected_size[1] * s)))
            if tw >= roi_img.shape[1] or th >= roi_img.shape[0]:
                continue  # template must fit the ROI
            tmpl = cv2.resize(self.working, (tw, th))
            resp = cv2.matchTemplate(roi_img, tmpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, mx_loc = cv2.minMaxLoc(resp)
            if mx > best_val:
                best_val, best_loc, best_size, best_resp = mx, mx_loc, (tw, th), resp
        if best_size is None:
            return out

        out.box = (float(rx + best_loc[0]), float(ry + best_loc[1]),
                   float(best_size[0]), float(best_size[1]))
        out.ncc = float(best_val)
        out.psr = compute_psr(best_resp, best_loc)
        out.valid = True
        return out


# --------------------------------------------------------------------------
class CvTrackerMatcher(IMatcher):
    """CSRT or KCF via cv2.Tracker, adapted to IMatcher."""

    def __init__(self, kind: str = "csrt") -> None:
        self.kind = kind.lower()
        self.tracker = None
        self.original: Optional[np.ndarray] = None
        self.need_reinit = False

    def _make(self):
        if self.kind == "kcf":
            return cv2.TrackerKCF_create()
        return cv2.TrackerCSRT_create()

    def has_template(self) -> bool:
        return self.tracker is not None

    def set_template(self, frame, box):
        x, y, w, h = intersect(box, (0, 0, frame.shape[1], frame.shape[0]))
        if w < 4 or h < 4:
            return
        self.original = to_gray(frame)[y:y + h, x:x + w].copy()
        self.tracker = self._make()
        self.tracker.init(frame, (x, y, w, h))
        self.need_reinit = False

    def update_template(self, frame, box, alpha):
        pass  # CSRT/KCF update themselves on every match()

    def use_original_template(self):
        self.need_reinit = True  # re-init at the Kalman-predicted spot next match

    def _score_at(self, gray, box) -> float:
        if self.original is None:
            return 0.0
        x, y, w, h = intersect(box, (0, 0, gray.shape[1], gray.shape[0]))
        if w < 4 or h < 4:
            return 0.0
        patch = cv2.resize(gray[y:y + h, x:x + w], (self.original.shape[1], self.original.shape[0]))
        r = cv2.matchTemplate(patch, self.original, cv2.TM_CCOEFF_NORMED)  # equal sizes -> 1x1
        return float(r[0, 0])

    def match(self, frame, search_region, expected_size, scales):
        out = MatchResult()
        if self.tracker is None:
            return out

        if self.need_reinit:
            cx = search_region[0] + search_region[2] * 0.5
            cy = search_region[1] + search_region[3] * 0.5
            rb = (int(round(cx - expected_size[0] * 0.5)), int(round(cy - expected_size[1] * 0.5)),
                  max(4, expected_size[0]), max(4, expected_size[1]))
            rb = intersect(rb, (0, 0, frame.shape[1], frame.shape[0]))
            if rb[2] >= 4 and rb[3] >= 4:
                self.tracker = self._make()
                self.tracker.init(frame, rb)
            self.need_reinit = False

        ok, box = self.tracker.update(frame)
        if not ok or box[2] < 2 or box[3] < 2:
            return out  # invalid -> LOST

        q = clamp01(self._score_at(to_gray(frame), tuple(int(v) for v in box)))
        out.box = tuple(float(v) for v in box)
        out.ncc = q
        out.psr = q * 12.0  # makes confidence ~= q under default 0.5/0.5 fusion
        out.valid = True
        return out
