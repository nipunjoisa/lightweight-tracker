"""Constant-velocity Kalman filter over a bounding box.

State: [cx, cy, vx, vy, w, h]. Measurement: [cx, cy, w, h].
"""
from __future__ import annotations

import cv2
import numpy as np

from .types import Box


def _to_box(cx: float, cy: float, w: float, h: float) -> Box:
    return (cx - w * 0.5, cy - h * 0.5, w, h)


class KalmanBoxTracker:
    def __init__(self) -> None:
        kf = cv2.KalmanFilter(6, 4, 0, cv2.CV_32F)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]], dtype=np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]], dtype=np.float32)
        kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        kf.errorCovPost = np.eye(6, dtype=np.float32)
        self._kf = kf

    def init(self, box: Box) -> None:
        x, y, w, h = box
        state = np.array([[x + w * 0.5], [y + h * 0.5], [0], [0], [w], [h]], dtype=np.float32)
        self._kf.statePost = state.copy()
        self._kf.statePre = state.copy()

    def predict(self) -> Box:
        p = self._kf.predict().ravel()
        return _to_box(p[0], p[1], p[4], p[5])

    def correct(self, meas: Box) -> Box:
        x, y, w, h = meas
        m = np.array([[x + w * 0.5], [y + h * 0.5], [w], [h]], dtype=np.float32)
        c = self._kf.correct(m).ravel()
        return _to_box(c[0], c[1], c[4], c[5])

    def state_box(self) -> Box:
        s = self._kf.statePost.ravel()
        return _to_box(s[0], s[1], s[4], s[5])

    def velocity(self) -> tuple:
        s = self._kf.statePost.ravel()
        return (float(s[2]), float(s[3]))
