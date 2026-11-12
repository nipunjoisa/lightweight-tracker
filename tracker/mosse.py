"""MOSSE adaptive correlation filter (Bolme et al., 2010) behind IMatcher.

A fixed-size complex filter is learned in the Fourier domain and updated online,
so it adapts to appearance change and tolerates orientation change far better
than rigid NCC. PSR is MOSSE's native quality signal and drives the confidence.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .components import clamp01, intersect
from .matchers import IMatcher, to_gray
from .types import MatchResult

_POOL = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)  # cv2 releases the GIL


def _pmap(fn, items):
    """Run fn over items across CPU cores (order preserved)."""
    return list(_POOL.map(fn, items))


def _sub_pix(gray, size, center):
    return cv2.getRectSubPix(gray, (int(size[0]), int(size[1])),
                             (float(center[0]), float(center[1])), patchType=cv2.CV_32F)


# --------------------------------------------------------------------------
class RotationEstimator:
    """Log-polar + phase correlation: rotation -> vertical shift, scale -> horizontal."""

    def estimate(self, ref: np.ndarray, cur: np.ndarray) -> Tuple[float, float, float]:
        """Return (angle_deg, scale, response)."""
        if ref is None or cur is None or ref.size == 0 or cur.size == 0:
            return 0.0, 1.0, 0.0
        sz = (64, 64)  # fixed modest resolution -> stable angular resolution
        ref = cv2.resize(ref.astype(np.float32), sz)
        cur = cv2.resize(cur.astype(np.float32), sz)
        center = (sz[0] * 0.5, sz[1] * 0.5)
        max_radius = min(sz) * 0.5
        flags = cv2.INTER_LINEAR + cv2.WARP_POLAR_LOG
        ref_lp = cv2.warpPolar(ref, sz, center, max_radius, flags)
        cur_lp = cv2.warpPolar(cur, sz, center, max_radius, flags)
        hann = cv2.createHanningWindow(sz, cv2.CV_32F)
        (sx, sy), response = cv2.phaseCorrelate(cur_lp, ref_lp, hann)
        angle = sy * 360.0 / sz[1]
        scale = math.exp(sx * math.log(max_radius) / sz[0])
        return angle, scale, response


# --------------------------------------------------------------------------
class MosseMatcher(IMatcher):
    def __init__(self, sigma: float = 2.0, init_warps: int = 8, anchor: float = 0.0,
                 prior_frac: float = 0.0, rot_enable: bool = True) -> None:
        self.sigma = sigma
        self.init_warps = init_warps
        self.anchor = anchor          # per-update pull back toward the original filter
        self.prior_frac = prior_frac  # motion-prior width (fraction of half-window); 0 = off
        self.rot_enable = rot_enable
        self.rot = RotationEstimator()

        self.fsize: Tuple[int, int] = (0, 0)  # (w, h)
        self.win: Optional[np.ndarray] = None
        self.G: Optional[np.ndarray] = None
        self.H1 = self.H2 = self.H = None
        self.H1_0 = self.H2_0 = None
        self.original: Optional[np.ndarray] = None
        self.prior: Optional[np.ndarray] = None
        self.velocity = (0.0, 0.0)
        self.reacquiring = False

    # ---- interface ---------------------------------------------------------
    def has_template(self) -> bool:
        return self.H is not None

    def set_motion_hint(self, velocity):
        self.velocity = velocity

    def set_reacquiring(self, reacquiring):
        self.reacquiring = reacquiring

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _rotate_patch(p: np.ndarray, angle_deg: float) -> np.ndarray:
        c = (p.shape[1] * 0.5, p.shape[0] * 0.5)
        M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
        return cv2.warpAffine(p, M, (p.shape[1], p.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def _rnd_warp(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Small random rotation+scale about the center for robust filter init."""
        h, w = a.shape[:2]
        ang = (rng.uniform() - 0.5) * 0.2
        scale = 1.0 + (rng.uniform() - 0.5) * 0.2
        c, s = math.cos(ang) * scale, math.sin(ang) * scale
        cx, cy = w * 0.5, h * 0.5
        T = np.array([[c, -s, cx - (c * cx - s * cy)],
                      [s, c, cy - (s * cx + c * cy)]], dtype=np.float64)
        return cv2.warpAffine(a, T, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)

    def _preprocess(self, patch: np.ndarray) -> np.ndarray:
        """log -> normalize -> cosine window (MOSSE preprocessing)."""
        f = np.log(patch.astype(np.float32) + 1.0)
        f = (f - f.mean()) / (f.std() + 1e-5)
        win = self.win
        if win.shape != f.shape:
            win = cv2.resize(win, (f.shape[1], f.shape[0]))
        return (f * win).astype(np.float32)

    def _update_kernel(self) -> None:
        eps = 1e-6
        den = (self.H2.real ** 2 + self.H2.imag ** 2) + eps
        self.H = np.conj(self.H1 * np.conj(self.H2) / den)

    def _build_prior(self, v) -> None:
        """Velocity-oriented teardrop prior over the response, centered at zero-shift.

        Generous along the velocity, tight across it, collapsed behind the target
        (where a wake/trail lives). With little motion it is a symmetric Gaussian.
        """
        if self.prior_frac <= 0.0:
            self.prior = None
            return
        w, h = self.fsize
        s0 = self.prior_frac * 0.5 * min(w, h)
        sig_par = max(2.0, s0)
        sig_perp = max(2.0, s0 * 0.6)
        tau = max(1.0, s0 * 0.4)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        dx, dy = xs - w * 0.5, ys - h * 0.5

        speed = math.hypot(v[0], v[1])
        if speed <= 1.5:  # not enough motion -> symmetric Gaussian
            prior = np.exp(-(dx ** 2 + dy ** 2) / (2.0 * s0 * s0 + 1e-6))
        else:
            ux, uy = v[0] / speed, v[1] / speed
            a = dx * ux + dy * uy                      # along-track (signed)
            px, py = dx - a * ux, dy - a * uy          # cross-track vector
            w_along = np.exp(-(a ** 2) / (2.0 * sig_par ** 2))
            w_cross = np.exp(-(px ** 2 + py ** 2) / (2.0 * sig_perp ** 2))
            back = np.where(a >= 0.0, 1.0, np.exp(np.minimum(a, 0.0) / tau))
            prior = w_along * w_cross * back
        self.prior = prior.astype(np.float32)

    # ---- template lifecycle -----------------------------------------------
    def set_template(self, frame, box):
        gray = to_gray(frame)
        x, y, w, h = intersect(box, (0, 0, gray.shape[1], gray.shape[0]))
        self.fsize = (max(8, w), max(8, h))
        center = (x + w * 0.5, y + h * 0.5)
        fw, fh = self.fsize

        self.win = cv2.createHanningWindow(self.fsize, cv2.CV_32F)

        # Desired correlation output: a Gaussian peak at the window center.
        ys, xs = np.mgrid[0:fh, 0:fw].astype(np.float32)
        g = np.exp(-((xs - fw * 0.5) ** 2 + (ys - fh * 0.5) ** 2) / (2.0 * self.sigma ** 2))
        self.G = np.fft.fft2(g.astype(np.float32))

        raw = _sub_pix(gray, self.fsize, center)
        self.original = raw.copy()
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)
        rng = np.random.default_rng(12345)
        for i in range(max(1, self.init_warps)):
            a = raw.copy() if i == 0 else self._rnd_warp(raw, rng)
            F = np.fft.fft2(self._preprocess(a))
            self.H1 += self.G * np.conj(F)
            self.H2 += F * np.conj(F)
        self.H1_0, self.H2_0 = self.H1.copy(), self.H2.copy()
        self._update_kernel()

    def use_original_template(self):
        if self.H1_0 is None:
            return
        self.H1, self.H2 = self.H1_0.copy(), self.H2_0.copy()
        self._update_kernel()

    def update_template(self, frame, box, alpha):
        if self.H is None:
            return
        gray = to_gray(frame)
        bs = (max(8, int(box[2])), max(8, int(box[3])))
        center = (box[0] + box[2] * 0.5, box[1] + box[3] * 0.5)
        raw = _sub_pix(gray, bs, center)
        if (raw.shape[1], raw.shape[0]) != self.fsize:
            raw = cv2.resize(raw, self.fsize)
        F = np.fft.fft2(self._preprocess(raw))
        A, B = self.G * np.conj(F), F * np.conj(F)

        a = clamp01(alpha)  # online blend; MOSSE typically ~0.125
        self.H1 = (1.0 - a) * self.H1 + a * A
        self.H2 = (1.0 - a) * self.H2 + a * B

        # Anti-drift: tether to the original frame-1 appearance.
        if self.anchor > 0.0 and self.H1_0 is not None:
            self.H1 = (1.0 - self.anchor) * self.H1 + self.anchor * self.H1_0
            self.H2 = (1.0 - self.anchor) * self.H2 + self.anchor * self.H2_0
        self._update_kernel()

    # ---- correlation --------------------------------------------------------
    def _correlate(self, patch: np.ndarray):
        """Return (response, (peak_x, peak_y), psr). `patch` is already preprocessed."""
        F = np.fft.fft2(patch)
        resp = np.fft.ifft2(F * np.conj(self.H)).real.astype(np.float32)

        # Bias peak selection toward the predicted location with the motion prior;
        # measure PSR on the RAW response so the prior doesn't inflate confidence.
        scored = resp if self.prior is None else resp * self.prior
        py, px = np.unravel_index(int(np.argmax(scored)), scored.shape)

        peak_val = float(resp[py, px])
        mask = np.ones(resp.shape, dtype=bool)
        win = 5
        mask[max(0, py - win):py + win + 1, max(0, px - win):px + win + 1] = False
        side = resp[mask]
        psr = (peak_val - float(side.mean())) / (float(side.std()) + 1e-6) if side.size else 0.0
        return resp, (int(px), int(py)), float(psr)

    def _patch_for(self, gray, center, scale):
        ss = (max(4, int(round(self.fsize[0] * scale))), max(4, int(round(self.fsize[1] * scale))))
        raw = _sub_pix(gray, ss, center)
        if (raw.shape[1], raw.shape[0]) != self.fsize:
            raw = cv2.resize(raw, self.fsize)
        return raw

    # ---- match --------------------------------------------------------------
    def match(self, frame, search_region, expected_size, scales) -> MatchResult:
        out = MatchResult()
        if self.H is None:
            return out
        gray = to_gray(frame)
        rx, ry, rw, rh = intersect(search_region, (0, 0, gray.shape[1], gray.shape[0]))
        if rw < 4 or rh < 4:
            return out

        self._build_prior(self.velocity)  # orient the prior along predicted velocity
        fw, fh = self.fsize

        # While locked the ROI is ~2x the box -> a single center with a full scale
        # sweep. Once widened for reacquisition, tile it with overlapping
        # filter-sized windows (MOSSE can't extend its search by downsampling).
        tile = rw > fw * 2.5 or rh > fh * 2.5
        if not tile:
            centers = [(rx + rw * 0.5, ry + rh * 0.5)]
            sweep = list(scales)
        else:
            sx, sy = max(1.0, fw * 0.5), max(1.0, fh * 0.5)
            centers = []
            cy = ry + fh * 0.5
            while cy <= ry + rh - fh * 0.5:
                cx = rx + fw * 0.5
                while cx <= rx + rw - fw * 0.5:
                    centers.append((cx, cy))
                    cx += sx
                cy += sy
            if not centers:
                centers = [(rx + rw * 0.5, ry + rh * 0.5)]
            sweep = [1.0]

        def search_one(c):
            best = (-1e9, 0.0, c, (0, 0))  # psr, scale, center, peak
            for s in sweep:
                _, peak, psr = self._correlate(self._preprocess(self._patch_for(gray, c, s)))
                if psr > best[0]:
                    best = (psr, s, c, peak)
            return best

        # Windows are independent -> correlate them in parallel across CPU cores.
        hits = _pmap(search_one, centers) if len(centers) > 1 else [search_one(centers[0])]
        best_psr, best_scale, best_center, best_peak = max(hits, key=lambda h: h[0])
        if best_scale <= 0.0:
            return out

        # Peak offset (relative to the filter center) maps back to pixels via scale.
        dx = (best_peak[0] - fw * 0.5) * best_scale
        dy = (best_peak[1] - fh * 0.5) * best_scale
        cx, cy = best_center[0] + dx, best_center[1] + dy
        bw, bh = fw * best_scale, fh * best_scale

        out.box = (cx - bw * 0.5, cy - bh * 0.5, bw, bh)
        out.psr = best_psr
        # MOSSE confidence is PSR-driven; encode it into ncc as well so the evaluator
        # (0.5*ncc + 0.5*psr/sat) yields ~clamp(psr/12) under the default weights.
        out.ncc = clamp01(best_psr / 12.0)
        out.valid = True

        # ---- Gated rotation recovery -----------------------------------------
        # Fire whenever the upright match is sub-strong and we are NOT re-detecting.
        # Estimate the angle vs the frozen original, de-rotate, re-correlate, and
        # keep it ONLY if PSR improves -- so the stage can never lower confidence.
        if self.rot_enable and not self.reacquiring and out.psr < 8.4 and self.original is not None:
            raw = self._patch_for(gray, (cx, cy), best_scale)
            angle, _, _ = self.rot.estimate(self.original, raw)
            if abs(angle) >= 3.0:
                improved = out.psr
                for cand in (angle, -angle):
                    _, _, psr = self._correlate(self._preprocess(self._rotate_patch(raw, cand)))
                    improved = max(improved, psr)
                if improved > out.psr:
                    out.psr = improved
                    out.ncc = clamp01(improved / 12.0)
        return out

    # ---- multi-target detection --------------------------------------------
    def detect(self, frame, region, max_peaks: int, psr_floor: float) -> List[MatchResult]:
        """Up to max_peaks candidate locations read from the correlation heat map.

        One shared filter lights up several peaks when look-alike objects are present.
        """
        if self.H is None or max_peaks <= 0:
            return []
        gray = to_gray(frame)
        rx, ry, rw, rh = intersect(region, (0, 0, gray.shape[1], gray.shape[0]))
        fw, fh = self.fsize
        if rw < fw or rh < fh:
            return []

        sx, sy = max(1.0, fw * 0.5), max(1.0, fh * 0.5)
        grid = []
        cy = ry + fh * 0.5
        while cy <= ry + rh - fh * 0.5 + 1.0:
            cx = rx + fw * 0.5
            while cx <= rx + rw - fw * 0.5 + 1.0:
                grid.append((cx, cy))
                cx += sx
            cy += sy
        if not grid:
            return []

        def one(c):
            _, peak, psr = self._correlate(self._preprocess(_sub_pix(gray, self.fsize, c)))
            return ((c[0] + (peak[0] - fw * 0.5), c[1] + (peak[1] - fh * 0.5)), psr)

        cands = sorted(_pmap(one, grid), key=lambda t: t[1], reverse=True)

        min_dist = 0.5 * min(fw, fh)
        out: List[MatchResult] = []
        for (cx, cy), psr in cands:
            if psr < psr_floor:
                break  # sorted desc -> nothing better left
            if any(math.hypot(k.box[0] + k.box[2] * 0.5 - cx, k.box[1] + k.box[3] * 0.5 - cy) < min_dist
                   for k in out):
                continue
            out.append(MatchResult(box=(cx - fw * 0.5, cy - fh * 0.5, float(fw), float(fh)),
                                   ncc=clamp01(psr / 12.0), psr=psr, valid=True))
            if len(out) >= max_peaks:
                break
        return out
