"""Shared types and configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

# A box is (x, y, w, h) in full-frame pixel coordinates (floats allowed).
Box = Tuple[float, float, float, float]


class TrackStatus(Enum):
    TRACKING = "TRACKING"
    UNCERTAIN = "UNCERTAIN"
    LOST = "LOST"
    REACQUIRING = "REACQUIRING"


@dataclass
class MatchResult:
    """Raw output of a matcher: candidate box + quality signals."""
    box: Box = (0.0, 0.0, 0.0, 0.0)
    ncc: float = 0.0   # peak normalized cross-correlation in [-1, 1]
    psr: float = 0.0   # peak-to-sidelobe ratio (how SHARP the peak is)
    valid: bool = False


@dataclass
class Evaluation:
    confidence: float = 0.0
    status: TrackStatus = TrackStatus.LOST
    gates_passed: bool = True


@dataclass
class TrackState:
    box: Box
    predicted: Box
    search_roi: Tuple[int, int, int, int]
    confidence: float = 0.0
    status: TrackStatus = TrackStatus.TRACKING
    lost_frames: int = 0
    fps: float = 0.0


@dataclass
class TrackerConfig:
    # Confidence thresholds
    strong_threshold: float = 0.70
    weak_threshold: float = 0.45
    reacquire_confidence: float = 0.70   # strict bar to re-lock after a loss

    # Confidence fusion
    ncc_weight: float = 0.5
    psr_weight: float = 0.5
    psr_saturation: float = 12.0

    # Search region
    search_expansion: float = 2.0
    reacq_expansion_step: float = 0.6
    reacq_max_expansion: float = 6.0

    # Multi-scale match
    scales: List[float] = field(default_factory=lambda: [0.85, 0.925, 1.0, 1.075, 1.15])

    # Physics gating
    max_speed: float = 120.0     # px/frame vs prediction
    max_size_ratio: float = 1.6  # area growth/shrink per frame

    # Template update
    template_learn_rate: float = 0.05
    adapt: bool = True

    # IO
    display: bool = True
    output_path: str = "outputs/videos/tracked.mp4"
    init_roi: Optional[Tuple[int, int, int, int]] = None
