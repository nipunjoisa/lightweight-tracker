#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace cot {

enum class TrackStatus { TRACKING, UNCERTAIN, LOST, REACQUIRING };

inline const char* toString(TrackStatus s) {
    switch (s) {
        case TrackStatus::TRACKING:    return "TRACKING";
        case TrackStatus::UNCERTAIN:   return "UNCERTAIN";
        case TrackStatus::LOST:        return "LOST";
        case TrackStatus::REACQUIRING: return "REACQUIRING";
    }
    return "UNKNOWN";
}

// Raw output of a matcher: candidate box in full-frame coords + quality signals.
struct MatchResult {
    cv::Rect2f box;
    float ncc = 0.0f;   // peak normalized cross-correlation in [-1, 1]
    float psr = 0.0f;   // peak-to-sidelobe ratio (how SHARP the peak is, not how tall)
    bool  valid = false;
};

// Result of turning a MatchResult into a decision.
struct Evaluation {
    float       confidence = 0.0f;  // fused, gated score in [0, 1]
    TrackStatus status = TrackStatus::LOST;
    bool        gatesPassed = true;
};

// Everything the visualizer / metrics need for one frame.
struct TrackState {
    cv::Rect2f  box;         // what we believe the target is (corrected or coasted)
    cv::Rect2f  predicted;   // raw Kalman prediction
    cv::Rect    searchRoi;   // region searched this frame
    float       confidence = 0.0f;
    TrackStatus status = TrackStatus::TRACKING;
    int         lostFrames = 0;
    double      fps = 0.0;
};

// Tunable parameters. v1: defaults in code / a few via CLI.
// v2: load these from configs/tracker.yaml (parsing not wired yet).
struct TrackerConfig {
    // Confidence thresholds
    float strongThreshold = 0.70f;
    float weakThreshold   = 0.45f;

    // Confidence fusion (correlation peak height vs peak sharpness)
    float nccWeight     = 0.5f;
    float psrWeight     = 0.5f;
    float psrSaturation = 12.0f;  // PSR value that maps to 1.0

    // Search region
    float searchExpansion    = 2.0f;  // ROI side = box side * this, while tracking
    float reacqExpansionStep = 0.6f;  // grow per lost frame
    float reacqMaxExpansion  = 6.0f;  // cap

    // Multi-scale match (scale factors applied to the predicted size)
    std::vector<float> scales = {0.85f, 0.925f, 1.0f, 1.075f, 1.15f};

    // Physics gating
    float maxSpeed     = 120.0f;  // max center displacement (px/frame) vs prediction
    float maxSizeRatio = 1.6f;    // max area growth/shrink per frame

    // Template update
    float templateLearnRate = 0.05f;  // blend factor, applied only when TRACKING
    bool  adapt = true;               // online template adaptation on/off

    // IO
    bool        display    = true;
    std::string outputPath = "outputs/videos/tracked.mp4";
};

} // namespace cot
