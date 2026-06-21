#include "ConfidenceEvaluator.hpp"
#include <algorithm>
#include <cmath>

namespace cot {

static float clamp01(float v) { return std::max(0.0f, std::min(1.0f, v)); }

static cv::Point2f center(const cv::Rect2f& r) {
    return cv::Point2f(r.x + r.width * 0.5f, r.y + r.height * 0.5f);
}

TrackStatus ConfidenceEvaluator::classify(float c) const {
    if (c >= cfg_.strongThreshold) return TrackStatus::TRACKING;
    if (c >= cfg_.weakThreshold)   return TrackStatus::UNCERTAIN;
    return TrackStatus::LOST;
}

Evaluation ConfidenceEvaluator::evaluate(const MatchResult& m,
                                         const cv::Rect2f& predicted,
                                         const cv::Rect2f& previous) const {
    Evaluation e;
    if (!m.valid) {
        e.confidence = 0.0f;
        e.status = TrackStatus::LOST;
        e.gatesPassed = false;
        return e;
    }

    // Fuse peak HEIGHT (clamped NCC) with peak SHARPNESS (saturated PSR).
    float nccTerm = clamp01(m.ncc);
    float psrTerm = clamp01(m.psr / cfg_.psrSaturation);
    float conf    = cfg_.nccWeight * nccTerm + cfg_.psrWeight * psrTerm;

    // ---- Physics gating: implausible candidates must not read as confident. ----
    bool gatesPassed = true;

    // Speed gate: the match shouldn't sit far from where Kalman expected it.
    cv::Point2f dp = center(m.box) - center(predicted);
    float disp = std::sqrt(dp.x * dp.x + dp.y * dp.y);
    if (disp > cfg_.maxSpeed) { conf *= 0.4f; gatesPassed = false; }

    // Size gate: area shouldn't balloon or collapse in one frame.
    float prevArea = std::max(1.0f, previous.width * previous.height);
    float curArea  = std::max(1.0f, m.box.width  * m.box.height);
    float ratio    = std::max(curArea / prevArea, prevArea / curArea);
    if (ratio > cfg_.maxSizeRatio) { conf *= 0.6f; gatesPassed = false; }

    e.confidence  = clamp01(conf);
    e.status      = classify(e.confidence);
    e.gatesPassed = gatesPassed;
    return e;
}

} // namespace cot
