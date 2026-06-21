#pragma once
#include "Types.hpp"

namespace cot {

// Fuses matcher signals into a [0,1] confidence, applies physics gates,
// then classifies the result into a TrackStatus.
class ConfidenceEvaluator {
public:
    explicit ConfidenceEvaluator(const TrackerConfig& cfg) : cfg_(cfg) {}

    Evaluation evaluate(const MatchResult& m,
                        const cv::Rect2f& predicted,
                        const cv::Rect2f& previous) const;

private:
    TrackStatus classify(float confidence) const;
    TrackerConfig cfg_;
};

} // namespace cot
