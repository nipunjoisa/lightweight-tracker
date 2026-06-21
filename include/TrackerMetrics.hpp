#pragma once
#include "Types.hpp"
#include <vector>

namespace cot {

// Self-consistency metrics for v1 (no ground truth). Built so IoU / center-error
// against labels can be added later without touching the engine.
class TrackerMetrics {
public:
    void record(TrackStatus status, float confidence, double frameMs);
    void printSummary(double sourceFps) const;

private:
    int total_ = 0;
    int tracking_ = 0, uncertain_ = 0, lost_ = 0, reacquiring_ = 0;
    int reacquisitions_ = 0;
    int curStreak_ = 0, longestStreak_ = 0;
    bool wasLost_ = false;
    double confSum_ = 0.0;
    std::vector<double> frameMs_;
};

} // namespace cot
