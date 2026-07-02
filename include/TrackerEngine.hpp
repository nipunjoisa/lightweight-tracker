#pragma once
#include "Types.hpp"
#include "VideoSource.hpp"
#include "KalmanBoxTracker.hpp"
#include "IMatcher.hpp"
#include "SearchRegion.hpp"
#include "ConfidenceEvaluator.hpp"
#include "ReacquisitionManager.hpp"
#include "TrackerMetrics.hpp"
#include "Visualizer.hpp"
#include <memory>

namespace cot {

// Wires every module into the predict -> search -> match -> evaluate ->
// correct/update -> visualize loop. Owns no CV algorithm itself.
class TrackerEngine {
public:
    TrackerEngine(TrackerConfig cfg, std::unique_ptr<IMatcher> matcher);

    bool run(VideoSource& source);  // false on fatal setup error

private:
    bool initializeTarget(VideoSource& source, cv::Mat& firstFrame, cv::Rect& roi);
    cv::Size cachedScreenSize();

    TrackerConfig             cfg_;
    std::unique_ptr<IMatcher> matcher_;
    KalmanBoxTracker          kalman_;
    SearchRegion              search_;
    ConfidenceEvaluator       confidence_;
    ReacquisitionManager      reacq_;
    TrackerMetrics            metrics_;
    Visualizer                viz_;
    cv::Size                  screenSize_{0, 0};  // cached, detected once on first use
};

} // namespace cot
