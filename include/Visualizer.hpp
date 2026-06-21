#pragma once
#include "Types.hpp"
#include <opencv2/core.hpp>

namespace cot {

// Paints the believed box, predicted box, search ROI, and a HUD onto a copy.
class Visualizer {
public:
    cv::Mat draw(const cv::Mat& frame, const TrackState& st) const;

private:
    static cv::Scalar statusColor(TrackStatus s);
};

} // namespace cot
