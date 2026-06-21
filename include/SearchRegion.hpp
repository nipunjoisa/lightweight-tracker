#pragma once
#include <opencv2/core.hpp>

namespace cot {

// Builds the local ROI to search, centered on a box and clamped to the frame.
class SearchRegion {
public:
    // expansion: ROI side = box side * expansion (centered on the box center).
    cv::Rect compute(const cv::Rect2f& box, float expansion, const cv::Size& frameSize) const;
};

} // namespace cot
