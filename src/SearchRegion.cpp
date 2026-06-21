#include "SearchRegion.hpp"
#include <algorithm>
#include <cmath>

namespace cot {

cv::Rect SearchRegion::compute(const cv::Rect2f& box, float expansion,
                               const cv::Size& frameSize) const {
    float cx = box.x + box.width  * 0.5f;
    float cy = box.y + box.height * 0.5f;
    float w  = std::max(8.0f, box.width  * expansion);
    float h  = std::max(8.0f, box.height * expansion);

    int x = static_cast<int>(std::lround(cx - w * 0.5f));
    int y = static_cast<int>(std::lround(cy - h * 0.5f));
    cv::Rect roi(x, y, static_cast<int>(std::lround(w)), static_cast<int>(std::lround(h)));
    return roi & cv::Rect(0, 0, frameSize.width, frameSize.height);
}

} // namespace cot
