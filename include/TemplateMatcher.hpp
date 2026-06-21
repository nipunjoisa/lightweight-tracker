#pragma once
#include "IMatcher.hpp"

namespace cot {

// NCC (TM_CCOEFF_NORMED) multi-scale matcher with peak-to-sidelobe sharpness.
// Keeps a pristine "original" template and an adaptively-updated "working" one.
class TemplateMatcher : public IMatcher {
public:
    void setTemplate(const cv::Mat& frame, const cv::Rect& box) override;
    void updateTemplate(const cv::Mat& frame, const cv::Rect& box, float alpha) override;
    void useOriginalTemplate() override;
    MatchResult match(const cv::Mat& frame,
                      const cv::Rect& searchRegion,
                      const cv::Size& expectedSize,
                      const std::vector<float>& scales) override;
    bool hasTemplate() const override { return !working_.empty(); }

private:
    static cv::Mat  toGray(const cv::Mat& frame);
    static cv::Rect clampRect(const cv::Rect& r, const cv::Size& bounds);
    static float    computePSR(const cv::Mat& response, const cv::Point& peak);

    cv::Mat original_;  // pristine gray template at capture size
    cv::Mat working_;   // adaptively-updated gray template
};

} // namespace cot
