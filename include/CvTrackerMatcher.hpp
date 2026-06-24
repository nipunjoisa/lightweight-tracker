#pragma once
#include "IMatcher.hpp"
#include <opencv2/tracking.hpp>

namespace cot {

// Wraps OpenCV's built-in discriminative trackers (CSRT / KCF) behind IMatcher,
// for a head-to-head comparison with the hand-rolled MOSSE. These trackers run
// their own internal local search and online appearance model; we supervise them
// with the Kalman state machine + gates, and derive a confidence proxy by
// correlating the tracked patch against the initial template (they expose no
// score of their own). Requires the opencv tracking module (present in
// `brew install opencv`).
class CvTrackerMatcher : public IMatcher {
public:
    enum class Kind { CSRT, KCF };
    explicit CvTrackerMatcher(Kind kind) : kind_(kind) {}

    void setTemplate(const cv::Mat& frame, const cv::Rect& box) override;
    void updateTemplate(const cv::Mat& frame, const cv::Rect& box, float alpha) override;
    void useOriginalTemplate() override;
    MatchResult match(const cv::Mat& frame,
                      const cv::Rect& searchRegion,
                      const cv::Size& expectedSize,
                      const std::vector<float>& scales) override;
    bool hasTemplate() const override { return static_cast<bool>(tracker_); }

private:
    cv::Ptr<cv::Tracker> makeTracker() const;
    static cv::Mat toGray(const cv::Mat& frame);
    float scoreAt(const cv::Mat& gray, const cv::Rect& box) const;

    Kind kind_;
    cv::Ptr<cv::Tracker> tracker_;
    cv::Mat original_;          // gray template at init size (confidence proxy)
    bool    needReinit_ = false;
};

} // namespace cot
