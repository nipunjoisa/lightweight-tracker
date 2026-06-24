#include "CvTrackerMatcher.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace cot {

static float clamp01(float v) { return std::max(0.0f, std::min(1.0f, v)); }

cv::Ptr<cv::Tracker> CvTrackerMatcher::makeTracker() const {
    if (kind_ == Kind::KCF) return cv::TrackerKCF::create();
    return cv::TrackerCSRT::create();
}

cv::Mat CvTrackerMatcher::toGray(const cv::Mat& frame) {
    if (frame.channels() == 1) return frame;
    cv::Mat g;
    cv::cvtColor(frame, g, cv::COLOR_BGR2GRAY);
    return g;
}

void CvTrackerMatcher::setTemplate(const cv::Mat& frame, const cv::Rect& box) {
    cv::Rect b = box & cv::Rect(0, 0, frame.cols, frame.rows);
    if (b.width < 4 || b.height < 4) return;
    toGray(frame)(b).copyTo(original_);
    tracker_ = makeTracker();
    tracker_->init(frame, b);     // CSRT/KCF take the color frame
    needReinit_ = false;
}

void CvTrackerMatcher::updateTemplate(const cv::Mat&, const cv::Rect&, float) {
    // CSRT/KCF update their own appearance model on every match(); nothing to do.
}

void CvTrackerMatcher::useOriginalTemplate() {
    needReinit_ = true;   // on loss, re-init at the Kalman-predicted spot next match
}

float CvTrackerMatcher::scoreAt(const cv::Mat& gray, const cv::Rect& box) const {
    if (original_.empty()) return 0.0f;
    cv::Rect b = box & cv::Rect(0, 0, gray.cols, gray.rows);
    if (b.width < 4 || b.height < 4) return 0.0f;
    cv::Mat patch;
    cv::resize(gray(b), patch, original_.size());
    cv::Mat r;
    cv::matchTemplate(patch, original_, r, cv::TM_CCOEFF_NORMED);  // equal sizes -> 1x1
    return r.at<float>(0, 0);
}

MatchResult CvTrackerMatcher::match(const cv::Mat& frame,
                                    const cv::Rect& searchRegion,
                                    const cv::Size& expectedSize,
                                    const std::vector<float>&) {
    MatchResult out;
    if (!tracker_) return out;

    // On loss, teleport the tracker to where Kalman expects the target, re-lock.
    if (needReinit_) {
        cv::Point2f c(searchRegion.x + searchRegion.width * 0.5f,
                      searchRegion.y + searchRegion.height * 0.5f);
        cv::Rect rb(cvRound(c.x - expectedSize.width * 0.5f),
                    cvRound(c.y - expectedSize.height * 0.5f),
                    std::max(4, expectedSize.width), std::max(4, expectedSize.height));
        rb &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (rb.width >= 4 && rb.height >= 4) {
            tracker_ = makeTracker();
            tracker_->init(frame, rb);
        }
        needReinit_ = false;
    }

    cv::Rect box;
    bool ok = tracker_->update(frame, box);
    if (!ok || box.width < 2 || box.height < 2) return out;  // invalid -> LOST

    float q = clamp01(scoreAt(toGray(frame), box));
    out.box = cv::Rect2f((float)box.x, (float)box.y, (float)box.width, (float)box.height);
    out.ncc = q;
    out.psr = q * 12.0f;   // makes confidence ~= q under the default 0.5/0.5 fusion
    out.valid = true;
    return out;
}

} // namespace cot
