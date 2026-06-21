#include "TemplateMatcher.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace cot {

cv::Mat TemplateMatcher::toGray(const cv::Mat& frame) {
    if (frame.channels() == 1) return frame;
    cv::Mat g;
    cv::cvtColor(frame, g, cv::COLOR_BGR2GRAY);
    return g;
}

cv::Rect TemplateMatcher::clampRect(const cv::Rect& r, const cv::Size& bounds) {
    return r & cv::Rect(0, 0, bounds.width, bounds.height);
}

void TemplateMatcher::setTemplate(const cv::Mat& frame, const cv::Rect& box) {
    cv::Mat gray = toGray(frame);
    cv::Rect b = clampRect(box, gray.size());
    if (b.width < 4 || b.height < 4) return;
    gray(b).copyTo(original_);
    original_.copyTo(working_);
}

void TemplateMatcher::useOriginalTemplate() {
    if (!original_.empty()) original_.copyTo(working_);
}

void TemplateMatcher::updateTemplate(const cv::Mat& frame, const cv::Rect& box, float alpha) {
    if (working_.empty()) return;
    cv::Mat gray = toGray(frame);
    cv::Rect b = clampRect(box, gray.size());
    if (b.width < 4 || b.height < 4) return;

    cv::Mat patch;
    cv::resize(gray(b), patch, working_.size());
    // Slow exponential blend: adapt to lighting/pose drift without snapping to background.
    cv::addWeighted(working_, 1.0 - alpha, patch, alpha, 0.0, working_);
}

// PSR = (peak - mean_sidelobe) / (std_sidelobe + eps). High when the peak is a
// lonely spike (a real, distinctive match); low when the surface is a plateau.
float TemplateMatcher::computePSR(const cv::Mat& response, const cv::Point& peak) {
    double peakVal = response.at<float>(peak);

    cv::Mat mask = cv::Mat::ones(response.size(), CV_8U);
    const int win = 5;  // exclude an 11x11 window around the peak
    cv::Rect excl(peak.x - win, peak.y - win, 2 * win + 1, 2 * win + 1);
    excl &= cv::Rect(0, 0, response.cols, response.rows);
    if (excl.area() > 0) mask(excl).setTo(0);

    if (cv::countNonZero(mask) < 8) return 0.0f;  // ROI too small to judge sharpness

    cv::Scalar mean, stddev;
    cv::meanStdDev(response, mean, stddev, mask);
    double s = stddev[0] + 1e-6;
    return static_cast<float>((peakVal - mean[0]) / s);
}

MatchResult TemplateMatcher::match(const cv::Mat& frame,
                                   const cv::Rect& searchRegion,
                                   const cv::Size& expectedSize,
                                   const std::vector<float>& scales) {
    MatchResult out;
    if (working_.empty()) return out;

    cv::Mat gray = toGray(frame);
    cv::Rect roi = clampRect(searchRegion, gray.size());
    if (roi.width < 8 || roi.height < 8) return out;
    cv::Mat roiImg = gray(roi);

    double   bestVal = -1e9;
    cv::Point bestLoc;
    cv::Size  bestSize(0, 0);
    cv::Mat   bestResp;

    for (float s : scales) {
        int tw = std::max(4, static_cast<int>(std::lround(expectedSize.width  * s)));
        int th = std::max(4, static_cast<int>(std::lround(expectedSize.height * s)));
        if (tw >= roiImg.cols || th >= roiImg.rows) continue;  // template must fit the ROI

        cv::Mat tmpl;
        cv::resize(working_, tmpl, cv::Size(tw, th));

        cv::Mat resp;
        cv::matchTemplate(roiImg, tmpl, resp, cv::TM_CCOEFF_NORMED);

        double mn, mx; cv::Point mnLoc, mxLoc;
        cv::minMaxLoc(resp, &mn, &mx, &mnLoc, &mxLoc);
        if (mx > bestVal) {
            bestVal  = mx;
            bestLoc  = mxLoc;
            bestSize = cv::Size(tw, th);
            bestResp = resp.clone();
        }
    }

    if (bestSize.width == 0) return out;  // nothing fit at any scale

    float psr = computePSR(bestResp, bestLoc);

    // bestLoc is the template's top-left WITHIN the ROI; lift to full-frame coords.
    float fx = static_cast<float>(roi.x + bestLoc.x);
    float fy = static_cast<float>(roi.y + bestLoc.y);
    out.box   = cv::Rect2f(fx, fy, static_cast<float>(bestSize.width),
                                   static_cast<float>(bestSize.height));
    out.ncc   = static_cast<float>(bestVal);
    out.psr   = psr;
    out.valid = true;
    return out;
}

} // namespace cot
