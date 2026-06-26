#include "RotationEstimator.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace cot {

RotationEstimator::Result RotationEstimator::estimate(const cv::Mat& refIn,
                                                      const cv::Mat& curIn) const {
    Result out;
    if (refIn.empty() || curIn.empty()) return out;

    // Work at a fixed, modest resolution so angular resolution is stable even for
    // tiny targets (360/64 ~ 5.6 deg per row).
    const cv::Size sz(64, 64);
    cv::Mat ref, cur;
    refIn.convertTo(ref, CV_32F);
    curIn.convertTo(cur, CV_32F);
    if (ref.size() != sz) cv::resize(ref, ref, sz);
    if (cur.size() != sz) cv::resize(cur, cur, sz);

    // Log-polar: rotation -> vertical shift, scale -> horizontal shift.
    const cv::Point2f center(sz.width * 0.5f, sz.height * 0.5f);
    const double maxRadius = std::min(sz.width, sz.height) * 0.5;
    const int flags = cv::INTER_LINEAR + cv::WARP_POLAR_LOG;
    cv::Mat refLP, curLP;
    cv::warpPolar(ref, refLP, sz, center, maxRadius, flags);
    cv::warpPolar(cur, curLP, sz, center, maxRadius, flags);

    // Phase correlation recovers the (log-radius, angle) shift in one FFT pass.
    cv::Mat hann;
    cv::createHanningWindow(hann, sz, CV_32F);
    double response = 0.0;
    cv::Point2d shift = cv::phaseCorrelate(curLP, refLP, hann, &response);

    out.angleDeg = shift.y * 360.0 / sz.height;
    out.scale    = std::exp(shift.x * std::log(maxRadius) / sz.width);
    out.response = response;
    return out;
}

} // namespace cot
