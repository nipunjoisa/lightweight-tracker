#pragma once
#include <opencv2/video/tracking.hpp>
#include <opencv2/core.hpp>

namespace cot {

// State  [cx, cy, vx, vy, w, h]   |   Measurement [cx, cy, w, h]
// Constant-velocity model on the center; size carried forward frame-to-frame.
// (Constant-acceleration was tried and destabilised the search center on noisy
// measurements, so we keep CV as the stable baseline.)
class KalmanBoxTracker {
public:
    KalmanBoxTracker();
    void init(const cv::Rect2f& box);
    cv::Rect2f predict();                        // advances state, returns predicted box
    cv::Rect2f correct(const cv::Rect2f& meas);  // fuses a measurement
    cv::Rect2f stateBox() const;                 // current posterior box
    cv::Point2f velocity() const;                // current (vx, vy), px/frame

private:
    static cv::Rect2f toBox(float cx, float cy, float w, float h);
    cv::KalmanFilter kf_;
    bool initialized_ = false;
};

} // namespace cot
