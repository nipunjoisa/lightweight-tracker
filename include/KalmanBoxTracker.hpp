#pragma once
#include <opencv2/video/tracking.hpp>
#include <opencv2/core.hpp>

namespace cot {

// State  [cx, cy, vx, vy, w, h]   |   Measurement [cx, cy, w, h]
// Constant-velocity model on the center; size carried forward frame-to-frame.
class KalmanBoxTracker {
public:
    KalmanBoxTracker();
    void init(const cv::Rect2f& box);
    cv::Rect2f predict();                        // advances state, returns predicted box
    cv::Rect2f correct(const cv::Rect2f& meas);  // fuses a measurement
    cv::Rect2f stateBox() const;                 // current posterior box

private:
    static cv::Rect2f toBox(float cx, float cy, float w, float h);
    cv::KalmanFilter kf_;
    bool initialized_ = false;
};

} // namespace cot
