#include "KalmanBoxTracker.hpp"

namespace cot {

KalmanBoxTracker::KalmanBoxTracker() {
    kf_.init(6, 4, 0, CV_32F);

    // Constant-velocity model on the center; size held steady frame-to-frame.
    kf_.transitionMatrix = (cv::Mat_<float>(6, 6) <<
        1, 0, 1, 0, 0, 0,
        0, 1, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1);

    // Measurement observes [cx, cy, w, h].
    kf_.measurementMatrix = (cv::Mat_<float>(4, 6) <<
        1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1);

    cv::setIdentity(kf_.processNoiseCov,     cv::Scalar::all(1e-2));
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kf_.errorCovPost,        cv::Scalar::all(1.0));
}

cv::Rect2f KalmanBoxTracker::toBox(float cx, float cy, float w, float h) {
    return cv::Rect2f(cx - w * 0.5f, cy - h * 0.5f, w, h);
}

void KalmanBoxTracker::init(const cv::Rect2f& box) {
    float cx = box.x + box.width  * 0.5f;
    float cy = box.y + box.height * 0.5f;
    kf_.statePost = (cv::Mat_<float>(6, 1) << cx, cy, 0, 0, box.width, box.height);
    kf_.statePre  = kf_.statePost.clone();
    initialized_  = true;
}

cv::Rect2f KalmanBoxTracker::predict() {
    cv::Mat p = kf_.predict();
    return toBox(p.at<float>(0), p.at<float>(1), p.at<float>(4), p.at<float>(5));
}

cv::Rect2f KalmanBoxTracker::correct(const cv::Rect2f& meas) {
    float cx = meas.x + meas.width  * 0.5f;
    float cy = meas.y + meas.height * 0.5f;
    cv::Mat m = (cv::Mat_<float>(4, 1) << cx, cy, meas.width, meas.height);
    cv::Mat c = kf_.correct(m);
    return toBox(c.at<float>(0), c.at<float>(1), c.at<float>(4), c.at<float>(5));
}

cv::Rect2f KalmanBoxTracker::stateBox() const {
    const cv::Mat& s = kf_.statePost;
    return toBox(s.at<float>(0), s.at<float>(1), s.at<float>(4), s.at<float>(5));
}

cv::Point2f KalmanBoxTracker::velocity() const {
    const cv::Mat& s = kf_.statePost;
    return cv::Point2f(s.at<float>(2), s.at<float>(3));
}

} // namespace cot
