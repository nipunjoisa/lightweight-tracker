#include "VideoSource.hpp"

namespace cot {

bool VideoSource::openFile(const std::string& path) {
    if (!cap_.open(path)) return false;
    cacheProperties();
    return true;
}

bool VideoSource::openCamera(int index) {
    if (!cap_.open(index)) return false;
    cacheProperties();
    return true;
}

void VideoSource::cacheProperties() {
    double f = cap_.get(cv::CAP_PROP_FPS);
    fps_ = (f > 1.0 && f < 1000.0) ? f : 30.0;  // cameras frequently report 0
    int w = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    size_ = cv::Size(w, h);
}

bool VideoSource::read(cv::Mat& frame) {
    return cap_.read(frame) && !frame.empty();
}

} // namespace cot
