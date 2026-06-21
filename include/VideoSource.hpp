#pragma once
#include <opencv2/videoio.hpp>
#include <string>

namespace cot {

// Uniform faucet over a video file OR a camera index. The engine never
// learns which one it is talking to.
class VideoSource {
public:
    bool openFile(const std::string& path);
    bool openCamera(int index);
    bool read(cv::Mat& frame);            // false when the stream is dry
    double fps() const { return fps_; }
    cv::Size frameSize() const { return size_; }
    bool isOpened() const { return cap_.isOpened(); }

private:
    void cacheProperties();
    cv::VideoCapture cap_;
    double fps_ = 30.0;
    cv::Size size_{0, 0};
};

} // namespace cot
