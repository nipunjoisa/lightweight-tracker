#include "Visualizer.hpp"
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <string>

namespace cot {

cv::Scalar Visualizer::statusColor(TrackStatus s) {
    switch (s) {  // BGR
        case TrackStatus::TRACKING:    return cv::Scalar(80, 220, 80);   // green
        case TrackStatus::UNCERTAIN:   return cv::Scalar(0, 200, 255);   // amber
        case TrackStatus::LOST:        return cv::Scalar(60, 60, 230);   // red
        case TrackStatus::REACQUIRING: return cv::Scalar(0, 140, 255);   // orange
    }
    return cv::Scalar(200, 200, 200);
}

cv::Mat Visualizer::draw(const cv::Mat& frame, const TrackState& st) const {
    cv::Mat out = frame.clone();
    cv::Scalar col = statusColor(st.status);

    cv::rectangle(out, st.searchRoi, cv::Scalar(120, 120, 120), 1);        // search ROI
    cv::rectangle(out, cv::Rect(st.predicted), cv::Scalar(200, 160, 60), 1); // raw prediction
    cv::rectangle(out, cv::Rect(st.box), col, 1);                          // believed target (thin)

    // Primary marker: a dot at the target center (SkyTracker style). The thin
    // box stays as a secondary cue, but the dot is what reads at small sizes.
    cv::Point2f c(st.box.x + st.box.width * 0.5f, st.box.y + st.box.height * 0.5f);
    cv::circle(out, c, 6, col, -1, cv::LINE_AA);
    cv::circle(out, c, 8, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

    int y = 24;
    auto put = [&](const std::string& text, cv::Scalar c) {
        cv::putText(out, text, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 0, 0), 3, cv::LINE_AA);  // outline for legibility
        cv::putText(out, text, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    c, 1, cv::LINE_AA);
        y += 26;
    };

    char buf[128];
    std::snprintf(buf, sizeof(buf), "STATUS: %s", toString(st.status));
    put(buf, col);
    std::snprintf(buf, sizeof(buf), "CONF  : %.2f", st.confidence);
    put(buf, cv::Scalar(255, 255, 255));
    std::snprintf(buf, sizeof(buf), "FPS   : %.1f", st.fps);
    put(buf, cv::Scalar(255, 255, 255));
    std::snprintf(buf, sizeof(buf), "LOST  : %d", st.lostFrames);
    put(buf, cv::Scalar(255, 255, 255));

    return out;
}

} // namespace cot
