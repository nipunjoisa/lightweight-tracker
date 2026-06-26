#include "TrackerEngine.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>

namespace cot {

static cv::Rect toRect(const cv::Rect2f& r) {
    return cv::Rect(static_cast<int>(std::lround(r.x)),
                    static_cast<int>(std::lround(r.y)),
                    static_cast<int>(std::lround(r.width)),
                    static_cast<int>(std::lround(r.height)));
}

TrackerEngine::TrackerEngine(TrackerConfig cfg, std::unique_ptr<IMatcher> matcher)
    : cfg_(std::move(cfg)),
      matcher_(std::move(matcher)),
      confidence_(cfg_),
      reacq_(cfg_) {}

bool TrackerEngine::initializeTarget(VideoSource& source, cv::Mat& firstFrame, cv::Rect& roi) {
    if (!source.read(firstFrame)) {
        std::fprintf(stderr, "[engine] could not read first frame\n");
        return false;
    }
    cv::Rect bounds(0, 0, firstFrame.cols, firstFrame.rows);
    if (cfg_.hasInitRoi) {
        roi = cfg_.initRoi & bounds;          // preset box -> identical init across runs
    } else {
        const char* win = "Select target  |  drag a box, ENTER/SPACE to confirm, c to cancel";
        roi = cv::selectROI(win, firstFrame, /*showCrosshair=*/false, /*fromCenter=*/false);
        cv::destroyWindow(win);
    }
    if (roi.width < 4 || roi.height < 4) {
        std::fprintf(stderr, "[engine] invalid/empty ROI\n");
        return false;
    }
    // Print so the same box can be reused across matchers for a fair comparison.
    std::printf("init roi: %d,%d,%d,%d  (reuse with --roi %d,%d,%d,%d)\n",
                roi.x, roi.y, roi.width, roi.height, roi.x, roi.y, roi.width, roi.height);
    return true;
}

bool TrackerEngine::run(VideoSource& source) {
    cv::Mat frame;
    cv::Rect roi;
    if (!initializeTarget(source, frame, roi)) return false;

    // ---- One-time setup: template + Kalman seed. ----
    matcher_->setTemplate(frame, roi);
    cv::Rect2f roiF(roi.x, roi.y, roi.width, roi.height);
    kalman_.init(roiF);
    reacq_.reset();

    // Output writer matches the source's size/fps so playback is honest.
    cv::VideoWriter writer;
    {
        std::error_code ec;
        auto parent = std::filesystem::path(cfg_.outputPath).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent, ec);
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(cfg_.outputPath, fourcc, source.fps(), source.frameSize());
        if (!writer.isOpened())
            std::fprintf(stderr, "[engine] warning: could not open writer at %s\n",
                         cfg_.outputPath.c_str());
    }

    cv::Rect2f believed      = roiF;  // current best estimate of the target box
    cv::Rect2f lastConfident = roiF;  // last box we actually measured (re-detect anchor)
    float expansion   = cfg_.searchExpansion;
    bool  lastWasLost = false;

    while (source.read(frame)) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // 1) Predict where the target should be.
        cv::Rect2f predicted = kalman_.predict();
        matcher_->setMotionHint(kalman_.velocity());  // orient the motion prior

        // 2) Build the local search ROI. While locked we center on the prediction.
        //    Once lost we RE-DETECT: center on the last position we actually
        //    measured (not the coasting prediction, which marches off in the stale
        //    direction) and let the matcher's widened tiled search scan around it.
        const bool searching = reacq_.lostFrames() > 0;
        matcher_->setReacquiring(searching);   // suppress rotation recovery while re-detecting
        const cv::Rect2f searchCenter = searching ? lastConfident : predicted;
        cv::Rect roiRect = search_.compute(searchCenter, expansion, frame.size());

        // 3) Multi-scale template match inside the ROI.
        cv::Size expectedSize(std::max(4, (int)std::lround(predicted.width)),
                              std::max(4, (int)std::lround(predicted.height)));
        MatchResult m = matcher_->match(frame, roiRect, expectedSize, cfg_.scales);

        // 4) Fuse confidence + gating. While re-detecting, gate against the last
        //    sighting and let the reachable radius grow with the time spent lost.
        const cv::Rect2f speedRef = searching ? lastConfident : predicted;
        const cv::Rect2f sizeRef  = searching ? lastConfident : believed;
        const float slack = searching ? std::min(8.0f, 1.0f + (float)reacq_.lostFrames()) : 1.0f;
        Evaluation ev = confidence_.evaluate(m, speedRef, sizeRef, slack);

        // 5) Accept or reject. Normal tracking accepts anything not LOST; while
        //    re-detecting we demand a STRONG, confirmed match before re-locking, so
        //    a weak false peak during the wide search can't hijack the track.
        bool accept = searching ? (ev.confidence >= cfg_.reacquireConfidence)
                                : (ev.status != TrackStatus::LOST);

        TrackStatus reported;
        if (accept) {
            believed = kalman_.correct(m.box);
            lastConfident = m.box;                       // remember the real sighting
            reported = (ev.confidence >= cfg_.strongThreshold) ? TrackStatus::TRACKING
                                                               : TrackStatus::UNCERTAIN;
            if (reported == TrackStatus::TRACKING && cfg_.adapt)
                matcher_->updateTemplate(frame, toRect(m.box), cfg_.templateLearnRate);
        } else {
            believed = predicted;                        // coast on prediction only
            if (!lastWasLost) matcher_->useOriginalTemplate();  // revert to pristine on first loss
            reported = lastWasLost ? TrackStatus::REACQUIRING : TrackStatus::LOST;
        }

        // 6) Reacquisition policy decides NEXT frame's search expansion.
        expansion   = reacq_.onStatus(reported);
        lastWasLost = !accept;

        // ---- timing / HUD ----
        auto t1 = std::chrono::high_resolution_clock::now();
        double frameMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double fps = frameMs > 0 ? 1000.0 / frameMs : 0.0;

        TrackState st;
        st.box        = believed;
        st.predicted  = predicted;
        st.searchRoi  = roiRect;
        st.confidence = ev.confidence;
        st.status     = reported;
        st.lostFrames = reacq_.lostFrames();
        st.fps        = fps;

        metrics_.record(reported, ev.confidence, frameMs);

        cv::Mat vis = viz_.draw(frame, st);
        if (writer.isOpened()) writer.write(vis);
        if (cfg_.display) {
            cv::imshow("cpu-object-tracker", vis);
            int key = cv::waitKey(1);
            if (key == 27 || key == 'q') break;  // ESC / q to quit early
        }
    }

    metrics_.printSummary(source.fps());
    if (cfg_.display) cv::destroyAllWindows();
    if (writer.isOpened())
        std::printf("saved output video to: %s\n", cfg_.outputPath.c_str());
    return true;
}

} // namespace cot
