#include "TrackerEngine.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
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
        roi = cfg_.initRoi & bounds;
    } else {
        float scale = std::min(900.0f / firstFrame.rows, 500.0f / firstFrame.cols);
        if (scale > 1.0f) scale = 1.0f;

        cv::Mat small;
        cv::resize(firstFrame, small, cv::Size(), scale, scale, cv::INTER_LINEAR);

        const char* win = "Select target  |  drag a box, ENTER/SPACE to confirm, c to cancel";
        cv::Rect smallRoi = cv::selectROI(win, small, false, false);
        cv::destroyWindow(win);

        roi = cv::Rect(
            static_cast<int>(std::lround(smallRoi.x / scale)),
            static_cast<int>(std::lround(smallRoi.y / scale)),
            static_cast<int>(std::lround(smallRoi.width / scale)),
            static_cast<int>(std::lround(smallRoi.height / scale))
        );
        roi &= bounds;
    }
    if (roi.width < 4 || roi.height < 4) {
        std::fprintf(stderr, "[engine] invalid/empty ROI\n");
        return false;
    }
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

    cv::Rect2f believed = roiF;     // current best estimate of the target box
    float expansion   = cfg_.searchExpansion;
    bool  lastWasLost = false;

    while (source.read(frame)) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // 1) Predict where the target should be.
        cv::Rect2f predicted = kalman_.predict();
        matcher_->setMotionHint(kalman_.velocity());  // orient the motion prior

        // 2) Build the local search ROI (widened if we've been lost).
        cv::Rect roiRect = search_.compute(predicted, expansion, frame.size());

        // 3) Multi-scale template match inside the ROI.
        cv::Size expectedSize(std::max(4, (int)std::lround(predicted.width)),
                              std::max(4, (int)std::lround(predicted.height)));
        MatchResult m = matcher_->match(frame, roiRect, expectedSize, cfg_.scales);

        // 4) Fuse confidence + physics gating, then classify.
        Evaluation ev = confidence_.evaluate(m, predicted, believed);

        // A loss with prior loss history is reported as REACQUIRING.
        TrackStatus reported = ev.status;
        if (ev.status == TrackStatus::LOST && lastWasLost)
            reported = TrackStatus::REACQUIRING;

        // 5) Branch on confidence.
        if (ev.status == TrackStatus::TRACKING) {
            believed = kalman_.correct(m.box);
            if (cfg_.adapt)
                matcher_->updateTemplate(frame, toRect(m.box), cfg_.templateLearnRate);  // slow update
        } else if (ev.status == TrackStatus::UNCERTAIN) {
            believed = kalman_.correct(m.box);  // correct cautiously, do NOT touch template
        } else {
            believed = predicted;               // LOST: coast on prediction only
            if (!lastWasLost) matcher_->useOriginalTemplate();  // revert to pristine on first loss
        }

        // 6) Reacquisition policy decides NEXT frame's search expansion.
        expansion   = reacq_.onStatus(reported);
        lastWasLost = (ev.status == TrackStatus::LOST);

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
    		cv::Mat display;
    		float scale = std::min(900.0f / vis.rows, 500.0f / vis.cols);
   		if (scale < 1.0f)
       			 cv::resize(vis, display, cv::Size(), scale, scale, cv::INTER_LINEAR);
    		else
        		display = vis;
    		cv::imshow("cpu-object-tracker", display);
    		int key = cv::waitKey(1);
    		if (key == 27 || key == 'q') break;
	}
    }

    metrics_.printSummary(source.fps());
    if (cfg_.display) cv::destroyAllWindows();
    if (writer.isOpened())
        std::printf("saved output video to: %s\n", cfg_.outputPath.c_str());
    return true;
}

} // namespace cot
