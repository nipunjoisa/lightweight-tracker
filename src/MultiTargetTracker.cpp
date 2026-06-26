#include "MultiTargetTracker.hpp"
#include "VideoSource.hpp"
#include "TrackerMetrics.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>

namespace cot {

static cv::Point2f centerOf(const cv::Rect2f& r) {
    return { r.x + r.width * 0.5f, r.y + r.height * 0.5f };
}
static cv::Rect toRectI(const cv::Rect2f& r) {
    return cv::Rect((int)std::lround(r.x), (int)std::lround(r.y),
                    (int)std::lround(r.width), (int)std::lround(r.height));
}

MultiTargetTracker::MultiTargetTracker(TrackerConfig cfg, std::unique_ptr<MosseMatcher> appearance)
    : cfg_(std::move(cfg)), app_(std::move(appearance)) {}

void MultiTargetTracker::init(const cv::Mat& frame, const cv::Rect& primaryRoi) {
    app_->setTemplate(frame, primaryRoi);
    targetSize_ = cv::Size(std::max(8, primaryRoi.width), std::max(8, primaryRoi.height));
    auto t = std::make_unique<MtTrack>();
    t->id = nextId_++;
    t->primary = true;
    t->box = cv::Rect2f(primaryRoi.x, primaryRoi.y, primaryRoi.width, primaryRoi.height);
    t->kf.init(t->box);
    t->hits = 1;
    tracks_.push_back(std::move(t));
}

const MtTrack* MultiTargetTracker::primary() const {
    for (const auto& t : tracks_) if (t->primary) return t.get();
    return tracks_.empty() ? nullptr : tracks_.front().get();
}

cv::Rect MultiTargetTracker::computeRegion(const cv::Size& frameSize) const {
    // Union of all predicted boxes, padded, so the shared heat map covers every
    // track plus room for a newcomer to appear.
    float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
    for (const auto& t : tracks_) {
        x0 = std::min(x0, t->box.x);
        y0 = std::min(y0, t->box.y);
        x1 = std::max(x1, t->box.x + t->box.width);
        y1 = std::max(y1, t->box.y + t->box.height);
    }
    float padX = targetSize_.width  * 2.0f;
    float padY = targetSize_.height * 2.0f;
    cv::Rect r((int)std::lround(x0 - padX), (int)std::lround(y0 - padY),
               (int)std::lround((x1 - x0) + 2 * padX), (int)std::lround((y1 - y0) + 2 * padY));
    return r & cv::Rect(0, 0, frameSize.width, frameSize.height);
}

MtTrack* MultiTargetTracker::spawn(const cv::Rect2f& box) {
    if ((int)tracks_.size() >= maxTracks_) return nullptr;
    auto t = std::make_unique<MtTrack>();
    t->id = nextId_++;
    t->box = box;
    t->kf.init(box);
    t->status = TrackStatus::REACQUIRING;
    MtTrack* raw = t.get();
    tracks_.push_back(std::move(t));
    return raw;
}

void MultiTargetTracker::step(const cv::Mat& frame) {
    if (tracks_.empty()) return;

    // 1) Predict every track.
    for (auto& t : tracks_) t->box = t->kf.predict();

    // 2) Read the shared heat map -> candidate detections (peaks).
    cv::Rect region = computeRegion(frame.size());
    const int want = (int)tracks_.size() + 2;
    std::vector<MatchResult> dets =
        app_->detect(frame, region, want, /*psrFloor=*/cfg_.weakThreshold * 12.0f);

    // 3) Greedy association by motion: each track (primary first) claims its nearest
    //    unclaimed detection within a speed-scaled gate. Contested peaks go to the
    //    track whose prediction is closest -- identity stays with motion, not looks.
    std::vector<int> claim(dets.size(), -1);                 // det -> track index
    std::vector<int> order(tracks_.size());
    for (size_t i = 0; i < tracks_.size(); ++i) order[i] = (int)i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return tracks_[a]->primary > tracks_[b]->primary;    // primary first
    });

    std::vector<int> trackDet(tracks_.size(), -1);
    for (int ti : order) {
        const cv::Point2f pc = centerOf(tracks_[ti]->box);
        const float gate = cfg_.maxSpeed * (1.0f + std::min(7, tracks_[ti]->missed));
        float best = std::numeric_limits<float>::max();
        int bestDet = -1;
        for (size_t di = 0; di < dets.size(); ++di) {
            if (claim[di] >= 0) continue;
            float d = (float)cv::norm(centerOf(dets[di].box) - pc);
            if (d < best && d <= gate) { best = d; bestDet = (int)di; }
        }
        if (bestDet >= 0) { claim[bestDet] = ti; trackDet[ti] = bestDet; }
    }

    // 4) Update matched, coast unmatched.
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        MtTrack& t = *tracks_[ti];
        if (trackDet[ti] >= 0) {
            const MatchResult& d = dets[trackDet[ti]];
            cv::Point2f c = centerOf(d.box);
            cv::Rect2f meas(c.x - targetSize_.width * 0.5f, c.y - targetSize_.height * 0.5f,
                            (float)targetSize_.width, (float)targetSize_.height);
            t.box  = t.kf.correct(meas);
            t.conf = std::max(0.0f, std::min(1.0f, d.psr / 12.0f));
            t.missed = 0;
            ++t.hits;
            t.status = (t.conf >= cfg_.strongThreshold) ? TrackStatus::TRACKING
                                                        : TrackStatus::UNCERTAIN;
            // Adapt the shared appearance only from the PRIMARY's confident match,
            // so a distractor can never drag the shared filter off the real target.
            if (t.primary && t.status == TrackStatus::TRACKING && cfg_.adapt)
                app_->updateTemplate(frame, toRectI(t.box), cfg_.templateLearnRate);
        } else {
            t.box = t.kf.stateBox();        // coast on prediction
            t.conf = 0.0f;
            ++t.missed;
            t.status = (t.missed > 1) ? TrackStatus::REACQUIRING : TrackStatus::LOST;
        }
    }

    // 5) Spawn a new track for any strong, unclaimed, well-separated detection.
    for (size_t di = 0; di < dets.size(); ++di) {
        if (claim[di] >= 0) continue;
        if (dets[di].psr < cfg_.strongThreshold * 12.0f) continue;   // strong only
        cv::Point2f c = centerOf(dets[di].box);
        bool nearExisting = false;
        for (const auto& t : tracks_)
            if (cv::norm(centerOf(t->box) - c) < std::min(targetSize_.width, targetSize_.height))
                { nearExisting = true; break; }
        if (nearExisting) continue;
        cv::Rect2f b(c.x - targetSize_.width * 0.5f, c.y - targetSize_.height * 0.5f,
                     (float)targetSize_.width, (float)targetSize_.height);
        spawn(b);
    }

    // 6) Cull stale distractors (never the primary -- it coasts indefinitely).
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                  [&](const std::unique_ptr<MtTrack>& t) {
                      return !t->primary && t->missed > maxMissed_;
                  }), tracks_.end());
}

// ----------------------------------------------------------------------------
// Run loop (multi-target). Mirrors TrackerEngine::run; draws every track so the
// identity-through-occlusion behaviour is visible.
// ----------------------------------------------------------------------------
static cv::Scalar statusColor(TrackStatus s) {
    switch (s) {
        case TrackStatus::TRACKING:    return cv::Scalar(0, 200, 0);
        case TrackStatus::UNCERTAIN:   return cv::Scalar(0, 200, 255);
        case TrackStatus::LOST:        return cv::Scalar(0, 0, 255);
        case TrackStatus::REACQUIRING: return cv::Scalar(0, 140, 255);
    }
    return cv::Scalar(200, 200, 200);
}

int runMultiTarget(VideoSource& source, const TrackerConfig& cfg) {
    cv::Mat frame;
    if (!source.read(frame)) { std::fprintf(stderr, "[multi] cannot read first frame\n"); return 3; }

    cv::Rect roi;
    if (cfg.hasInitRoi) {
        roi = cfg.initRoi;
    } else {
        const char* win = "Select target  |  drag a box, ENTER/SPACE to confirm";
        roi = cv::selectROI(win, frame, false, false);
        cv::destroyWindow(win);
    }
    if (roi.width < 4 || roi.height < 4) { std::fprintf(stderr, "[multi] invalid ROI\n"); return 3; }
    std::printf("init roi: %d,%d,%d,%d  (reuse with --roi %d,%d,%d,%d)\n",
                roi.x, roi.y, roi.width, roi.height, roi.x, roi.y, roi.width, roi.height);

    auto app = std::make_unique<MosseMatcher>(2.0, 8, /*anchor=*/0.0, /*prior=*/0.0,
                                              /*rotEnable=*/false);
    MultiTargetTracker mtt(cfg, std::move(app));
    mtt.init(frame, roi);

    cv::VideoWriter writer;
    {
        std::error_code ec;
        auto parent = std::filesystem::path(cfg.outputPath).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent, ec);
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(cfg.outputPath, fourcc, source.fps(), source.frameSize());
    }

    TrackerMetrics metrics;
    while (source.read(frame)) {
        auto t0 = std::chrono::high_resolution_clock::now();
        mtt.step(frame);
        auto t1 = std::chrono::high_resolution_clock::now();
        double frameMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        cv::Mat vis = frame.clone();
        const MtTrack* prim = mtt.primary();
        for (const auto& t : mtt.tracks()) {
            cv::Scalar col = statusColor(t->status);
            int thick = t->primary ? 2 : 1;
            cv::rectangle(vis, toRectI(t->box), col, thick, cv::LINE_AA);
            cv::Point2f c = centerOf(t->box);
            cv::circle(vis, c, t->primary ? 5 : 3, col, -1, cv::LINE_AA);
            char lbl[32]; std::snprintf(lbl, sizeof(lbl), "%s%d", t->primary ? "P" : "D", t->id);
            cv::putText(vis, lbl, cv::Point((int)t->box.x, (int)t->box.y - 4),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv::LINE_AA);
        }
        double fps = frameMs > 0 ? 1000.0 / frameMs : 0.0;
        char hud[160];
        std::snprintf(hud, sizeof(hud), "MULTI  tracks:%zu  %s  conf:%.2f  fps:%.0f",
                      mtt.tracks().size(), prim ? toString(prim->status) : "-",
                      prim ? prim->conf : 0.0f, fps);
        cv::putText(vis, hud, cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

        if (prim) metrics.record(prim->status, prim->conf, frameMs);
        if (writer.isOpened()) writer.write(vis);
        if (cfg.display) {
            cv::imshow("cpu-object-tracker [multi]", vis);
            int key = cv::waitKey(1);
            if (key == 27 || key == 'q') break;
        }
    }

    metrics.printSummary(source.fps());
    if (cfg.display) cv::destroyAllWindows();
    if (writer.isOpened()) std::printf("saved output video to: %s\n", cfg.outputPath.c_str());
    return 0;
}

} // namespace cot
