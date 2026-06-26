#pragma once
#include "Types.hpp"
#include "KalmanBoxTracker.hpp"
#include "MosseMatcher.hpp"
#include <memory>
#include <vector>

namespace cot {

// One motion track. Identity is carried by the Kalman state (a distinct velocity),
// NOT by appearance -- which is the whole point for near-identical objects.
struct MtTrack {
    int                id       = 0;
    KalmanBoxTracker   kf;
    cv::Rect2f         box;
    float              conf     = 0.0f;
    int                missed   = 0;           // consecutive frames without a match
    int                hits     = 0;
    bool               primary  = false;       // the user-initialized target
    TrackStatus        status   = TrackStatus::TRACKING;
};

// Multi-target tracker: ONE shared MOSSE appearance model feeds a heat map whose
// peaks are detections; a Kalman track per object claims a detection by motion
// consistency, so two look-alike objects keep separate identities through an
// overlap (they coast on different velocities and re-claim the right peak after).
class MultiTargetTracker {
public:
    MultiTargetTracker(TrackerConfig cfg, std::unique_ptr<MosseMatcher> appearance);

    void init(const cv::Mat& frame, const cv::Rect& primaryRoi);
    void step(const cv::Mat& frame);

    const std::vector<std::unique_ptr<MtTrack>>& tracks() const { return tracks_; }
    const MtTrack* primary() const;

private:
    cv::Rect computeRegion(const cv::Size& frameSize) const;  // union search area
    MtTrack* spawn(const cv::Rect2f& box);

    TrackerConfig                          cfg_;
    std::unique_ptr<MosseMatcher>          app_;       // shared appearance / heat map
    std::vector<std::unique_ptr<MtTrack>>  tracks_;
    cv::Size                               targetSize_{0, 0};
    int                                    nextId_     = 0;
    int                                    maxTracks_  = 4;
    int                                    maxMissed_  = 30;   // cull a distractor after this
};

// Drives the multi-target tracker over a video (mirrors TrackerEngine::run for the
// single-target path). Kept separate so the single-target path is untouched.
int runMultiTarget(class VideoSource& source, const TrackerConfig& cfg);

} // namespace cot
