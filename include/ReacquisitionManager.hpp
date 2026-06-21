#pragma once
#include "Types.hpp"

namespace cot {

// Owns the "what to do when lost" policy: grow the search region every lost
// frame (up to a cap) and signal when to fall back to the pristine template.
class ReacquisitionManager {
public:
    explicit ReacquisitionManager(const TrackerConfig& cfg) : cfg_(cfg) {}

    void reset();
    // Feed the current frame's status; returns the search expansion factor to
    // use for the NEXT search.
    float onStatus(TrackStatus status);

    int  lostFrames() const { return lostFrames_; }
    bool justEnteredLoss() const { return justEnteredLoss_; }

private:
    TrackerConfig cfg_;
    int   lostFrames_ = 0;
    float expansion_  = 0.0f;
    bool  justEnteredLoss_ = false;
};

} // namespace cot
