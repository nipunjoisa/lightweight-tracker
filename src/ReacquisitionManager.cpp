#include "ReacquisitionManager.hpp"
#include <algorithm>

namespace cot {

void ReacquisitionManager::reset() {
    lostFrames_      = 0;
    expansion_       = cfg_.searchExpansion;
    justEnteredLoss_ = false;
}

float ReacquisitionManager::onStatus(TrackStatus status) {
    justEnteredLoss_ = false;

    if (status == TrackStatus::LOST || status == TrackStatus::REACQUIRING) {
        if (lostFrames_ == 0) justEnteredLoss_ = true;
        ++lostFrames_;
        // Widen the net the longer we stay lost, up to a hard cap.
        expansion_ = std::min(cfg_.reacqMaxExpansion,
                              cfg_.searchExpansion + cfg_.reacqExpansionStep * lostFrames_);
    } else {
        lostFrames_ = 0;
        expansion_  = cfg_.searchExpansion;
    }
    return expansion_;
}

} // namespace cot
