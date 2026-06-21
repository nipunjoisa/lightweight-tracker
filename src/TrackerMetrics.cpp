#include "TrackerMetrics.hpp"
#include <algorithm>
#include <numeric>
#include <cstdio>

namespace cot {

void TrackerMetrics::record(TrackStatus status, float confidence, double frameMs) {
    ++total_;
    confSum_ += confidence;
    frameMs_.push_back(frameMs);

    const bool locked = (status == TrackStatus::TRACKING || status == TrackStatus::UNCERTAIN);

    switch (status) {
        case TrackStatus::TRACKING:    ++tracking_;    break;
        case TrackStatus::UNCERTAIN:   ++uncertain_;   break;
        case TrackStatus::LOST:        ++lost_;        break;
        case TrackStatus::REACQUIRING: ++reacquiring_; break;
    }

    if (locked) {
        if (wasLost_) ++reacquisitions_;  // recovered from a loss
        ++curStreak_;
        longestStreak_ = std::max(longestStreak_, curStreak_);
        wasLost_ = false;
    } else {
        curStreak_ = 0;
        wasLost_   = true;
    }
}

void TrackerMetrics::printSummary(double sourceFps) const {
    if (total_ == 0) { std::printf("\n[metrics] no frames processed\n"); return; }

    std::vector<double> ms = frameMs_;
    std::sort(ms.begin(), ms.end());
    double sum    = std::accumulate(ms.begin(), ms.end(), 0.0);
    double meanMs = sum / ms.size();
    double medMs  = ms[ms.size() / 2];
    double meanFps = meanMs > 0 ? 1000.0 / meanMs : 0.0;
    double medFps  = medMs  > 0 ? 1000.0 / medMs  : 0.0;

    auto pct = [&](int n) { return 100.0 * n / total_; };

    std::printf("\n================ tracker metrics ================\n");
    std::printf("frames processed     : %d\n", total_);
    std::printf("source fps           : %.1f\n", sourceFps);
    std::printf("processing fps (mean): %.1f\n", meanFps);
    std::printf("processing fps (med) : %.1f\n", medFps);
    std::printf("mean confidence      : %.3f\n", confSum_ / total_);
    std::printf("TRACKING             : %d (%.1f%%)\n", tracking_,    pct(tracking_));
    std::printf("UNCERTAIN            : %d (%.1f%%)\n", uncertain_,   pct(uncertain_));
    std::printf("LOST                 : %d (%.1f%%)\n", lost_,        pct(lost_));
    std::printf("REACQUIRING          : %d (%.1f%%)\n", reacquiring_, pct(reacquiring_));
    std::printf("longest locked streak: %d frames\n", longestStreak_);
    std::printf("reacquisitions       : %d\n", reacquisitions_);
    std::printf("=================================================\n");
}

} // namespace cot
