#pragma once
#include <opencv2/core.hpp>
#include "Types.hpp"

namespace cot {

// Backend-agnostic matcher contract.
// v1 ships TemplateMatcher (NCC). A MOSSE/KCF correlation-filter backend can
// drop in behind this interface in v2 with no change to the engine.
class IMatcher {
public:
    virtual ~IMatcher() = default;

    // Capture the reference template from a frame region. Sets BOTH the
    // working template and the pristine "original" used during reacquisition.
    virtual void setTemplate(const cv::Mat& frame, const cv::Rect& box) = 0;

    // Slowly blend current appearance into the working template.
    virtual void updateTemplate(const cv::Mat& frame, const cv::Rect& box, float alpha) = 0;

    // Restore the working template to the pristine original (called on loss).
    virtual void useOriginalTemplate() = 0;

    // Optional per-frame motion context (predicted velocity, px/frame). Backends
    // that use a motion prior orient it with this; others ignore it.
    virtual void setMotionHint(const cv::Point2f& velocity) { (void)velocity; }

    // Optional hint: the supervisor is in re-detection/search mode (target may be
    // absent/occluded). Backends can suppress risky recovery heuristics.
    virtual void setReacquiring(bool reacquiring) { (void)reacquiring; }

    // Optional per-frame hint: are we in re-detection (post-loss) mode? Backends
    // can suppress appearance tricks (e.g. rotation recovery) that would otherwise
    // manufacture confidence on a distractor during an occlusion.
    virtual void setRedetecting(bool redetecting) { (void)redetecting; }

    // Search inside searchRegion (full-frame rect), trying scales around expectedSize.
    // Returned box is in FULL-FRAME coordinates.
    virtual MatchResult match(const cv::Mat& frame,
                              const cv::Rect& searchRegion,
                              const cv::Size& expectedSize,
                              const std::vector<float>& scales) = 0;

    virtual bool hasTemplate() const = 0;
};

} // namespace cot
