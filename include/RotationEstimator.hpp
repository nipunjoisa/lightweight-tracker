#pragma once
#include <opencv2/core.hpp>

namespace cot {

// Estimates in-plane rotation (and scale) between two equally-sized single-channel
// patches via a log-polar transform + phase correlation. In log-polar space an
// in-plane rotation becomes a vertical (angular) shift and a scale change becomes a
// horizontal (log-radius) shift, both recovered by one FFT-based phase correlation
// (the Fourier-Mellin / LDES idea). Reuses the same FFT machinery the tracker runs.
class RotationEstimator {
public:
    struct Result {
        double angleDeg = 0.0;   // estimated rotation of `cur` relative to `ref`
        double scale    = 1.0;   // estimated scale of `cur` relative to `ref`
        double response = 0.0;   // phase-correlation peak strength (0..1-ish)
    };
    Result estimate(const cv::Mat& ref, const cv::Mat& cur) const;
};

} // namespace cot
