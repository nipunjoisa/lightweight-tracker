#pragma once
#include "IMatcher.hpp"

namespace cot {

// MOSSE adaptive correlation filter (Bolme et al., 2010) behind IMatcher.
// A fixed-size complex filter is learned in the Fourier domain and updated
// online, so it adapts to appearance change (the bike) and tolerates
// orientation change (the missile turn) far better than rigid NCC.
// PSR is MOSSE's native quality signal and drives the confidence directly.
class MosseMatcher : public IMatcher {
public:
    explicit MosseMatcher(double sigma = 2.0, int initWarps = 8,
                          double anchor = 0.02, double priorFrac = 1.0);

    void setTemplate(const cv::Mat& frame, const cv::Rect& box) override;
    void updateTemplate(const cv::Mat& frame, const cv::Rect& box, float alpha) override;
    void useOriginalTemplate() override;
    MatchResult match(const cv::Mat& frame,
                      const cv::Rect& searchRegion,
                      const cv::Size& expectedSize,
                      const std::vector<float>& scales) override;
    bool hasTemplate() const override { return !H_.empty(); }

private:
    static cv::Mat toGray(const cv::Mat& frame);
    static cv::Mat divSpectrums(const cv::Mat& A, const cv::Mat& B);  // complex A/B
    static cv::Mat rndWarp(const cv::Mat& a, cv::RNG& rng);
    cv::Mat preprocess(const cv::Mat& patch32f) const;               // log + normalize + window
    void    updateKernel();                                          // H_ = conj(H1_/H2_)
    void    correlate(const cv::Mat& patch32f,
                      cv::Mat& resp, cv::Point& peak, float& psr) const;

    cv::Size fsize_;          // filter / target size
    cv::Mat  win_;            // Hanning window (CV_32F, fsize_)
    cv::Mat  G_;              // FFT of the desired gaussian response (CV_32FC2)
    cv::Mat  H1_, H2_, H_;    // running numerator, denominator, kernel (CV_32FC2)
    cv::Mat  H1_0_, H2_0_;    // pristine init, restored on reacquisition
    cv::Mat  prior_;          // Gaussian motion prior over the response (may be empty)
    double   sigma_;
    int      initWarps_;
    double   anchor_;         // per-update pull back toward the original filter
    double   priorFrac_;      // motion-prior width as a fraction of the half-window
};

} // namespace cot
