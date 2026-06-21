#include "MosseMatcher.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace cot {

static float clamp01(float v) { return std::max(0.0f, std::min(1.0f, v)); }

MosseMatcher::MosseMatcher(double sigma, int initWarps, double anchor, double priorFrac)
    : sigma_(sigma), initWarps_(initWarps), anchor_(anchor), priorFrac_(priorFrac) {}

cv::Mat MosseMatcher::toGray(const cv::Mat& frame) {
    if (frame.channels() == 1) return frame;
    cv::Mat g;
    cv::cvtColor(frame, g, cv::COLOR_BGR2GRAY);
    return g;
}

// Element-wise complex division of two CV_32FC2 spectra: (a+bi)/(c+di).
cv::Mat MosseMatcher::divSpectrums(const cv::Mat& A, const cv::Mat& B) {
    CV_Assert(A.size() == B.size() && A.type() == CV_32FC2 && B.type() == CV_32FC2);
    cv::Mat out(A.size(), CV_32FC2);
    const float eps = 1e-6f;
    for (int y = 0; y < A.rows; ++y) {
        const cv::Vec2f* a = A.ptr<cv::Vec2f>(y);
        const cv::Vec2f* b = B.ptr<cv::Vec2f>(y);
        cv::Vec2f* o = out.ptr<cv::Vec2f>(y);
        for (int x = 0; x < A.cols; ++x) {
            float den = b[x][0] * b[x][0] + b[x][1] * b[x][1] + eps;
            o[x][0] = (a[x][0] * b[x][0] + a[x][1] * b[x][1]) / den;
            o[x][1] = (a[x][1] * b[x][0] - a[x][0] * b[x][1]) / den;
        }
    }
    return out;
}

// Small random affine (rotation + scale about the center) for robust filter init.
cv::Mat MosseMatcher::rndWarp(const cv::Mat& a, cv::RNG& rng) {
    int w = a.cols, h = a.rows;
    double ang   = (rng.uniform(0.0, 1.0) - 0.5) * 0.2;
    double scale = 1.0 + (rng.uniform(0.0, 1.0) - 0.5) * 0.2;
    double c = std::cos(ang) * scale, s = std::sin(ang) * scale;
    cv::Mat T = (cv::Mat_<double>(2, 3) << c, -s, 0, s, c, 0);
    double cx = w * 0.5, cy = h * 0.5;
    T.at<double>(0, 2) = cx - (c * cx - s * cy);
    T.at<double>(1, 2) = cy - (s * cx + c * cy);
    cv::Mat out;
    cv::warpAffine(a, out, T, a.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    return out;
}

cv::Mat MosseMatcher::preprocess(const cv::Mat& patch) const {
    cv::Mat f;
    patch.convertTo(f, CV_32F);
    cv::add(f, cv::Scalar::all(1.0), f);
    cv::Mat l;
    cv::log(f, l);                       // log compress dynamic range (MOSSE preprocessing)
    cv::Scalar mean, stddev;
    cv::meanStdDev(l, mean, stddev);
    l = (l - mean[0]) / (stddev[0] + 1e-5);
    if (win_.size() != l.size()) {       // safety (sizes always match in practice)
        cv::Mat w; cv::resize(win_, w, l.size());
        return l.mul(w);
    }
    return l.mul(win_);                  // cosine window kills the FFT edge effects
}

void MosseMatcher::updateKernel() {
    H_ = divSpectrums(H1_, H2_);
    std::vector<cv::Mat> ch;             // conjugate (mirrors the OpenCV MOSSE sample)
    cv::split(H_, ch);
    ch[1] *= -1.0f;
    cv::merge(ch, H_);
}

void MosseMatcher::setTemplate(const cv::Mat& frame, const cv::Rect& box) {
    cv::Mat gray = toGray(frame);
    cv::Rect b = box & cv::Rect(0, 0, gray.cols, gray.rows);
    fsize_ = cv::Size(std::max(8, b.width), std::max(8, b.height));
    cv::Point2f center(b.x + b.width * 0.5f, b.y + b.height * 0.5f);

    cv::createHanningWindow(win_, fsize_, CV_32F);

    // Motion prior: a Gaussian over the response centered at zero-shift (the
    // Kalman-predicted location). Multiplying the response by this biases peak
    // selection toward the predicted path, so a persistent off-center pull
    // (smoke trail below the missile, an off-path bike) can't slowly drag the
    // lock away. priorFrac_ <= 0 disables it.
    if (priorFrac_ > 0.0) {
        prior_ = cv::Mat(fsize_, CV_32F);
        double sig = priorFrac_ * 0.5 * std::min(fsize_.width, fsize_.height);
        double ps2 = 2.0 * sig * sig + 1e-6;
        float pcx = fsize_.width * 0.5f, pcy = fsize_.height * 0.5f;
        for (int y = 0; y < fsize_.height; ++y) {
            float* pp = prior_.ptr<float>(y);
            for (int x = 0; x < fsize_.width; ++x) {
                float dx = x - pcx, dy = y - pcy;
                pp[x] = static_cast<float>(std::exp(-(dx * dx + dy * dy) / ps2));
            }
        }
    } else {
        prior_.release();
    }

    // Desired correlation output: a Gaussian peak at the window center.
    cv::Mat g(fsize_, CV_32F);
    float gx = fsize_.width * 0.5f, gy = fsize_.height * 0.5f;
    double s2 = 2.0 * sigma_ * sigma_;
    for (int y = 0; y < fsize_.height; ++y) {
        float* gp = g.ptr<float>(y);
        for (int x = 0; x < fsize_.width; ++x) {
            float dx = x - gx, dy = y - gy;
            gp[x] = static_cast<float>(std::exp(-(dx * dx + dy * dy) / s2));
        }
    }
    cv::dft(g, G_, cv::DFT_COMPLEX_OUTPUT);

    // Initialize the filter from the patch plus several random warps.
    cv::Mat raw;
    cv::getRectSubPix(gray, fsize_, center, raw, CV_32F);
    H1_ = cv::Mat::zeros(G_.size(), CV_32FC2);
    H2_ = cv::Mat::zeros(G_.size(), CV_32FC2);
    cv::RNG rng(12345);
    for (int i = 0; i < std::max(1, initWarps_); ++i) {
        cv::Mat a = (i == 0) ? raw.clone() : rndWarp(raw, rng);
        cv::Mat F; cv::dft(preprocess(a), F, cv::DFT_COMPLEX_OUTPUT);
        cv::Mat A, B;
        cv::mulSpectrums(G_, F, A, 0, true);   // G * conj(F)
        cv::mulSpectrums(F,  F, B, 0, true);   // F * conj(F)
        H1_ += A;
        H2_ += B;
    }
    H1_0_ = H1_.clone();
    H2_0_ = H2_.clone();
    updateKernel();
}

void MosseMatcher::useOriginalTemplate() {
    if (H1_0_.empty()) return;
    H1_ = H1_0_.clone();
    H2_ = H2_0_.clone();
    updateKernel();
}

void MosseMatcher::updateTemplate(const cv::Mat& frame, const cv::Rect& box, float alpha) {
    if (H_.empty()) return;
    cv::Mat gray = toGray(frame);
    cv::Size bs(std::max(8, box.width), std::max(8, box.height));
    cv::Point2f center(box.x + box.width * 0.5f, box.y + box.height * 0.5f);

    cv::Mat raw;
    cv::getRectSubPix(gray, bs, center, raw, CV_32F);
    if (raw.size() != fsize_) { cv::Mat r; cv::resize(raw, r, fsize_); raw = r; }

    cv::Mat F; cv::dft(preprocess(raw), F, cv::DFT_COMPLEX_OUTPUT);
    cv::Mat A, B;
    cv::mulSpectrums(G_, F, A, 0, true);
    cv::mulSpectrums(F,  F, B, 0, true);

    double a = clamp01(alpha);           // online blend; MOSSE typically ~0.125
    cv::addWeighted(H1_, 1.0 - a, A, a, 0.0, H1_);
    cv::addWeighted(H2_, 1.0 - a, B, a, 0.0, H2_);

    // Anti-drift: tether the filter to the original frame-1 appearance so it
    // can't be slowly dragged onto an adjacent high-contrast feature (e.g. a
    // missile's exhaust trail) while adapting.
    if (anchor_ > 0.0 && !H1_0_.empty()) {
        cv::addWeighted(H1_, 1.0 - anchor_, H1_0_, anchor_, 0.0, H1_);
        cv::addWeighted(H2_, 1.0 - anchor_, H2_0_, anchor_, 0.0, H2_);
    }
    updateKernel();
}

void MosseMatcher::correlate(const cv::Mat& patch, cv::Mat& resp,
                             cv::Point& peak, float& psr) const {
    cv::Mat F; cv::dft(patch, F, cv::DFT_COMPLEX_OUTPUT);
    cv::Mat C; cv::mulSpectrums(F, H_, C, 0, true);          // F * conj(H_)
    cv::idft(C, resp, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

    // Bias peak selection toward the predicted location with the motion prior;
    // measure PSR on the RAW response so the prior doesn't inflate confidence.
    cv::Mat scored = prior_.empty() ? resp : resp.mul(prior_);
    double mn, mx; cv::Point mnLoc, mxLoc;
    cv::minMaxLoc(scored, &mn, &mx, &mnLoc, &mxLoc);
    peak = mxLoc;

    double peakVal = resp.at<float>(peak);
    cv::Mat mask = cv::Mat::ones(resp.size(), CV_8U);
    const int win = 5;
    cv::Rect excl(peak.x - win, peak.y - win, 2 * win + 1, 2 * win + 1);
    excl &= cv::Rect(0, 0, resp.cols, resp.rows);
    if (excl.area() > 0) mask(excl).setTo(0);
    cv::Scalar m, sd;
    cv::meanStdDev(resp, m, sd, mask);
    psr = static_cast<float>((peakVal - m[0]) / (sd[0] + 1e-6));
}

MatchResult MosseMatcher::match(const cv::Mat& frame,
                                const cv::Rect& searchRegion,
                                const cv::Size& expectedSize,
                                const std::vector<float>& scales) {
    (void)expectedSize;  // MOSSE scale is handled via `scales` around fsize_
    MatchResult out;
    if (H_.empty()) return out;

    cv::Mat gray = toGray(frame);
    cv::Rect roi = searchRegion & cv::Rect(0, 0, gray.cols, gray.rows);
    if (roi.width < 4 || roi.height < 4) return out;

    // Candidate eval centers. While locked the ROI is ~2x the box -> a single
    // center (the prediction) with a full scale sweep. Once the ROI is widened
    // for reacquisition we tile it with overlapping filter-sized windows and
    // scan for the target, since MOSSE can't extend its search by downsampling.
    std::vector<cv::Point2f> centers;
    const bool tile = (roi.width > fsize_.width * 2.5) || (roi.height > fsize_.height * 2.5);
    if (!tile) {
        centers.emplace_back(roi.x + roi.width * 0.5f, roi.y + roi.height * 0.5f);
    } else {
        float sx = std::max(1.0f, fsize_.width * 0.5f);
        float sy = std::max(1.0f, fsize_.height * 0.5f);
        for (float cy = roi.y + fsize_.height * 0.5f; cy <= roi.y + roi.height - fsize_.height * 0.5f; cy += sy)
            for (float cx = roi.x + fsize_.width * 0.5f; cx <= roi.x + roi.width - fsize_.width * 0.5f; cx += sx)
                centers.emplace_back(cx, cy);
        if (centers.empty())
            centers.emplace_back(roi.x + roi.width * 0.5f, roi.y + roi.height * 0.5f);
    }
    const std::vector<float> scaleSweep = tile ? std::vector<float>{1.0f} : scales;

    float bestPsr = -1e9f, bestScale = 0.0f;
    cv::Point2f bestCenter;
    cv::Point bestPeak;

    for (const cv::Point2f& c : centers) {
        for (float s : scaleSweep) {
            cv::Size ss(std::max(4, (int)std::lround(fsize_.width * s)),
                        std::max(4, (int)std::lround(fsize_.height * s)));
            cv::Mat raw; cv::getRectSubPix(gray, ss, c, raw, CV_32F);
            if (raw.size() != fsize_) { cv::Mat r; cv::resize(raw, r, fsize_); raw = r; }

            cv::Mat resp; cv::Point peak; float psr;
            correlate(preprocess(raw), resp, peak, psr);
            if (psr > bestPsr) {
                bestPsr = psr; bestScale = s; bestCenter = c; bestPeak = peak;
            }
        }
    }
    if (bestScale <= 0.0f) return out;

    // Peak offset (relative to the filter center) maps back to image pixels via
    // the scale at which we sampled.
    float dx = (bestPeak.x - fsize_.width  * 0.5f) * bestScale;
    float dy = (bestPeak.y - fsize_.height * 0.5f) * bestScale;
    float cx = bestCenter.x + dx, cy = bestCenter.y + dy;
    float bw = fsize_.width * bestScale, bh = fsize_.height * bestScale;

    out.box = cv::Rect2f(cx - bw * 0.5f, cy - bh * 0.5f, bw, bh);
    out.psr = bestPsr;
    // MOSSE confidence is PSR-driven. Encode it into ncc as well so the existing
    // ConfidenceEvaluator (0.5*ncc + 0.5*psr/sat) yields ~clamp(psr/12) with the
    // default weights/saturation. Lower psrSaturation in Types.hpp to loosen.
    out.ncc = clamp01(bestPsr / 12.0f);
    out.valid = true;
    return out;
}

} // namespace cot
