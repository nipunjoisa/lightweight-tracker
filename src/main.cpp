#include "TrackerEngine.hpp"
#include "TemplateMatcher.hpp"
#include "MosseMatcher.hpp"
#include "CvTrackerMatcher.hpp"
#include "VideoSource.hpp"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

using namespace cot;

static void usage(const char* prog) {
    std::printf("cpu-object-tracker  (C++17 / OpenCV, CPU-only)\n");
    std::printf("usage:\n");
    std::printf("  %s <video_path> [options]\n", prog);
    std::printf("  %s --camera <index> [options]\n", prog);
    std::printf("options:\n");
    std::printf("  --camera <i>      use camera index i instead of a file\n");
    std::printf("  --matcher <name>  appearance model: mosse (default) | csrt | kcf | ncc\n");
    std::printf("  --roi x,y,w,h     preset target box (skip manual selection; for fair A/B)\n");
    std::printf("  --learn <f>       online learn rate (default 0.05; MOSSE likes ~0.125)\n");
    std::printf("  --anchor <f>      MOSSE pull-back toward frame-1 filter (default 0 = off)\n");
    std::printf("  --prior <f>       velocity-oriented motion prior (default 0 = off)\n");
    std::printf("  --no-rotation     disable the log-polar rotation-recovery stage (MOSSE)\n");
    std::printf("  --no-adapt        freeze the template (diagnostic: isolates drift)\n");
    std::printf("  --output <path>   output video (default outputs/videos/tracked.mp4)\n");
    std::printf("  --reacq-max <f>   max search expansion when lost (default 6; lower = tighter\n");
    std::printf("                    re-search, better for cluttered/look-alike scenes)\n");
    std::printf("  --reacq-step <f>  search growth per lost frame (default 0.6)\n");
    std::printf("  --no-display      run headless (still writes the output video)\n");
    std::printf("  -h, --help        show this help\n");
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    TrackerConfig cfg;
    std::string videoPath;
    std::string matcherName = "mosse";
    double mosseAnchor = 0.0;   // off by default (plain MOSSE was best)
    double mossePrior  = 0.0;   // off by default
    bool   mosseRotation = true;  // log-polar rotation recovery on by default
    int cameraIndex = -1;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--camera" && i + 1 < argc)       cameraIndex = std::atoi(argv[++i]);
        else if (a == "--matcher" && i + 1 < argc) matcherName = argv[++i];
        else if (a == "--learn" && i + 1 < argc)   cfg.templateLearnRate = (float)std::atof(argv[++i]);
        else if (a == "--anchor" && i + 1 < argc)  mosseAnchor = std::atof(argv[++i]);
        else if (a == "--prior" && i + 1 < argc)   mossePrior = std::atof(argv[++i]);
        else if (a == "--reacq-max" && i + 1 < argc)  cfg.reacqMaxExpansion  = (float)std::atof(argv[++i]);
        else if (a == "--reacq-step" && i + 1 < argc) cfg.reacqExpansionStep = (float)std::atof(argv[++i]);
        else if (a == "--roi" && i + 1 < argc) {
            int x, y, w, h;
            if (std::sscanf(argv[++i], "%d,%d,%d,%d", &x, &y, &w, &h) == 4) {
                cfg.initRoi = cv::Rect(x, y, w, h);
                cfg.hasInitRoi = true;
            } else {
                std::fprintf(stderr, "bad --roi (expected x,y,w,h)\n");
            }
        }
        else if (a == "--no-adapt")                cfg.adapt = false;
        else if (a == "--no-rotation")             mosseRotation = false;
        else if (a == "--output" && i + 1 < argc)  cfg.outputPath = argv[++i];
        else if (a == "--no-display")              cfg.display = false;
        else if (a == "-h" || a == "--help")     { usage(argv[0]); return 0; }
        else if (a.rfind("--", 0) != 0)            videoPath = a;
        else std::fprintf(stderr, "unknown option: %s\n", a.c_str());
    }

    VideoSource source;
    bool ok = (cameraIndex >= 0) ? source.openCamera(cameraIndex)
                                 : source.openFile(videoPath);
    if (!ok || !source.isOpened()) {
        std::fprintf(stderr, "error: could not open %s\n",
                     cameraIndex >= 0 ? "camera" : videoPath.c_str());
        return 2;
    }

    std::unique_ptr<IMatcher> matcher;
    if (matcherName == "ncc")       matcher = std::make_unique<TemplateMatcher>();
    else if (matcherName == "csrt") matcher = std::make_unique<CvTrackerMatcher>(CvTrackerMatcher::Kind::CSRT);
    else if (matcherName == "kcf")  matcher = std::make_unique<CvTrackerMatcher>(CvTrackerMatcher::Kind::KCF);
    else                            matcher = std::make_unique<MosseMatcher>(2.0, 8, mosseAnchor, mossePrior, mosseRotation);
    std::printf("matcher: %s\n", matcherName.c_str());

    TrackerEngine engine(cfg, std::move(matcher));
    return engine.run(source) ? 0 : 3;
}
