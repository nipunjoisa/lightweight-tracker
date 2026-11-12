"""Command-line entry point: `python -m tracker [video] [options]`."""
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List, Optional

from .engine import TrackerEngine, VideoSource
from .matchers import CvTrackerMatcher, TemplateMatcher
from .mosse import MosseMatcher
from .multi import run_multi_target
from .types import TrackerConfig


def pick_video_file() -> str:
    """Native file-picker dialog (macOS via osascript, Linux via zenity).

    Falls back to tkinter if neither is available. Returns "" if cancelled.
    """
    try:
        if sys.platform == "darwin":
            script = ('POSIX path of (choose file with prompt "Select a video to track" '
                      'of type {"public.movie", "public.video"})')
            cmd = ["osascript", "-e", script]
        else:
            cmd = ["zenity", "--file-selection", "--title=Select a video to track"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.stdout.strip() if res.returncode == 0 else ""
    except FileNotFoundError:
        try:
            import tkinter
            from tkinter import filedialog
            root = tkinter.Tk()
            root.withdraw()
            return filedialog.askopenfilename(title="Select a video to track")
        except Exception:
            return ""


def parse_roi(text: str):
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected x,y,w,h")
    return tuple(parts)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tracker",
        description="CPU-only object tracker (Python / OpenCV). "
                    "No video path = open a file picker.")
    p.add_argument("video", nargs="?", help="path to a video file (omit for file picker)")
    p.add_argument("--camera", type=int, default=None, help="use camera index instead of a file")
    p.add_argument("--matcher", default="mosse", choices=["mosse", "csrt", "kcf", "ncc"],
                   help="appearance model (default: mosse)")
    p.add_argument("--roi", type=parse_roi, metavar="x,y,w,h",
                   help="preset target box (skip manual selection; for fair A/B)")
    p.add_argument("--learn", type=float, default=0.05,
                   help="online learn rate (default 0.05; MOSSE likes ~0.125)")
    p.add_argument("--anchor", type=float, default=0.0,
                   help="MOSSE pull-back toward the frame-1 filter (0 = off)")
    p.add_argument("--prior", type=float, default=0.0,
                   help="velocity-oriented motion prior width (0 = off)")
    p.add_argument("--no-rotation", action="store_true",
                   help="disable the log-polar rotation-recovery stage (MOSSE)")
    p.add_argument("--multi", action="store_true",
                   help="multi-target mode: keeps identities apart by motion")
    p.add_argument("--no-adapt", action="store_true",
                   help="freeze the template (diagnostic: isolates drift)")
    p.add_argument("--output", default="outputs/videos/tracked.mp4", help="output video path")
    p.add_argument("--reacq-max", type=float, default=6.0,
                   help="max search expansion when lost (default 6)")
    p.add_argument("--reacq-step", type=float, default=0.6,
                   help="search growth per lost frame (default 0.6)")
    p.add_argument("--no-display", action="store_true",
                   help="run headless (still writes the output video)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = TrackerConfig(
        template_learn_rate=args.learn,
        reacq_max_expansion=args.reacq_max,
        reacq_expansion_step=args.reacq_step,
        adapt=not args.no_adapt,
        display=not args.no_display,
        output_path=args.output,
        init_roi=args.roi,
    )

    video = args.video
    if args.camera is None and not video:
        print("no video given - opening file picker...")
        video = pick_video_file()
        if not video:
            print("no video selected.", file=sys.stderr)
            build_parser().print_usage()
            return 1
        print(f"selected: {video}")

    source = VideoSource()
    ok = source.open_camera(args.camera) if args.camera is not None else source.open_file(video)
    if not ok or not source.is_opened():
        print(f"error: could not open {'camera' if args.camera is not None else video}",
              file=sys.stderr)
        return 2

    if args.multi:
        print("matcher: mosse (multi-target)")
        return run_multi_target(source, cfg)

    if args.matcher == "ncc":
        matcher = TemplateMatcher()
    elif args.matcher in ("csrt", "kcf"):
        matcher = CvTrackerMatcher(args.matcher)
    else:
        matcher = MosseMatcher(2.0, 8, args.anchor, args.prior, not args.no_rotation)
    print(f"matcher: {args.matcher}")

    return 0 if TrackerEngine(cfg, matcher).run(source) else 3


if __name__ == "__main__":
    sys.exit(main())
