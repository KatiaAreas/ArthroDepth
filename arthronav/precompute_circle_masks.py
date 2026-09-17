"""
Detect the endoscope circle on EVERY frame of every prepared real-knee
clip (unlike SOBONE's precompute_crop_boxes.py, which detects once per
sequence -- the notch position moves during real surgery, so the circle
genuinely changes frame to frame here, confirmed empirically before
building this).

Run standalone, before any script that imports torch -- same reason as
SOBONE's version: opencv-contrib-python's bundled cuDNN clashes with
PyTorch's cuDNN if both load in the same process
(CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH).

Writes one JSON per clip: {frame_idx: {center, radius, box}} or
{frame_idx: null} for frames where detection failed on all attempts
(including the nearest-neighbor fallback -- these need a manual check
or should be excluded from training, not silently guessed at).

Usage:
    python -m arthronav.precompute_circle_masks --patient 2602446F_G --view medial
    python -m arthronav.precompute_circle_masks --all-prepared-clips
"""

import argparse
import glob
import json
import os
import re

import cv2
import numpy as np
from PIL import Image
from areas_theta_compute.utils.circle_detector import CircleDetector

DATA_ROOT = "/mnt/areas_nas/SLAM/real_knee_dataset"
PATCH_SIZE = 14
FRAME_RE = re.compile(r"frame_(\d{6})\.jpg$")


def detect_on_frame(detector, rgb_path):
    img = np.array(Image.open(rgb_path).convert("RGB"))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    result = detector.detect_image_from_array(bgr)
    if not result or result["circle_count"] == 0:
        return None
    circle = max(result["circles"], key=lambda c: c["confidence"])
    return circle["center"][0], circle["center"][1], circle["radius"], img.shape[:2]


def compute_box(cx, cy, radius, frame_hw):
    h, w = frame_hw
    diameter = int(2 * radius)
    side = (diameter // PATCH_SIZE) * PATCH_SIZE

    x0 = int(cx - side / 2)
    y0 = int(cy - side / 2)
    x1, y1 = x0 + side, y0 + side

    if x0 < 0:
        x1 -= x0; x0 = 0
    if y0 < 0:
        y1 -= y0; y0 = 0
    if x1 > w:
        x0 -= (x1 - w); x1 = w
    if y1 > h:
        y0 -= (y1 - h); y1 = h

    return [x0, y0, x1, y1]


def process_clip(detector, clip_dir, clip_name):
    rgb_dir = os.path.join(clip_dir, "rgb")
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
    frame_indices = sorted(int(FRAME_RE.search(f).group(1)) for f in rgb_files if FRAME_RE.search(f))

    print(f"\n=== {clip_name}: {len(frame_indices)} frames ===")
    results = {}
    last_good = None
    n_direct, n_fallback, n_failed = 0, 0, 0

    for i, frame_idx in enumerate(frame_indices):
        rgb_path = os.path.join(rgb_dir, f"frame_{frame_idx:06d}.jpg")
        detected = detect_on_frame(detector, rgb_path)

        if detected is not None:
            cx, cy, radius, frame_hw = detected
            box = compute_box(cx, cy, radius, frame_hw)
            results[frame_idx] = {"center": [cx, cy], "radius": radius, "box": box, "fallback": False}
            last_good = results[frame_idx]
            n_direct += 1
        elif last_good is not None:
            results[frame_idx] = {**last_good, "fallback": True}
            n_fallback += 1
        else:
            results[frame_idx] = None
            n_failed += 1

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(frame_indices)} | direct={n_direct} fallback={n_fallback} failed={n_failed}")

    print(f"Done: direct={n_direct}, fallback={n_fallback}, failed={n_failed} (of {len(frame_indices)})")
    if n_failed > 0:
        print(f"  WARNING: {n_failed} frames had no detection at all (no prior successful "
              f"frame to fall back on, e.g. failures right at the start of the clip) -- "
              f"these need a manual check before being used for training.")

    out_path = os.path.join(clip_dir, "circle_boxes.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", type=str, default=None)
    ap.add_argument("--view", type=str, default=None, choices=["lateral", "medial"])
    ap.add_argument("--all-prepared-clips", action="store_true")
    ap.add_argument("--data-root", type=str, default=DATA_ROOT)
    args = ap.parse_args()

    detector = CircleDetector()

    if args.all_prepared_clips:
        clip_dirs = sorted(glob.glob(os.path.join(args.data_root, "*")))
        clip_dirs = [d for d in clip_dirs if os.path.isdir(os.path.join(d, "rgb"))]
        for clip_dir in clip_dirs:
            process_clip(detector, clip_dir, os.path.basename(clip_dir))
    else:
        if not args.patient or not args.view:
            raise SystemExit("Provide --patient and --view, or use --all-prepared-clips")
        clip_name = f"{args.patient}_{args.view}"
        clip_dir = os.path.join(args.data_root, clip_name)
        if not os.path.isdir(clip_dir):
            raise SystemExit(f"No such clip: {clip_dir}")
        process_clip(detector, clip_dir, clip_name)


if __name__ == "__main__":
    main()
