"""
Detect the endoscope circle once per sequence, using the FIRST N frames
as candidates (falls back to the next frame if one fails detection),
and cache the result to a JSON file. Run this standalone -- it never
imports torch, deliberately, so opencv-contrib-python's bundled cuDNN
never shares a process with PyTorch's cuDNN (that clash is what broke
the training runs).

Usage:
    python -m arthronav.precompute_crop_boxes
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from areas_theta_compute.utils.circle_detector import CircleDetector

SOBONE_ROOT = Path("/mnt/areas_nas/SLAM/sobone_dataset")
PATCH_SIZE = 14
CANDIDATE_FRAMES_TO_TRY = 5  # try up to this many frames per sequence before giving up

SEQUENCES = [
    "tour_lateral_0_cartilage", "tour_lateral_1_cartilage",
    "tour_lateral_2_cartilage", "tour_lateral_3_cartilage",
    "tour_medial_0_cartilage", "tour_medial_1_cartilage",
    "tour_medial_2_cartilage", "tour_medial_3_cartilage",
]


def detect_on_frame(detector, rgb_path):
    import cv2
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


def main():
    detector = CircleDetector()
    results = {}

    for seq_name in SEQUENCES:
        seq_dir = SOBONE_ROOT / seq_name
        rgb_files = sorted((seq_dir / "rgb").glob("*.png"))[:CANDIDATE_FRAMES_TO_TRY]

        detected = None
        for rgb_path in rgb_files:
            detected = detect_on_frame(detector, rgb_path)
            if detected is not None:
                cx, cy, radius, frame_hw = detected
                box = compute_box(cx, cy, radius, frame_hw)
                results[seq_name] = {
                    "center": [cx, cy], "radius": radius,
                    "box": box, "detected_on": rgb_path.name,
                }
                print(f"{seq_name}: OK on {rgb_path.name}, center=({cx:.1f},{cy:.1f}), "
                      f"radius={radius:.1f}, box={box}")
                break

        if detected is None:
            print(f"{seq_name}: FAILED on all {len(rgb_files)} candidate frames -- needs manual check")

    out_path = Path(__file__).parent / "sobone_crop_boxes.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path} ({len(results)}/{len(SEQUENCES)} sequences succeeded)")


if __name__ == "__main__":
    main()
