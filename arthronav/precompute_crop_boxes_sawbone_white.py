"""
Detect the endoscope circle once per sequence for the NEW white sawbone
dataset (case_sawbone_white_2, a different physical phantom than the
original red sawbone -- confirmed empirically that the circle is static
within a sequence here too, center/radius stable within a few pixels
across 5 widely-spread frames of the same sequence, same as the
original sawbone benchtop setup, unlike the real surgical footage where
it moves).

Run standalone, before any script imports torch -- same cuDNN-conflict
reason as precompute_crop_boxes.py (opencv-contrib-python's bundled
cuDNN clashes with PyTorch's own).

Usage:
    python -m arthronav.precompute_crop_boxes_sawbone_white
"""

import json
from pathlib import Path
import zipfile
import io

import numpy as np
from PIL import Image
from areas_theta_compute.utils.circle_detector import CircleDetector

ZIP_PATH = "/mnt/areas_nas/SLAM/sawbone_white_depth.zip"
PATCH_SIZE = 14
CANDIDATE_FRAMES_TO_TRY = 5

SEQUENCES = [
    "20260917-134858-eeeeeeee", "20260917-134939-eeeeeeee", "20260917-135011-eeeeeeee",
    "20260917-135034-eeeeeeee", "20260917-135100-eeeeeeee", "20260917-135126-eeeeeeee",
    "20260917-135156-eeeeeeee", "20260917-135219-eeeeeeee", "20260917-135304-eeeeeeee",
    "20260917-135325-eeeeeeee",
]


def detect_on_frame_bytes(detector, data):
    import cv2
    img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
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
    z = zipfile.ZipFile(ZIP_PATH)
    results = {}

    for seq_name in SEQUENCES:
        rgb_names = sorted(n for n in z.namelist()
                            if f"depth/{seq_name}/rgb/" in n and n.endswith(".png"))
        candidates = rgb_names[:CANDIDATE_FRAMES_TO_TRY]

        detected = None
        used_name = None
        for name in candidates:
            data = z.read(name)
            detected = detect_on_frame_bytes(detector, data)
            if detected is not None:
                used_name = name
                break

        if detected is None:
            print(f"{seq_name}: FAILED on all {len(candidates)} candidate frames -- needs manual check")
            continue

        cx, cy, radius, frame_hw = detected
        box = compute_box(cx, cy, radius, frame_hw)
        results[seq_name] = {
            "center": [cx, cy], "radius": radius,
            "box": box, "detected_on": used_name,
        }
        print(f"{seq_name}: OK on {used_name}, center=({cx:.1f},{cy:.1f}), "
              f"radius={radius:.1f}, box={box}")

    out_path = Path(__file__).parent / "sawbone_white_crop_boxes.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path} ({len(results)}/{len(SEQUENCES)} sequences succeeded)")


if __name__ == "__main__":
    main()
