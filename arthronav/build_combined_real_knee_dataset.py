"""
Build the unified, combined real-knee training dataset from all 14
prepared clips (5 autriche patients x mostly 2 views + 3 austria
patients x 1-2 views each), applying, uniformly across every clip:
  - per-frame circle crop (using each clip's own circle_boxes.json,
    since real surgical footage genuinely moves the circle frame to
    frame, unlike the static benchtop sawbone setup)
  - resize to 1022x1022 (this project's established convention)
  - conversion from the existing millimeters convention to real
    meters (matching DA3's own native output scale, and the now-
    established meters convention used for sobone red/white -- a
    deliberate departure from every earlier real-knee script, which
    stayed in mm throughout; empirically, from-scratch training in mm
    still produced reasonable results there, since a freshly-
    initialized LoRA has no pre-existing scale to conflict with, but
    meters keeps this dataset consistent with everything else and
    avoids ever repeating the scale-mismatch bug found on sawbone)
  - keep 1 frame out of every 3 (by position within the clip, so
    exactly 1/3 is kept regardless of gaps)

Split is by PATIENT (not clip), derived from random.Random(42) over
the 8 patients with real ground truth:
    train (5): 2602446F_G, 2602484M, 2602425F_D, 2602432N_D, 2602345F
    val   (1): 2601253M_D
    test  (2): 2509457F, 26011317M

Source clips must already exist under SOURCE_ROOT (rgb/, depth/
folders, from prepare_real_knee_data.py) with a matching
circle_boxes.json (from precompute_circle_masks.py) in the same
directory.

Usage:
    python -m arthronav.build_combined_real_knee_dataset
"""

import json
import os
import re

import cv2
import numpy as np

SOURCE_ROOT = "/mnt/areas_nas/SLAM/real_knee_dataset"
OUT_ROOT = "/mnt/areas_nas/SLAM/real_knee_combined_dataset"
TARGET_SIZE = 1022
KEEP_EVERY_N = 3

TRAIN_CLIPS = [
    ("2602446F_G", "lateral"), ("2602446F_G", "medial"),
    ("2602484M", "lateral"), ("2602484M", "medial"),
    ("2602425F_D", "tour_medial"),
    ("2602432N_D", "antero_medial"), ("2602432N_D", "tour_medial"),
    ("2602345F", "lateral"), ("2602345F", "medial"),
]
VAL_CLIPS = [
    ("2601253M_D", "antero_lateral"),
]
TEST_CLIPS = [
    ("2509457F", "lateral"), ("2509457F", "medial"),
    ("26011317M", "lateral"), ("26011317M", "medial"),
]

FRAME_RE = re.compile(r"frame_(\d{6})\.depth\.npy$")


def crop_and_resize(image, box, target_size, interpolation):
    x0, y0, x1, y1 = box
    cropped = image[y0:y1, x0:x1]
    return cv2.resize(cropped, (target_size, target_size), interpolation=interpolation)


def process_clip(patient, view, out_dir):
    clip_name = f"{patient}_{view}"
    clip_dir = os.path.join(SOURCE_ROOT, clip_name)
    circle_path = os.path.join(clip_dir, "circle_boxes.json")

    if not os.path.isdir(clip_dir):
        print(f"  SKIPPING {clip_name}: source clip not found at {clip_dir}")
        return 0
    if not os.path.exists(circle_path):
        print(f"  SKIPPING {clip_name}: no circle_boxes.json -- run "
              f"precompute_circle_masks.py for this clip first")
        return 0

    with open(circle_path) as f:
        circle_boxes = json.load(f)

    rgb_dir = os.path.join(clip_dir, "rgb")
    depth_dir = os.path.join(clip_dir, "depth")
    depth_files = sorted(os.listdir(depth_dir))
    frame_indices = sorted(set(int(m.group(1)) for f in depth_files
                                if (m := FRAME_RE.search(f))))

    out_rgb_dir = os.path.join(out_dir, "rgb")
    out_depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)

    kept, no_box, no_rgb = 0, 0, 0
    for i, frame_idx in enumerate(frame_indices):
        if i % KEEP_EVERY_N != 0:
            continue

        frame_key = str(frame_idx)
        box_entry = circle_boxes.get(frame_key)
        if box_entry is None:
            no_box += 1
            continue
        box = box_entry["box"]

        rgb_path = os.path.join(rgb_dir, f"frame_{frame_idx:06d}.jpg")
        if not os.path.exists(rgb_path):
            no_rgb += 1
            continue

        rgb_bgr = cv2.imread(rgb_path)
        rgb_cropped = crop_and_resize(rgb_bgr, box, TARGET_SIZE, cv2.INTER_LINEAR)

        depth_mm = np.load(os.path.join(depth_dir, f"frame_{frame_idx:06d}.depth.npy")).astype(np.float32)
        valid = np.load(os.path.join(depth_dir, f"frame_{frame_idx:06d}.valid.npy"))

        depth_m = depth_mm / 1000.0
        depth_cropped = crop_and_resize(depth_m, box, TARGET_SIZE, cv2.INTER_NEAREST)
        valid_cropped = crop_and_resize(valid.astype(np.uint8), box, TARGET_SIZE, cv2.INTER_NEAREST)

        depth_out = np.where(valid_cropped > 0, depth_cropped, 0.0).astype(np.float32)

        out_id = f"{patient}_{view}_{frame_idx:06d}"
        cv2.imwrite(os.path.join(out_rgb_dir, f"{out_id}.jpg"), rgb_cropped)
        np.save(os.path.join(out_depth_dir, f"{out_id}.depth.npy"), depth_out)
        np.save(os.path.join(out_depth_dir, f"{out_id}.valid.npy"), (valid_cropped > 0))
        kept += 1

    print(f"  {clip_name}: kept {kept} / {len(frame_indices)} frames "
          f"(1-in-{KEEP_EVERY_N}), {no_box} missing circle box, {no_rgb} missing rgb")
    return kept


def main():
    totals = {"train": 0, "val": 0, "test": 0}
    for split_name, clips in [("train", TRAIN_CLIPS), ("val", VAL_CLIPS), ("test", TEST_CLIPS)]:
        print(f"\n=== {split_name} ({len(clips)} clips) ===")
        out_dir = os.path.join(OUT_ROOT, split_name)
        for patient, view in clips:
            totals[split_name] += process_clip(patient, view, out_dir)

    print(f"\nDone. Frame totals: train={totals['train']}, val={totals['val']}, "
          f"test={totals['test']}, overall={sum(totals.values())}")
    print(f"Saved to {OUT_ROOT}")


if __name__ == "__main__":
    main()
