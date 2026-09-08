"""
I/O for the preprocessed real-knee dataset (see prepare_real_knee_data.py
for how the raw S3 zips/videos become this local structure).

Layout expected under `root`:
    <root>/<PATIENT>_<view>/
        rgb/frame_NNNNNN.png
        depth/frame_NNNNNN.depth.npy   (real millimeters, no correction needed)
        depth/frame_NNNNNN.sigma.npy   (real per-pixel uncertainty, mm)
        depth/frame_NNNNNN.valid.npy
        metadata.json

Split is by PATIENT (not frame), so both views of a given patient always
land in the same split, keeping the same held-out generalization test
that matters clinically (never seen this patient's anatomy at all), and
avoiding the leakage a frame-level split would risk given how similar
consecutive frames from the same patient are. 5 patients total: 3
train / 1 val / 1 test. Assignment is derived from a fixed seed (42, the
same seed used everywhere else in this project) rather than a hardcoded
list, so it's transparent and reproducible from this file alone.
"""

import json
import os
import random
import re

ALL_PATIENTS = ["2509457F", "26011317M", "2602345F", "2602446F_G", "2602484M"]
VIEWS = ["lateral", "medial"]

FRAME_RE = re.compile(r"frame_(\d{6})\.depth\.npy$")


def split_patients(seed: int = 42):
    """Returns (train_patients, val_patients, test_patients), 3/1/1."""
    shuffled = ALL_PATIENTS.copy()
    random.Random(seed).shuffle(shuffled)
    return shuffled[:3], shuffled[3:4], shuffled[4:5]


def clip_dir(root, patient, view):
    return os.path.join(root, f"{patient}_{view}")


def build_frame_list(root, patients, views=VIEWS):
    """
    Returns a list of dicts: {clip_dir, patient, view, frame_idx, rgb_path,
    depth_path, sigma_path, valid_path}, for every frame that has both an
    RGB image and depth ground truth already extracted.
    """
    frames = []
    for patient in patients:
        for view in views:
            cdir = clip_dir(root, patient, view)
            depth_dir = os.path.join(cdir, "depth")
            rgb_dir = os.path.join(cdir, "rgb")
            if not os.path.isdir(depth_dir):
                continue  # this patient/view combo not prepared yet, skip rather than error

            for fname in sorted(os.listdir(depth_dir)):
                m = FRAME_RE.search(fname)
                if m is None:
                    continue
                frame_idx = int(m.group(1))
                rgb_path = os.path.join(rgb_dir, f"frame_{frame_idx:06d}.jpg")
                if not os.path.exists(rgb_path):
                    continue  # depth exists but matching rgb missing -- skip, don't crash
                frames.append({
                    "clip_dir": cdir,
                    "patient": patient,
                    "view": view,
                    "frame_idx": frame_idx,
                    "rgb_path": rgb_path,
                    "depth_path": os.path.join(depth_dir, f"frame_{frame_idx:06d}.depth.npy"),
                    "sigma_path": os.path.join(depth_dir, f"frame_{frame_idx:06d}.sigma.npy"),
                    "valid_path": os.path.join(depth_dir, f"frame_{frame_idx:06d}.valid.npy"),
                })
    return frames


def load_metadata(cdir):
    with open(os.path.join(cdir, "metadata.json")) as f:
        return json.load(f)
