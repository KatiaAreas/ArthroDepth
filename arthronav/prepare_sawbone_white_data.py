"""
Prepare the new white sawbone dataset (case_sawbone_white_2) for
training: crop each frame to its sequence's detected circle (from
precompute_crop_boxes_sawbone_white.py), resize to 518x518 (this
project's own validated DA3-native resolution for circle-cropped
data -- confirmed earlier that 518x518 loses negligible accuracy vs.
native resolution while training ~4.7x faster), and keep 1 frame out
of every 3 (by position within the sequence, not by raw frame number,
so exactly 1/3 is kept regardless of any gaps).

Ground truth: depth_png only (uint16, depth_mm = raw * 0.01, invalid=0
-- verified directly, same convention as the original sawbone dataset).
depth_npy exists in the source but is not used, matching the original
sawbone convention of depth_png as the confirmed source.

Sequences and split (train=7, val=1, test=2), derived from
random.Random(42) over the 10 sequence names, the same seed convention
used throughout this project:
    train: 135219, 135034, 135011, 135304, 135126, 135156, 135325
    val:   135100
    test:  134858, 134939

Usage:
    python -m arthronav.prepare_sawbone_white_data
"""

import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ZIP_PATH = "/mnt/areas_nas/SLAM/sawbone_white_depth.zip"
OUT_ROOT = Path("/mnt/areas_nas/SLAM/sawbone_white_dataset")
CROP_BOXES_PATH = Path(__file__).parent / "sawbone_white_crop_boxes.json"
TARGET_SIZE = 1022  # matches the original red-sawbone convention (1022x1022, meters output)
KEEP_EVERY_N = 3

TRAIN_SEQUENCES = ["20260917-135219-eeeeeeee", "20260917-135034-eeeeeeee",
                   "20260917-135011-eeeeeeee", "20260917-135304-eeeeeeee",
                   "20260917-135126-eeeeeeee", "20260917-135156-eeeeeeee",
                   "20260917-135325-eeeeeeee"]
VAL_SEQUENCES = ["20260917-135100-eeeeeeee"]
TEST_SEQUENCES = ["20260917-134858-eeeeeeee", "20260917-134939-eeeeeeee"]
ALL_SEQUENCES = TRAIN_SEQUENCES + VAL_SEQUENCES + TEST_SEQUENCES


def crop_and_resize(image, box, target_size, interpolation):
    x0, y0, x1, y1 = box
    cropped = image[y0:y1, x0:x1]
    return cv2.resize(cropped, (target_size, target_size), interpolation=interpolation)


def process_sequence(z, seq_name, box, out_dir):
    rgb_names = sorted(n for n in z.namelist()
                        if f"depth/{seq_name}/rgb/" in n and n.endswith(".png"))
    depth_names = sorted(n for n in z.namelist()
                          if f"depth/{seq_name}/depth_png/" in n and n.endswith(".png"))

    if len(rgb_names) != len(depth_names):
        print(f"  WARNING: {seq_name} has {len(rgb_names)} rgb but {len(depth_names)} "
              f"depth_png files -- mismatch, check before trusting this sequence's pairing")

    rgb_dir = out_dir / "rgb"
    depth_dir = out_dir / "depth_png"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    for i, (rgb_name, depth_name) in enumerate(zip(rgb_names, depth_names)):
        if i % KEEP_EVERY_N != 0:
            continue

        rgb_bytes = z.read(rgb_name)
        rgb_img = np.array(Image.open(io.BytesIO(rgb_bytes)).convert("RGB"))
        rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        rgb_cropped = crop_and_resize(rgb_bgr, box, TARGET_SIZE, cv2.INTER_LINEAR)

        depth_bytes = z.read(depth_name)
        depth_img = np.array(Image.open(io.BytesIO(depth_bytes)))  # uint16
        depth_cropped = crop_and_resize(depth_img, box, TARGET_SIZE, cv2.INTER_NEAREST)

        frame_id = Path(rgb_name).stem  # e.g. "000653"
        cv2.imwrite(str(rgb_dir / f"{frame_id}.png"), rgb_cropped)
        cv2.imwrite(str(depth_dir / f"{frame_id}.png"), depth_cropped.astype(np.uint16))
        kept += 1

    print(f"  {seq_name}: kept {kept} / {len(rgb_names)} frames (1-in-{KEEP_EVERY_N})")
    return kept


def main():
    with open(CROP_BOXES_PATH) as f:
        crop_boxes = json.load(f)

    z = zipfile.ZipFile(ZIP_PATH)
    totals = {"train": 0, "val": 0, "test": 0}

    for split_name, sequences in [("train", TRAIN_SEQUENCES), ("val", VAL_SEQUENCES),
                                    ("test", TEST_SEQUENCES)]:
        print(f"\n=== {split_name} ({len(sequences)} sequences) ===")
        for seq_name in sequences:
            if seq_name not in crop_boxes:
                print(f"  SKIPPING {seq_name}: no precomputed crop box -- run "
                      f"precompute_crop_boxes_sawbone_white.py first")
                continue
            box = tuple(crop_boxes[seq_name]["box"])
            out_dir = OUT_ROOT / split_name / seq_name
            kept = process_sequence(z, seq_name, box, out_dir)
            totals[split_name] += kept

    print(f"\nDone. Frame totals: train={totals['train']}, val={totals['val']}, "
          f"test={totals['test']}, overall={sum(totals.values())}")
    print(f"Saved to {OUT_ROOT}")


if __name__ == "__main__":
    main()
