"""
Scan every .h5 file the training pipeline would read, and report any that
are corrupted or truncated, so we can exclude them proactively instead of
discovering them one crash at a time.

Usage:
    python -m arthronav.scan_h5_integrity
    python -m arthronav.scan_h5_integrity --out bad_files.txt
"""

import argparse

import h5py
from tqdm import tqdm

from arthronav.scared_io import build_frame_list

H5_ROOT = "/mnt/areas_nas/SLAM/scared_dataset_full_copy/depth_anything_preprocessed_data/train_depth_anything"
JSON_ROOT = "/mnt/areas_nas/SLAM/scared_dataset_full_copy/frame_trajectory_data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="bad_h5_files.txt")
    args = ap.parse_args()

    print("Building frame list...")
    frames = build_frame_list(H5_ROOT, JSON_ROOT)
    print(f"Scanning {len(frames)} files...")

    bad_files = []
    for entry in tqdm(frames, desc="scanning"):
        path = entry["h5_path"]
        try:
            with h5py.File(path, "r") as f:
                # actually touch the data, not just open the container,
                # since a truncated file can sometimes open but fail on read
                _ = f["rgb"].shape
                _ = f["depth"].shape
        except Exception as e:
            bad_files.append((path, str(e)))

    print(f"\nFound {len(bad_files)} corrupted / unreadable files out of {len(frames)}")
    with open(args.out, "w") as f:
        for path, err in bad_files:
            f.write(f"{path}\t{err}\n")
    print(f"Wrote list to {args.out}")


if __name__ == "__main__":
    main()
