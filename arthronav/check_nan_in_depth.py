"""
Check whether the real-knee depth/sigma ground truth stores invalid
pixels as NaN, and whether those NaN locations actually match the
separate `valid` mask (they should -- a mismatch would mean some
"valid" pixels are secretly NaN, a real landmine for training: NaN
propagates silently through any unmasked operation, even ones that look
harmless, like resizing, visualization, or a future sigma-weighted loss
that touches the full tensor before masking).

Checks a handful of frames across every prepared clip, not just one,
since a single frame's absence of NaN wouldn't prove the whole dataset
is clean.

Usage:
    python -m arthronav.check_nan_in_depth --data-root /mnt/areas_nas/SLAM/real_knee_dataset
"""

import argparse
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--frames-per-clip", type=int, default=5)
    args = ap.parse_args()

    clip_dirs = sorted(glob.glob(os.path.join(args.data_root, "*")))
    clip_dirs = [d for d in clip_dirs if os.path.isdir(os.path.join(d, "depth"))]

    total_frames_checked = 0
    total_nan_in_depth = 0
    total_nan_in_sigma = 0
    total_mismatch = 0

    for clip_dir in clip_dirs:
        depth_dir = os.path.join(clip_dir, "depth")
        depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.depth.npy")))
        sample = depth_files[:: max(1, len(depth_files) // args.frames_per_clip)][:args.frames_per_clip]

        clip_nan_depth = 0
        clip_mismatch = 0

        for depth_path in sample:
            base = os.path.basename(depth_path).replace(".depth.npy", "")
            sigma_path = os.path.join(depth_dir, f"{base}.sigma.npy")
            valid_path = os.path.join(depth_dir, f"{base}.valid.npy")

            depth = np.load(depth_path)
            sigma = np.load(sigma_path)
            valid = np.load(valid_path).astype(bool)

            nan_depth = np.isnan(depth)
            nan_sigma = np.isnan(sigma)

            nan_but_valid = nan_depth & valid
            notnan_but_invalid = (~nan_depth) & (~valid)

            total_frames_checked += 1
            total_nan_in_depth += nan_depth.sum()
            total_nan_in_sigma += nan_sigma.sum()
            total_mismatch += nan_but_valid.sum()
            clip_nan_depth += nan_depth.sum()
            clip_mismatch += nan_but_valid.sum()

            if nan_but_valid.sum() > 0:
                print(f"  ALERT: {depth_path} has {nan_but_valid.sum()} pixels marked "
                      f"valid=True but depth=NaN")

        print(f"{os.path.basename(clip_dir):<25} checked {len(sample)} frames, "
              f"NaN-in-depth pixels: {clip_nan_depth:,}, "
              f"valid-but-NaN mismatches: {clip_mismatch:,}")

    print("\n=== Summary ===")
    print(f"Frames checked: {total_frames_checked}")
    print(f"Total NaN pixels in depth arrays: {total_nan_in_depth:,}")
    print(f"Total NaN pixels in sigma arrays: {total_nan_in_sigma:,}")
    print(f"Total valid=True-but-NaN mismatches: {total_mismatch:,}")

    if total_nan_in_depth == 0:
        print("\nNo NaN found in depth at all -- invalid pixels must be represented some "
              "other way (check what value sits there), or all sampled frames happened "
              "to be fully valid (unlikely given the ~50% valid_fraction seen in "
              "metrics.csv -- worth double-checking).")
    elif total_mismatch == 0:
        print("\nGood: every NaN in depth aligns with valid=False. The current training "
              "pipeline is safe as long as everything always applies the mask before any "
              "operation on depth. Still worth replacing NaN with 0 in storage as a "
              "defensive measure against future code that forgets to mask.")
    else:
        print("\nREAL PROBLEM: some pixels are marked valid=True but contain NaN depth. "
              "This would silently break training (NaN loss/gradients). Needs fixing "
              "before any further training runs use this data.")


if __name__ == "__main__":
    main()
