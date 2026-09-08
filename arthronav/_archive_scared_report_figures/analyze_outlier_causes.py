"""
Before concluding the loss function needs to change, check whether the
extreme-error pixels are actually explained by something in the data:
dark/black RGB regions (no real signal to predict from), or pixels near
the edge of the valid ground-truth mask (where SCARED's reprojection
from keyframe structured-light data is least reliable).

For each frame in the validation set:
  - compute the per-pixel error map (valid pixels only)
  - compute RGB brightness at every pixel
  - compute each valid pixel's distance to the nearest invalid pixel
    (cv2.distanceTransform), as a proxy for "near the edge of the mask"
Then compares these two properties between the extreme-error pixels
(top 1%) and the rest of the valid pixels, pooled across all frames.

Usage:
    python -m arthronav.analyze_outlier_causes \
        --checkpoint checkpoints/scared_training_checkpoints/checkpoints_long_run_full_v2/epoch_2.pt \
        --lora-mode uniform
"""

import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3

from arthronav.lora import inject_lora, inject_vector_lora
from arthronav.scared_io import build_frame_list, split_frames
from arthronav.scared_dataset import SCAREDDataset

H5_ROOT = "/mnt/areas_nas/SLAM/scared_dataset_full_copy/depth_anything_preprocessed_data/train_depth_anything"
JSON_ROOT = "/mnt/areas_nas/SLAM/scared_dataset_full_copy/frame_trajectory_data"
TARGET_SIZE = (1022, 1274)
UNIT_CORRECTION = 0.256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--lora-mode", type=str, default="uniform", choices=["uniform", "vector"])
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--dark-threshold", type=float, default=0.08,
                     help="mean RGB below this (0-1 scale) counts as 'dark'")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    wrapper = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = wrapper.model
    if args.lora_mode == "vector":
        inject_vector_lora(net)
    else:
        inject_lora(net, rank=args.lora_rank)
    net = net.to(device)
    net.eval()

    state = torch.load(args.checkpoint, map_location=device)
    _, unexpected = net.load_state_dict(state, strict=False)
    loaded = len(state) - len(unexpected)
    if loaded == 0:
        print("WARNING: 0 tensors matched, check --lora-mode")
    print(f"Loaded {args.checkpoint} (mode={args.lora_mode}, {loaded} tensors matched)")

    print("Building frame list...")
    frames = build_frame_list(H5_ROOT, JSON_ROOT)
    _, val_frames = split_frames(frames)
    import random
    rng = random.Random(42)
    n_keep = max(1, int(len(val_frames) * 0.1))
    val_subset = rng.sample(val_frames, n_keep)
    ds = SCAREDDataset(val_subset, bad_files_path="bad_h5_files.txt")
    val_loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    print(f"Analyzing {len(ds)} frames (same split used throughout this project)")

    all_errors = []
    all_brightness = []
    all_dist_to_edge = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="frames"):
            rgb = F.interpolate(batch["rgb"], size=TARGET_SIZE, mode="bilinear", align_corners=False)
            rgb_in = rgb.unsqueeze(1).to(device)
            depth_gt = F.interpolate(batch["depth"].unsqueeze(1), size=TARGET_SIZE, mode="nearest")
            depth_gt = (depth_gt.squeeze(1) * UNIT_CORRECTION).to(device)
            valid_mask = depth_gt > (1e-4 * UNIT_CORRECTION)

            output = net(rgb_in, export_feat_layers=[])
            pred = output.depth.squeeze(1) * UNIT_CORRECTION
            err = (pred - depth_gt).abs()

            mask_np = valid_mask[0].cpu().numpy().astype(np.uint8)
            # distance (in pixels) from each valid pixel to the nearest invalid pixel
            dist = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)

            brightness = rgb[0].mean(dim=0).numpy()  # mean over RGB channels, (H, W)

            m = mask_np.astype(bool)
            all_errors.append(err[0].cpu().numpy()[m])
            all_brightness.append(brightness[m])
            all_dist_to_edge.append(dist[m])

    all_errors = np.concatenate(all_errors)
    all_brightness = np.concatenate(all_brightness)
    all_dist_to_edge = np.concatenate(all_dist_to_edge)

    n_total = len(all_errors)
    outlier_thresh = np.percentile(all_errors, 99)  # top 1% = "extreme" for this test
    is_outlier = all_errors >= outlier_thresh

    print(f"\nTotal valid pixels: {n_total:,}")
    print(f"Top-1% error threshold: {outlier_thresh*1000:.2f} mm")
    print(f"Number of outlier pixels (top 1%): {is_outlier.sum():,}")

    print("\n--- RGB brightness: outliers vs rest ---")
    print(f"  Mean brightness, outlier pixels: {all_brightness[is_outlier].mean():.4f}")
    print(f"  Mean brightness, other pixels:   {all_brightness[~is_outlier].mean():.4f}")
    dark = all_brightness < args.dark_threshold
    frac_dark_in_outliers = dark[is_outlier].mean() * 100
    frac_dark_overall = dark.mean() * 100
    print(f"  Fraction of pixels below dark threshold ({args.dark_threshold}), among outliers: {frac_dark_in_outliers:.2f}%")
    print(f"  Fraction of pixels below dark threshold, overall:                 {frac_dark_overall:.2f}%")

    print("\n--- Distance to mask edge (pixels): outliers vs rest ---")
    print(f"  Mean distance to edge, outlier pixels: {all_dist_to_edge[is_outlier].mean():.2f} px")
    print(f"  Mean distance to edge, other pixels:   {all_dist_to_edge[~is_outlier].mean():.2f} px")
    near_edge = all_dist_to_edge < 5  # within 5 px of the valid-mask boundary
    frac_near_edge_in_outliers = near_edge[is_outlier].mean() * 100
    frac_near_edge_overall = near_edge.mean() * 100
    print(f"  Fraction within 5px of mask edge, among outliers: {frac_near_edge_in_outliers:.2f}%")
    print(f"  Fraction within 5px of mask edge, overall:         {frac_near_edge_overall:.2f}%")


if __name__ == "__main__":
    main()
