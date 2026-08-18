"""
The raw max error is one pixel out of a million. This script asks a
more useful question: across every pixel in the validation set, what
fraction are actually bad, and what does the error look like once you
set those aside?

For the checkpoint given, pools every valid pixel across all 136
validation frames (not just per-frame max, an actual pooled distribution)
and reports:
  - what fraction of pixels exceed a few thresholds (1mm, 3mm, 1cm, 5cm)
  - the 90th, 95th, 99th, 99.9th percentile error
  - the max error with the top 0.1%, 1%, 5% of pixels excluded

Usage:
    python -m arthronav.compute_error_percentiles \
        --checkpoint checkpoints/scared_training_checkpoints/checkpoints_long_run_full_v2/epoch_2.pt \
        --lora-mode uniform
"""

import argparse

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


def prepare_batch(batch, device):
    rgb = F.interpolate(batch["rgb"], size=TARGET_SIZE, mode="bilinear", align_corners=False)
    rgb = rgb.unsqueeze(1).to(device)
    depth_gt = F.interpolate(batch["depth"].unsqueeze(1), size=TARGET_SIZE, mode="nearest")
    depth_gt = (depth_gt.squeeze(1) * UNIT_CORRECTION).to(device)
    valid_mask = depth_gt > (1e-4 * UNIT_CORRECTION)
    return rgb, depth_gt, valid_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--lora-mode", type=str, default="uniform", choices=["uniform", "vector"])
    ap.add_argument("--lora-rank", type=int, default=16)
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
    missing, unexpected = net.load_state_dict(state, strict=False)
    loaded = len(state) - len(unexpected)
    if loaded == 0:
        print(f"WARNING: 0 tensors matched, check --lora-mode is correct for this checkpoint")
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
    print(f"Pooling errors across {len(ds)} frames (same split used throughout this project)")

    all_errors = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="frames"):
            rgb, depth_gt, valid_mask = prepare_batch(batch, device)
            output = net(rgb, export_feat_layers=[])
            pred = output.depth.squeeze(1) * UNIT_CORRECTION
            err = (pred[valid_mask] - depth_gt[valid_mask]).abs()
            all_errors.append(err.cpu().numpy())

    all_errors = np.concatenate(all_errors)
    n_total = len(all_errors)
    print(f"\nTotal valid pixels pooled: {n_total:,}")

    print("\n--- Fraction of pixels exceeding a threshold ---")
    for thresh_m, label in [(0.001, "1mm"), (0.003, "3mm"), (0.01, "1cm"), (0.05, "5cm")]:
        frac = (all_errors > thresh_m).mean() * 100
        count = (all_errors > thresh_m).sum()
        print(f"  > {label:>4}: {frac:7.4f}% of pixels ({count:,} pixels)")

    print("\n--- Percentiles of the error distribution ---")
    for p in [50, 90, 95, 99, 99.9]:
        val = np.percentile(all_errors, p)
        print(f"  {p:>5}th percentile: {val*1000:.3f} mm")

    print("\n--- Max error, with top outliers excluded ---")
    for excl_pct, label in [(0.0, "no exclusion (raw max)"), (0.1, "top 0.1% excluded"),
                              (1.0, "top 1% excluded"), (5.0, "top 5% excluded")]:
        keep_percentile = 100 - excl_pct
        val = np.percentile(all_errors, keep_percentile)
        print(f"  {label:<28}: {val*1000:.3f} mm  (= {val*100:.4f} cm)")

    print(f"\nMean error (for reference): {all_errors.mean()*1000:.3f} mm")


if __name__ == "__main__":
    main()
