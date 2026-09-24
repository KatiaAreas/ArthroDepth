"""
Validate real-knee-combined checkpoints against the held-out val or
test splits.

The model itself outputs and is compared internally in real meters
(matching training). Displayed results are converted to millimeters
at the print stage only, for readability -- AbsRel is unitless and
unaffected; RMSE/mean/max/min error are multiplied by 1000 just before
printing. The underlying computation stays in meters.

Usage:
    python -m arthronav.validate_real_knee_combined \
        --checkpoint checkpoints/real_knee_combined_from_scratch/epoch_2.pt \
        --split val --label "Combined from scratch, epoch 2"
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from depth_anything_3.api import DepthAnything3

from arthronav.lora import inject_vector_lora
from arthronav.metrics import abs_rel, rmse, abs_error_stats

DATA_ROOT = "/mnt/areas_nas/SLAM/real_knee_combined_dataset"


def build_frame_list(split_dir):
    rgb_dir = os.path.join(split_dir, "rgb")
    depth_dir = os.path.join(split_dir, "depth")
    frames = []
    for rgb_path in sorted(glob.glob(os.path.join(rgb_dir, "*.jpg"))):
        frame_id = os.path.splitext(os.path.basename(rgb_path))[0]
        depth_path = os.path.join(depth_dir, f"{frame_id}.depth.npy")
        valid_path = os.path.join(depth_dir, f"{frame_id}.valid.npy")
        if os.path.exists(depth_path) and os.path.exists(valid_path):
            frames.append({"rgb_path": rgb_path, "depth_path": depth_path, "valid_path": valid_path})
    return frames


class CombinedRealKneeDataset(Dataset):
    def __init__(self, frame_list):
        self.frames = frame_list

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        entry = self.frames[idx]
        rgb_bgr = cv2.imread(entry["rgb_path"])
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        depth_m = np.load(entry["depth_path"]).astype(np.float32)
        valid = np.load(entry["valid_path"])

        return {"rgb": rgb_t, "depth": torch.from_numpy(depth_m).float(),
                "valid": torch.from_numpy(valid)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--label", type=str, default=None)
    ap.add_argument("--data-root", type=str, default=DATA_ROOT)
    ap.add_argument("--split", type=str, default="val", choices=["val", "test"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    label = args.label or args.checkpoint

    print("Loading model...")
    wrapper = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = wrapper.model
    inject_vector_lora(net)
    net = net.to(device)
    net.eval()

    state = torch.load(args.checkpoint, map_location=device)
    _, unexpected = net.load_state_dict(state, strict=False)
    loaded = len(state) - len(unexpected)
    if loaded == 0:
        print("WARNING: 0 tensors matched -- results below would be the untouched base model.")
    print(f"Loaded {args.checkpoint} ({loaded} tensors matched)")

    frames = build_frame_list(os.path.join(args.data_root, args.split))
    if len(frames) == 0:
        raise RuntimeError(f"No frames found under {args.data_root}/{args.split}")

    ds = CombinedRealKneeDataset(frames)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    print(f"Validating on {len(ds)} frames ({args.split} split)")

    all_abs_rel, all_rmse = [], []
    all_min, all_max, all_mean = [], [], []
    skipped_empty = 0

    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].unsqueeze(1).to(device)
            depth_gt = batch["depth"].to(device)
            valid_mask = batch["valid"].to(device)

            if valid_mask.sum().item() == 0:
                skipped_empty += 1
                continue

            output = net(rgb, export_feat_layers=[])
            pred = output.depth.squeeze(1)

            all_abs_rel.append(abs_rel(pred, depth_gt, valid_mask).item())
            all_rmse.append(rmse(pred, depth_gt, valid_mask).item())

            stats = abs_error_stats(pred, depth_gt, valid_mask)
            all_min.append(stats["min_error_m"])
            all_max.append(stats["max_error_m"])
            all_mean.append(stats["mean_error_m"])

    if skipped_empty > 0:
        print(f"Skipped {skipped_empty} frames with zero valid pixels")

    MM_PER_M = 1000.0
    print(f"\n=== {label} ===")
    print(f"  AbsRel:     {sum(all_abs_rel)/len(all_abs_rel):.4f}")
    print(f"  RMSE:       {sum(all_rmse)/len(all_rmse) * MM_PER_M:.3f} mm")
    print(f"  Mean error: {sum(all_mean)/len(all_mean) * MM_PER_M:.3f} mm")
    print(f"  Max error:  {max(all_max) * MM_PER_M:.3f} mm")
    print(f"  Min error:  {min(all_min) * MM_PER_M:.3f} mm")


if __name__ == "__main__":
    main()
