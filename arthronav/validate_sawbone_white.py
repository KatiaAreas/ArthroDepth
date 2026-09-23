"""
Validate white sawbone checkpoints against the held-out validation
sequence (20260917-135100-eeeeeeee, 360 frames after 1-in-3 decimation,
never used in training by either run).

Ground truth is real meters (depth_png, uint16, raw*0.01 = mm, /1000 =
meters), matching the original red-sawbone convention -- confirmed
empirically that the red-sawbone checkpoint's raw output magnitude
matches ground truth in meters, not millimeters.
Data is already circle-cropped and resized to 1022x1022 by
prepare_sawbone_white_data.py, so no further resize needed here.

Usage:
    python -m arthronav.validate_sawbone_white \
        --checkpoint checkpoints/sawbone_white_from_sobone_v2/epoch_3.pt \
        --label "From sobone v2, epoch 3"
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

DATA_ROOT = "/mnt/areas_nas/SLAM/sawbone_white_dataset"


def build_frame_list(split_dir):
    frames = []
    seq_dirs = sorted(glob.glob(os.path.join(split_dir, "*")))
    for seq_dir in seq_dirs:
        rgb_dir = os.path.join(seq_dir, "rgb")
        depth_dir = os.path.join(seq_dir, "depth_png")
        if not os.path.isdir(rgb_dir):
            continue
        for rgb_path in sorted(glob.glob(os.path.join(rgb_dir, "*.png"))):
            frame_id = os.path.splitext(os.path.basename(rgb_path))[0]
            depth_path = os.path.join(depth_dir, f"{frame_id}.png")
            if os.path.exists(depth_path):
                frames.append({"rgb_path": rgb_path, "depth_path": depth_path})
    return frames


class SawboneWhiteDataset(Dataset):
    def __init__(self, frame_list):
        self.frames = frame_list

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        entry = self.frames[idx]
        rgb_bgr = cv2.imread(entry["rgb_path"])
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        depth_raw = cv2.imread(entry["depth_path"], cv2.IMREAD_UNCHANGED)
        depth_m = depth_raw.astype(np.float32) * 0.01 / 1000.0
        depth_t = torch.from_numpy(depth_m).float()
        valid_t = torch.from_numpy(depth_raw > 0)

        return {"rgb": rgb_t, "depth": depth_t, "valid": valid_t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--label", type=str, default=None)
    ap.add_argument("--data-root", type=str, default=DATA_ROOT)
    ap.add_argument("--split", type=str, default="val", choices=["val", "test"],
                     help="'val' = the held-out validation sequence used throughout development "
                          "(360 frames, 1 sequence); 'test' = the fully held-out test sequences "
                          "(786 frames, 2 sequences), never touched -- use only for a final check")
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
        print(f"WARNING: 0 tensors matched -- results below would be the untouched base model.")
    print(f"Loaded {args.checkpoint} ({loaded} tensors matched)")

    val_frames = build_frame_list(os.path.join(args.data_root, args.split))
    if len(val_frames) == 0:
        raise RuntimeError(f"No frames found under {args.data_root}/{args.split}")

    val_ds = SawboneWhiteDataset(val_frames)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
    print(f"Validating on {len(val_ds)} frames")

    all_abs_rel, all_rmse = [], []
    all_min, all_max, all_mean = [], [], []
    skipped_empty = 0

    with torch.no_grad():
        for batch in val_loader:
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

    print(f"\n=== {label} ===")
    print(f"  AbsRel:     {sum(all_abs_rel)/len(all_abs_rel):.4f}")
    print(f"  RMSE:       {sum(all_rmse)/len(all_rmse):.4f} m")
    print(f"  Mean error: {sum(all_mean)/len(all_mean):.4f} m")
    print(f"  Max error:  {max(all_max):.4f} m")
    print(f"  Min error:  {min(all_min):.4f} m")


if __name__ == "__main__":
    main()
