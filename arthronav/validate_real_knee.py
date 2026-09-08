"""
Validate a real-knee checkpoint (or several) on the held-out validation
patient (2602484M, from real_knee_io.split_patients(), never seen during
training regardless of which run produced the checkpoint).

Reports AbsRel, RMSE, and raw min/max/mean absolute error, all in real
millimeters directly -- this dataset needs NO 0.256-style correction
factor (verified: 12-32mm depth range, matches metadata's sigma_floor_mm
exactly).

Usage:
    python -m arthronav.validate_real_knee \
        --checkpoint checkpoints/real_knee_training/transfer_from_scared_vector/epoch_2.pt \
        --vector-lora \
        --label "Transfer, epoch 2"
"""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from depth_anything_3.api import DepthAnything3

from arthronav.lora import inject_lora, inject_vector_lora
from arthronav.metrics import abs_rel, rmse, abs_error_stats
from arthronav.real_knee_io import split_patients, build_frame_list
from arthronav.real_knee_dataset import RealKneeDataset

TARGET_SIZE = (1078, 1918)


def prepare_batch(batch, device):
    rgb = F.interpolate(batch["rgb"], size=TARGET_SIZE, mode="bilinear", align_corners=False)
    rgb = rgb.unsqueeze(1).to(device)
    depth_gt = F.interpolate(batch["depth"].unsqueeze(1), size=TARGET_SIZE, mode="nearest")
    depth_gt = depth_gt.squeeze(1).to(device)
    valid = F.interpolate(batch["valid"].unsqueeze(1).float(), size=TARGET_SIZE, mode="nearest")
    valid_mask = (valid.squeeze(1) > 0.5).to(device)
    return rgb, depth_gt, valid_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--vector-lora", action="store_true")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--label", type=str, default=None)
    ap.add_argument("--data-root", type=str, default="/mnt/areas_nas/SLAM/real_knee_dataset",
                     help="use /data/real_knee_dataset on AWS")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    label = args.label or args.checkpoint

    print("Loading model...")
    wrapper = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = wrapper.model
    if args.vector_lora:
        inject_vector_lora(net)
    else:
        inject_lora(net, rank=args.lora_rank)
    net = net.to(device)
    net.eval()

    state = torch.load(args.checkpoint, map_location=device)
    _, unexpected = net.load_state_dict(state, strict=False)
    loaded = len(state) - len(unexpected)
    if loaded == 0:
        print(f"WARNING: 0 tensors matched -- check --vector-lora matches how this "
              f"checkpoint was actually trained. Results below would be the untouched "
              f"base model, not this checkpoint.")
    print(f"Loaded {args.checkpoint} ({loaded} tensors matched)")

    train_patients, val_patients, test_patients = split_patients()
    print(f"Validating on held-out patient(s): {val_patients} (never seen during training)")

    val_frames = build_frame_list(args.data_root, val_patients)
    if len(val_frames) == 0:
        raise RuntimeError(f"No validation frames found under {args.data_root} for {val_patients}")

    val_ds = RealKneeDataset(val_frames, load_sigma=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
    print(f"Validating on {len(val_ds)} frames")

    all_abs_rel, all_rmse = [], []
    all_min, all_max, all_mean = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            rgb, depth_gt, valid_mask = prepare_batch(batch, device)
            output = net(rgb, export_feat_layers=[])
            pred = output.depth.squeeze(1)

            all_abs_rel.append(abs_rel(pred, depth_gt, valid_mask).item())
            all_rmse.append(rmse(pred, depth_gt, valid_mask).item())

            stats = abs_error_stats(pred, depth_gt, valid_mask)
            all_min.append(stats["min_error_m"])
            all_max.append(stats["max_error_m"])
            all_mean.append(stats["mean_error_m"])

    print(f"\n=== {label} ===")
    print(f"  AbsRel:     {sum(all_abs_rel)/len(all_abs_rel):.4f}")
    print(f"  RMSE:       {sum(all_rmse)/len(all_rmse):.3f} mm")
    print(f"  Mean error: {sum(all_mean)/len(all_mean):.3f} mm")
    print(f"  Max error:  {max(all_max):.3f} mm")
    print(f"  Min error:  {min(all_min):.3f} mm")


if __name__ == "__main__":
    main()
