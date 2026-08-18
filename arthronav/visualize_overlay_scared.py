"""
Superpose ground-truth depth (as contour lines) directly on top of the
predicted depth map, on a genuinely unseen validation frame.

Supports both LoRA modes -- pass --lora-mode uniform (default) or
--lora-mode vector to match whichever checkpoint you're loading. Loading
a checkpoint built with one mode into a model injected with the other
mode silently loads zero matching tensors under strict=False (no error,
just a quietly wrong result showing the untouched base model instead) --
this version checks for that and warns loudly if it happens.

Usage:
    python -m arthronav.visualize_overlay_scared --frame-index 0 \
        --checkpoint checkpoints/scared_training_checkpoints/checkpoints_long_run_full_v2/epoch_2.pt \
        --lora-mode uniform

    python -m arthronav.visualize_overlay_scared --frame-index 0 \
        --checkpoint checkpoints/scared_training_checkpoints/checkpoints_vector_lora_full/epoch_3.pt \
        --lora-mode vector --out overlay_vector_lora.png
"""

import argparse

import matplotlib
matplotlib.use("Agg")  # no display over SSH
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from depth_anything_3.api import DepthAnything3

from arthronav.lora import inject_lora, inject_vector_lora
from arthronav.scared_io import build_frame_list, split_frames
from arthronav.scared_dataset import SCAREDDataset

H5_ROOT = "/mnt/areas_nas/SLAM/scared_dataset_full_copy/depth_anything_preprocessed_data/train_depth_anything"
JSON_ROOT = "/mnt/areas_nas/SLAM/scared_dataset_full_copy/frame_trajectory_data"
TARGET_SIZE = (1022, 1274)
UNIT_CORRECTION = 0.256  # verified: raw h5 depth x 0.256 = real meters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-index", type=int, default=0,
                     help="index into the held-out validation split (unseen during training)")
    ap.add_argument("--checkpoint", type=str, default=None,
                     help="path to a LoRA checkpoint; omit to use the base pretrained model")
    ap.add_argument("--lora-mode", type=str, default="uniform", choices=["uniform", "vector"],
                     help="must match how the checkpoint was actually trained")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--num-contour-levels", type=int, default=6)
    ap.add_argument("--out", type=str, default="overlay_sample.png")
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

    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location=device)
        missing, unexpected = net.load_state_dict(state, strict=False)
        loaded_count = len(state) - len(unexpected)
        if loaded_count == 0:
            print(f"WARNING: 0 matching tensors loaded from {args.checkpoint} "
                  f"with --lora-mode {args.lora_mode}. This checkpoint was probably "
                  f"trained with the other mode -- the results below are the "
                  f"UNTOUCHED BASE MODEL, not the checkpoint you asked for.")
        print(f"Loaded checkpoint: {args.checkpoint} (mode={args.lora_mode}, {loaded_count} tensors matched)")
    else:
        print("No checkpoint given, using base pretrained model.")

    print("Building frame list...")
    frames = build_frame_list(H5_ROOT, JSON_ROOT)
    _, val_frames = split_frames(frames)
    ds = SCAREDDataset(val_frames, bad_files_path="bad_h5_files.txt")

    sample = ds[args.frame_index]
    print(f"Using validation frame index {args.frame_index} (unseen during training)")

    rgb_orig = sample["rgb"].unsqueeze(0)
    rgb_in = F.interpolate(rgb_orig, size=TARGET_SIZE, mode="bilinear", align_corners=False)
    rgb_in = rgb_in.unsqueeze(1).to(device)

    depth_gt = sample["depth"].unsqueeze(0).unsqueeze(0)
    depth_gt = F.interpolate(depth_gt, size=TARGET_SIZE, mode="nearest").squeeze(0).squeeze(0)
    depth_gt = (depth_gt * UNIT_CORRECTION).to(device)
    valid_mask = depth_gt > (1e-4 * UNIT_CORRECTION)

    with torch.no_grad():
        output = net(rgb_in, export_feat_layers=[])
    depth_pred = output.depth.squeeze(0).squeeze(0) * UNIT_CORRECTION

    error_map = torch.zeros_like(depth_gt)
    error_map[valid_mask] = (depth_pred[valid_mask] - depth_gt[valid_mask]).abs()
    valid_errors = error_map[valid_mask]
    err_min, err_max, err_mean = valid_errors.min().item(), valid_errors.max().item(), valid_errors.mean().item()

    gt_np = depth_gt.cpu().numpy()
    pred_np = depth_pred.cpu().numpy()
    mask_np = valid_mask.cpu().numpy()
    gt_masked = np.where(mask_np, gt_np, np.nan)
    pred_masked = np.where(mask_np, pred_np, np.nan)

    vmin, vmax = np.nanmin(gt_masked), np.nanmax(gt_masked)
    levels = np.linspace(vmin, vmax, args.num_contour_levels)

    fig, ax = plt.subplots(figsize=(9, 7))

    im = ax.imshow(pred_masked, cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Predicted depth (m)")

    contour = ax.contour(gt_masked, levels=levels, colors="red", linewidths=1.0)
    ax.clabel(contour, inline=True, fontsize=7, fmt="%.3f")

    ax.set_title(
        f"Predicted depth (color) with ground-truth iso-depth contours (red)\n"
        f"Frame {args.frame_index}, unseen validation split | "
        f"error min={err_min:.4f} m, max={err_max:.4f} m, mean={err_mean:.4f} m",
        fontsize=10,
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved overlay figure to {args.out}")
    print(f"Error, real meters (valid pixels only): min={err_min:.4f} m, max={err_max:.4f} m, mean={err_mean:.4f} m")


if __name__ == "__main__":
    main()
