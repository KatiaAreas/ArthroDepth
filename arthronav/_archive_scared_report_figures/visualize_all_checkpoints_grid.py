"""
For every checkpoint trained so far, run inference on the same unseen
validation frame and plot RGB next to predicted depth (corrected to
real meters via the verified 0.256 factor), stacked as rows so the
progression across training stages is directly visible.

Usage:
    python -m arthronav.visualize_all_checkpoints_grid --frame-index 0
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from depth_anything_3.api import DepthAnything3

from arthronav.lora import inject_lora
from arthronav.scared_io import build_frame_list, split_frames
from arthronav.scared_dataset import SCAREDDataset

H5_ROOT = "/mnt/areas_nas/SLAM/scared_dataset_full_copy/depth_anything_preprocessed_data/train_depth_anything"
JSON_ROOT = "/mnt/areas_nas/SLAM/scared_dataset_full_copy/frame_trajectory_data"
TARGET_SIZE = (1022, 1274)
BASE_CKPT_DIR = "checkpoints/scared_training_checkpoints"
UNIT_CORRECTION = 0.256  # verified: h5_depth * 0.256 = real meters

CHECKPOINTS = [
    ("Base pretrained model", None),
    ("1% subset, 1 epoch", f"{BASE_CKPT_DIR}/checkpoints_small/epoch_0.pt"),
    ("Medium run (20%), epoch 0", f"{BASE_CKPT_DIR}/checkpoints_medium_run_20subset/epoch_0.pt"),
    ("Medium run (20%), epoch 1", f"{BASE_CKPT_DIR}/checkpoints_medium_run_20subset/epoch_1.pt"),
    ("Medium run (20%), epoch 2", f"{BASE_CKPT_DIR}/checkpoints_medium_run_20subset/epoch_2.pt"),
    ("Full run, epoch 0", f"{BASE_CKPT_DIR}/checkpoints_long_run_full_v2/epoch_0.pt"),
    ("Full run, epoch 1", f"{BASE_CKPT_DIR}/checkpoints_long_run_full_v2/epoch_1.pt"),
    ("Full run, epoch 2", f"{BASE_CKPT_DIR}/checkpoints_long_run_full_v2/epoch_2.pt"),
    ("Full run, epoch 3", f"{BASE_CKPT_DIR}/checkpoints_long_run_full_v2/epoch_3.pt"),
    ("Full run, epoch 4", f"{BASE_CKPT_DIR}/checkpoints_long_run_full_v2/epoch_4.pt"),
]


def run_one(checkpoint_path, rgb_in, device, lora_rank=16):
    wrapper = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = wrapper.model
    inject_lora(net, rank=lora_rank)
    net = net.to(device)
    net.eval()

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(state, strict=False)

    with torch.no_grad():
        output = net(rgb_in, export_feat_layers=[])
    depth_pred = output.depth.squeeze(0).squeeze(0).cpu().numpy()

    del net
    torch.cuda.empty_cache()
    return depth_pred * UNIT_CORRECTION  # corrected to real meters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-index", type=int, default=0,
                     help="index into the held-out validation split (unseen during training)")
    ap.add_argument("--out", type=str, default="all_checkpoints_grid.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Building frame list...")
    frames = build_frame_list(H5_ROOT, JSON_ROOT)
    _, val_frames = split_frames(frames)
    ds = SCAREDDataset(val_frames, bad_files_path="bad_h5_files.txt")

    sample = ds[args.frame_index]
    print(f"Using validation frame index {args.frame_index} (unseen during training)")

    rgb_orig = sample["rgb"].unsqueeze(0)
    rgb_display = rgb_orig.squeeze(0).permute(1, 2, 0).numpy()

    rgb_in = F.interpolate(rgb_orig, size=TARGET_SIZE, mode="bilinear", align_corners=False)
    rgb_in = rgb_in.unsqueeze(1).to(device)

    depth_gt = sample["depth"].unsqueeze(0).unsqueeze(0)
    depth_gt = F.interpolate(depth_gt, size=TARGET_SIZE, mode="nearest").squeeze(0).squeeze(0)
    valid_mask = (depth_gt > 1e-4).numpy()

    print("Running inference for every checkpoint...")
    all_preds = []
    for label, ckpt in CHECKPOINTS:
        print(f"  {label}...")
        pred = run_one(ckpt, rgb_in, device)
        all_preds.append(pred)

    # shared color scale across every row, so rows are genuinely comparable
    all_valid_values = np.concatenate([p[valid_mask] for p in all_preds])
    vmin, vmax = np.percentile(all_valid_values, [1, 99])  # robust to outliers

    n = len(CHECKPOINTS)
    fig, axes = plt.subplots(n, 2, figsize=(9, 3.2 * n))

    for i, ((label, _), pred) in enumerate(zip(CHECKPOINTS, all_preds)):
        axes[i, 0].imshow(rgb_display)
        axes[i, 0].set_title(f"{label}\nRGB (unseen validation frame)", fontsize=9)
        axes[i, 0].axis("off")

        pred_display = np.where(valid_mask, pred, np.nan)
        im = axes[i, 1].imshow(pred_display, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"Predicted depth (m), corrected scale", fontsize=9)
        axes[i, 1].axis("off")

    fig.colorbar(im, ax=axes[:, 1], fraction=0.02, pad=0.02, label="Depth (m)")
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Saved grid to {args.out}")


if __name__ == "__main__":
    main()
