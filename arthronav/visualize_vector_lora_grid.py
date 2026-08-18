"""
Same idea as visualize_all_checkpoints_grid.py, but for the 5 Vector-LoRA
epochs specifically: RGB next to predicted depth (corrected to real
meters), same unseen validation frame, shared color scale across rows.

Usage:
    python -m arthronav.visualize_vector_lora_grid --frame-index 0
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from depth_anything_3.api import DepthAnything3

from arthronav.lora import inject_vector_lora
from arthronav.scared_io import build_frame_list, split_frames
from arthronav.scared_dataset import SCAREDDataset

H5_ROOT = "/mnt/areas_nas/SLAM/scared_dataset_full_copy/depth_anything_preprocessed_data/train_depth_anything"
JSON_ROOT = "/mnt/areas_nas/SLAM/scared_dataset_full_copy/frame_trajectory_data"
TARGET_SIZE = (1022, 1274)
BASE_CKPT_DIR = "checkpoints/scared_training_checkpoints/checkpoints_vector_lora_full"
UNIT_CORRECTION = 0.256

CHECKPOINTS = [
    ("Vector-LoRA, epoch 0", f"{BASE_CKPT_DIR}/epoch_0.pt"),
    ("Vector-LoRA, epoch 1", f"{BASE_CKPT_DIR}/epoch_1.pt"),
    ("Vector-LoRA, epoch 2", f"{BASE_CKPT_DIR}/epoch_2.pt"),
    ("Vector-LoRA, epoch 3", f"{BASE_CKPT_DIR}/epoch_3.pt"),
    ("Vector-LoRA, epoch 4", f"{BASE_CKPT_DIR}/epoch_4.pt"),
]


def run_one(checkpoint_path, rgb_in, device):
    wrapper = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = wrapper.model
    inject_vector_lora(net)
    net = net.to(device)
    net.eval()

    state = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = net.load_state_dict(state, strict=False)
    loaded = len(state) - len(unexpected)
    if loaded == 0:
        print(f"  WARNING: 0 tensors matched for {checkpoint_path}")

    with torch.no_grad():
        output = net(rgb_in, export_feat_layers=[])
    depth_pred = output.depth.squeeze(0).squeeze(0).cpu().numpy()

    del net
    torch.cuda.empty_cache()
    return depth_pred * UNIT_CORRECTION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-index", type=int, default=0)
    ap.add_argument("--out", type=str, default="vector_lora_grid.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Building frame list...")
    frames = build_frame_list(H5_ROOT, JSON_ROOT)
    _, val_frames = split_frames(frames)
    ds = SCAREDDataset(val_frames, bad_files_path="bad_h5_files.txt")

    sample = ds[args.frame_index]
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
        all_preds.append(run_one(ckpt, rgb_in, device))

    all_valid_values = np.concatenate([p[valid_mask] for p in all_preds])
    vmin, vmax = np.percentile(all_valid_values, [1, 99])

    n = len(CHECKPOINTS)
    fig, axes = plt.subplots(n, 2, figsize=(9, 3.2 * n))

    for i, ((label, _), pred) in enumerate(zip(CHECKPOINTS, all_preds)):
        axes[i, 0].imshow(rgb_display)
        axes[i, 0].set_title(f"{label}\nRGB (unseen validation frame)", fontsize=9)
        axes[i, 0].axis("off")

        pred_display = np.where(valid_mask, pred, np.nan)
        im = axes[i, 1].imshow(pred_display, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title("Predicted depth (m), corrected scale", fontsize=9)
        axes[i, 1].axis("off")

    fig.colorbar(im, ax=axes[:, 1], fraction=0.02, pad=0.02, label="Depth (m)")
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Saved grid to {args.out}")


if __name__ == "__main__":
    main()
