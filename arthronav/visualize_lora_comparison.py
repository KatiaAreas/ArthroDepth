"""
Focused comparison: uniform LoRA's best checkpoint next to Vector-LoRA's
best checkpoint, same unseen validation frame, same color scale, so the
two methods are directly comparable in one image rather than buried in
the full 15-row grid.

Usage:
    python -m arthronav.visualize_lora_comparison --frame-index 0
"""

import argparse

import matplotlib
matplotlib.use("Agg")
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
BASE_CKPT_DIR = "checkpoints/scared_training_checkpoints"
UNIT_CORRECTION = 0.256

ROWS = [
    ("Uniform LoRA (epoch 2, best)", f"{BASE_CKPT_DIR}/checkpoints_long_run_full_v2/epoch_2.pt", "uniform"),
    ("Vector-LoRA (epoch 3, best)", f"{BASE_CKPT_DIR}/checkpoints_vector_lora_full/epoch_3.pt", "vector"),
]


def run_one(checkpoint_path, lora_mode, rgb_in, device):
    wrapper = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = wrapper.model
    if lora_mode == "vector":
        inject_vector_lora(net)
    else:
        inject_lora(net, rank=16)
    net = net.to(device)
    net.eval()

    state = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state, strict=False)

    with torch.no_grad():
        output = net(rgb_in, export_feat_layers=[])
    depth_pred = output.depth.squeeze(0).squeeze(0).cpu().numpy()

    del net
    torch.cuda.empty_cache()
    return depth_pred * UNIT_CORRECTION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-index", type=int, default=0)
    ap.add_argument("--out", type=str, default="lora_comparison.png")
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
    depth_gt = (depth_gt * UNIT_CORRECTION).numpy()
    valid_mask = depth_gt > (1e-4 * UNIT_CORRECTION)

    preds = []
    for label, ckpt, mode in ROWS:
        print(f"Running {label}...")
        preds.append(run_one(ckpt, mode, rgb_in, device))

    all_valid = np.concatenate([p[valid_mask] for p in preds])
    vmin, vmax = np.percentile(all_valid, [1, 99])

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for i, ((label, _, _), pred) in enumerate(zip(ROWS, preds)):
        axes[i, 0].imshow(rgb_display)
        axes[i, 0].set_title(f"{label}\nRGB (unseen validation frame)", fontsize=9)
        axes[i, 0].axis("off")

        pred_display = np.where(valid_mask, pred, np.nan)
        im = axes[i, 1].imshow(pred_display, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title("Predicted depth (m)", fontsize=9)
        axes[i, 1].axis("off")

    fig.colorbar(im, ax=axes[:, 1], fraction=0.04, pad=0.03, label="Depth (m)")
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
