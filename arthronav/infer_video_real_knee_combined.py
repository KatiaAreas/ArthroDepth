"""
Build a side-by-side video (RGB | predicted depth | error vs real
ground truth) for one held-out test clip from the combined real-knee
dataset. Real depth ground truth exists here, so this shows genuine
accuracy on real patient tissue, not a consistency proxy.

Data is already circle-cropped and resized to 1022x1022, real meters
(converted from the original mm convention during the combine step).

Usage:
    python -m arthronav.infer_video_real_knee_combined \
        --patient 2509457F --view lateral \
        --checkpoint checkpoints/real_knee_combined_from_scratch/epoch_1.pt \
        --error-threshold-m 0.005 \
        --out real_knee_combined_test_2509457F_lateral.mp4
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

from depth_anything_3.api import DepthAnything3

from arthronav.lora import inject_vector_lora

DATA_ROOT = "/mnt/areas_nas/SLAM/real_knee_combined_dataset"


def load_model(checkpoint_path, device):
    print("Loading model...")
    wrapper = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = wrapper.model
    inject_vector_lora(net)
    net = net.to(device)
    net.eval()

    state = torch.load(checkpoint_path, map_location=device)
    _, unexpected = net.load_state_dict(state, strict=False)
    loaded = len(state) - len(unexpected)
    if loaded == 0:
        print("WARNING: 0 tensors matched -- check this checkpoint was trained with Vector-LoRA")
    print(f"Loaded {checkpoint_path} ({loaded} tensors matched)")
    return net


def make_panel(rgb_display, pred_m, gt_m, valid_mask, err_mean_m, err_max_m,
                threshold_m, fig_dpi=100):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=fig_dpi)

    axes[0].imshow(rgb_display)
    axes[0].set_title("Input RGB", fontsize=10)
    axes[0].axis("off")

    gt_masked = np.where(valid_mask, gt_m, np.nan)
    if valid_mask.any():
        vmin, vmax = np.nanmin(gt_masked), np.nanmax(gt_masked)
    else:
        vmin, vmax = 0, 1

    pred_masked = np.where(valid_mask, pred_m, np.nan)
    axes[1].imshow(pred_masked, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Predicted depth (m)", fontsize=10)
    axes[1].axis("off")

    err_map = np.where(valid_mask, np.abs(pred_m - gt_m), np.nan)
    im = axes[2].imshow(err_map, cmap="hot", vmin=0, vmax=threshold_m * 2)
    highlight = valid_mask & (np.abs(pred_m - gt_m) > threshold_m)
    if highlight.any():
        red_overlay = np.zeros((*err_map.shape, 4))
        red_overlay[highlight] = [0.0, 1.0, 0.0, 1.0]
        axes[2].imshow(red_overlay)
    axes[2].set_title(
        f"|error| vs real GT (m)\nmean {err_mean_m:.4f}, max {err_max_m:.4f}, "
        f"green > {threshold_m:.4f}m", fontsize=9)
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.04, pad=0.03, label="Error (m)")

    plt.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    panel = buf[:, :, :3].copy()
    plt.close(fig)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True)
    ap.add_argument("--view", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--error-threshold-m", type=float, default=0.005)
    ap.add_argument("--out", default="real_knee_combined_video.mp4")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_model(args.checkpoint, device)

    prefix = f"{args.patient}_{args.view}_"
    rgb_dir = os.path.join(args.data_root, args.split, "rgb")
    depth_dir = os.path.join(args.data_root, args.split, "depth")

    all_rgb = glob.glob(os.path.join(rgb_dir, f"{prefix}*.jpg"))
    frame_re = re.compile(re.escape(prefix) + r"(\d{6})\.jpg$")
    indexed = []
    for p in all_rgb:
        m = frame_re.search(os.path.basename(p))
        if m:
            indexed.append((int(m.group(1)), p))
    indexed.sort()

    if not indexed:
        raise RuntimeError(f"No frames found for prefix '{prefix}' under {rgb_dir}. "
                            f"Check --patient/--view/--split match an actual clip.")

    rgb_paths = [p for _, p in indexed]
    if args.max_frames:
        rgb_paths = rgb_paths[:args.max_frames]

    print(f"Processing {len(rgb_paths)} frames for {args.patient}_{args.view} ({args.split})")

    writer = None
    all_mean_errs, all_max_errs = [], []

    with torch.no_grad():
        for i, rgb_path in enumerate(rgb_paths):
            frame_id = os.path.splitext(os.path.basename(rgb_path))[0]
            depth_path = os.path.join(depth_dir, f"{frame_id}.depth.npy")
            valid_path = os.path.join(depth_dir, f"{frame_id}.valid.npy")

            rgb_bgr = cv2.imread(rgb_path)
            rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

            depth_gt = np.load(depth_path).astype(np.float32)
            valid = np.load(valid_path)

            rgb_t = torch.from_numpy(rgb_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            rgb_in = rgb_t.unsqueeze(1).to(device)

            output = net(rgb_in, export_feat_layers=[])
            pred_m = output.depth.squeeze(0).squeeze(0).cpu().numpy()

            if valid.any():
                err_map = np.abs(pred_m - depth_gt)
                err_mean = err_map[valid].mean()
                err_max = err_map[valid].max()
            else:
                err_mean = err_max = 0.0
            all_mean_errs.append(err_mean)
            all_max_errs.append(err_max)

            panel = make_panel(rgb_rgb, pred_m, depth_gt, valid,
                                err_mean, err_max, args.error_threshold_m)

            if writer is None:
                h, w = panel.shape[:2]
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(rgb_paths)} frames | running mean err: "
                      f"{np.mean(all_mean_errs):.4f} m")

    if writer is not None:
        writer.release()

    print(f"\nDone. {len(rgb_paths)} frames processed.")
    print(f"Overall mean error: {np.mean(all_mean_errs):.4f} m")
    print(f"Overall max error:  {np.max(all_max_errs):.4f} m")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
