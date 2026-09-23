"""
Build a side-by-side video (RGB | predicted depth | error vs real ground
truth) for a white sawbone test sequence. Real depth ground truth exists
here, so this shows genuine accuracy, not a consistency proxy.

Data is already circle-cropped and resized to 1022x1022, real meters
(depth_png, uint16, raw*0.01 = mm, /1000 = meters, matching the
original red-sawbone convention) -- no further preprocessing needed here.

Usage:
    python -m arthronav.infer_video_sawbone_white \
        --seq-dir /mnt/areas_nas/SLAM/sawbone_white_dataset/test/20260917-134858-eeeeeeee \
        --checkpoint checkpoints/sawbone_white_from_sobone_v2/epoch_2.pt \
        --error-threshold-m 0.005 \
        --out sawbone_white_test_best_v2.mp4
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

from depth_anything_3.api import DepthAnything3

from arthronav.lora import inject_vector_lora


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
    vmin, vmax = np.nanmin(gt_masked), np.nanmax(gt_masked)

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
    ap.add_argument("--seq-dir", required=True,
                     help="e.g. .../sawbone_white_dataset/test/20260917-134858-eeeeeeee")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--error-threshold-m", type=float, default=0.005)
    ap.add_argument("--out", default="sawbone_white_test.mp4")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_model(args.checkpoint, device)

    rgb_dir = os.path.join(args.seq_dir, "rgb")
    depth_dir = os.path.join(args.seq_dir, "depth_png")
    rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    if args.max_frames:
        rgb_paths = rgb_paths[:args.max_frames]

    print(f"Processing {len(rgb_paths)} frames from {args.seq_dir}")

    writer = None
    all_mean_errs, all_max_errs = [], []

    with torch.no_grad():
        for i, rgb_path in enumerate(rgb_paths):
            frame_id = os.path.splitext(os.path.basename(rgb_path))[0]
            depth_path = os.path.join(depth_dir, f"{frame_id}.png")

            rgb_bgr = cv2.imread(rgb_path)
            rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth_gt = depth_raw.astype(np.float32) * 0.01 / 1000.0
            valid = depth_raw > 0

            rgb_t = torch.from_numpy(rgb_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            rgb_in = rgb_t.unsqueeze(1).to(device)

            output = net(rgb_in, export_feat_layers=[])
            pred_m = output.depth.squeeze(0).squeeze(0).cpu().numpy()

            err_map = np.abs(pred_m - depth_gt)
            err_mean = err_map[valid].mean() if valid.any() else 0.0
            err_max = err_map[valid].max() if valid.any() else 0.0
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
