"""
Build a side-by-side video (RGB | predicted depth | error vs real ground
truth) for the held-out TEST patient, which has never been used in
training or validation for any run. This test patient has real depth
ground truth, so this shows genuine accuracy, not a consistency proxy.

Optionally applies the same optical-flow-based temporal smoothing used
in infer_video_depth_stats.py, and -- since real ground truth exists
here -- reports BOTH raw and filtered error against it, so we can see
directly whether smoothing helps or hurts actual accuracy, not just
visual flicker. The video displays the filtered prediction when
smoothing is enabled; both numbers are always printed/logged so the
tradeoff is visible either way.

Frames are the ones already extracted by prepare_real_knee_data.py
(only frames with matching depth ground truth; playback may be slightly
choppy if any real video frames were skipped during reconstruction).

Usage:
    python -m arthronav.infer_video_real_knee \
        --clip-dir /mnt/areas_nas/SLAM/real_knee_dataset/2509457F_medial \
        --checkpoint checkpoints/real_knee_training/from_scratch_vector/epoch_0.pt \
        --vector-lora \
        --error-threshold-mm 5.0 \
        --temporal-smoothing 0.4 \
        --out test_patient_medial_depth_smoothed.mp4
"""

import argparse
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from depth_anything_3.api import DepthAnything3

from arthronav.lora import inject_lora, inject_vector_lora

TARGET_SIZE = (1078, 1918)
FRAME_RE = re.compile(r"frame_(\d{6})\.depth\.npy$")


def load_model(checkpoint_path, vector_lora, lora_rank, device):
    print("Loading model...")
    wrapper = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = wrapper.model
    if vector_lora:
        inject_vector_lora(net)
    else:
        inject_lora(net, rank=lora_rank)
    net = net.to(device)
    net.eval()

    state = torch.load(checkpoint_path, map_location=device)
    _, unexpected = net.load_state_dict(state, strict=False)
    loaded = len(state) - len(unexpected)
    if loaded == 0:
        print("WARNING: 0 tensors matched -- check --vector-lora matches this checkpoint")
    print(f"Loaded {checkpoint_path} ({loaded} tensors matched)")
    return net


def warp_prev_depth(prev_depth, prev_gray, curr_gray):
    """Same optical-flow warp as infer_video_depth_stats.py -- see that
    file's docstring for the rationale and limits."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    h, w = prev_depth.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(prev_depth, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    return warped


def make_panel(rgb_display, pred_mm, gt_mm, valid_mask, err_mean_mm, err_max_mm,
                threshold_mm, smoothing_label, fig_dpi=100):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=fig_dpi)

    axes[0].imshow(rgb_display)
    axes[0].set_title("Input RGB", fontsize=10)
    axes[0].axis("off")

    gt_masked = np.where(valid_mask, gt_mm, np.nan)
    vmin, vmax = np.nanmin(gt_masked), np.nanmax(gt_masked)

    pred_masked = np.where(valid_mask, pred_mm, np.nan)
    axes[1].imshow(pred_masked, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Predicted depth (mm){smoothing_label}", fontsize=10)
    axes[1].axis("off")

    err_map = np.where(valid_mask, np.abs(pred_mm - gt_mm), np.nan)
    im = axes[2].imshow(err_map, cmap="hot", vmin=0, vmax=threshold_mm * 2)
    highlight = valid_mask & (np.abs(pred_mm - gt_mm) > threshold_mm)
    if highlight.any():
        red_overlay = np.zeros((*err_map.shape, 4))
        red_overlay[highlight] = [0.0, 1.0, 0.0, 1.0]
        axes[2].imshow(red_overlay)
    axes[2].set_title(
        f"|error| vs real GT (mm){smoothing_label}\nmean {err_mean_mm:.2f}, max {err_max_mm:.2f}, "
        f"green > {threshold_mm:.1f}mm", fontsize=9)
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.04, pad=0.03, label="Error (mm)")

    plt.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    panel = buf[:, :, :3].copy()
    plt.close(fig)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-dir", required=True, help="e.g. .../real_knee_dataset/2509457F_medial")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vector-lora", action="store_true")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--error-threshold-mm", type=float, default=5.0)
    ap.add_argument("--out", default="test_patient_depth.mp4")
    ap.add_argument("--csv-out", default=None,
                     help="optional path to write per-frame raw vs filtered error CSV")
    ap.add_argument("--fps", type=float, default=10.0,
                     help="display fps -- frames aren't temporally sequential in the "
                          "original video, so this is a reasonable playback rate, not "
                          "the source video's real fps")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--temporal-smoothing", type=float, default=0.0,
                     help="0.0 = off (default). Blends each frame's raw prediction with "
                          "the previous frame's depth, motion-compensated via optical "
                          "flow. The video displays the FILTERED prediction; both raw "
                          "and filtered error against real ground truth are printed at "
                          "the end for direct comparison.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_model(args.checkpoint, args.vector_lora, args.lora_rank, device)

    depth_dir = os.path.join(args.clip_dir, "depth")
    rgb_dir = os.path.join(args.clip_dir, "rgb")
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.depth.npy")))
    frame_indices = sorted(int(FRAME_RE.search(f).group(1)) for f in depth_files if FRAME_RE.search(f))
    if args.max_frames:
        frame_indices = frame_indices[:args.max_frames]

    smoothing_label = f" (filtered, w={args.temporal_smoothing})" if args.temporal_smoothing > 0 else " (raw)"
    print(f"Processing {len(frame_indices)} frames from {args.clip_dir}"
          f"{' with temporal smoothing ' + str(args.temporal_smoothing) if args.temporal_smoothing > 0 else ''}")

    writer = None
    raw_mean_errs, raw_max_errs = [], []
    filt_mean_errs, filt_max_errs = [], []
    csv_rows = []

    prev_pred_native = None
    prev_gray = None

    with torch.no_grad():
        for i, frame_idx in enumerate(frame_indices):
            rgb_bgr = cv2.imread(os.path.join(rgb_dir, f"frame_{frame_idx:06d}.jpg"))
            rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

            depth_gt = np.load(os.path.join(depth_dir, f"frame_{frame_idx:06d}.depth.npy")).astype(np.float32)
            valid = np.load(os.path.join(depth_dir, f"frame_{frame_idx:06d}.valid.npy")).astype(bool)

            rgb_t = torch.from_numpy(rgb_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            rgb_t = F.interpolate(rgb_t, size=TARGET_SIZE, mode="bilinear", align_corners=False)
            rgb_in = rgb_t.unsqueeze(1).to(device)

            output = net(rgb_in, export_feat_layers=[])
            pred_mm = output.depth.squeeze(0).squeeze(0).cpu().numpy()
            pred_mm_native = cv2.resize(pred_mm, (depth_gt.shape[1], depth_gt.shape[0]),
                                         interpolation=cv2.INTER_LINEAR)

            raw_err_map = np.abs(pred_mm_native - depth_gt)
            raw_err_mean = raw_err_map[valid].mean() if valid.any() else 0.0
            raw_err_max = raw_err_map[valid].max() if valid.any() else 0.0
            raw_mean_errs.append(raw_err_mean)
            raw_max_errs.append(raw_err_max)

            curr_gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
            if args.temporal_smoothing > 0.0 and prev_pred_native is not None:
                warped_prev = warp_prev_depth(prev_pred_native, prev_gray, curr_gray)
                filt_pred = (args.temporal_smoothing * warped_prev
                             + (1 - args.temporal_smoothing) * pred_mm_native)
            else:
                filt_pred = pred_mm_native

            filt_err_map = np.abs(filt_pred - depth_gt)
            filt_err_mean = filt_err_map[valid].mean() if valid.any() else 0.0
            filt_err_max = filt_err_map[valid].max() if valid.any() else 0.0
            filt_mean_errs.append(filt_err_mean)
            filt_max_errs.append(filt_err_max)

            prev_pred_native = pred_mm_native
            prev_gray = curr_gray

            panel = make_panel(rgb_rgb, filt_pred, depth_gt, valid,
                                filt_err_mean, filt_err_max, args.error_threshold_mm, smoothing_label)

            if writer is None:
                h, w = panel.shape[:2]
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

            csv_rows.append([frame_idx, raw_err_mean, raw_err_max, filt_err_mean, filt_err_max])

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(frame_indices)} frames | raw mean err: {np.mean(raw_mean_errs):.2f} mm "
                      f"| filtered mean err: {np.mean(filt_mean_errs):.2f} mm")

    if writer is not None:
        writer.release()

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_index", "raw_mean_err_mm", "raw_max_err_mm", "filtered_mean_err_mm", "filtered_max_err_mm"])
            w.writerows(csv_rows)
        print(f"Per-frame CSV saved to {args.csv_out}")

    print(f"\nDone. {len(frame_indices)} frames processed.")
    print(f"--- RAW prediction vs ground truth ---")
    print(f"  Overall mean error: {np.mean(raw_mean_errs):.3f} mm")
    print(f"  Overall max error:  {np.max(raw_max_errs):.3f} mm")
    print(f"--- FILTERED prediction vs ground truth (temporal-smoothing={args.temporal_smoothing}) ---")
    print(f"  Overall mean error: {np.mean(filt_mean_errs):.3f} mm")
    print(f"  Overall max error:  {np.max(filt_max_errs):.3f} mm")
    print(f"Saved video to {args.out}")


if __name__ == "__main__":
    main()
