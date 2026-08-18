"""
Run DA3 inference frame-by-frame on a real video that has no ground
truth, and build a side-by-side output video: RGB next to predicted
depth, with the region exceeding a configurable frame-to-frame error
threshold highlighted in solid red.

"Error" here is frame-to-frame depth consistency (how much predicted
depth at a given pixel changes between consecutive frames), not error
against ground truth, there isn't any for this video. It's a practical
stability/noise indicator for display purposes, not a rigorous metric:
real camera motion also shifts a pixel's true depth from one frame to
the next, and this doesn't correct for that (no pose/motion pipeline
exists yet, see the ATE item in the roadmap). Treat it as "how much did
this pixel's predicted depth jump", not "how wrong is this pixel".

Usage:
    python -m arthronav.infer_video \
        --video tour_lateral_0.mp4 \
        --checkpoint checkpoints/scared_training_checkpoints/checkpoints_long_run_full_v2/epoch_2.pt \
        --lora-mode uniform \
        --error-threshold-m 0.02 \
        --out tour_lateral_0_depth.mp4
"""

import argparse

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from depth_anything_3.api import DepthAnything3

from arthronav.lora import inject_lora, inject_vector_lora

TARGET_SIZE = (1022, 1274)  # nearest multiples of 14 below DA3's expected input
UNIT_CORRECTION = 0.256  # verified: raw h5-trained depth x 0.256 = real meters


def load_model(checkpoint_path, lora_mode, lora_rank, device):
    print("Loading model...")
    wrapper = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = wrapper.model
    if lora_mode == "vector":
        inject_vector_lora(net)
    else:
        inject_lora(net, rank=lora_rank)
    net = net.to(device)
    net.eval()

    state = torch.load(checkpoint_path, map_location=device)
    _, unexpected = net.load_state_dict(state, strict=False)
    loaded = len(state) - len(unexpected)
    if loaded == 0:
        print(f"WARNING: 0 tensors matched, check --lora-mode is correct for this checkpoint")
    print(f"Loaded {checkpoint_path} (mode={lora_mode}, {loaded} tensors matched)")
    return net


def predict_depth(net, frame_rgb_uint8, device):
    """frame_rgb_uint8: (H, W, 3) uint8, RGB order. Returns depth in real meters, (H, W)."""
    rgb = torch.from_numpy(frame_rgb_uint8).float() / 255.0
    rgb = rgb.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    rgb = F.interpolate(rgb, size=TARGET_SIZE, mode="bilinear", align_corners=False)
    rgb_in = rgb.unsqueeze(1).to(device)  # (1, 1, 3, H, W)

    with torch.no_grad():
        output = net(rgb_in, export_feat_layers=[])
    depth = output.depth.squeeze(0).squeeze(0).cpu().numpy() * UNIT_CORRECTION
    return depth


def make_panel(rgb_display, depth_m, err_mean_mm, err_max_mm, threshold_m, highlight_mask, fig_dpi=100):
    """Builds one side-by-side RGB | depth (with red highlight) panel as an (H, W, 3) uint8 array."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=fig_dpi)

    axes[0].imshow(rgb_display)
    axes[0].set_title("Input RGB", fontsize=10)
    axes[0].axis("off")

    vmin, vmax = np.percentile(depth_m, [1, 99])
    im = axes[1].imshow(depth_m, cmap="viridis", vmin=vmin, vmax=vmax)

    if highlight_mask is not None and highlight_mask.any():
        red_overlay = np.zeros((*depth_m.shape, 4))
        red_overlay[highlight_mask] = [1.0, 0.0, 0.0, 1.0]  # solid red, fully opaque
        axes[1].imshow(red_overlay)

    axes[1].set_title(
        f"Predicted depth (m) | frame-to-frame err: mean {err_mean_mm:.2f} mm, "
        f"max {err_max_mm:.2f} mm | red > {threshold_m*1000:.0f} mm",
        fontsize=9,
    )
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.04, pad=0.03, label="Depth (m)")

    plt.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    panel = buf[:, :, :3].copy()  # drop alpha
    plt.close(fig)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--lora-mode", type=str, default="uniform", choices=["uniform", "vector"])
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--error-threshold-m", type=float, default=0.02,
                     help="frame-to-frame depth change (meters) above which a pixel is highlighted red")
    ap.add_argument("--out", type=str, default="video_depth_output.mp4")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after this many processed frames")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_model(args.checkpoint, args.lora_mode, args.lora_rank, device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {args.video} | {total_frames} frames @ {src_fps:.1f} fps | stride={args.stride}")

    writer = None
    prev_depth = None
    frame_idx = 0
    processed = 0
    all_mean_errs, all_max_errs = [], []

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue
        if args.max_frames is not None and processed >= args.max_frames:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        depth_m = predict_depth(net, frame_rgb, device)

        if prev_depth is not None:
            err_map = np.abs(depth_m - prev_depth)
            err_mean_mm = err_map.mean() * 1000
            err_max_mm = err_map.max() * 1000
            highlight_mask = err_map > args.error_threshold_m
        else:
            err_map = None
            err_mean_mm = 0.0
            err_max_mm = 0.0
            highlight_mask = None

        panel = make_panel(frame_rgb, depth_m, err_mean_mm, err_max_mm, args.error_threshold_m, highlight_mask)

        if writer is None:
            h, w = panel.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), src_fps / args.stride, (w, h))

        writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

        if prev_depth is not None:
            all_mean_errs.append(err_mean_mm)
            all_max_errs.append(err_max_mm)

        prev_depth = depth_m
        frame_idx += 1
        processed += 1
        if processed % 20 == 0:
            print(f"  processed {processed} frames (mean err so far: "
                  f"{np.mean(all_mean_errs) if all_mean_errs else 0:.2f} mm)")

    cap.release()
    if writer is not None:
        writer.release()

    print(f"\nDone. Processed {processed} frames.")
    if all_mean_errs:
        print(f"Overall mean of per-frame mean err: {np.mean(all_mean_errs):.2f} mm")
        print(f"Overall max of per-frame max err:    {np.max(all_max_errs):.2f} mm")
    print(f"Saved output video to {args.out}")


if __name__ == "__main__":
    main()
