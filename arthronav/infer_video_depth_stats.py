"""
Run depth inference on one or more raw videos that have no ground
truth. For each video, builds a side-by-side output video: RGB next to
predicted depth, with real per-frame min/max depth reported, and the
fraction of pixels ABOVE a configurable threshold highlighted in a
configurable color.

No error metric here since there's no ground truth -- this is a
visualization/deployment tool, not an accuracy evaluation. For accuracy
evaluation, see validate_real_knee.py (needs ground truth) or
infer_video.py (frame-to-frame consistency proxy, also no ground truth
needed but measures stability, not this script's min/max/threshold view).

Default checkpoint: from-scratch, epoch 0, our current best real-knee
result (AbsRel 0.1015, mean error 1.754mm on the held-out validation
patient -- see the report for the full comparison against transfer
learning from SCARED).

Usage:
    # single video
    python -m arthronav.infer_video_depth_stats \
        --videos my_video.mp4 \
        --threshold-mm 30 \
        --out-dir depth_outputs/

    # multiple videos in one run
    python -m arthronav.infer_video_depth_stats \
        --videos video1.mp4 video2.mp4 video3.mp4 \
        --threshold-mm 30 \
        --out-dir depth_outputs/

    # or point at a whole directory of .mp4 files
    python -m arthronav.infer_video_depth_stats \
        --video-dir /path/to/videos/ \
        --threshold-mm 30 \
        --out-dir depth_outputs/

    # with temporal smoothing to reduce flicker during slow motion
    python -m arthronav.infer_video_depth_stats \
        --videos my_video.mp4 \
        --threshold-mm 30 \
        --temporal-smoothing 0.4 \
        --out-dir depth_outputs/
"""

import argparse
import csv
import glob
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from depth_anything_3.api import DepthAnything3

from arthronav.lora import inject_lora, inject_vector_lora

TARGET_SIZE = (1078, 1918)  # nearest multiples of 14 below 1080x1920; adjust if source differs

DEFAULT_CHECKPOINT = "checkpoints/real_knee_training/from_scratch_vector/epoch_0.pt"


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
        print(f"WARNING: 0 tensors matched from {checkpoint_path} -- check --vector-lora "
              f"matches how this checkpoint was actually trained.")
    print(f"Loaded {checkpoint_path} ({loaded} tensors matched)")
    return net


def make_panel(rgb_display, depth, threshold, highlight_color_rgb, fig_dpi=100):
    d_min, d_max = float(depth.min()), float(depth.max())
    pct_above = 100.0 * (depth > threshold).mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=fig_dpi)

    axes[0].imshow(rgb_display)
    axes[0].set_title("Input RGB", fontsize=10)
    axes[0].axis("off")

    vmin, vmax = np.percentile(depth, [1, 99])
    im = axes[1].imshow(depth, cmap="viridis", vmin=vmin, vmax=vmax)

    highlight = depth > threshold
    if highlight.any():
        overlay = np.zeros((*depth.shape, 4))
        overlay[highlight] = [*highlight_color_rgb, 1.0]
        axes[1].imshow(overlay)

    axes[1].set_title(
        f"Predicted depth | min {d_min:.2f}, max {d_max:.2f} | "
        f"{pct_above:.1f}% > {threshold:.1f} (highlighted)",
        fontsize=9,
    )
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.04, pad=0.03, label="Depth")

    plt.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    panel = buf[:, :, :3].copy()
    plt.close(fig)
    return panel, d_min, d_max, pct_above


def warp_prev_depth(prev_depth, prev_gray, curr_gray):
    """
    Estimates 2D pixel motion from prev_gray to curr_gray via optical flow
    (Farneback, no extra dependencies), then warps prev_depth to align
    with the current frame's viewpoint. This approximates temporal
    consistency without needing known camera pose -- works on any video,
    not just the patients we have OptiTrack tracking for. Breaks down
    for fast motion or large occlusions (inherent limit of flow-based
    warping, not specific to this implementation) -- best suited to the
    slow, continuous motion this was asked about.
    """
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


def process_video(video_path, net, device, threshold, highlight_color_rgb, out_dir,
                   stride, max_frames, fps_override, temporal_smoothing=0.0):
    name = os.path.splitext(os.path.basename(video_path))[0]
    out_video_path = os.path.join(out_dir, f"{name}_depth.mp4")
    out_csv_path = os.path.join(out_dir, f"{name}_depth_stats.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"WARNING: could not open {video_path}, skipping")
        return
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = fps_override if fps_override else src_fps / stride
    print(f"\n=== {video_path} ===")
    print(f"{total_frames} frames @ {src_fps:.1f} fps, stride={stride}, output fps={out_fps:.1f}")

    writer = None
    frame_idx = 0
    processed = 0
    csv_rows = []
    prev_depth = None
    prev_gray = None

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            if max_frames is not None and processed >= max_frames:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb_t = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            rgb_t = F.interpolate(rgb_t, size=TARGET_SIZE, mode="bilinear", align_corners=False)
            rgb_in = rgb_t.unsqueeze(1).to(device)

            output = net(rgb_in, export_feat_layers=[])
            depth_raw = output.depth.squeeze(0).squeeze(0).cpu().numpy()

            if temporal_smoothing > 0.0 and prev_depth is not None:
                curr_gray = cv2.cvtColor(cv2.resize(frame_bgr, (depth_raw.shape[1], depth_raw.shape[0])),
                                          cv2.COLOR_BGR2GRAY)
                warped_prev = warp_prev_depth(prev_depth, prev_gray, curr_gray)
                depth = temporal_smoothing * warped_prev + (1 - temporal_smoothing) * depth_raw
            else:
                depth = depth_raw
                curr_gray = cv2.cvtColor(cv2.resize(frame_bgr, (depth_raw.shape[1], depth_raw.shape[0])),
                                          cv2.COLOR_BGR2GRAY)

            prev_depth = depth
            prev_gray = curr_gray

            panel, d_min, d_max, pct_above = make_panel(
                frame_rgb, depth, threshold, highlight_color_rgb)

            if writer is None:
                h, w = panel.shape[:2]
                writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                          out_fps, (w, h))
            writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

            csv_rows.append([frame_idx, d_min, d_max, pct_above])
            frame_idx += 1
            processed += 1
            if processed % 50 == 0:
                print(f"  {processed} frames processed")

    if writer is not None:
        writer.release()
    cap.release()

    with open(out_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "min_depth", "max_depth", "pct_pixels_above_threshold"])
        w.writerows(csv_rows)

    print(f"Done: {processed} frames -> {out_video_path}")
    print(f"Stats CSV -> {out_csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=None, help="one or more video paths")
    ap.add_argument("--video-dir", default=None, help="alternative to --videos: a directory of .mp4 files")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--vector-lora", action="store_true", default=True,
                     help="default True to match the default checkpoint's mode; "
                          "pass --no-vector-lora if using a uniform-LoRA checkpoint instead")
    ap.add_argument("--no-vector-lora", dest="vector_lora", action="store_false")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--threshold-mm", type=float, required=True,
                     help="pixels with predicted depth above this value get highlighted")
    ap.add_argument("--highlight-color", default="1.0,0.0,0.0",
                     help="R,G,B floats 0-1, default red")
    ap.add_argument("--out-dir", default="depth_outputs")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps", type=float, default=None, help="override output fps")
    ap.add_argument("--temporal-smoothing", type=float, default=0.0,
                     help="0.0 = off (default). 0.3-0.5 is a reasonable starting range: "
                          "blends each frame's raw prediction with the previous frame's "
                          "depth, motion-compensated via optical flow, to reduce flicker "
                          "during slow, continuous motion. Higher = smoother but more lag "
                          "behind real changes; breaks down under fast motion since optical "
                          "flow itself gets less reliable.")
    args = ap.parse_args()

    if args.videos is None and args.video_dir is None:
        raise SystemExit("Provide either --videos (one or more paths) or --video-dir")

    video_list = list(args.videos) if args.videos else sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    if not video_list:
        raise SystemExit("No videos found to process")

    os.makedirs(args.out_dir, exist_ok=True)
    highlight_color_rgb = tuple(float(x) for x in args.highlight_color.split(","))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_model(args.checkpoint, args.vector_lora, args.lora_rank, device)

    print(f"\nProcessing {len(video_list)} video(s), threshold={args.threshold_mm}, "
          f"highlight color (RGB)={highlight_color_rgb}")

    for video_path in video_list:
        process_video(video_path, net, device, args.threshold_mm, highlight_color_rgb,
                      args.out_dir, args.stride, args.max_frames, args.fps,
                      args.temporal_smoothing)

    print("\nAll videos processed.")


if __name__ == "__main__":
    main()
