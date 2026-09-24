"""
Build a comparison video over a held-out sawbone sequence: RGB alongside
zero-shot / transfer_scared / from_scratch depth predictions, with
per-frame error metrics overlaid and pixels exceeding 1cm/5cm error
highlighted in orange/red.

Usage:
    python -m arthronav.make_comparison_video --sequence tour_lateral_3_cartilage
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.cm as cm

from depth_anything_3.api import DepthAnything3
from arthronav.lora import inject_vector_lora
from arthronav.sawbone_io import build_frame_list
from arthronav.sawbone_dataset import SawboneDataset
from arthronav.test_sawbone import prepare_batch, SAWBONE_ROOT

PANEL_LABELS = {"zero_shot": "Zero-shot", "transfer_scared": "Transfer (SCARED)", "from_scratch": "From scratch"}


def load_model(mode, device, checkpoint_path=None):
    wrapper = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = wrapper.model
    if mode != "zero_shot":
        inject_vector_lora(net)
        state = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(state, strict=False)
    net = net.to(device)
    net.eval()
    return net


def depth_to_color(depth_m, vmin, vmax):
    norm = np.clip((depth_m - vmin) / (vmax - vmin + 1e-8), 0, 1)
    colored = (cm.viridis(norm)[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)


def overlay_outliers(color_img, error_m, threshold_1cm=0.01, threshold_5cm=0.05):
    out = color_img.copy()
    mask_1cm = (error_m > threshold_1cm) & (error_m <= threshold_5cm)
    mask_5cm = error_m > threshold_5cm
    out[mask_1cm] = [0, 100, 255]
    out[mask_5cm] = [0, 0, 255]
    return out


def put_metrics_text(img, lines, origin=(10, 25)):
    for i, line in enumerate(lines):
        y = origin[1] + i * 22
        cv2.putText(img, line, (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", type=str, default="tour_lateral_3_cartilage")
    ap.add_argument("--transfer-checkpoint", type=str,
                     default="checkpoints/sawbone_transfer_scared/epoch_4.pt")
    ap.add_argument("--scratch-checkpoint", type=str,
                     default="checkpoints/sawbone_from_scratch/epoch_4.pt")
    ap.add_argument("--out", type=str, default="sawbone_comparison.mp4")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models on {device}...")

    models = {
        "zero_shot": load_model("zero_shot", device),
        "transfer_scared": load_model("transfer_scared", device, args.transfer_checkpoint),
        "from_scratch": load_model("from_scratch", device, args.scratch_checkpoint),
    }

    frames = build_frame_list(Path(SAWBONE_ROOT), [args.sequence])
    if args.max_frames:
        frames = frames[:args.max_frames]
    ds = SawboneDataset(frames)
    print(f"Running on {len(ds)} frames from {args.sequence}")

    all_gt_vals = []
    for i in range(len(ds)):
        depth = ds[i]["depth"].numpy()
        valid = depth > 1e-4
        if valid.any():
            all_gt_vals.append(depth[valid])
    all_gt_vals = np.concatenate(all_gt_vals)
    vmin, vmax = np.percentile(all_gt_vals, [1, 99])
    print(f"Depth color scale: {vmin:.4f} - {vmax:.4f} m")

    writer = None
    for idx in range(len(ds)):
        sample = ds[idx]
        rgb_display = (sample["rgb"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        rgb_display_bgr = cv2.cvtColor(rgb_display, cv2.COLOR_RGB2BGR)

        batch = {k: v.unsqueeze(0) for k, v in sample.items()}
        rgb_in, depth_gt, valid_mask = prepare_batch(batch, device)

        panels = [cv2.resize(rgb_display_bgr, (rgb_in.shape[-1], rgb_in.shape[-2]))]

        for mode_name in ["zero_shot", "transfer_scared", "from_scratch"]:
            with torch.no_grad():
                output = models[mode_name](rgb_in, export_feat_layers=[])
            pred = output.depth.squeeze(1)

            error = torch.zeros_like(depth_gt)
            error[valid_mask] = (pred[valid_mask] - depth_gt[valid_mask]).abs()

            pred_np = pred.squeeze(0).cpu().numpy()
            error_np = error.squeeze(0).cpu().numpy()
            mask_np = valid_mask.squeeze(0).cpu().numpy()

            color = depth_to_color(np.where(mask_np, pred_np, vmin), vmin, vmax)
            color = overlay_outliers(color, np.where(mask_np, error_np, 0.0))

            if mask_np.sum() > 0:
                mean_err_mm = error_np[mask_np].mean() * 1000
                pct_1cm = (error_np[mask_np] > 0.01).mean() * 100
                pct_5cm = (error_np[mask_np] > 0.05).mean() * 100
            else:
                mean_err_mm, pct_1cm, pct_5cm = float("nan"), float("nan"), float("nan")

            color = put_metrics_text(color, [
                PANEL_LABELS[mode_name],
                f"mean err: {mean_err_mm:.2f}mm",
                f">1cm: {pct_1cm:.1f}%  >5cm: {pct_5cm:.2f}%",
            ])
            panels.append(color)

        panels[0] = put_metrics_text(panels[0].copy(), ["RGB", f"frame {idx+1}/{len(ds)}"])

        top = np.hstack([panels[0], panels[1]])
        bottom = np.hstack([panels[2], panels[3]])
        grid = np.vstack([top, bottom])

        if writer is None:
            h, w = grid.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

        writer.write(grid)
        if (idx + 1) % 20 == 0:
            print(f"  frame {idx+1}/{len(ds)}")

    writer.release()
    print(f"\nSaved video to {args.out}")


if __name__ == "__main__":
    main()
