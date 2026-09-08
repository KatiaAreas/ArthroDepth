"""
Two-pass still selection for the report:
  Pass 1: compute per-frame mean error (all three modes) across a full
          sequence, fast, no image rendering.
  Pass 2: render only the selected representative frames (best, median,
          worst) as full-resolution standalone PNGs, matching the
          4-panel layout used in make_comparison_video.py.

Usage:
    python -m arthronav.select_and_render_stills --sequence tour_lateral_3_cartilage
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.cm as cm

from depth_anything_3.api import DepthAnything3
from arthronav.lora import inject_vector_lora
from arthronav.sobone_io import build_frame_list
from arthronav.sobone_dataset import SoboneDataset
from arthronav.test_sobone import prepare_batch, SOBONE_ROOT

MODES = ["zero_shot", "transfer_scared", "from_scratch"]
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


def put_metrics_text(img, lines, origin=(10, 30)):
    for i, line in enumerate(lines):
        y = origin[1] + i * 30
        cv2.putText(img, line, (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, line, (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def compute_frame_metrics(models, ds, device):
    """Pass 1: per-frame mean error for every mode, no image rendering."""
    rows = []
    for idx in range(len(ds)):
        sample = ds[idx]
        batch = {k: v.unsqueeze(0) for k, v in sample.items()}
        rgb_in, depth_gt, valid_mask = prepare_batch(batch, device)

        row = {"frame_idx": idx}
        for mode_name in MODES:
            with torch.no_grad():
                output = models[mode_name](rgb_in, export_feat_layers=[])
            pred = output.depth.squeeze(1)
            if valid_mask.sum() > 0:
                err = (pred[valid_mask] - depth_gt[valid_mask]).abs()
                row[f"{mode_name}_mean_err_mm"] = err.mean().item() * 1000
            else:
                row[f"{mode_name}_mean_err_mm"] = float("nan")
        rows.append(row)
        if (idx + 1) % 20 == 0:
            print(f"  pass 1: {idx+1}/{len(ds)}")
    return rows


def render_frame_grid(models, ds, idx, vmin, vmax, device):
    """Pass 2: full-resolution 4-panel render for one specific frame."""
    sample = ds[idx]
    rgb_display = (sample["rgb"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    rgb_display_bgr = cv2.cvtColor(rgb_display, cv2.COLOR_RGB2BGR)

    batch = {k: v.unsqueeze(0) for k, v in sample.items()}
    rgb_in, depth_gt, valid_mask = prepare_batch(batch, device)

    panels = [cv2.resize(rgb_display_bgr, (rgb_in.shape[-1], rgb_in.shape[-2]))]
    panels[0] = put_metrics_text(panels[0].copy(), ["RGB"])

    for mode_name in MODES:
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

    top = np.hstack([panels[0], panels[1]])
    bottom = np.hstack([panels[2], panels[3]])
    return np.vstack([top, bottom])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", type=str, default="tour_lateral_3_cartilage")
    ap.add_argument("--transfer-checkpoint", type=str,
                     default="checkpoints/sobone_transfer_scared/epoch_4.pt")
    ap.add_argument("--scratch-checkpoint", type=str,
                     default="checkpoints/sobone_from_scratch/epoch_4.pt")
    ap.add_argument("--out-dir", type=str, default="sobone_stills")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models on {device}...")
    models = {
        "zero_shot": load_model("zero_shot", device),
        "transfer_scared": load_model("transfer_scared", device, args.transfer_checkpoint),
        "from_scratch": load_model("from_scratch", device, args.scratch_checkpoint),
    }

    frames = build_frame_list(Path(SOBONE_ROOT), [args.sequence])
    ds = SoboneDataset(frames)
    print(f"Running pass 1 (metrics only) on {len(ds)} frames from {args.sequence}...")
    rows = compute_frame_metrics(models, ds, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"{args.sequence}_frame_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx"] + [f"{m}_mean_err_mm" for m in MODES])
        writer.writeheader()
        writer.writerows(rows)

    # select representative frames based on transfer_scared's mean error
    # (the mode ultimately closest to your headline result)
    errs = [(r["frame_idx"], r["transfer_scared_mean_err_mm"]) for r in rows if not np.isnan(r["transfer_scared_mean_err_mm"])]
    errs_sorted = sorted(errs, key=lambda x: x[1])

    best_idx = errs_sorted[0][0]
    worst_idx = errs_sorted[-1][0]
    median_idx = errs_sorted[len(errs_sorted) // 2][0]

    selected = {"best": best_idx, "median": median_idx, "worst": worst_idx}
    print(f"\nSelected frames: {selected}")
    print(f"  best   (idx {best_idx}):   {dict(errs)[best_idx]:.3f}mm")
    print(f"  median (idx {median_idx}): {dict(errs)[median_idx]:.3f}mm")
    print(f"  worst  (idx {worst_idx}):  {dict(errs)[worst_idx]:.3f}mm")

    # color scale from the full sequence's GT range, for consistency
    all_gt_vals = []
    for i in range(len(ds)):
        depth = ds[i]["depth"].numpy()
        valid = depth > 1e-4
        if valid.any():
            all_gt_vals.append(depth[valid])
    all_gt_vals = np.concatenate(all_gt_vals)
    vmin, vmax = np.percentile(all_gt_vals, [1, 99])

    print("\nRendering selected frames at full resolution...")
    for label, idx in selected.items():
        grid = render_frame_grid(models, ds, idx, vmin, vmax, device)
        out_path = out_dir / f"{args.sequence}_{label}_frame{idx}.png"
        cv2.imwrite(str(out_path), grid)
        print(f"  saved {out_path}")

    print(f"\nDone. Frame metrics CSV and stills are in {out_dir}/")


if __name__ == "__main__":
    main()
