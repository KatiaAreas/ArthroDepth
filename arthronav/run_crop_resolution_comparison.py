"""
Two from-scratch training runs on the SAME crop, different resolutions:
  run A: 1022x1022 (crop, no further downsampling)
  run B: 1022x1022 cropped, then downsampled to 518x518 for training/inference,
         prediction upsampled back to 1022x1022 before scoring against GT --
         so both runs are graded on identical footing, same GT resolution.

Crop boxes are precomputed offline (precompute_crop_boxes.py, run once in
a separate Python 3.12 venv with areas_theta_compute installed) and read
from sawbone_crop_boxes.json -- this script never imports cv2 or
areas_theta_compute, both of which caused real problems when loaded
alongside torch in the same process.

Usage:
    python -m arthronav.run_crop_resolution_comparison --resolution 1022
    python -m arthronav.run_crop_resolution_comparison --resolution 518
"""
import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3
from arthronav.lora import inject_vector_lora
from arthronav.losses import masked_l1_loss
from arthronav.metrics import compute_all_metrics, abs_error_stats, error_distribution
from arthronav.sawbone_io import build_frame_list, split_frames
from arthronav.sawbone_dataset import SawboneDataset

SAWBONE_ROOT = "/mnt/areas_nas/SLAM/sawbone_dataset"
SCORING_SIZE = 1022

ALL_SEQUENCES = [
    "tour_lateral_0_cartilage", "tour_lateral_1_cartilage",
    "tour_lateral_2_cartilage", "tour_lateral_3_cartilage",
    "tour_medial_0_cartilage", "tour_medial_1_cartilage",
    "tour_medial_2_cartilage", "tour_medial_3_cartilage",
]
HOLDOUT = ["tour_lateral_3_cartilage", "tour_medial_3_cartilage"]


def save_trainable_checkpoint(net, path):
    trainable_names = {name for name, p in net.named_parameters() if p.requires_grad}
    state = {name: t for name, t in net.state_dict().items() if name in trainable_names}
    torch.save(state, path)


def prepare_batch(batch, device, train_resolution):
    rgb = batch["rgb"]
    depth_gt = batch["depth"]

    if train_resolution != SCORING_SIZE:
        rgb = F.interpolate(rgb, size=(train_resolution, train_resolution),
                             mode="bilinear", align_corners=False)
        depth_gt_train = F.interpolate(depth_gt.unsqueeze(1), size=(train_resolution, train_resolution),
                                        mode="nearest").squeeze(1)
    else:
        depth_gt_train = depth_gt

    rgb = rgb.unsqueeze(1).to(device)
    depth_gt_train = depth_gt_train.to(device)
    valid_mask_train = depth_gt_train > 1e-4
    return rgb, depth_gt_train, valid_mask_train


def upsample_pred_to_scoring_size(pred, train_resolution):
    if train_resolution == SCORING_SIZE:
        return pred
    return F.interpolate(pred.unsqueeze(1), size=(SCORING_SIZE, SCORING_SIZE),
                          mode="bilinear", align_corners=False).squeeze(1)


def train(net, loader, device, epochs, checkpoint_dir, train_resolution, lr=1e-4, min_lr=1e-6):
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    total_steps = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)

    log_path = os.path.join(checkpoint_dir, "loss_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "lr", "loss"])

    net.train()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"[res={train_resolution}] epoch {epoch}")
        step = 0
        for batch in pbar:
            rgb, depth_gt, valid_mask = prepare_batch(batch, device, train_resolution)
            output = net(rgb, export_feat_layers=[])
            pred = output.depth.squeeze(1)

            loss = masked_l1_loss(pred, depth_gt, valid_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, step, current_lr, loss.item()])

        ckpt_path = f"{checkpoint_dir}/epoch_{epoch}.pt"
        save_trainable_checkpoint(net, ckpt_path)
        print(f"saved checkpoint: {ckpt_path}")


def evaluate(net, loader, device, train_resolution):
    net.eval()
    all_abs_rel, all_rmse, all_min, all_max, all_mean = [], [], [], [], []
    all_preds, all_gts, all_masks = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[res={train_resolution}] evaluating (scored @ {SCORING_SIZE})"):
            rgb, depth_gt_train, valid_mask_train = prepare_batch(batch, device, train_resolution)
            output = net(rgb, export_feat_layers=[])
            pred_train_res = output.depth.squeeze(1)

            pred_scored = upsample_pred_to_scoring_size(pred_train_res, train_resolution)
            depth_gt_scored = batch["depth"].to(device)
            valid_mask_scored = depth_gt_scored > 1e-4

            m = compute_all_metrics(pred_scored, depth_gt_scored, valid_mask_scored, unit_label="m")
            s = abs_error_stats(pred_scored, depth_gt_scored, valid_mask_scored, unit_label="m")
            all_abs_rel.append(m["AbsRel"])
            all_rmse.append(m["RMSE_m"])
            all_min.append(s["min_error_m"])
            all_max.append(s["max_error_m"])
            all_mean.append(s["mean_error_m"])

            all_preds.append(pred_scored.cpu())
            all_gts.append(depth_gt_scored.cpu())
            all_masks.append(valid_mask_scored.cpu())

    pooled_pred = torch.cat([p.flatten() for p in all_preds])
    pooled_gt = torch.cat([g.flatten() for g in all_gts])
    pooled_mask = torch.cat([m.flatten() for m in all_masks])
    dist = error_distribution(pooled_pred, pooled_gt, pooled_mask)

    return {
        "AbsRel": sum(all_abs_rel) / len(all_abs_rel),
        "RMSE_m": sum(all_rmse) / len(all_rmse),
        "min_error_m": min(all_min),
        "max_error_m": max(all_max),
        "mean_error_m": sum(all_mean) / len(all_mean),
        **dist,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, required=True, choices=[1022, 518],
                     help="Training/inference resolution; always scored at 1022 regardless")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0,
                     help="0 avoids forking DataLoader workers after CUDA init")
    ap.add_argument("--out-csv", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, training resolution: {args.resolution}, scoring resolution: {SCORING_SIZE}")

    frames = build_frame_list(Path(SAWBONE_ROOT), ALL_SEQUENCES)
    train_frames, val_frames = split_frames(frames, HOLDOUT)
    print(f"Train: {len(train_frames)} frames, Val (held out): {len(val_frames)} frames")

    train_ds = SawboneDataset(train_frames, crop_size=SCORING_SIZE)
    val_ds = SawboneDataset(val_frames, crop_size=SCORING_SIZE)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    wrapper = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = wrapper.model
    inject_vector_lora(net)
    net = net.to(device)

    checkpoint_dir = f"checkpoints/sawbone_crop_res{args.resolution}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    train(net, train_loader, device, args.epochs, checkpoint_dir, args.resolution)

    results = evaluate(net, val_loader, device, args.resolution)

    print(f"\n=== resolution={args.resolution} results (scored @ {SCORING_SIZE}) ===")
    for k, v in results.items():
        suffix = " (% of pixels)" if "pct_gt" in k else ""
        print(f"  {k}: {v:.4f}{suffix}" if isinstance(v, float) else f"  {k}: {v}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["resolution"] + list(results.keys()))
            writer.writeheader()
            writer.writerow({"resolution": args.resolution, **results})
        print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
