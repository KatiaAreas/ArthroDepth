"""
Train the ENTIRE DA3 backbone from random initialization (no pretrained
weights, no LoRA, no frozen layers) -- a deliberate scientific control to
compare against pretrained+LoRA results, per explicit request despite
disagreement about its practical value given this dataset's size (~2k
frames vs. the 336M-parameter model's real data appetite).

Randomization mechanism: walks every submodule of the pretrained model
and calls reset_parameters() where that method exists (standard for
nn.Linear/Conv2d/LayerNorm/etc.). Modules WITHOUT reset_parameters()
are reported explicitly at startup -- if any show up, those specific
submodules still hold pretrained weights, and this is not a true full
random init. Check that list before trusting the result.

Usage:
    python -m arthronav.run_from_scratch_full_backbone --epochs 5
"""
import argparse
import csv
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3
from arthronav.losses import masked_l1_loss
from arthronav.metrics import compute_all_metrics, abs_error_stats, error_distribution
from arthronav.sobone_io import build_frame_list, split_frames
from arthronav.sobone_dataset import SoboneDataset

SOBONE_ROOT = "/mnt/areas_nas/SLAM/sobone_dataset"
CROP_SIZE = 1022

ALL_SEQUENCES = [
    "tour_lateral_0_cartilage", "tour_lateral_1_cartilage",
    "tour_lateral_2_cartilage", "tour_lateral_3_cartilage",
    "tour_medial_0_cartilage", "tour_medial_1_cartilage",
    "tour_medial_2_cartilage", "tour_medial_3_cartilage",
]
HOLDOUT = ["tour_lateral_3_cartilage", "tour_medial_3_cartilage"]


def randomize_all_weights(model):
    """Reinitializes every submodule with reset_parameters(). Returns
    (n_reset, n_skipped, skipped_module_types) so the caller can verify
    coverage rather than assume it."""
    reset_count = 0
    skipped = []

    def _reset(module):
        nonlocal reset_count
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
            reset_count += 1
        elif len(list(module.parameters(recurse=False))) > 0:
            # has its own parameters but no reset method -- flagged, not silently skipped
            skipped.append(type(module).__name__)

    model.apply(_reset)
    return reset_count, skipped


def prepare_batch(batch, device):
    rgb = batch["rgb"].unsqueeze(1).to(device)
    depth_gt = batch["depth"].to(device)
    valid_mask = depth_gt > 1e-4
    return rgb, depth_gt, valid_mask


def save_full_checkpoint(net, path):
    torch.save(net.state_dict(), path)


def train(net, loader, device, epochs, checkpoint_dir, lr=1e-4, min_lr=1e-6):
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    print(f"Training {sum(p.numel() for p in trainable_params):,} parameters "
          f"(full model, no freezing)")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    total_steps = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)

    log_path = os.path.join(checkpoint_dir, "loss_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "lr", "loss"])

    net.train()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"[full-backbone-scratch] epoch {epoch}")
        step = 0
        for batch in pbar:
            rgb, depth_gt, valid_mask = prepare_batch(batch, device)
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
        save_full_checkpoint(net, ckpt_path)
        print(f"saved checkpoint: {ckpt_path}")


def evaluate(net, loader, device):
    net.eval()
    all_abs_rel, all_rmse, all_min, all_max, all_mean = [], [], [], [], []
    all_preds, all_gts, all_masks = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating"):
            rgb, depth_gt, valid_mask = prepare_batch(batch, device)
            output = net(rgb, export_feat_layers=[])
            pred = output.depth.squeeze(1)

            m = compute_all_metrics(pred, depth_gt, valid_mask, unit_label="m")
            s = abs_error_stats(pred, depth_gt, valid_mask, unit_label="m")
            all_abs_rel.append(m["AbsRel"])
            all_rmse.append(m["RMSE_m"])
            all_min.append(s["min_error_m"])
            all_max.append(s["max_error_m"])
            all_mean.append(s["mean_error_m"])
            all_preds.append(pred.cpu())
            all_gts.append(depth_gt.cpu())
            all_masks.append(valid_mask.cpu())

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
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--out-csv", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    frames = build_frame_list(Path(SOBONE_ROOT), ALL_SEQUENCES)
    train_frames, val_frames = split_frames(frames, HOLDOUT)
    print(f"Train: {len(train_frames)} frames, Val (held out): {len(val_frames)} frames")

    train_ds = SoboneDataset(train_frames, crop_size=CROP_SIZE)
    val_ds = SoboneDataset(val_frames, crop_size=CROP_SIZE)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Loading DA3 architecture (weights will be discarded and randomized)...")
    wrapper = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = wrapper.model

    reset_count, skipped = randomize_all_weights(net)
    print(f"\nRandomized {reset_count} submodules via reset_parameters().")
    if skipped:
        print(f"WARNING: {len(skipped)} parameterized submodules had NO reset_parameters() "
              f"method and still hold PRETRAINED weights: {set(skipped)}")
        print("This run is NOT a fully clean random init unless this list is empty.")
    else:
        print("No modules skipped -- full model successfully randomized.")

    for p in net.parameters():
        p.requires_grad = True
    net = net.to(device)

    checkpoint_dir = "checkpoints/sobone_full_backbone_scratch"
    os.makedirs(checkpoint_dir, exist_ok=True)
    train(net, train_loader, device, args.epochs, checkpoint_dir)

    results = evaluate(net, val_loader, device)
    print(f"\n=== full-backbone-scratch results ===")
    for k, v in results.items():
        suffix = " (% of pixels)" if "pct_gt" in k else ""
        print(f"  {k}: {v:.4f}{suffix}" if isinstance(v, float) else f"  {k}: {v}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results.keys()))
            writer.writeheader()
            writer.writerow(results)
        print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
