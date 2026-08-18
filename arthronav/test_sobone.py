"""
Zero-shot / transfer-from-SCARED / from-scratch comparison on the
sobone/cartilage dataset (currently tour_lateral_0_cartilage only).

Modes:
  zero_shot        : base DA3METRIC-LARGE, no LoRA, no fine-tuning.
  transfer_scared   : start from the best SCARED Vector-LoRA checkpoint
                      (epoch 3), continue fine-tuning on cartilage data.
  from_scratch      : fresh Vector-LoRA init on base DA3 weights, trained
                      directly on cartilage data, no SCARED step.

Note: pools all 8 sequences by default, holding out tour_lateral_3 and
tour_medial_3 entirely (one lateral, one medial) as genuine unseen
validation -- the first real generalization test, not a smoke test.
in sobone_io.split_frames), there's no held-out val set yet -- this is
a pipeline smoke test, not a generalization claim. Metrics reported are
on the same 343 frames used for training in the two fine-tune modes.

Usage:
    python -m arthronav.test_sobone --mode zero_shot
    python -m arthronav.test_sobone --mode transfer_scared --epochs 5
    python -m arthronav.test_sobone --mode from_scratch --epochs 5
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
from arthronav.sobone_io import build_frame_list, split_frames
from arthronav.sobone_dataset import SoboneDataset

SOBONE_ROOT = "/mnt/areas_nas/SLAM/sobone_dataset"
TARGET_SIZE = (1078, 1918)

VECTOR_LORA_SCARED_CKPT = (
    "checkpoints/scared_training_checkpoints/checkpoints_vector_lora_full/epoch_3.pt"
)


def prepare_batch(batch, device):
    rgb = F.interpolate(batch["rgb"], size=TARGET_SIZE, mode="bilinear", align_corners=False)
    rgb = rgb.unsqueeze(1).to(device)

    depth_gt = F.interpolate(batch["depth"].unsqueeze(1), size=TARGET_SIZE, mode="nearest")
    depth_gt = depth_gt.squeeze(1).to(device)

    valid_mask = depth_gt > 1e-4
    return rgb, depth_gt, valid_mask


def build_model(mode, device, checkpoint_path=None):
    wrapper = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = wrapper.model

    if mode == "zero_shot":
        pass
    elif mode in ("transfer_scared", "from_scratch"):
        inject_vector_lora(net)
        if mode == "transfer_scared":
            state = torch.load(checkpoint_path, map_location=device)
            net.load_state_dict(state, strict=False)
            print(f"Loaded SCARED Vector-LoRA checkpoint: {checkpoint_path}")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return net.to(device)


def save_trainable_checkpoint(net, path):
    trainable_names = {name for name, p in net.named_parameters() if p.requires_grad}
    state = {name: t for name, t in net.state_dict().items() if name in trainable_names}
    torch.save(state, path)


def train(net, loader, device, epochs, checkpoint_dir, lr=1e-4, min_lr=1e-6):
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    total_steps = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)

    net.train()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for batch in pbar:
            rgb, depth_gt, valid_mask = prepare_batch(batch, device)
            output = net(rgb, export_feat_layers=[])
            pred = output.depth.squeeze(1)

            loss = masked_l1_loss(pred, depth_gt, valid_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        ckpt_path = f"{checkpoint_dir}/epoch_{epoch}.pt"
        save_trainable_checkpoint(net, ckpt_path)
        print(f"saved checkpoint: {ckpt_path}")


def evaluate(net, loader, device):
    net.eval()
    all_abs_rel, all_rmse = [], []
    all_min, all_max, all_mean = [], [], []
    all_preds, all_gts, all_masks = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating"):
            rgb, depth_gt, valid_mask = prepare_batch(batch, device)
            output = net(rgb, export_feat_layers=[])
            pred = output.depth.squeeze(1)

            metrics = compute_all_metrics(pred, depth_gt, valid_mask, unit_label="m")
            all_abs_rel.append(metrics["AbsRel"])
            all_rmse.append(metrics["RMSE_m"])

            stats = abs_error_stats(pred, depth_gt, valid_mask, unit_label="m")
            all_min.append(stats["min_error_m"])
            all_max.append(stats["max_error_m"])
            all_mean.append(stats["mean_error_m"])

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
    ap.add_argument("--mode", required=True, choices=["zero_shot", "transfer_scared", "from_scratch"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--sequences", nargs="+", default=[
        "tour_lateral_0_cartilage", "tour_lateral_1_cartilage",
        "tour_lateral_2_cartilage", "tour_lateral_3_cartilage",
        "tour_medial_0_cartilage", "tour_medial_1_cartilage",
        "tour_medial_2_cartilage", "tour_medial_3_cartilage",
    ])
    ap.add_argument("--holdout-sequences", nargs="+", default=[
        "tour_lateral_3_cartilage", "tour_medial_3_cartilage",
    ], help="Entire sequences held out for validation, never used in training.")
    ap.add_argument("--scared-checkpoint", type=str, default=VECTOR_LORA_SCARED_CKPT)
    ap.add_argument("--checkpoint-dir", type=str, default=None)
    ap.add_argument("--out-csv", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    frames = build_frame_list(Path(SOBONE_ROOT), args.sequences)
    train_frames, val_frames = split_frames(frames, args.holdout_sequences)
    print(f"Total frames: {len(frames)} from {args.sequences}")
    print(f"Train: {len(train_frames)} frames (sequences: {[s for s in args.sequences if s not in args.holdout_sequences]})")
    print(f"Val (held out, unseen): {len(val_frames)} frames (sequences: {args.holdout_sequences})")

    train_ds = SoboneDataset(train_frames)
    val_ds = SoboneDataset(val_frames)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    net = build_model(args.mode, device, checkpoint_path=args.scared_checkpoint)

    if args.mode != "zero_shot":
        checkpoint_dir = args.checkpoint_dir or f"checkpoints/sobone_{args.mode}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        train(net, train_loader, device, args.epochs, checkpoint_dir)

    results = evaluate(net, val_loader, device)

    print(f"\n=== {args.mode} results ===")
    for k, v in results.items():
        suffix = " (% of pixels)" if "pct_gt" in k else ""
        print(f"  {k}: {v:.4f}{suffix}" if isinstance(v, float) else f"  {k}: {v}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["mode"] + list(results.keys()))
            writer.writeheader()
            writer.writerow({"mode": args.mode, **results})
        print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
