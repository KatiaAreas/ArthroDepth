"""
Fine-tune DA3METRIC-LARGE on SCARED with LoRA adapters.

Usage:
    python -m arthronav.train_scared --epochs 5 --batch-size 1 --subset-fraction 1.0

Notes:
    - batch-size 1 is currently the max this GPU (24GB) can hold at full
      1022x1274 resolution with backprop. Batch size 2 runs out of memory.
    - Full training set is 15,846 frames -> ~5-5.5h/epoch at batch-size 1.
      Use --subset-fraction (e.g. 0.2) to train on a random subset for
      faster iteration while still validating the pipeline; drop it (or
      set to 1.0) for a real full-dataset training run.
    - Only the trainable subset (LoRA adapters) gets checkpointed each
      epoch, not the full 336M-parameter model, a few MB instead of
      over a gigabyte.
    - Cosine LR schedule: decays smoothly from --lr down to --min-lr over
      the whole run (all epochs, stepped every batch), added after the
      medium run showed a dip at epoch 2 with a flat learning rate.
    - loss_log.csv now has an lr column alongside loss, so both can be
      plotted together afterward.
    - train_ds now uses bad_files_path="bad_h5_files.txt" to skip the one
      known-corrupted frame found by scan_h5_integrity.py, plus a runtime
      retry safety net in SCAREDDataset itself.
"""

import argparse
import csv
import os
import random
from datetime import datetime

import torch
# torch.backends.cudnn.enabled = False  # workaround for the cuDNN version conflict on this machine

import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3

from arthronav.lora import inject_lora, inject_vector_lora
from arthronav.losses import masked_l1_loss, gradient_loss
from arthronav.scared_io import build_frame_list, split_frames
from arthronav.scared_dataset import SCAREDDataset

H5_ROOT = "/mnt/areas_nas/SLAM/scared_dataset_full_copy/depth_anything_preprocessed_data/train_depth_anything"
JSON_ROOT = "/mnt/areas_nas/SLAM/scared_dataset_full_copy/frame_trajectory_data"
TARGET_SIZE = (1022, 1274)  # nearest multiples of 14 below the native 1024x1280


def prepare_batch(batch, device):
    rgb = F.interpolate(batch["rgb"], size=TARGET_SIZE, mode="bilinear", align_corners=False)
    rgb = rgb.unsqueeze(1).to(device)  # (B, 1, 3, H, W)

    depth_gt = F.interpolate(batch["depth"].unsqueeze(1), size=TARGET_SIZE, mode="nearest")
    depth_gt = depth_gt.squeeze(1).to(device)  # (B, H, W)

    valid_mask = depth_gt > 1e-4
    return rgb, depth_gt, valid_mask


def save_trainable_checkpoint(net, path):
    """Saves only parameters with requires_grad=True (the LoRA adapters)."""
    trainable_names = {name for name, p in net.named_parameters() if p.requires_grad}
    state = {name: tensor for name, tensor in net.state_dict().items() if name in trainable_names}
    torch.save(state, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4, help="starting learning rate")
    ap.add_argument("--min-lr", type=float, default=1e-6, help="floor the cosine schedule decays toward")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--vector-lora", action="store_true",
                     help="use Vector-LoRA (decreasing rank per block) instead of uniform rank")
    ap.add_argument("--subset-fraction", type=float, default=1.0,
                     help="fraction of training frames to use (0 < f <= 1), for faster iteration")
    ap.add_argument("--checkpoint-dir", type=str, default=None,
                     help="directory to save checkpoints; defaults to checkpoints/run_<timestamp>")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--bad-files", type=str, default="bad_h5_files.txt",
                     help="path to the known-bad-files list from scan_h5_integrity.py")
    ap.add_argument("--grad-loss-weight", type=float, default=0.0,
                     help="weight for the gradient-matching loss term; 0.0 = pure L1 (baseline)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model...")
    wrapper = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = wrapper.model
    if args.vector_lora:
        adapted = inject_vector_lora(net)  # DARES's real schedule, stretched to this backbone's block count
        print(f"Vector-LoRA (DARES-style, q/v only, no scaling): {len(adapted)} layers adapted")
    else:
        adapted = inject_lora(net, rank=args.lora_rank)
    net = net.to(device)
    net.train()
    print(f"LoRA injected into {len(adapted)} layers")

    if args.checkpoint_dir is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        args.checkpoint_dir = os.path.join("checkpoints", run_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {args.checkpoint_dir}")

    log_path = os.path.join(args.checkpoint_dir, "loss_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "lr", "loss"])

    print("Building frame list...")
    frames = build_frame_list(H5_ROOT, JSON_ROOT)
    train_frames, val_frames = split_frames(frames)

    if args.subset_fraction < 1.0:
        rng = random.Random(42)
        n_keep = max(1, int(len(train_frames) * args.subset_fraction))
        train_frames = rng.sample(train_frames, n_keep)
        print(f"Using subset: {n_keep} / {len(frames)} frames ({args.subset_fraction:.0%})")

    train_ds = SCAREDDataset(train_frames, bad_files_path=args.bad_files)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)

    trainable_params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.min_lr
    )
    print(f"Cosine LR schedule: {args.lr} -> {args.min_lr} over {total_steps} steps")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            rgb, depth_gt, valid_mask = prepare_batch(batch, device)

            output = net(rgb, export_feat_layers=[])
            pred = output.depth.squeeze(1)

            loss_l1 = masked_l1_loss(pred, depth_gt, valid_mask)
            if args.grad_loss_weight > 0:
                loss_grad = gradient_loss(pred, depth_gt, valid_mask)
                loss = loss_l1 + args.grad_loss_weight * loss_grad
            else:
                loss = loss_l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, n_batches, current_lr, loss.item()])

        print(f"epoch {epoch} | avg loss: {epoch_loss / n_batches:.4f} | lr: {current_lr:.2e}")

        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
        save_trainable_checkpoint(net, ckpt_path)
        print(f"saved checkpoint: {ckpt_path}")

    print("Done.")


if __name__ == "__main__":
    main()
