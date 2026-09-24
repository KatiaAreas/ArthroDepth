"""
Fine-tune DA3METRIC-LARGE (Vector-LoRA) on the new white sawbone
dataset (case_sawbone_white_2), prepared by prepare_sawbone_white_data.py
(circle-cropped, resized to 1022x1022 -- matches the original
red-sawbone resolution exactly).

Two modes:
  - From scratch: omit --init-checkpoint, starts from base DA3METRIC-LARGE.
  - Continue from best existing sawbone model: --init-checkpoint pointing
    at the best original (red) sawbone checkpoint, sawbone_from_scratch
    epoch_4 (AbsRel 0.0597 on that dataset's own held-out set, the
    slightly-better of the two original sawbone runs).

Ground truth is real meters (depth_png, uint16, raw*0.01 = mm, then /1000
= meters), matching the original red-sawbone convention exactly --
confirmed empirically: the red-sawbone checkpoint's raw output magnitude
matches ground truth in meters (~0.02-0.09), not millimeters. An earlier
version of this script used millimeters directly, a real inconsistency
introduced when continuing fine-tuning from a meters-calibrated
checkpoint -- fixed here to keep the same unit throughout the whole
fine-tuning chain (DA3 -> SCARED -> sawbone red -> sawbone white).

Usage:
    # from scratch (AWS)
    python -m arthronav.train_sawbone_white --epochs 5 \
        --checkpoint-dir checkpoints/sawbone_white_from_scratch

    # continuing from the best original sawbone checkpoint (Grenoble)
    python -m arthronav.train_sawbone_white --epochs 5 \
        --init-checkpoint checkpoints/sawbone_from_scratch/epoch_4.pt \
        --checkpoint-dir checkpoints/sawbone_white_from_sawbone
"""

import argparse
import csv
import glob
import os
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3

from arthronav.lora import inject_vector_lora
from arthronav.losses import masked_l1_loss

DATA_ROOT = "/mnt/areas_nas/SLAM/sawbone_white_dataset"


def build_frame_list(split_dir):
    frames = []
    seq_dirs = sorted(glob.glob(os.path.join(split_dir, "*")))
    for seq_dir in seq_dirs:
        rgb_dir = os.path.join(seq_dir, "rgb")
        depth_dir = os.path.join(seq_dir, "depth_png")
        if not os.path.isdir(rgb_dir):
            continue
        for rgb_path in sorted(glob.glob(os.path.join(rgb_dir, "*.png"))):
            frame_id = os.path.splitext(os.path.basename(rgb_path))[0]
            depth_path = os.path.join(depth_dir, f"{frame_id}.png")
            if os.path.exists(depth_path):
                frames.append({"rgb_path": rgb_path, "depth_path": depth_path,
                                "seq": os.path.basename(seq_dir), "frame_id": frame_id})
    return frames


class SawboneWhiteDataset(Dataset):
    def __init__(self, frame_list):
        self.frames = frame_list

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        entry = self.frames[idx]
        rgb_bgr = cv2.imread(entry["rgb_path"])
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3, H, W)

        depth_raw = cv2.imread(entry["depth_path"], cv2.IMREAD_UNCHANGED)  # uint16
        depth_m = depth_raw.astype(np.float32) * 0.01 / 1000.0
        depth_t = torch.from_numpy(depth_m).float()  # (H, W), real meters
        valid_t = torch.from_numpy(depth_raw > 0)  # (H, W), bool

        return {"rgb": rgb_t, "depth": depth_t, "valid": valid_t}


def save_trainable_checkpoint(net, path):
    trainable_names = {name for name, p in net.named_parameters() if p.requires_grad}
    state = {name: tensor for name, tensor in net.state_dict().items() if name in trainable_names}
    torch.save(state, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--init-checkpoint", type=str, default=None,
                     help="path to an existing checkpoint to continue from; omit for from-scratch")
    ap.add_argument("--checkpoint-dir", type=str, default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--data-root", type=str, default=DATA_ROOT)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model...")
    wrapper = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = wrapper.model
    adapted = inject_vector_lora(net)
    print(f"Vector-LoRA: {len(adapted)} layers adapted")

    if args.init_checkpoint is not None:
        state = torch.load(args.init_checkpoint, map_location=device)
        _, unexpected = net.load_state_dict(state, strict=False)
        loaded = len(state) - len(unexpected)
        if loaded == 0:
            print(f"WARNING: 0 tensors matched from {args.init_checkpoint} -- this would silently "
                  f"train from the untouched base model, not a real continuation. Stopping.")
            raise SystemExit(1)
        print(f"Continuing from {args.init_checkpoint} ({loaded} tensors matched)")
    else:
        print("From scratch: starting from base DA3METRIC-LARGE")

    net = net.to(device)
    net.train()

    if args.checkpoint_dir is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        args.checkpoint_dir = os.path.join("checkpoints", "sawbone_white", run_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {args.checkpoint_dir}")

    log_path = os.path.join(args.checkpoint_dir, "loss_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "lr", "loss"])

    train_frames = build_frame_list(os.path.join(args.data_root, "train"))
    print(f"Training on {len(train_frames)} frames (7 sequences, 1-in-3 decimated)")
    if len(train_frames) == 0:
        raise RuntimeError(f"No training frames found under {args.data_root}/train")

    train_ds = SawboneWhiteDataset(train_frames)
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
            rgb = batch["rgb"].unsqueeze(1).to(device)       # (B, 1, 3, H, W)
            depth_gt = batch["depth"].to(device)               # (B, H, W)
            valid_mask = batch["valid"].to(device)              # (B, H, W)

            output = net(rgb, export_feat_layers=[])
            pred = output.depth.squeeze(1)

            loss = masked_l1_loss(pred, depth_gt, valid_mask)

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

        print(f"epoch {epoch} | avg loss (m): {epoch_loss / n_batches:.4f} | lr: {current_lr:.2e}")

        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
        save_trainable_checkpoint(net, ckpt_path)
        print(f"saved checkpoint: {ckpt_path}")

    print("Done.")


if __name__ == "__main__":
    main()
