"""
Fine-tune DA3METRIC-LARGE on the real knee dataset, with LoRA adapters.

Two modes, both supported by this one script:
  - Transfer learning: start from a SCARED checkpoint (--init-checkpoint),
    continue training on real knee data. Recommended starting point:
    Vector-LoRA, SCARED full run epoch 3 (tied with uniform LoRA on
    accuracy, less than half the trainable parameters -- see the report).
  - From scratch: omit --init-checkpoint, starts from base DA3METRIC-LARGE
    exactly like every SCARED run did.

Data: real millimeters directly (12-32mm range for this dataset,
verified against metadata's sigma_floor_mm). Do NOT apply SCARED's 0.256
correction factor here -- that's specific to SCARED's h5 encoding and
does not apply to this dataset.

Train/val split is by PATIENT (see real_knee_io.split_patients()), not
frame, so validation tests generalization to an unseen patient, not just
unseen frames from patients already seen in training.

Per-pixel sigma (real uncertainty from the depth reconstruction) is
loaded but NOT used in the loss yet -- this run is a plain masked L1
baseline, matching the SCARED methodology exactly for a fair before/
after comparison. Confidence-weighting using sigma is a natural next
step once this baseline exists.

Usage:
    # transfer learning from SCARED's best Vector-LoRA checkpoint
    python -m arthronav.train_real_knee --epochs 5 --vector-lora \
        --init-checkpoint checkpoints/scared_training_checkpoints/checkpoints_vector_lora_full/epoch_3.pt \
        --checkpoint-dir checkpoints/real_knee_training/transfer_from_scared_vector

    # from scratch (no SCARED checkpoint)
    python -m arthronav.train_real_knee --epochs 5 --vector-lora \
        --checkpoint-dir checkpoints/real_knee_training/from_scratch_vector
"""

import argparse
import csv
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3

from arthronav.lora import inject_lora, inject_vector_lora
from arthronav.losses import masked_l1_loss
from arthronav.real_knee_io import split_patients, build_frame_list
from arthronav.real_knee_dataset import RealKneeDataset

# Grenoble:
DATA_ROOT = "/mnt/areas_nas/SLAM/real_knee_dataset"
# AWS: swap for
# DATA_ROOT = "/data/real_knee_dataset"

TARGET_SIZE = (1078, 1918)  # nearest multiples of 14 below the native 1080x1920


def prepare_batch(batch, device):
    rgb = F.interpolate(batch["rgb"], size=TARGET_SIZE, mode="bilinear", align_corners=False)
    rgb = rgb.unsqueeze(1).to(device)  # (B, 1, 3, H, W)

    depth_gt = F.interpolate(batch["depth"].unsqueeze(1), size=TARGET_SIZE, mode="nearest")
    depth_gt = depth_gt.squeeze(1).to(device)  # (B, H, W), real mm

    valid = F.interpolate(batch["valid"].unsqueeze(1).float(), size=TARGET_SIZE, mode="nearest")
    valid_mask = (valid.squeeze(1) > 0.5).to(device)

    return rgb, depth_gt, valid_mask


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
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--vector-lora", action="store_true",
                     help="use Vector-LoRA (DARES-style, q/v-only) instead of uniform rank")
    ap.add_argument("--init-checkpoint", type=str, default=None,
                     help="path to a SCARED checkpoint to start from (transfer learning); "
                          "omit for from-scratch training. Must match --vector-lora setting "
                          "(the checkpoint's mode and this run's mode must be the same).")
    ap.add_argument("--checkpoint-dir", type=str, default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--data-root", type=str, default=DATA_ROOT)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model...")
    wrapper = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = wrapper.model
    if args.vector_lora:
        adapted = inject_vector_lora(net)
        print(f"Vector-LoRA: {len(adapted)} layers adapted")
    else:
        adapted = inject_lora(net, rank=args.lora_rank)
        print(f"Uniform LoRA: {len(adapted)} layers adapted")

    if args.init_checkpoint is not None:
        state = torch.load(args.init_checkpoint, map_location=device)
        missing, unexpected = net.load_state_dict(state, strict=False)
        loaded_count = len(state) - len(unexpected)
        if loaded_count == 0:
            print(f"WARNING: 0 tensors matched from {args.init_checkpoint} -- check "
                  f"--vector-lora matches how that checkpoint was actually trained. "
                  f"Proceeding would silently train from the untouched base model, "
                  f"not a real transfer-learning run.")
            raise SystemExit(1)
        print(f"Transfer learning: loaded {loaded_count} tensors from {args.init_checkpoint}")
    else:
        print("From scratch: starting from base DA3METRIC-LARGE (no SCARED checkpoint)")

    net = net.to(device)
    net.train()

    if args.checkpoint_dir is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        args.checkpoint_dir = os.path.join("checkpoints", "real_knee_training", run_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {args.checkpoint_dir}")

    log_path = os.path.join(args.checkpoint_dir, "loss_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "lr", "loss", "patient", "view"])

    train_patients, val_patients, test_patients = split_patients()
    print(f"Train patients: {train_patients}")
    print(f"Val patients:   {val_patients}")
    print(f"Test patients:  {test_patients} (held out, not used here)")

    train_frames = build_frame_list(args.data_root, train_patients)
    print(f"Training on {len(train_frames)} frames across {len(train_patients)} patients")
    if len(train_frames) == 0:
        raise RuntimeError(
            f"No training frames found under {args.data_root} for patients {train_patients}. "
            f"Has data prep finished for these patients on this machine?"
        )

    train_ds = RealKneeDataset(train_frames, load_sigma=False)  # not used in loss yet
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
                csv.writer(f).writerow([epoch, n_batches, current_lr, loss.item(),
                                         batch["patient"][0], batch["view"][0]])

        print(f"epoch {epoch} | avg loss (mm): {epoch_loss / n_batches:.4f} | lr: {current_lr:.2e}")

        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
        save_trainable_checkpoint(net, ckpt_path)
        print(f"saved checkpoint: {ckpt_path}")

    print("Done.")


if __name__ == "__main__":
    main()
