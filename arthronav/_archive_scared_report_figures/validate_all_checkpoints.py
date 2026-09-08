"""
Validate every checkpoint trained so far, on the identical 136-frame
held-out split, reporting AbsRel, RMSE, and raw min/max/mean absolute
error in meters (the metric that actually reveals worst-case error,
which AbsRel/RMSE average away).

Handles both LoRA modes: uniform-rank LoRA (qkv+proj, LoRALinear) and
Vector-LoRA (q/v-only fused-qkv, QKVLoRA). Each checkpoint's entry says
which mode it needs, so the right injection function runs before
loading that checkpoint's state dict, since the two modes produce
differently-named parameters and loading one into the other's structure
would silently do nothing or error.

Usage:
    python -m arthronav.validate_all_checkpoints
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3

from arthronav.lora import inject_lora, inject_vector_lora
from arthronav.metrics import abs_rel, rmse, abs_error_stats
from arthronav.scared_io import build_frame_list, split_frames
from arthronav.scared_dataset import SCAREDDataset

H5_ROOT = "/mnt/areas_nas/SLAM/scared_dataset_full_copy/depth_anything_preprocessed_data/train_depth_anything"
JSON_ROOT = "/mnt/areas_nas/SLAM/scared_dataset_full_copy/frame_trajectory_data"
TARGET_SIZE = (1022, 1274)
BASE_CKPT_DIR = "checkpoints/scared_training_checkpoints"

# (label, checkpoint_path_or_None, lora_mode)
# lora_mode: "uniform" or "vector" -- ignored when checkpoint is None (base model)
CHECKPOINTS = [
    ("Base pretrained model", None, "uniform"),
    ("1% subset, 1 epoch", f"{BASE_CKPT_DIR}/checkpoints_small/epoch_0.pt", "uniform"),
    ("Medium run (20%), epoch 0", f"{BASE_CKPT_DIR}/checkpoints_medium_run_20subset/epoch_0.pt", "uniform"),
    ("Medium run (20%), epoch 1", f"{BASE_CKPT_DIR}/checkpoints_medium_run_20subset/epoch_1.pt", "uniform"),
    ("Medium run (20%), epoch 2", f"{BASE_CKPT_DIR}/checkpoints_medium_run_20subset/epoch_2.pt", "uniform"),
    ("Full run, epoch 0", f"{BASE_CKPT_DIR}/checkpoints_long_run_full_v2/epoch_0.pt", "uniform"),
    ("Full run, epoch 1", f"{BASE_CKPT_DIR}/checkpoints_long_run_full_v2/epoch_1.pt", "uniform"),
    ("Full run, epoch 2", f"{BASE_CKPT_DIR}/checkpoints_long_run_full_v2/epoch_2.pt", "uniform"),
    ("Full run, epoch 3", f"{BASE_CKPT_DIR}/checkpoints_long_run_full_v2/epoch_3.pt", "uniform"),
    ("Full run, epoch 4", f"{BASE_CKPT_DIR}/checkpoints_long_run_full_v2/epoch_4.pt", "uniform"),
    ("Vector-LoRA, epoch 0", f"{BASE_CKPT_DIR}/checkpoints_vector_lora_full/epoch_0.pt", "vector"),
    ("Vector-LoRA, epoch 1", f"{BASE_CKPT_DIR}/checkpoints_vector_lora_full/epoch_1.pt", "vector"),
    ("Vector-LoRA, epoch 2", f"{BASE_CKPT_DIR}/checkpoints_vector_lora_full/epoch_2.pt", "vector"),
    ("Vector-LoRA, epoch 3", f"{BASE_CKPT_DIR}/checkpoints_vector_lora_full/epoch_3.pt", "vector"),
    ("Vector-LoRA, epoch 4", f"{BASE_CKPT_DIR}/checkpoints_vector_lora_full/epoch_4.pt", "vector"),
    ("Uniform + gradient loss, epoch 0", f"{BASE_CKPT_DIR}/checkpoints_uniform_gradloss_full/epoch_0.pt", "uniform"),
    ("Uniform + gradient loss, epoch 1", f"{BASE_CKPT_DIR}/checkpoints_uniform_gradloss_full/epoch_1.pt", "uniform"),
    ("Uniform + gradient loss, epoch 2", f"{BASE_CKPT_DIR}/checkpoints_uniform_gradloss_full/epoch_2.pt", "uniform"),
    ("Uniform + gradient loss, epoch 3", f"{BASE_CKPT_DIR}/checkpoints_uniform_gradloss_full/epoch_3.pt", "uniform"),
    ("Uniform + gradient loss, epoch 4", f"{BASE_CKPT_DIR}/checkpoints_uniform_gradloss_full/epoch_4.pt", "uniform"),
    ("Vector-LoRA + gradient loss, epoch 0", f"{BASE_CKPT_DIR}/checkpoints_vector_gradloss_full/epoch_0.pt", "vector"),
    ("Vector-LoRA + gradient loss, epoch 1", f"{BASE_CKPT_DIR}/checkpoints_vector_gradloss_full/epoch_1.pt", "vector"),
    ("Vector-LoRA + gradient loss, epoch 2", f"{BASE_CKPT_DIR}/checkpoints_vector_gradloss_full/epoch_2.pt", "vector"),
    ("Vector-LoRA + gradient loss, epoch 3", f"{BASE_CKPT_DIR}/checkpoints_vector_gradloss_full/epoch_3.pt", "vector"),
    ("Vector-LoRA + gradient loss, epoch 4", f"{BASE_CKPT_DIR}/checkpoints_vector_gradloss_full/epoch_4.pt", "vector"),
    ("Uniform + gradient loss (w=0.1), epoch 0", f"{BASE_CKPT_DIR}/checkpoints_uniform_gradloss_w01_full/epoch_0.pt", "uniform"),
    ("Uniform + gradient loss (w=0.1), epoch 1", f"{BASE_CKPT_DIR}/checkpoints_uniform_gradloss_w01_full/epoch_1.pt", "uniform"),
    ("Uniform + gradient loss (w=0.1), epoch 2", f"{BASE_CKPT_DIR}/checkpoints_uniform_gradloss_w01_full/epoch_2.pt", "uniform"),
    ("Uniform + gradient loss (w=0.1), epoch 3", f"{BASE_CKPT_DIR}/checkpoints_uniform_gradloss_w01_full/epoch_3.pt", "uniform"),
    ("Uniform + gradient loss (w=0.1), epoch 4", f"{BASE_CKPT_DIR}/checkpoints_uniform_gradloss_w01_full/epoch_4.pt", "uniform"),
]


def prepare_batch(batch, device):
    rgb = F.interpolate(batch["rgb"], size=TARGET_SIZE, mode="bilinear", align_corners=False)
    rgb = rgb.unsqueeze(1).to(device)
    depth_gt = F.interpolate(batch["depth"].unsqueeze(1), size=TARGET_SIZE, mode="nearest")
    depth_gt = depth_gt.squeeze(1).to(device)
    valid_mask = depth_gt > 1e-4
    return rgb, depth_gt, valid_mask


def validate_one(checkpoint_path, lora_mode, val_loader, device, lora_rank=16):
    wrapper = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = wrapper.model

    if lora_mode == "vector":
        inject_vector_lora(net)
    else:
        inject_lora(net, rank=lora_rank)

    net = net.to(device)
    net.eval()

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        missing, unexpected = net.load_state_dict(state, strict=False)
        # sanity check: a mismatched checkpoint would load nothing useful,
        # so flag it loudly rather than silently reporting base-model numbers
        loaded_count = len(state) - len(unexpected)
        if loaded_count == 0:
            print(f"  WARNING: 0 matching tensors loaded from {checkpoint_path} "
                  f"(mode={lora_mode}) -- check the mode is correct for this checkpoint")

    all_abs_rel, all_rmse = [], []
    all_min, all_max, all_mean = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            rgb, depth_gt, valid_mask = prepare_batch(batch, device)
            output = net(rgb, export_feat_layers=[])
            pred = output.depth.squeeze(1)

            all_abs_rel.append(abs_rel(pred, depth_gt, valid_mask).item())
            all_rmse.append(rmse(pred, depth_gt, valid_mask).item())

            stats = abs_error_stats(pred, depth_gt, valid_mask)
            all_min.append(stats["min_error_m"])
            all_max.append(stats["max_error_m"])
            all_mean.append(stats["mean_error_m"])

    del net
    torch.cuda.empty_cache()

    return {
        "AbsRel": sum(all_abs_rel) / len(all_abs_rel),
        "RMSE": sum(all_rmse) / len(all_rmse),
        "min_error_m": min(all_min),
        "max_error_m": max(all_max),
        "mean_error_m": sum(all_mean) / len(all_mean),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Building frame list...")
    frames = build_frame_list(H5_ROOT, JSON_ROOT)
    _, val_frames = split_frames(frames)

    import random
    rng = random.Random(42)
    n_keep = max(1, int(len(val_frames) * 0.1))
    val_subset = rng.sample(val_frames, n_keep)
    ds = SCAREDDataset(val_subset, bad_files_path="bad_h5_files.txt")
    val_loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    print(f"Validating on {len(ds)} frames (same split used throughout this project)")

    results = []
    for label, ckpt, mode in tqdm(CHECKPOINTS, desc="checkpoints"):
        r = validate_one(ckpt, mode, val_loader, device)
        r["label"] = label
        results.append(r)

    print("\n" + "=" * 100)
    print(f"{'Stage':<32} | {'AbsRel':>8} | {'RMSE (m)':>9} | {'min err (m)':>11} | {'max err (m)':>11} | {'mean err (m)':>12}")
    print("-" * 100)
    for r in results:
        print(f"{r['label']:<32} | {r['AbsRel']:>8.4f} | {r['RMSE']:>9.4f} | "
              f"{r['min_error_m']:>11.4f} | {r['max_error_m']:>11.4f} | {r['mean_error_m']:>12.4f}")
    print("=" * 100)

    with open("all_checkpoints_validation.csv", "w") as f:
        f.write("label,abs_rel,rmse,min_error_m,max_error_m,mean_error_m\n")
        for r in results:
            f.write(f"{r['label']},{r['AbsRel']},{r['RMSE']},{r['min_error_m']},{r['max_error_m']},{r['mean_error_m']}\n")
    print("Wrote all_checkpoints_validation.csv")


if __name__ == "__main__":
    main()
