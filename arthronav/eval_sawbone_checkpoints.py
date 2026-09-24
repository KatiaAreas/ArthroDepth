"""
Evaluate every epoch checkpoint from both sawbone fine-tuning runs against
the same held-out validation frames, to compare convergence speed with
real validation metrics rather than training-loss noise.

Usage:
    python -m arthronav.eval_sawbone_checkpoints
"""
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3
from arthronav.lora import inject_vector_lora
from arthronav.metrics import compute_all_metrics, abs_error_stats
from arthronav.sawbone_io import build_frame_list, split_frames
from arthronav.sawbone_dataset import SawboneDataset
from arthronav.test_sawbone import prepare_batch, SAWBONE_ROOT

ALL_SEQUENCES = [
    "tour_lateral_0_cartilage", "tour_lateral_1_cartilage",
    "tour_lateral_2_cartilage", "tour_lateral_3_cartilage",
    "tour_medial_0_cartilage", "tour_medial_1_cartilage",
    "tour_medial_2_cartilage", "tour_medial_3_cartilage",
]
HOLDOUT = ["tour_lateral_3_cartilage", "tour_medial_3_cartilage"]

RUNS = {
    "transfer_scared": "checkpoints/sawbone_transfer_scared",
    "from_scratch": "checkpoints/sawbone_from_scratch",
}


def evaluate_checkpoint(ckpt_path, val_loader, device):
    wrapper = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = wrapper.model
    inject_vector_lora(net)
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state, strict=False)
    net = net.to(device)
    net.eval()

    all_abs_rel, all_rmse, all_mean = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            rgb, depth_gt, valid_mask = prepare_batch(batch, device)
            output = net(rgb, export_feat_layers=[])
            pred = output.depth.squeeze(1)
            m = compute_all_metrics(pred, depth_gt, valid_mask, unit_label="m")
            s = abs_error_stats(pred, depth_gt, valid_mask, unit_label="m")
            all_abs_rel.append(m["AbsRel"])
            all_rmse.append(m["RMSE_m"])
            all_mean.append(s["mean_error_m"])

    del net
    torch.cuda.empty_cache()
    return {
        "AbsRel": sum(all_abs_rel) / len(all_abs_rel),
        "RMSE_mm": (sum(all_rmse) / len(all_rmse)) * 1000,
        "mean_error_mm": (sum(all_mean) / len(all_mean)) * 1000,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = build_frame_list(Path(SAWBONE_ROOT), ALL_SEQUENCES)
    _, val_frames = split_frames(frames, HOLDOUT)
    print(f"Validating on {len(val_frames)} held-out frames: {HOLDOUT}")
    val_ds = SawboneDataset(val_frames)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    results = []
    for run_name, ckpt_dir in RUNS.items():
        for epoch in range(5):
            ckpt_path = Path(ckpt_dir) / f"epoch_{epoch}.pt"
            if not ckpt_path.exists():
                print(f"missing {ckpt_path}, skipping")
                continue
            print(f"\n{run_name} epoch {epoch}...")
            r = evaluate_checkpoint(ckpt_path, val_loader, device)
            r["run"] = run_name
            r["epoch"] = epoch
            results.append(r)
            print(f"  AbsRel={r['AbsRel']:.4f}  RMSE={r['RMSE_mm']:.3f}mm  mean_err={r['mean_error_mm']:.3f}mm")

    print("\n" + "=" * 70)
    print(f"{'Run':<18} {'Epoch':<6} {'AbsRel':>8} {'RMSE_mm':>10} {'mean_err_mm':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['run']:<18} {r['epoch']:<6} {r['AbsRel']:>8.4f} {r['RMSE_mm']:>10.3f} {r['mean_error_mm']:>12.3f}")

    with open("sawbone_convergence_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "epoch", "AbsRel", "RMSE_mm", "mean_error_mm"])
        writer.writeheader()
        writer.writerows(results)
    print("\nWrote sawbone_convergence_comparison.csv")


if __name__ == "__main__":
    main()
