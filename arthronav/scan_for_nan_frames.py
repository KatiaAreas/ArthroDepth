"""
Scan every frame of the given patient's clips, running the model on each
and checking only for NaN/Inf in the prediction (no full metric
computation, this is a fast, targeted search for WHICH frame(s) trigger
NaN, not a validation run).

Usage:
    python -m arthronav.scan_for_nan_frames \
        --checkpoint checkpoints/real_knee_training/from_scratch_vector/epoch_7.pt \
        --vector-lora --patient 2602484M
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from depth_anything_3.api import DepthAnything3

from arthronav.lora import inject_lora, inject_vector_lora
from arthronav.real_knee_io import build_frame_list

TARGET_SIZE = (1078, 1918)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vector-lora", action="store_true")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--patient", required=True)
    ap.add_argument("--data-root", default="/mnt/areas_nas/SLAM/real_knee_dataset")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    wrapper = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = wrapper.model
    if args.vector_lora:
        inject_vector_lora(net)
    else:
        inject_lora(net, rank=args.lora_rank)
    net = net.to(device)
    net.eval()

    state = torch.load(args.checkpoint, map_location=device)
    _, unexpected = net.load_state_dict(state, strict=False)
    print(f"Loaded {len(state) - len(unexpected)} tensors")

    frames = build_frame_list(args.data_root, [args.patient])
    print(f"Scanning {len(frames)} frames for patient {args.patient}...")

    bad_frames = []
    with torch.no_grad():
        for i, entry in enumerate(frames):
            rgb_bgr = cv2.imread(entry["rgb_path"])
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            rgb_t = F.interpolate(rgb_t, size=TARGET_SIZE, mode="bilinear", align_corners=False)
            rgb_in = rgb_t.unsqueeze(1).to(device)

            output = net(rgb_in, export_feat_layers=[])
            pred = output.depth

            n_nan = torch.isnan(pred).sum().item()
            n_inf = torch.isinf(pred).sum().item()

            input_nan = torch.isnan(rgb_t).sum().item()

            if n_nan > 0 or n_inf > 0 or input_nan > 0:
                bad_frames.append((entry["view"], entry["frame_idx"], n_nan, n_inf, input_nan))
                print(f"  BAD: view={entry['view']} frame={entry['frame_idx']} "
                      f"pred_nan={n_nan} pred_inf={n_inf} input_nan={input_nan}")

            if (i + 1) % 500 == 0:
                print(f"  ...{i+1}/{len(frames)} scanned, {len(bad_frames)} bad so far")

    print(f"\nDone. {len(bad_frames)} bad frames out of {len(frames)} total.")
    if bad_frames:
        print("Bad frames:", bad_frames)


if __name__ == "__main__":
    main()
