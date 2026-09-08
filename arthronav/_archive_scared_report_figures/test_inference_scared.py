"""
Quick sanity check: load DA3METRIC-LARGE, run it on N real SCARED frames,
report predicted depth ranges. Not training -- just confirms the model
and data pipeline agree on shapes/values.

Usage:

    cd ~/Depth-Anything-3
    python -m arthronav.test_inference_scared -n 5 -v

    python arthronav/test_inference_scared.py -n 5 -v
    python arthronav/test_inference_scared.py --num-samples 20
"""
import argparse

import torch
torch.backends.cudnn.enabled = False
import torch.nn.functional as F
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3
from arthronav.scared_io import build_frame_list, split_frames
from arthronav.scared_dataset import SCAREDDataset

H5_ROOT = "/mnt/areas_nas/SLAM/scared_dataset_full_copy/depth_anything_preprocessed_data/train_depth_anything"
JSON_ROOT = "/mnt/areas_nas/SLAM/scared_dataset_full_copy/frame_trajectory_data"

# nearest multiples of 14 (DINOv2 patch size) below the native 1024x1280
TARGET_SIZE = (1022, 1274)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num-samples", type=int, default=5, help="how many frames to test")
    ap.add_argument("-v", "--verbose", action="store_true", help="print per-frame details")
    args = ap.parse_args()

    print(f"Using device: {DEVICE}")

    print("Loading model...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    model.eval()
    model = model.to(DEVICE)

    print("Building frame list...")
    frames = build_frame_list(H5_ROOT, JSON_ROOT)
    train, val = split_frames(frames)
    ds = SCAREDDataset(train)

    n = min(args.num_samples, len(ds))
    print(f"Running inference on {n} frames...")

    for i in tqdm(range(n), desc="testing frames"):
        sample = ds[i]
        rgb = sample["rgb"].unsqueeze(0)
        rgb = F.interpolate(rgb, size=TARGET_SIZE, mode="bilinear", align_corners=False)
        rgb = rgb.unsqueeze(1)
        rgb = rgb.to(DEVICE)

        with torch.inference_mode():
            output = model(rgb, export_feat_layers=[])

        if args.verbose:
            d = output.depth
            tqdm.write(
                f"frame {i} | {ds.frames[i]['dataset']}/{ds.frames[i]['keyframe']}/{ds.frames[i]['frame_idx']} "
                f"| pred depth min={d.min().item():.4f} max={d.max().item():.4f}"
            )

    print("Done.")


if __name__ == "__main__":
    main() 
