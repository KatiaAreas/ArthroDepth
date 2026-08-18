"""
Sobone/cartilage dataset (PNG-depth-backed). Wraps sobone_io.py's frame list,
mirroring SCAREDDataset's shape and conventions so the training/eval loop
code can stay unified across both datasets.
"""

import torch
from torch.utils.data import Dataset

from arthronav.sobone_io import load_frame_depth_png, load_rgb


class SoboneDataset(Dataset):
    def __init__(self, frame_list):
        self.frames = list(frame_list)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        entry = self.frames[idx]
        rgb = load_rgb(entry["rgb"])          # (H, W, 3) float32 [0,1]
        depth_m, valid_mask = load_frame_depth_png(entry["depth"])  # (H, W) each, METERS

        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()  # (3, H, W)
        depth = torch.from_numpy(depth_m).float()               # (H, W)

        return {
            "rgb": rgb,
            "depth": depth,
        }
