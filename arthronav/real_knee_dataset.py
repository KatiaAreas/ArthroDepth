"""
PyTorch Dataset for the preprocessed real-knee dataset.

IMPORTANT: depth/sigma here are already real millimeters. Do NOT apply
SCARED's 0.256 correction factor to this data -- that factor is specific
to SCARED's h5 encoding (raw_mm / 256) and does not apply here. Verified
directly for this dataset: depth range 12-32mm for a real knee joint,
sigma values matching metadata's sigma_floor_mm exactly. Applying the
SCARED correction here would silently produce wrong numbers that still
look plausible -- exactly the kind of mistake this note exists to
prevent.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RealKneeDataset(Dataset):
    def __init__(self, frame_list, load_sigma: bool = True):
        self.frames = list(frame_list)
        self.load_sigma = load_sigma

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        entry = self.frames[idx]

        rgb_bgr = cv2.imread(entry["rgb_path"])
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3, H, W), [0,1]

        depth = np.load(entry["depth_path"])
        valid = np.load(entry["valid_path"]).astype(bool)

        depth = torch.from_numpy(depth).float()   # (H, W), real millimeters
        valid = torch.from_numpy(valid).bool()     # (H, W)

        sample = {
            "rgb": rgb,
            "depth": depth,
            "valid": valid,
            "patient": entry["patient"],
            "view": entry["view"],
            "frame_idx": entry["frame_idx"],
        }

        if self.load_sigma:
            sigma = np.load(entry["sigma_path"])
            sample["sigma"] = torch.from_numpy(sigma).float()  # (H, W), real mm uncertainty

        return sample
