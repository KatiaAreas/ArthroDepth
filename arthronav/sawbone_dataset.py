"""
Sawbone/cartilage dataset (PNG-depth-backed). Wraps sawbone_io.py's frame list,
mirroring SCAREDDataset's shape and conventions so the training/eval loop
code can stay unified across both datasets.
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from arthronav.sawbone_io import load_frame_depth_png, load_rgb
from arthronav.crop_utils import get_crop_box_for_sequence, crop_and_resize


class SawboneDataset(Dataset):
    """
    crop_size: if set, detects the endoscope circle once per sequence
    (cached), crops RGB+depth to a square around it, then resizes to
    crop_size x crop_size. depth is cropped with the SAME box as RGB
    (nearest-neighbor resize, to avoid inventing fractional depth values
    at the mask boundary) so pred/gt stay spatially aligned.
    If crop_size is None, behaves exactly as before (no cropping).
    """
    def __init__(self, frame_list, crop_size=None):
        self.frames = list(frame_list)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        entry = self.frames[idx]
        rgb = load_rgb(entry["rgb"])          # (H, W, 3) float32 [0,1]
        depth_m, valid_mask = load_frame_depth_png(entry["depth"])  # (H, W) each, METERS

        if self.crop_size is not None:
            seq_dir = Path(entry["rgb"]).parent.parent  # .../<sequence>/rgb/xxx.png -> .../<sequence>
            box = get_crop_box_for_sequence(seq_dir, entry["rgb"])
            rgb = crop_and_resize((rgb * 255).astype(np.uint8), box, self.crop_size,
                                   is_depth=False).astype(np.float32) / 255.0
            depth_m = crop_and_resize(depth_m, box, self.crop_size, is_depth=True)

        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()  # (3, H, W)
        depth = torch.from_numpy(depth_m).float()               # (H, W)

        return {
            "rgb": rgb,
            "depth": depth,
        }
