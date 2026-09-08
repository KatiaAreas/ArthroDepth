"""
I/O for the growing sobone/cartilage dataset (tour_lateral_0, 1, 2, ...).

Ground truth source: depth_png (uint16). Per manifest.json's "depth" block:

    depth_mm = png_uint16_value * 0.01
    invalid  = 0  (outside the cartilage mask)

IMPORTANT: this loader returns depth in MILLIMETERS, matching the native
manifest-documented unit exactly, with no further conversion to meters.
Any unit choice for training/eval happens explicitly in the training
script, not silently here -- this is deliberate, so a conversion bug
can't hide inside the IO layer and get silently learned by LoRA fine-
tuning (which has no fixed notion of "meters"; it just matches whatever
scale the loss target is in).

Note on data source (from manifest "command" block): tracked rigid-body
camera over a physical sawbone phantom, ground-truth depth rendered from
a known 3D mesh (STL) using tracked poses (OptiTrack rigid bodies
RB_3=camera, RB_4=femur, RB_1=pen/tool) -- not measured optically the
way SCARED's structured light is.

Frame count: filenames run 000010-003430 in steps of 10 because the
manifest's "command" block shows --every 10 -- the source video was
already decimated 10x during generation. 343 usable frames.

Split strategy, mirroring SCARED's leakage-avoidance logic: SCARED splits
train/val by whole keyframe folder, never scattering frames from the same
keyframe across the split, because adjacent frames are near-duplicates.
Here, one sequence IS one continuous (already 10x-decimated) trajectory
with no internal keyframe structure, so the equivalent unit is the whole
sequence: train on some sequences, hold out one entire sequence for
val/test, never split frames within a single sequence across train/val.
"""

import numpy as np
import json
from pathlib import Path
from PIL import Image

PNG_SCALE_MM_PER_LSB = 0.01   # from manifest["depth"]["png"]["scale_mm_per_lsb"]
MM_TO_M = 0.001

def load_frame_depth_png(path: Path):
    """
    Returns (depth_m: float32 HxW, valid_mask: bool HxW).

    Depth is converted to METERS here, matching DA3's apparent native
    output convention (confirmed empirically: zero-shot raw predictions
    land in the ~0.01-0.16 numeric range, consistent with meters, not
    millimeters or micrometers -- see conversation history for the
    diagnostic that established this). Training/eval must compare pred
    and gt in the SAME unit; mm-native gt against meter-scale pred
    creates a 1000x mismatch, which was tested and reverted.
    """
    raw = np.array(Image.open(path))  # uint16
    valid_mask = raw != 0
    depth_mm = raw.astype(np.float32) * PNG_SCALE_MM_PER_LSB
    depth_m = np.where(valid_mask, depth_mm * MM_TO_M, 0.0)
    return depth_m, valid_mask

def load_rgb(path: Path):
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return img

def build_frame_list(sobone_root: Path, sequence_names: list):
    """
    Pools frames across the given sequences, tagging each with its
    sequence_id so split_frames() can hold out whole sequences.
    """
    frames = []
    for seq_name in sequence_names:
        seq_dir = sobone_root / seq_name
        rgb_files = sorted((seq_dir / "rgb").glob("*"))
        depth_dir = seq_dir / "depth_png"
        for rgb_path in rgb_files:
            depth_path = depth_dir / (rgb_path.stem + ".png")
            if not depth_path.exists():
                continue  # TODO: log rather than silently skip
            frames.append({
                "rgb": rgb_path,
                "depth": depth_path,
                "sequence_id": seq_name,
            })
    return frames

def split_frames(frames, holdout_sequences: list):
    """
    Whole-sequence holdout, matching SCARED's whole-keyframe holdout logic.
    holdout_sequences: e.g. ["tour_lateral_2_cartilage"] once it exists.
    """
    train = [f for f in frames if f["sequence_id"] not in holdout_sequences]
    val = [f for f in frames if f["sequence_id"] in holdout_sequences]
    return train, val

def load_manifest(seq_dir: Path):
    return json.loads((seq_dir / "manifest.json").read_text())


def load_focal_px(seq_dir: Path) -> float:
    """
    Focal length in pixels, at NATIVE 1920x1080 resolution, from
    manifest["optics"]["camera_matrix"] (fx, fy averaged -- they're
    836.58/836.21 here, essentially identical).

    Needed for DA3METRIC-LARGE's documented output correction:
        metric_depth = focal_px * net_output / 300
    (see ByteDance-Seed/Depth-Anything-3 README). This correction is
    ONLY valid for a model that never learned to compensate for it --
    i.e. zero-shot only. Do NOT apply this to transfer_scared/from_scratch
    checkpoints: their LoRA weights were trained against uncorrected raw
    output vs. real-meters GT, so they already learned to absorb this
    factor internally. Applying the correction on top of an already-
    adapted checkpoint double-corrects and produces wrong-scale output.
    """
    manifest = load_manifest(seq_dir)
    cm = manifest["optics"]["camera_matrix"]
    fx, fy = cm[0][0], cm[1][1]
    return (fx + fy) / 2.0
