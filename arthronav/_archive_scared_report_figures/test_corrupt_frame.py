"""
Isolated test on the one known-corrupted file, to directly confirm:
  1. it really crashes when loaded the old way (no protection)
  2. the resilient SCAREDDataset (with bad_files_path) skips it cleanly

Usage:
    python -m arthronav.test_corrupt_frame
"""

from arthronav.scared_io import build_frame_list, load_frame_h5
from arthronav.scared_dataset import SCAREDDataset

H5_ROOT = "/mnt/areas_nas/SLAM/scared_dataset_full_copy/depth_anything_preprocessed_data/train_depth_anything"
JSON_ROOT = "/mnt/areas_nas/SLAM/scared_dataset_full_copy/frame_trajectory_data"

BAD_PATH = "/mnt/areas_nas/SLAM/scared_dataset_full_copy/depth_anything_preprocessed_data/train_depth_anything/dataset_2/keyframe_4/001653.h5"


def main():
    print("Building frame list (dataset_2/keyframe_4 only, for speed)...")
    all_frames = build_frame_list(H5_ROOT, JSON_ROOT)
    small_frames = [f for f in all_frames if f["dataset"] == "dataset_2" and f["keyframe"] == "keyframe_4"]
    print(f"Using {len(small_frames)} frames from dataset_2/keyframe_4 (includes the bad one)")

    print("\n--- Test 1: raw load_frame_h5 directly on the known-bad file ---")
    try:
        rgb, depth = load_frame_h5(BAD_PATH)
        print("Unexpectedly succeeded, no crash. rgb shape:", rgb.shape)
    except Exception as e:
        print(f"Crashed as expected: {type(e).__name__}: {e}")

    print("\n--- Test 2: resilient SCAREDDataset, WITHOUT bad_files_path (no filtering) ---")
    ds_unprotected = SCAREDDataset(small_frames)
    bad_idx = next(i for i, f in enumerate(small_frames) if f["h5_path"] == BAD_PATH)
    print(f"Bad file is at index {bad_idx} in this small dataset")
    sample = ds_unprotected[bad_idx]
    print("Retry logic kicked in and returned a valid frame instead of crashing.")
    print("rgb shape:", sample["rgb"].shape, "| depth shape:", sample["depth"].shape)

    print("\n--- Test 3: resilient SCAREDDataset, WITH bad_files_path (excluded up front) ---")
    ds_protected = SCAREDDataset(small_frames, bad_files_path="bad_h5_files.txt")
    print(f"Frame count before exclusion: {len(small_frames)}, after: {len(ds_protected)}")

    print("\nAll tests completed without an unhandled crash.")


if __name__ == "__main__":
    main()
