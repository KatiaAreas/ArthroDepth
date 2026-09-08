"""
Build a lean, preprocessed local copy of one real-knee clip (patient +
view), the same philosophy as SCARED's depth_anything_preprocessed_data:
extract ONLY the video frames that actually have matching depth ground
truth (roughly 40% of the raw video, per the metadata's valid_fraction/
frames_exported counts), so training never has to seek through the full
raw video, and never touches S3 directly during a training run.

Storage: depth/sigma are re-saved as float16 (not the original float32).
Checked as safe: float16's precision at these depth values (~12-32mm) is
about 0.02mm per step, roughly 40x finer than this dataset's own 0.8mm
sigma floor, so nothing meaningful is lost. RGB is re-encoded as JPEG
(quality 95) instead of PNG. Together this cuts per-frame size from
~21-25MB to roughly 4-5MB, which matters directly for AWS's limited
local disk (84GB free, no NAS access) -- 6 training clips at the
original size would be ~95-110GB, over budget; at this reduced size,
roughly 25-30GB, comfortably fits.

Source layout (S3):
    s3://datascientists-data/depth/autriche_<PATIENT>_<view>.zip
        -> <name>/depth_gt/frame_NNNNNN.{depth,sigma,valid}.npy (+ .png previews)
        -> <name>/depth_gt/metadata.json, metrics.csv
    s3://datascientists-data/raw/autriche_2026-07-24/knee_<PATIENT>_G/MVS/tour_<view>/video/tour_<view>_0.mp4
        -> the "_0" video is the one actually used for the depth reconstruction

Output layout (local):
    <out_root>/<PATIENT>_<view>/
        rgb/frame_NNNNNN.jpg          (only frames with matching depth_gt, JPEG q=95)
        depth/frame_NNNNNN.depth.npy  (float16, real millimeters, NO correction factor
                                        needed -- verified directly: 12-32mm range for
                                        this dataset, matches metadata's sigma_floor_mm)
        depth/frame_NNNNNN.sigma.npy  (float16, real per-pixel uncertainty, mm)
        depth/frame_NNNNNN.valid.npy  (unchanged, already compact uint8)
        metadata.json                 (copied as-is, has camera intrinsics)

Usage:
    python -m arthronav.prepare_real_knee_data \
        --patient 2509457F --view medial \
        --s3-depth-zip s3://datascientists-data/depth/autriche_2509457F_medial.zip \
        --s3-video s3://datascientists-data/raw/autriche_2026-07-24/knee_2509457F_G/MVS/tour_medial/video/tour_medial_0.mp4 \
        --out-root /mnt/areas_nas/SLAM/real_knee_dataset
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile

import cv2
import numpy as np


def s3_download(s3_uri, local_path, profile=None):
    cmd = ["aws", "s3", "cp", s3_uri, local_path]
    if profile:
        cmd += ["--profile", profile]
    print(f"Downloading {s3_uri} ...")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True, help="e.g. 2509457F")
    ap.add_argument("--view", required=True, choices=["lateral", "medial"])
    ap.add_argument("--s3-depth-zip", required=True)
    ap.add_argument("--s3-video", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--aws-profile", default=None,
                     help="pass --profile areas-datascientist on Grenoble; omit on AWS (uses instance role)")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args()

    clip_name = f"{args.patient}_{args.view}"
    out_dir = os.path.join(args.out_root, clip_name)
    rgb_dir = os.path.join(out_dir, "rgb")
    depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "depth.zip")
        video_path = os.path.join(tmp, "video.mp4")

        s3_download(args.s3_depth_zip, zip_path, args.aws_profile)
        s3_download(args.s3_video, video_path, args.aws_profile)

        print("Extracting depth_gt files (re-saving depth/sigma as float16)...")
        frame_indices = set()
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            metadata_name = next((n for n in names if n.endswith("depth_gt/metadata.json")), None)
            if metadata_name is None:
                raise RuntimeError("Could not find metadata.json in the zip")
            with zf.open(metadata_name) as f:
                metadata = json.load(f)
            shutil.copyfile(zf.extract(metadata_name, tmp), os.path.join(out_dir, "metadata.json"))

            npy_pattern = re.compile(r"frame_(\d{6})\.(depth|sigma|valid)\.npy$")
            for n in names:
                m = npy_pattern.search(n)
                if m is None:
                    continue
                frame_idx = int(m.group(1))
                field = m.group(2)
                frame_indices.add(frame_idx)
                dest = os.path.join(depth_dir, os.path.basename(n))

                if field == "valid":
                    with zf.open(n) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                else:
                    extracted_path = zf.extract(n, tmp)
                    arr = np.load(extracted_path).astype(np.float16)
                    np.save(dest, arr)
                    os.remove(extracted_path)

        print(f"Extracted depth ground truth for {len(frame_indices)} frames")
        print(f"metadata: intrinsics={metadata.get('intrinsics')}, "
              f"depth_convention={metadata.get('depth_convention')}, "
              f"frames_exported={metadata.get('frames_exported')}, "
              f"frames_skipped={metadata.get('frames_skipped')}")

        print("Extracting matching RGB frames from video (re-encoding as JPEG)...")
        cap = cv2.VideoCapture(video_path)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        expected_total = metadata.get("frames_exported", 0) + metadata.get("frames_skipped", 0)
        if total_video_frames != expected_total:
            print(f"WARNING: video has {total_video_frames} frames, but metadata says "
                  f"frames_exported+frames_skipped={expected_total}. Frame indices may not "
                  f"align correctly -- double check this is really the right video file "
                  f"(the '_0' vs '_1' take) before trusting the extracted pairs.")

        extracted = 0
        frame_idx = 0
        target_indices = sorted(frame_indices)
        target_set = set(target_indices)
        jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in target_set:
                out_path = os.path.join(rgb_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(out_path, frame, jpeg_params)
                extracted += 1
            frame_idx += 1
        cap.release()

        print(f"Extracted {extracted} / {len(target_indices)} matching RGB frames")
        if extracted != len(target_indices):
            print("WARNING: extracted count doesn't match expected -- some depth_gt frames "
                  "have no corresponding video frame. Check frame index alignment.")

    print(f"\nDone. Preprocessed clip saved to {out_dir}")
    print(f"  rgb/:   {len(os.listdir(rgb_dir))} files")
    print(f"  depth/: {len(os.listdir(depth_dir))} files")
    total_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(out_dir) for f in fs
    )
    print(f"  total size: {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
