"""
Aggregate frames_exported / frames_skipped across all real-knee clips,
grouped by patient, to check whether the ~39% yield seen on
2509457F_medial is representative or an outlier.

Usage:
    cd ~/Depth-Anything-3
    python -m arthronav.audit_real_knee_yield
"""

import json
from pathlib import Path

ROOT = "/mnt/areas_nas/SLAM/real_knee_dataset"
KNOWN_VIEWS = {"medial", "lateral"}


def patient_and_view(dirname):
    """
    Splits 'PATIENT_view' into (patient, view) if the trailing token is a
    known view name; otherwise returns (dirname, None) rather than
    guessing, so unexpected naming shows up plainly instead of being
    silently merged into the wrong bucket.
    """
    parts = dirname.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in KNOWN_VIEWS:
        return parts[0], parts[1]
    return dirname, None


def main():
    root = Path(ROOT)
    clip_dirs = sorted(d for d in root.iterdir() if d.is_dir())

    print(f"Found {len(clip_dirs)} clip directories under {ROOT}\n")

    rows = []
    for d in clip_dirs:
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            print(f"WARNING: no metadata.json in {d.name}, skipping")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        exported = meta.get("frames_exported")
        skipped = meta.get("frames_skipped")
        if exported is None or skipped is None:
            print(f"WARNING: {d.name} metadata.json missing frames_exported/frames_skipped keys "
                  f"(available keys: {sorted(meta.keys())})")
            continue

        total = exported + skipped
        yield_pct = 100.0 * exported / total if total else float("nan")

        patient, view = patient_and_view(d.name)
        rows.append({
            "clip": d.name,
            "patient": patient,
            "view": view,
            "exported": exported,
            "skipped": skipped,
            "total": total,
            "yield_pct": yield_pct,
        })

    if not rows:
        print("No usable metadata found -- check ROOT path and key names above.")
        return

    print(f"{'Clip':<28} | {'Exported':>9} | {'Skipped':>9} | {'Total':>7} | {'Yield %':>8}")
    print("-" * 72)
    for r in rows:
        print(f"{r['clip']:<28} | {r['exported']:>9} | {r['skipped']:>9} | {r['total']:>7} | {r['yield_pct']:>7.1f}%")

    print("\nPer-patient aggregate (all views pooled):")
    print(f"{'Patient':<16} | {'Exported':>9} | {'Skipped':>9} | {'Total':>7} | {'Yield %':>8}")
    print("-" * 60)
    patients = sorted(set(r["patient"] for r in rows))
    for p in patients:
        p_rows = [r for r in rows if r["patient"] == p]
        exp = sum(r["exported"] for r in p_rows)
        skp = sum(r["skipped"] for r in p_rows)
        tot = exp + skp
        pct = 100.0 * exp / tot if tot else float("nan")
        print(f"{p:<16} | {exp:>9} | {skp:>9} | {tot:>7} | {pct:>7.1f}%")

    overall_exp = sum(r["exported"] for r in rows)
    overall_skp = sum(r["skipped"] for r in rows)
    overall_tot = overall_exp + overall_skp
    overall_pct = 100.0 * overall_exp / overall_tot if overall_tot else float("nan")
    print("\nOverall (all patients, all clips pooled):")
    print(f"exported={overall_exp}, skipped={overall_skp}, total={overall_tot}, yield={overall_pct:.1f}%")


if __name__ == "__main__":
    main()
