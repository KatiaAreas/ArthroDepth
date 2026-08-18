"""
Standard depth-estimation metrics, matching the convention used across
the endoscopic-depth literature (EndoDAC, Endo-FASt3r, etc.) so numbers
are directly comparable.
"""

import numpy as np
import torch


def abs_rel(pred, gt, mask):
    """Mean absolute relative error: mean(|pred - gt| / gt), valid pixels only."""
    if mask.sum() == 0:
        return torch.tensor(float("nan"))
    return (torch.abs(pred[mask] - gt[mask]) / gt[mask]).mean()


def rmse(pred, gt, mask):
    """Root mean squared error, valid pixels only."""
    if mask.sum() == 0:
        return torch.tensor(float("nan"))
    return torch.sqrt(((pred[mask] - gt[mask]) ** 2).mean())


def compute_all_metrics(pred, gt, mask, unit_label="m"):
    """
    AbsRel is a genuinely unitless ratio (mean(|pred-gt|/gt)) -- reported
    as-is regardless of unit_label. RMSE is reported in whatever unit
    pred/gt are already in (label it via unit_label, e.g. "m" or "mm").
    """
    return {
        "AbsRel": abs_rel(pred, gt, mask).item(),
        f"RMSE_{unit_label}": rmse(pred, gt, mask).item(),
    }


def abs_error_stats(pred, gt, mask, unit_label="m"):
    """
    Raw absolute error (min, max, mean), over valid pixels only.
    UNIT-AGNOSTIC: pred/gt must already be in whichever unit you're
    working in. unit_label only affects the returned dict's key names,
    it does not convert anything -- pass "mm" for millimeter-scale data.
    """
    if mask.sum() == 0:
        return {
            f"min_error_{unit_label}": float("nan"),
            f"max_error_{unit_label}": float("nan"),
            f"mean_error_{unit_label}": float("nan"),
        }
    errors = (pred[mask] - gt[mask]).abs()
    return {
        f"min_error_{unit_label}": errors.min().item(),
        f"max_error_{unit_label}": errors.max().item(),
        f"mean_error_{unit_label}": errors.mean().item(),
    }


def error_distribution(pred, gt, mask):
    """
    Pooled percentile error distribution over valid pixels, in meters.
    Promoted from the ad hoc report code (Sec. 3.5) into reusable form.
    """
    if mask.sum() == 0:
        return {}
    errors = (pred[mask] - gt[mask]).abs()
    errors_np = errors.detach().cpu().numpy()
    return {
        "median_m": float(np.median(errors_np)),
        "p90_m": float(np.percentile(errors_np, 90)),
        "p95_m": float(np.percentile(errors_np, 95)),
        "p99_m": float(np.percentile(errors_np, 99)),
        "pct_gt_3mm": float((errors_np > 0.003).mean() * 100),
        "pct_gt_1cm": float((errors_np > 0.01).mean() * 100),
        "pct_gt_5cm": float((errors_np > 0.05).mean() * 100),
    }
