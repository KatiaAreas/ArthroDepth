"""
Loss functions for training. Starting simple: plain L1 on valid depth
pixels. We can swap in something fancier (scale-invariant log loss,
edge-aware terms) once this baseline is confirmed working.
"""

import torch


def masked_l1_loss(pred, gt, mask):
    """
    pred, gt: (B, H, W)
    mask: (B, H, W) bool, True where gt is valid
    """
    if mask.sum() == 0:
        return pred.sum() * 0.0  # no valid pixels in this batch
    return (pred[mask] - gt[mask]).abs().mean()


def gradient_loss(pred, gt, mask):
    """
    L1 loss on depth gradients (how fast depth changes pixel to pixel),
    not the raw depth values. Targets local softness/blur rather than
    per-pixel accuracy, complements masked_l1_loss rather than replacing
    it.

    A gradient at a given pixel is only valid if both it and its
    neighbor (in that direction) have valid ground truth.
    """
    pred_dx = pred[:, :, 1:] - pred[:, :, :-1]
    gt_dx = gt[:, :, 1:] - gt[:, :, :-1]
    mask_dx = mask[:, :, 1:] & mask[:, :, :-1]

    pred_dy = pred[:, 1:, :] - pred[:, :-1, :]
    gt_dy = gt[:, 1:, :] - gt[:, :-1, :]
    mask_dy = mask[:, 1:, :] & mask[:, :-1, :]

    loss_dx = (pred_dx[mask_dx] - gt_dx[mask_dx]).abs().mean() if mask_dx.any() else pred.sum() * 0.0
    loss_dy = (pred_dy[mask_dy] - gt_dy[mask_dy]).abs().mean() if mask_dy.any() else pred.sum() * 0.0
    return (loss_dx + loss_dy) / 2
