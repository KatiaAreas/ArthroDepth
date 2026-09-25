<div align="center">
<h1 style="border-bottom: none; margin-bottom: 0px">ArthroDepth</h1>
<h3 style="border-top: none; margin-top: 3px;">Monocular metric depth and keypoint tracking for arthroscopic knee navigation</h3>
</div>

ArthroDepth is AREAS SAS's ARTHRONAV project: adapting [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (DA3METRIC-LARGE), via Vector-LoRA fine-tuning, to real-time monocular metric depth estimation on arthroscopic knee video, plus a keypoint-tracking layer (LiteTracker, CoTracker-based) for the eventual navigation pipeline.

This repository is a fork of [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3). The upstream model and codebase (src/depth_anything_3/) are used as-is; all of ArthroDepth's own work lives under arthronav/.

WHAT'S HERE

- arthronav/ -- training, data preparation, and evaluation code for every dataset used: SCARED (endoscopic domain bridge), sawbone phantom (red and white), and real patient arthroscopic data (AREAS's own cohort, 8 patients across two recording sites).
- Reports -- full writeups of method, results, and the roadmap: the depth fine-tuning report (SCARED to sawbone to real patient data to combined dataset) and the LiteTracker occlusion-robustness report.
- checkpoints/ -- trained LoRA adapters (a few MB each, only the trainable weights) for each dataset and training regime.

Key results so far: Vector-LoRA matches uniform LoRA at under half the trainable parameters; transfer learning from SCARED reverses on a true held-out test patient despite looking worse in validation, a leave-one-patient-out study is the current top priority; and continuing from the red-sawbone checkpoint onto white sawbone gives the cleanest, largest transfer-learning gain found in the whole project (about 2x). See the depth report for the full picture, including the negative results and the unit-consistency bugs found and fixed along the way.

INSTALLATION

Same as upstream DA3:

    pip install xformers torch>=2 torchvision
    pip install -e .
    pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70

Plus arthronav's own dependencies (addict, opencv-python, areas_theta_compute for circle detection -- install this last one standalone, before importing torch anywhere in the same process, see the note in precompute_circle_masks.py).

BASE MODEL: DA3METRIC-LARGE

ArthroDepth fine-tunes DA3METRIC-LARGE specifically (https://huggingface.co/depth-anything/DA3METRIC-LARGE, 0.35B params, monocular metric depth, Apache 2.0). Basic upstream usage:

    import torch
    from depth_anything_3.api import DepthAnything3

    device = torch.device("cuda")
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    model = model.to(device=device)

Metric depth scale note (upstream FAQ): to obtain metric depth in meters from a model that doesn't already output meters directly, use metric_depth = focal * net_output / 300. Every ArthroDepth checkpoint is trained to output real meters directly (verified empirically per dataset, see the depth report's units sections), so this conversion is not needed for our own checkpoints, only relevant if working with a raw/unfine-tuned upstream model.

UPSTREAM PROJECT

Depth Anything 3, from ByteDance Seed: paper (https://arxiv.org/abs/2511.10647), project page (https://depth-anything-3.github.io), original repo (https://github.com/ByteDance-Seed/Depth-Anything-3). See the upstream repository for the full model zoo, CLI/API documentation, and benchmark evaluation pipeline, none of that is duplicated here.

CITATION

If referencing the base model, cite the original paper:

    @article{depthanything3,
      title={Depth Anything 3: Recovering the visual space from any views},
      author={Haotong Lin and Sili Chen and Jun Hao Liew and Donny Y. Chen and Zhenyu Li and Guang Shi and Jiashi Feng and Bingyi Kang},
      journal={arXiv preprint arXiv:2511.10647},
      year={2025}
    }
