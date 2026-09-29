# Deep Learning-Based Low-Light Image and Video Enhancement During Night Surveillance

Working implementation of the hybrid **Transformer + RetinexNet + Zero-DCE++**
pipeline described in the project deck (Team-8), with the dual-pipeline
routing for images and the chunk/keyframe streaming pipeline for videos.

## What's implemented

| Deck concept | File | Notes |
|---|---|---|
| Zero-DCE++ (0.003s, fastest branch) | `models/zero_dce.py` | Depthwise-separable curve estimation net, no-reference training |
| RetinexNet (best color preservation) | `models/retinex_net.py` | DecomNet (reflectance/illumination split) + EnhanceNet |
| Transformer (highest PSNR/SSIM) | `models/transformer_enhancer.py` | Compact Swin-style windowed self-attention |
| Non-reference losses for Zero-DCE++ | `losses.py` | Spatial consistency, exposure control, color constancy, illumination smoothness |
| Paired losses for RetinexNet/Transformer | `losses.py` | Reconstruction + edge-aware smoothness / gradient loss |
| Brightness Classifier (routing) | `brightness_classifier.py` | Heuristic (luminance + shadow-percentile) + optional learnable CNN |
| Image dual-pipeline + tiling | `pipeline_image.py` | Slightly-dark → Zero-DCE++ only; very-dark → fusion of all 3; tiles + feather-blends large/4K images |
| Video chunk pipeline | `pipeline_video.py` | FFmpeg-backed streaming, 5-sec overlapping chunks, keyframe (every 10th) vs. non-keyframe (Zero-DCE++ + optical-flow propagation) routing, audio re-mux |
| PSNR / SSIM / ALV metrics | `metrics.py` | ALV = average luminance variance, the flicker metric quoted on slides 5/18 |
| Dataset loader | `datasets.py` | LOL-v2 / VE-LOL paired layout (`low/`, `high/` folders) + unpaired loader for Zero-DCE++ |

Every file has been run and verified in this environment (see the `if
__name__ == "__main__":` block in each — they're runnable smoke tests, not
just docstrings).

## Honest scope notes (read before your viva)

- The Transformer is a **compact** Swin-style block (a few attention layers),
  not the full multi-stage Swin-Transformer from the paper it's inspired by —
  sized so it can actually be trained on LOL-v2 without a multi-GPU cluster.
  Say so if asked; it's a legitimate engineering trade-off, not a shortcut to hide.
- **ONNX export + TensorRT FP16** (mentioned on slide 5) aren't wired up here —
  that's a deployment/optimization step you'd add once the PyTorch models are
  trained and validated. `torch.onnx.export()` on each model is the natural
  next step; happy to add it once you have trained weights.
- The **brightness classifier** ships as a fast heuristic (luminance + shadow
  check) by default. A learnable CNN version is included in the same file if
  the heuristic misclassifies too often on your actual footage.
- Optical-flow propagation uses classic **Farneback dense flow** (OpenCV,
  CPU-friendly) rather than a learned flow network — matches the "TBEFN idea"
  referenced on slide 5 without needing another model to train.
- All three networks currently run with **randomly-initialized weights**
  until you train them (see below) — the pipelines are correct and tested,
  but enhancement quality depends entirely on training.

## Setup

```bash
pip install -r requirements.txt
sudo apt-get install ffmpeg   # required for the video pipeline
```

## Data layout (LOL-v2 / VE-LOL)

```
dataset_root/
    low/
        00001.png
        00002.png
    high/
        00001.png   # same filename as its low-light pair
        00002.png
```

## Training

```bash
# Zero-DCE++ -- no ground truth needed, just a folder of low-light images
python train_zero_dce.py --data_root /path/to/low_images --epochs 50 \
    --out checkpoints/zero_dce.pt

# RetinexNet -- needs paired low/high data
python train_retinex.py --data_root /path/to/dataset_root --epochs 100 \
    --out checkpoints/retinex.pt

# Transformer -- needs paired low/high data
python train_transformer.py --data_root /path/to/dataset_root --epochs 200 \
    --out checkpoints/transformer.pt
```

## Inference

```bash
# Single image or a folder of images
python infer_image.py --input samples/in --output samples/out \
    --zero_dce_ckpt checkpoints/zero_dce.pt \
    --retinex_ckpt checkpoints/retinex.pt \
    --transformer_ckpt checkpoints/transformer.pt

# Video (chunk-streamed, so length is not limited by RAM)
python infer_video.py --input night_cctv.mp4 --output night_cctv_enhanced.mp4 \
    --zero_dce_ckpt checkpoints/zero_dce.pt \
    --retinex_ckpt checkpoints/retinex.pt \
    --transformer_ckpt checkpoints/transformer.pt \
    --chunk_seconds 5 --keyframe_interval 10
```

Checkpoint arguments are optional — omitting them runs the pipeline with
random-init weights, which is useful for confirming the wiring works before
you've trained anything (that's exactly what `samples/` in this repo does).

## Evaluating against the deck's results table

`metrics.py` gives you `compute_psnr`, `compute_ssim`, and
`average_luminance_variance` — run these on your test split's outputs vs.
ground truth to reproduce a table in the same shape as slide 16/17:

```python
from metrics import evaluate_batch, average_luminance_variance
# evaluate_batch(enhanced_batch, gt_batch) -> {"psnr": ..., "ssim": ...}
# average_luminance_variance(list_of_frames) -> flicker score (lower=better)
```

## Project layout

```
lowlight_enhancement/
├── models/
│   ├── zero_dce.py
│   ├── retinex_net.py
│   └── transformer_enhancer.py
├── losses.py
├── brightness_classifier.py
├── datasets.py
├── metrics.py
├── pipeline_image.py
├── pipeline_video.py
├── train_zero_dce.py
├── train_retinex.py
├── train_transformer.py
├── infer_image.py
├── infer_video.py
├── requirements.txt
└── samples/            # tiny synthetic smoke-test in/out (safe to delete)
```
