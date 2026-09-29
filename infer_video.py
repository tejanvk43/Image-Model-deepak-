"""
Run the hybrid video pipeline on a single video file.

Usage:
    python infer_video.py --input path/to/video.mp4 --output path/to/enhanced.mp4 \
        --zero_dce_ckpt checkpoints/zero_dce.pt \
        --retinex_ckpt checkpoints/retinex.pt \
        --transformer_ckpt checkpoints/transformer.pt \
        --chunk_seconds 5 --keyframe_interval 10
"""
import argparse
from pathlib import Path

import torch

from infer_image import load_checkpoint
from models.retinex_net import RetinexNet
from models.transformer_enhancer import TransformerEnhancer
from models.zero_dce import ZeroDCEPlusPlus
from pipeline_video import VideoEnhancementPipeline, VideoPipelineConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--zero_dce_ckpt", default=None)
    parser.add_argument("--retinex_ckpt", default=None)
    parser.add_argument("--transformer_ckpt", default=None)
    parser.add_argument("--chunk_seconds", type=float, default=5.0)
    parser.add_argument("--chunk_overlap_seconds", type=float, default=0.5)
    parser.add_argument("--keyframe_interval", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    zero_dce = load_checkpoint(ZeroDCEPlusPlus(), args.zero_dce_ckpt, device)
    retinex = load_checkpoint(RetinexNet(), args.retinex_ckpt, device)
    transformer = load_checkpoint(TransformerEnhancer(), args.transformer_ckpt, device)

    config = VideoPipelineConfig(
        chunk_seconds=args.chunk_seconds,
        chunk_overlap_seconds=args.chunk_overlap_seconds,
        keyframe_interval=args.keyframe_interval,
        device=device,
    )
    pipeline = VideoEnhancementPipeline(zero_dce=zero_dce, retinex=retinex, transformer=transformer, config=config)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    def progress(done, total):
        pct = 100 * done / max(total, 1)
        print(f"  ...{done}/{total} frames ({pct:.1f}%)", end="\r")

    stats = pipeline.enhance_video(args.input, args.output, progress_cb=progress)
    print(
        f"\nDone: {stats['frames_processed']} frames in {stats['elapsed_seconds']:.2f}s "
        f"({stats['fps_achieved']:.1f} FPS) -> {args.output}"
    )


if __name__ == "__main__":
    main()
