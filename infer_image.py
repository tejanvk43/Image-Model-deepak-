"""
Run the hybrid image pipeline on a single image or a folder of images.

Usage:
    python infer_image.py --input path/to/image_or_folder --output path/to/out_folder \
        --zero_dce_ckpt checkpoints/zero_dce.pt \
        --retinex_ckpt checkpoints/retinex.pt \
        --transformer_ckpt checkpoints/transformer.pt

Checkpoint args are optional -- omit them to run with randomly-initialized
weights (useful for smoke-testing the pipeline wiring before you've trained).
"""
import argparse
import time
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from models.retinex_net import RetinexNet
from models.transformer_enhancer import TransformerEnhancer
from models.zero_dce import ZeroDCEPlusPlus
from pipeline_image import ImageEnhancementPipeline, PipelineConfig


def load_checkpoint(model, path, device):
    if path and Path(path).exists():
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"  loaded {path}")
    else:
        print(f"  no checkpoint given/found for {model.__class__.__name__} -- using random init")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image file or folder of images")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--zero_dce_ckpt", default=None)
    parser.add_argument("--retinex_ckpt", default=None)
    parser.add_argument("--transformer_ckpt", default=None)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    zero_dce = load_checkpoint(ZeroDCEPlusPlus(), args.zero_dce_ckpt, device)
    retinex = load_checkpoint(RetinexNet(), args.retinex_ckpt, device)
    transformer = load_checkpoint(TransformerEnhancer(), args.transformer_ckpt, device)

    pipeline = ImageEnhancementPipeline(
        zero_dce=zero_dce, retinex=retinex, transformer=transformer,
        config=PipelineConfig(tile_size=args.tile_size, device=device),
    )

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [in_path] if in_path.is_file() else sorted(
        p for p in in_path.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not files:
        raise FileNotFoundError(f"No images found at {args.input}")

    total_start = time.time()
    for f in files:
        img = Image.open(f).convert("RGB")
        tensor = TF.to_tensor(img)
        result = pipeline.enhance(tensor)

        out_img = TF.to_pil_image(result["output"][0].cpu())
        out_path = out_dir / f.name
        out_img.save(out_path)
        print(f"{f.name}: {result['num_tiles']} tile(s), {result['elapsed_seconds']:.3f}s -> {out_path}")

    total = time.time() - total_start
    print(f"\nDone: {len(files)} image(s) in {total:.2f}s ({total/len(files):.3f}s/image avg)")


if __name__ == "__main__":
    main()
