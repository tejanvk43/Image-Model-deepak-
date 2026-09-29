"""
Evaluate trained models on the held-out test split.

Runs each model on its own (Zero-DCE++, RetinexNet, Transformer) plus the
full hybrid pipeline, and reports mean PSNR / SSIM / seconds-per-image.
This is the honest evaluation -- the val numbers printed during training
come from a random training batch and are only a sanity check.

Usage:
    python evaluate.py --data_root data/test \
        --zero_dce_ckpt checkpoints/zero_dce.pt \
        --retinex_ckpt checkpoints/retinex.pt \
        --transformer_ckpt checkpoints/transformer.pt

    # only real-captured images (filenames contain "_real_"):
    python evaluate.py --data_root data/test ... --filter real
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from infer_image import load_checkpoint
from metrics import compute_psnr, compute_ssim
from models.retinex_net import RetinexNet
from models.transformer_enhancer import TransformerEnhancer
from models.zero_dce import ZeroDCEPlusPlus
from pipeline_image import ImageEnhancementPipeline, PipelineConfig


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Folder with low/ and high/ (e.g. data/test)")
    ap.add_argument("--zero_dce_ckpt", default=None)
    ap.add_argument("--retinex_ckpt", default=None)
    ap.add_argument("--transformer_ckpt", default=None)
    ap.add_argument("--filter", default=None, help="Only files whose name contains this (e.g. real, synth)")
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only the first N pairs")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = args.device
    zero_dce = load_checkpoint(ZeroDCEPlusPlus(), args.zero_dce_ckpt, dev).to(dev).eval()
    retinex = load_checkpoint(RetinexNet(), args.retinex_ckpt, dev).to(dev).eval()
    transformer = load_checkpoint(TransformerEnhancer(), args.transformer_ckpt, dev).to(dev).eval()
    pipeline = ImageEnhancementPipeline(
        zero_dce=zero_dce, retinex=retinex, transformer=transformer,
        config=PipelineConfig(device=dev),
    )

    methods = {
        "Zero-DCE++": lambda x: zero_dce(x)[0],
        "RetinexNet": lambda x: retinex(x)[0],
        "Transformer": lambda x: transformer(x),
        "Hybrid (proposed)": lambda x: pipeline.enhance(x)["output"],
    }

    root = Path(args.data_root)
    names = sorted(p.name for p in (root / "low").glob("*") if (root / "high" / p.name).exists())
    if args.filter:
        names = [n for n in names if args.filter in n]
    if args.limit:
        names = names[: args.limit]
    if not names:
        raise SystemExit(f"No matching low/high pairs under {root}")
    print(f"Evaluating {len(names)} pairs on {dev}\n")

    # warm-up so the first timed image isn't penalised by CUDA init
    with torch.no_grad():
        warm = TF.to_tensor(Image.open(root / "low" / names[0]).convert("RGB")).unsqueeze(0).to(dev)
        for fn in methods.values():
            fn(warm)
        sync(dev)

    rows, agg = [], {m: {"psnr": [], "ssim": [], "time": []} for m in methods}
    with torch.no_grad():
        for i, name in enumerate(names, 1):
            low = TF.to_tensor(Image.open(root / "low" / name).convert("RGB")).unsqueeze(0).to(dev)
            high = TF.to_tensor(Image.open(root / "high" / name).convert("RGB")).unsqueeze(0).to(dev)
            for m, fn in methods.items():
                sync(dev)
                t0 = time.time()
                out = fn(low).clamp(0, 1)
                sync(dev)
                dt = time.time() - t0
                p, s = compute_psnr(out, high), compute_ssim(out, high)
                agg[m]["psnr"].append(p); agg[m]["ssim"].append(s); agg[m]["time"].append(dt)
                rows.append([name, m, f"{p:.3f}", f"{s:.4f}", f"{dt:.4f}"])
            if i % 20 == 0 or i == len(names):
                print(f"  {i}/{len(names)} done", end="\r")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "method", "psnr_db", "ssim", "seconds"])
        w.writerows(rows)

    print("\n\n{:<20} {:>10} {:>8} {:>12}".format("Method", "PSNR (dB)", "SSIM", "Time (s/img)"))
    print("-" * 54)
    for m, d in agg.items():
        print("{:<20} {:>10.2f} {:>8.3f} {:>12.4f}".format(
            m, np.mean(d["psnr"]), np.mean(d["ssim"]), np.mean(d["time"])))
    print(f"\nPer-image results saved to {args.out}")


if __name__ == "__main__":
    main()