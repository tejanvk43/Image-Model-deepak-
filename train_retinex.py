"""
Train RetinexNet on paired LOL-v2 / VE-LOL data.

Usage:
    python train_retinex.py --data_root /path/to/dataset_root \
        --epochs 100 --batch_size 8 --out checkpoints/retinex.pt

dataset_root must contain low/ and high/ subfolders with matching filenames
(see datasets.py for the exact layout).
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import PairedLowLightDataset
from losses import RetinexLoss
from metrics import evaluate_batch
from models.retinex_net import RetinexNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--out", default="checkpoints/retinex.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = PairedLowLightDataset(args.data_root, patch_size=args.patch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = RetinexNet().to(args.device)
    loss_fn = RetinexLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_psnr, running_ssim, n_batches = 0.0, 0.0, 0.0, 0

        for low, high, _ in loader:
            low, high = low.to(args.device), high.to(args.device)
            enhanced, reflectance, illumination, enhanced_illum = model(low)
            loss, parts = loss_fn(enhanced, high, reflectance, enhanced_illum)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            low, high, _ = next(iter(loader))
            low, high = low.to(args.device), high.to(args.device)
            enhanced, *_ = model(low)
            m = evaluate_batch(enhanced, high)
            running_psnr, running_ssim = m["psnr"], m["ssim"]

        print(
            f"[RetinexNet] epoch {epoch}/{args.epochs} "
            f"loss={running_loss/n_batches:.4f} "
            f"val_psnr={running_psnr:.2f}dB val_ssim={running_ssim:.4f}"
        )

        if epoch % 10 == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), args.out)
            print(f"  saved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
