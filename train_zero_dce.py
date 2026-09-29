"""
Train Zero-DCE++ on a folder of low-light images (no ground truth needed).

Usage:
    python train_zero_dce.py --data_root /path/to/low-light/images \
        --epochs 50 --batch_size 8 --out checkpoints/zero_dce.pt
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import UnpairedLowLightDataset
from losses import ZeroDCELoss
from models.zero_dce import ZeroDCEPlusPlus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Folder of low-light images")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--target_exposure", type=float, default=0.6)
    parser.add_argument("--out", default="checkpoints/zero_dce.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = UnpairedLowLightDataset(args.data_root, patch_size=args.patch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = ZeroDCEPlusPlus().to(args.device)
    loss_fn = ZeroDCELoss(target_exposure=args.target_exposure).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"total": 0.0, "spa": 0.0, "exp": 0.0, "col": 0.0, "tv": 0.0}
        for imgs, _ in loader:
            imgs = imgs.to(args.device)
            enhanced, curve_params = model(imgs)
            loss, parts = loss_fn(enhanced, imgs, curve_params)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            running["total"] += loss.item()
            for k, v in parts.items():
                running[k] += v

        n = len(loader)
        print(
            f"[Zero-DCE++] epoch {epoch}/{args.epochs} "
            f"loss={running['total']/n:.4f} "
            f"(spa={running['spa']/n:.4f} exp={running['exp']/n:.4f} "
            f"col={running['col']/n:.4f} tv={running['tv']/n:.4f})"
        )

        if epoch % 5 == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), args.out)
            print(f"  saved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()