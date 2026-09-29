"""
Train the Transformer enhancer on paired LOL-v2 / VE-LOL data.

Usage:
    python train_transformer.py --data_root /path/to/dataset_root \
        --epochs 200 --batch_size 4 --out checkpoints/transformer.pt

Note: batch_size is kept small by default since window-attention memory
scales with H*W; increase patch_size/batch_size only once you've confirmed
your GPU can hold it (see the memory note in models/transformer_enhancer.py).
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import PairedLowLightDataset
from losses import TransformerLoss
from metrics import evaluate_batch
from models.transformer_enhancer import TransformerEnhancer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--out", default="checkpoints/transformer.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = PairedLowLightDataset(args.data_root, patch_size=args.patch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = TransformerEnhancer().to(args.device)
    loss_fn = TransformerLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, n_batches = 0.0, 0

        for low, high, _ in loader:
            low, high = low.to(args.device), high.to(args.device)
            enhanced = model(low)
            loss, parts = loss_fn(enhanced, high)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            low, high, _ = next(iter(loader))
            low, high = low.to(args.device), high.to(args.device)
            enhanced = model(low)
            m = evaluate_batch(enhanced, high)

        print(
            f"[Transformer] epoch {epoch}/{args.epochs} "
            f"loss={running_loss/n_batches:.4f} "
            f"val_psnr={m['psnr']:.2f}dB val_ssim={m['ssim']:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if epoch % 20 == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), args.out)
            print(f"  saved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
