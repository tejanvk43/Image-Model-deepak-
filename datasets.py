"""
Paired low-light / normal-light dataset loader.

Expects a directory layout matching LOL-v2 / VE-LOL after extraction:

    dataset_root/
        low/
            00001.png
            00002.png
            ...
        high/
            00001.png     <- same filename as its low-light counterpart
            00002.png
            ...

Slide 7-8 numbers this project targets:
    LOL-v2   : 689 real train + 100 real test + 900 synth train + 221 synth test
    VE-LOL   : 1,500 real pairs + 1,000 synthetic pairs (+ 100+ video clips)
    Combined training split used in the deck: 2,689 pairs (80%)
    Combined testing split used in the deck : 600 pairs (20%)
"""
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


class PairedLowLightDataset(Dataset):
    def __init__(self, root: str, patch_size: int = 256, augment: bool = True):
        self.root = Path(root)
        self.low_dir = self.root / "low"
        self.high_dir = self.root / "high"
        self.patch_size = patch_size
        self.augment = augment

        low_files = sorted(p.name for p in self.low_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        high_files = set(p.name for p in self.high_dir.glob("*"))
        self.pairs = [f for f in low_files if f in high_files]
        if not self.pairs:
            raise FileNotFoundError(
                f"No matching low/high pairs found under {self.root}. "
                f"Expected {self.root}/low/*.png and {self.root}/high/*.png with matching filenames."
            )

    def __len__(self):
        return len(self.pairs)

    def _random_crop(self, low: Image.Image, high: Image.Image):
        w, h = low.size
        ps = self.patch_size
        if w < ps or h < ps:
            low = low.resize((max(w, ps), max(h, ps)))
            high = high.resize((max(w, ps), max(h, ps)))
            w, h = low.size
        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)
        box = (x, y, x + ps, y + ps)
        return low.crop(box), high.crop(box)

    def __getitem__(self, idx):
        name = self.pairs[idx]
        low = Image.open(self.low_dir / name).convert("RGB")
        high = Image.open(self.high_dir / name).convert("RGB")

        if self.patch_size:
            low, high = self._random_crop(low, high)

        if self.augment and random.random() < 0.5:
            low = TF.hflip(low)
            high = TF.hflip(high)
        if self.augment and random.random() < 0.5:
            low = TF.vflip(low)
            high = TF.vflip(high)

        low_t = TF.to_tensor(low)
        high_t = TF.to_tensor(high)
        return low_t, high_t, name


class UnpairedLowLightDataset(Dataset):
    """For Zero-DCE++ training, which needs no ground truth -- just a
    folder of low-light images (e.g. the low/ split, or raw ExDark)."""

    def __init__(self, root: str, patch_size: int = 256):
        self.files = sorted(
            p for p in Path(root).glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        if not self.files:
            raise FileNotFoundError(f"No images found under {root}")
        self.patch_size = patch_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.patch_size:
            w, h = img.size
            ps = self.patch_size
            if w < ps or h < ps:
                img = img.resize((max(w, ps), max(h, ps)))
                w, h = img.size
            x = random.randint(0, w - ps)
            y = random.randint(0, h - ps)
            img = img.crop((x, y, x + ps, y + ps))
        return TF.to_tensor(img), self.files[idx].name
