"""
Brightness Classifier
----------------------
Slide 5: "Brightness Classifier -> if slightly dark = only Zero-DCE++,
if very dark = Fusion of all 3." This is the routing decision at the top
of the image pipeline.

Two implementations are given:
  - HeuristicBrightnessClassifier: instant, no training needed, based on
    mean luminance + a low-percentile check (catches images that are
    "fine on average" but have very dark shadow regions). Good default.
  - LearnedBrightnessClassifier: a tiny CNN you can train if the heuristic
    misclassifies too often on your own footage.
"""
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class BrightnessLevel(str, Enum):
    NORMAL = "normal"          # skip enhancement entirely
    SLIGHTLY_DARK = "slightly_dark"
    VERY_DARK = "very_dark"


class HeuristicBrightnessClassifier:
    """
    Fast, dependency-free classifier based on mean luminance (0-1 scale)
    and the 10th-percentile luminance (to catch heavy shadow regions that
    the mean would hide).
    """

    def __init__(self, normal_thresh: float = 0.45, dark_thresh: float = 0.25):
        self.normal_thresh = normal_thresh
        self.dark_thresh = dark_thresh

    def classify(self, image: torch.Tensor) -> BrightnessLevel:
        """
        Args:
            image: (3, H, W) or (1, 3, H, W) tensor in [0, 1]
        """
        if image.dim() == 4:
            image = image[0]
        gray = (0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2])
        mean_lum = gray.mean().item()
        p10_lum = torch.quantile(gray.flatten(), 0.10).item()

        # Whichever signal is darker drives the decision -- a bright average
        # with very dark shadows still needs the stronger (fused) branch.
        effective = min(mean_lum, p10_lum + 0.15)

        if effective >= self.normal_thresh:
            return BrightnessLevel.NORMAL
        elif effective >= self.dark_thresh:
            return BrightnessLevel.SLIGHTLY_DARK
        return BrightnessLevel.VERY_DARK


class LearnedBrightnessClassifier(nn.Module):
    """Optional tiny CNN classifier (3-way) if you'd rather train on your
    own labeled dark/slightly-dark/normal footage than tune thresholds."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, 3)  # normal, slightly_dark, very_dark

    def forward(self, x: torch.Tensor):
        feat = self.features(x).flatten(1)
        return self.classifier(feat)

    @torch.no_grad()
    def classify(self, image: torch.Tensor) -> BrightnessLevel:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        logits = self.forward(image)
        idx = logits.argmax(dim=1).item()
        return [BrightnessLevel.NORMAL, BrightnessLevel.SLIGHTLY_DARK, BrightnessLevel.VERY_DARK][idx]


if __name__ == "__main__":
    clf = HeuristicBrightnessClassifier()
    dark_img = torch.rand(3, 64, 64) * 0.1
    bright_img = torch.rand(3, 64, 64) * 0.3 + 0.5
    mid_img = torch.rand(3, 64, 64) * 0.2 + 0.25
    print("very dark ->", clf.classify(dark_img))
    print("normal    ->", clf.classify(bright_img))
    print("slight    ->", clf.classify(mid_img))
