"""
Evaluation metrics: PSNR, SSIM, and Average Luminance Variance (ALV) --
the flicker metric quoted on slides 5 and 18 (ALV 117.22 -> 58.69).
"""
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def tensor_to_numpy(img: torch.Tensor) -> np.ndarray:
    """(3, H, W) tensor in [0,1] -> (H, W, 3) uint8 numpy array."""
    if img.dim() == 4:
        img = img[0]
    arr = img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255).astype(np.uint8)


def compute_psnr(enhanced: torch.Tensor, ground_truth: torch.Tensor) -> float:
    e = tensor_to_numpy(enhanced)
    g = tensor_to_numpy(ground_truth)
    return float(peak_signal_noise_ratio(g, e, data_range=255))


def compute_ssim(enhanced: torch.Tensor, ground_truth: torch.Tensor) -> float:
    e = tensor_to_numpy(enhanced)
    g = tensor_to_numpy(ground_truth)
    return float(structural_similarity(g, e, channel_axis=2, data_range=255))


def average_luminance_variance(frames: list) -> float:
    """
    ALV: frame-to-frame variance of mean luminance across a video clip.
    Lower = less flickering. `frames` is a list of (3, H, W) tensors in [0,1].
    """
    means = []
    for f in frames:
        gray = 0.299 * f[0] + 0.587 * f[1] + 0.114 * f[2]
        means.append(gray.mean().item() * 255)  # scale to 0-255 to match slide units
    means = np.array(means)
    diffs = np.diff(means)
    return float(np.var(diffs))


def evaluate_batch(enhanced_batch: torch.Tensor, gt_batch: torch.Tensor) -> dict:
    """Average PSNR/SSIM over a batch of (B, 3, H, W) tensors."""
    psnrs, ssims = [], []
    for i in range(enhanced_batch.shape[0]):
        psnrs.append(compute_psnr(enhanced_batch[i], gt_batch[i]))
        ssims.append(compute_ssim(enhanced_batch[i], gt_batch[i]))
    return {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims))}


if __name__ == "__main__":
    a = torch.rand(3, 64, 64)
    b = a + torch.randn(3, 64, 64) * 0.02
    b = b.clamp(0, 1)
    print("PSNR:", compute_psnr(b, a))
    print("SSIM:", compute_ssim(b, a))
    frames = [torch.rand(3, 32, 32) * 0.3 + 0.4 for _ in range(10)]
    print("ALV:", average_luminance_variance(frames))
