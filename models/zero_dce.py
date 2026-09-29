"""
Zero-DCE / Zero-DCE++ (Zero-Reference Deep Curve Estimation)
--------------------------------------------------------------
Fastest branch of the hybrid pipeline. No paired ground truth is needed —
the network learns to predict a per-pixel light-enhancement curve and
applies it iteratively to brighten the image. Zero-DCE++ swaps the plain
convolutions for depthwise-separable convolutions, which is what gets it
down to ~0.003s/image and ~85 GFLOPs as quoted on slide 9 of the deck.
"""
import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise + pointwise conv — the speed trick behind Zero-DCE++."""

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=True
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ZeroDCEPlusPlus(nn.Module):
    """
    7-layer symmetric-skip curve estimation network.

    Args:
        num_iterations: number of times the learned curve is applied
            (paper default: 8). More iterations = more aggressive brightening.
        channels: hidden channel width (paper default: 32).
    """

    def __init__(self, num_iterations: int = 8, channels: int = 32):
        super().__init__()
        self.num_iterations = num_iterations
        n = channels
        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = DepthwiseSeparableConv(3, n)
        self.e_conv2 = DepthwiseSeparableConv(n, n)
        self.e_conv3 = DepthwiseSeparableConv(n, n)
        self.e_conv4 = DepthwiseSeparableConv(n, n)
        self.e_conv5 = DepthwiseSeparableConv(n * 2, n)
        self.e_conv6 = DepthwiseSeparableConv(n * 2, n)
        self.e_conv7 = DepthwiseSeparableConv(n * 2, 3)  # curve params (shared across iterations)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) low-light image in [0, 1]
        Returns:
            enhanced: (B, 3, H, W) brightened image in [0, 1]
            curve_params: (B, 3, H, W) the learned per-pixel curve (used by
                the illumination-smoothness loss during training)
        """
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))
        curve_params = torch.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))

        enhanced = x
        for _ in range(self.num_iterations):
            enhanced = enhanced + curve_params * (torch.pow(enhanced, 2) - enhanced)
        enhanced = torch.clamp(enhanced, 0.0, 1.0)
        return enhanced, curve_params


if __name__ == "__main__":
    model = ZeroDCEPlusPlus()
    dummy = torch.rand(1, 3, 400, 600)
    out, curves = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Zero-DCE++ params: {n_params:,} | output shape: {tuple(out.shape)}")
