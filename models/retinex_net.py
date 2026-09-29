"""
RetinexNet
----------
Based on Retinex theory: Image = Reflectance * Illumination.
  - DecomNet splits the low-light input into a light-independent
    Reflectance map (R) and a structure-aware Illumination map (L).
  - EnhanceNet brightens the illumination map only, so color/texture
    (carried by R) is left untouched -> best color-preservation branch
    of the hybrid (slide 9).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecomNet(nn.Module):
    """Splits input into reflectance (3ch) and illumination (1ch)."""

    def __init__(self, channels: int = 64, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        # First layer sees the raw image concatenated with its max channel
        # (a rough illumination prior), as in the original RetinexNet paper.
        big_k = kernel_size * 3
        self.net1_conv0 = nn.Conv2d(4, channels, big_k, padding=big_k // 2)

        self.net1_convs = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad), nn.ReLU(inplace=True),
        )
        self.net1_recon = nn.Conv2d(channels, 4, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor):
        illum_prior = torch.max(x, dim=1, keepdim=True)[0]
        inp = torch.cat([illum_prior, x], dim=1)
        feats = self.net1_conv0(inp)
        feats = self.net1_convs(feats)
        out = self.net1_recon(feats)
        reflectance = torch.sigmoid(out[:, 0:3, :, :])
        illumination = torch.sigmoid(out[:, 3:4, :, :])
        return reflectance, illumination


class EnhanceNet(nn.Module):
    """Brightens the illumination map, conditioned on reflectance features."""

    def __init__(self, channels: int = 64, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv0 = nn.Conv2d(4, channels, kernel_size, padding=pad)

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride=2, padding=pad)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride=2, padding=pad)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size, stride=2, padding=pad)

        self.deconv1 = nn.Conv2d(channels * 2, channels, kernel_size, padding=pad)
        self.deconv2 = nn.Conv2d(channels * 2, channels, kernel_size, padding=pad)
        self.deconv3 = nn.Conv2d(channels * 2, channels, kernel_size, padding=pad)

        self.fusion = nn.Conv2d(channels * 3, channels, 1)
        self.output = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, reflectance: torch.Tensor, illumination: torch.Tensor):
        x = torch.cat([reflectance, illumination], dim=1)
        f0 = F.relu(self.conv0(x))
        f1 = F.relu(self.conv1(f0))
        f2 = F.relu(self.conv2(f1))
        f3 = F.relu(self.conv3(f2))

        f3_up = F.interpolate(f3, size=f2.shape[2:], mode="nearest")
        d1 = F.relu(self.deconv1(torch.cat([f3_up, f2], dim=1)))
        d1_up = F.interpolate(d1, size=f1.shape[2:], mode="nearest")
        d2 = F.relu(self.deconv2(torch.cat([d1_up, f1], dim=1)))
        d2_up = F.interpolate(d2, size=f0.shape[2:], mode="nearest")
        d3 = F.relu(self.deconv3(torch.cat([d2_up, f0], dim=1)))

        d1_resized = F.interpolate(d1, size=f0.shape[2:], mode="nearest")
        d2_resized = F.interpolate(d2, size=f0.shape[2:], mode="nearest")
        fused = self.fusion(torch.cat([d1_resized, d2_resized, d3], dim=1))
        enhanced_illum = torch.sigmoid(self.output(fused))
        return enhanced_illum


class RetinexNet(nn.Module):
    """End-to-end: decompose -> enhance illumination -> recombine with R."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.decom = DecomNet(channels)
        self.enhance = EnhanceNet(channels)

    def forward(self, x: torch.Tensor):
        reflectance, illumination = self.decom(x)
        enhanced_illum = self.enhance(reflectance, illumination)
        enhanced = reflectance * enhanced_illum.expand_as(reflectance)
        return torch.clamp(enhanced, 0.0, 1.0), reflectance, illumination, enhanced_illum


if __name__ == "__main__":
    model = RetinexNet()
    dummy = torch.rand(1, 3, 400, 600)
    enhanced, r, l, l_enh = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"RetinexNet params: {n_params:,} | output shape: {tuple(enhanced.shape)}")
