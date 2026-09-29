"""
Loss functions for the three branches.

Zero-DCE++ trains with NO paired ground truth (that's the whole point —
"zero-reference"), so it uses four hand-designed losses instead:
  - Spatial Consistency Loss : preserve local contrast/edges between input & output
  - Exposure Control Loss    : push mean local brightness toward a target level
  - Color Constancy Loss     : keep R/G/B channel means close to each other
  - Illumination Smoothness  : TV-loss on the predicted curve map (no blockiness)

RetinexNet and the Transformer branch DO use paired LOL-v2/VE-LOL data, so
they train with standard reconstruction + structural losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Zero-DCE++ non-reference losses
# --------------------------------------------------------------------------
class SpatialConsistencyLoss(nn.Module):
    """Penalizes changes in local gradient direction/magnitude between the
    original and enhanced image (4-neighbour differences)."""

    def __init__(self, patch_size: int = 4):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        kernels = {
            "left": [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
            "right": [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
            "up": [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
            "down": [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
        }
        for name, k in kernels.items():
            kernel = torch.tensor(k, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.register_buffer(f"kernel_{name}", kernel)

    def _grad(self, x, name):
        kernel = getattr(self, f"kernel_{name}")
        return F.conv2d(x, kernel, padding=1)

    def forward(self, enhanced: torch.Tensor, original: torch.Tensor):
        orig_gray = self.pool(original.mean(dim=1, keepdim=True))
        enh_gray = self.pool(enhanced.mean(dim=1, keepdim=True))

        loss = 0.0
        for direction in ("left", "right", "up", "down"):
            d_orig = self._grad(orig_gray, direction)
            d_enh = self._grad(enh_gray, direction)
            loss = loss + (d_orig - d_enh) ** 2
        return loss.mean()


class ExposureControlLoss(nn.Module):
    """Pushes average local brightness toward a target exposure level E."""

    def __init__(self, patch_size: int = 16, target_exposure: float = 0.6):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.target = target_exposure

    def forward(self, enhanced: torch.Tensor):
        gray = enhanced.mean(dim=1, keepdim=True)
        local_mean = self.pool(gray)
        return ((local_mean - self.target) ** 2).mean()


class ColorConstancyLoss(nn.Module):
    """Keeps the average R, G, B roughly equal (grey-world assumption)."""

    def forward(self, enhanced: torch.Tensor):
        mean_rgb = enhanced.mean(dim=[2, 3])  # (B, 3)
        r, g, b = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        loss = (r - g) ** 2 + (g - b) ** 2 + (b - r) ** 2
        return loss.mean()


class IlluminationSmoothnessLoss(nn.Module):
    """Total-variation loss on the predicted curve parameter map so
    neighbouring pixels get similar enhancement (no blocky artifacts)."""

    def forward(self, curve_params: torch.Tensor):
        h_diff = curve_params[:, :, 1:, :] - curve_params[:, :, :-1, :]
        w_diff = curve_params[:, :, :, 1:] - curve_params[:, :, :, :-1]
        return (h_diff ** 2).mean() + (w_diff ** 2).mean()


class ZeroDCELoss(nn.Module):
    """Weighted combination of the four losses above (paper's default weights)."""

    def __init__(self, w_spa=1.0, w_exp=10.0, w_col=5.0, w_tv=200.0, target_exposure=0.6):
        super().__init__()
        self.spa = SpatialConsistencyLoss()
        self.exp = ExposureControlLoss(target_exposure=target_exposure)
        self.col = ColorConstancyLoss()
        self.tv = IlluminationSmoothnessLoss()
        self.w_spa, self.w_exp, self.w_col, self.w_tv = w_spa, w_exp, w_col, w_tv

    def forward(self, enhanced: torch.Tensor, original: torch.Tensor, curve_params: torch.Tensor):
        l_spa = self.spa(enhanced, original)
        l_exp = self.exp(enhanced)
        l_col = self.col(enhanced)
        l_tv = self.tv(curve_params)
        total = self.w_spa * l_spa + self.w_exp * l_exp + self.w_col * l_col + self.w_tv * l_tv
        return total, {"spa": l_spa.item(), "exp": l_exp.item(), "col": l_col.item(), "tv": l_tv.item()}


# --------------------------------------------------------------------------
# Paired-supervision losses (RetinexNet, Transformer)
# --------------------------------------------------------------------------
def gradient(x: torch.Tensor):
    """Simple horizontal/vertical gradient for structure-preservation loss."""
    gx = x[:, :, :, 1:] - x[:, :, :, :-1]
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return gx, gy


class RetinexLoss(nn.Module):
    """
    Reconstruction loss (L1) + illumination smoothness (weighted by
    reflectance gradient, so smoothing doesn't wash out real edges) +
    reflectance consistency between low/normal-light decompositions.
    """

    def __init__(self, w_recon=1.0, w_smooth=0.1):
        super().__init__()
        self.w_recon = w_recon
        self.w_smooth = w_smooth

    def forward(self, enhanced, target, reflectance, illumination):
        l_recon = F.l1_loss(enhanced, target)

        r_gray = reflectance.mean(dim=1, keepdim=True)
        gx_r, gy_r = gradient(r_gray)
        gx_l, gy_l = gradient(illumination)
        # illumination should be smooth EXCEPT where reflectance has edges
        l_smooth = (gx_l.abs() * torch.exp(-10 * gx_r.abs())).mean() + \
                   (gy_l.abs() * torch.exp(-10 * gy_r.abs())).mean()

        total = self.w_recon * l_recon + self.w_smooth * l_smooth
        return total, {"recon": l_recon.item(), "smooth": l_smooth.item()}


class TransformerLoss(nn.Module):
    """L1 pixel loss + gradient (structural) loss for the transformer branch."""

    def __init__(self, w_l1=1.0, w_grad=0.5):
        super().__init__()
        self.w_l1 = w_l1
        self.w_grad = w_grad

    def forward(self, enhanced, target):
        l1 = F.l1_loss(enhanced, target)
        gx_e, gy_e = gradient(enhanced)
        gx_t, gy_t = gradient(target)
        l_grad = F.l1_loss(gx_e, gx_t) + F.l1_loss(gy_e, gy_t)
        total = self.w_l1 * l1 + self.w_grad * l_grad
        return total, {"l1": l1.item(), "grad": l_grad.item()}
