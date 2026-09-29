"""
Transformer-based Enhancer (Swin-style windowed self-attention)
-----------------------------------------------------------------
Self-attention over local windows lets each patch draw context from the
whole image, which is what gives this branch the highest PSNR/SSIM of the
three (slide 9) at the cost of being the slowest -> used only on keyframes
in the video pipeline (every 10th frame) and in the "very dark" fusion
branch for images.

This is a compact SwinIR-style block, not the full Swin-Transformer paper,
sized to be practical to train from LOL-v2 without a multi-GPU cluster.
"""
import torch
import torch.nn as nn


def window_partition(x: torch.Tensor, window_size: int):
    """(B, H, W, C) -> (num_windows*B, window_size, window_size, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
    """Inverse of window_partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (num_windows*B, N, C) where N = window_size**2
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


class SwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x, H, W):
        # x: (B, H*W, C)
        B, L, C = x.shape
        ws = self.window_size
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        windows = window_partition(x, ws).view(-1, ws * ws, C)
        attn_windows = self.attn(windows)
        attn_windows = attn_windows.view(-1, ws, ws, C)
        x = window_reverse(attn_windows, ws, H, W).view(B, H * W, C)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEnhancer(nn.Module):
    """
    Shallow feature extraction -> stack of Swin blocks -> reconstruction,
    with a learned residual added back to the input (image = input + delta),
    which stabilizes training on small paired datasets like LOL-v2.
    """

    def __init__(self, dim: int = 60, depth: int = 4, num_heads: int = 4, window_size: int = 8):
        super().__init__()
        self.window_size = window_size
        self.patch_embed = nn.Conv2d(3, dim, 3, padding=1)
        self.blocks = nn.ModuleList(
            [SwinBlock(dim, num_heads, window_size) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.reconstruct = nn.Conv2d(dim, 3, 3, padding=1)

    def _pad_to_window(self, x):
        _, _, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, pad_h, pad_w

    def forward(self, x: torch.Tensor):
        orig_H, orig_W = x.shape[2], x.shape[3]
        x_padded, pad_h, pad_w = self._pad_to_window(x)

        feat = self.patch_embed(x_padded)
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)

        for block in self.blocks:
            tokens = block(tokens, H, W)
        tokens = self.norm(tokens)

        feat = tokens.transpose(1, 2).view(B, C, H, W)
        delta = self.reconstruct(feat)
        out = x_padded + delta

        if pad_h or pad_w:
            out = out[:, :, :orig_H, :orig_W]
        return torch.clamp(out, 0.0, 1.0)


if __name__ == "__main__":
    model = TransformerEnhancer()
    dummy = torch.rand(1, 3, 400, 600)  # not a multiple of window_size -> tests padding
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TransformerEnhancer params: {n_params:,} | output shape: {tuple(out.shape)}")
