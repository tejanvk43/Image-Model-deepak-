"""
Image Pipeline (slide 5, "For Images" branch)
------------------------------------------------
    Brightness Classifier
        -> slightly dark : Zero-DCE++ only            (fast path, 0.003s)
        -> very dark     : Fusion of all 3 models      (quality path)
    + Tiling for large/4K images so the Transformer branch's memory use
      stays bounded, processed as parallel tiles and stitched back with
      overlap blending to avoid seam artifacts.
"""
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.zero_dce import ZeroDCEPlusPlus
from models.retinex_net import RetinexNet
from models.transformer_enhancer import TransformerEnhancer
from brightness_classifier import HeuristicBrightnessClassifier, BrightnessLevel


@dataclass
class PipelineConfig:
    tile_size: int = 512          # tile edge length for large/4K images
    tile_overlap: int = 32        # overlap so seams blend smoothly
    fusion_weights: tuple = (0.5, 0.3, 0.2)  # (transformer, retinex, zero_dce) for "very dark"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ImageEnhancementPipeline:
    def __init__(
        self,
        zero_dce: ZeroDCEPlusPlus = None,
        retinex: RetinexNet = None,
        transformer: TransformerEnhancer = None,
        config: PipelineConfig = None,
    ):
        self.config = config or PipelineConfig()
        self.zero_dce = (zero_dce or ZeroDCEPlusPlus()).to(self.config.device).eval()
        self.retinex = (retinex or RetinexNet()).to(self.config.device).eval()
        self.transformer = (transformer or TransformerEnhancer()).to(self.config.device).eval()
        self.classifier = HeuristicBrightnessClassifier()

    # ---------------------------------------------------------------- #
    # Tiling: split -> enhance each tile -> blend back with overlap
    # ---------------------------------------------------------------- #
    def _make_tiles(self, image: torch.Tensor):
        """image: (1, 3, H, W). Returns list of (tile, (y0, y1, x0, x1))."""
        _, _, H, W = image.shape
        ts, ov = self.config.tile_size, self.config.tile_overlap
        if H <= ts and W <= ts:
            return [(image, (0, H, 0, W))]

        tiles = []
        y = 0
        while y < H:
            y1 = min(y + ts, H)
            x = 0
            while x < W:
                x1 = min(x + ts, W)
                tiles.append((image[:, :, y:y1, x:x1], (y, y1, x, x1)))
                if x1 == W:
                    break
                x = x1 - ov
            if y1 == H:
                break
            y = y1 - ov
        return tiles

    def _blend_tiles(self, enhanced_tiles, positions, out_shape):
        """Feather-blend overlapping tiles back into a full-size canvas."""
        _, C, H, W = out_shape
        canvas = torch.zeros(1, C, H, W, device=self.config.device)
        weight = torch.zeros(1, 1, H, W, device=self.config.device)

        for tile, (y0, y1, x0, x1) in zip(enhanced_tiles, positions):
            th, tw = y1 - y0, x1 - x0
            # simple linear feather at the edges of each tile
            wy = torch.ones(th, device=self.config.device)
            wx = torch.ones(tw, device=self.config.device)
            ov = self.config.tile_overlap
            ramp = torch.linspace(0, 1, ov, device=self.config.device) if ov > 0 else torch.tensor([])
            if ov > 0 and th > ov:
                wy[:ov] = ramp
                wy[-ov:] = ramp.flip(0)
            if ov > 0 and tw > ov:
                wx[:ov] = ramp
                wx[-ov:] = ramp.flip(0)
            w2d = (wy.unsqueeze(1) * wx.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

            canvas[:, :, y0:y1, x0:x1] += tile * w2d
            weight[:, :, y0:y1, x0:x1] += w2d

        weight = weight.clamp(min=1e-6)
        return (canvas / weight).clamp(0, 1)

    # ---------------------------------------------------------------- #
    # Per-tile enhancement: routes by brightness, fuses when very dark
    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def _enhance_tile(self, tile: torch.Tensor) -> torch.Tensor:
        level = self.classifier.classify(tile)

        if level == BrightnessLevel.NORMAL:
            return tile

        if level == BrightnessLevel.SLIGHTLY_DARK:
            enhanced, _ = self.zero_dce(tile)
            return enhanced

        # VERY_DARK -> fusion of all three
        w_t, w_r, w_z = self.config.fusion_weights
        t_out = self.transformer(tile)
        r_out, _, _, _ = self.retinex(tile)
        z_out, _ = self.zero_dce(tile)
        fused = w_t * t_out + w_r * r_out + w_z * z_out
        return fused.clamp(0, 1)

    @torch.no_grad()
    def enhance(self, image: torch.Tensor) -> dict:
        """
        Args:
            image: (1, 3, H, W) or (3, H, W) tensor in [0, 1]
        Returns:
            dict with 'output' tensor and 'elapsed_seconds'
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.config.device)

        start = time.time()
        tiles_with_pos = self._make_tiles(image)
        enhanced_tiles = [self._enhance_tile(t) for t, _ in tiles_with_pos]
        positions = [p for _, p in tiles_with_pos]
        output = self._blend_tiles(enhanced_tiles, positions, image.shape)
        elapsed = time.time() - start

        return {"output": output, "elapsed_seconds": elapsed, "num_tiles": len(tiles_with_pos)}


if __name__ == "__main__":
    pipeline = ImageEnhancementPipeline()

    # simulate a "very dark" 4K-ish frame to exercise both tiling and fusion
    dark_4k = torch.rand(1, 3, 1080, 1920) * 0.15
    result = pipeline.enhance(dark_4k)
    print(f"4K very-dark image -> {result['num_tiles']} tiles, "
          f"{result['elapsed_seconds']:.3f}s, output shape {tuple(result['output'].shape)}")

    # simulate a "slightly dark" small image -> should take the fast Zero-DCE-only path
    slightly_dark = torch.rand(1, 3, 400, 600) * 0.25 + 0.2
    result2 = pipeline.enhance(slightly_dark)
    print(f"Slightly-dark image -> {result2['num_tiles']} tile(s), "
          f"{result2['elapsed_seconds']:.3f}s")
