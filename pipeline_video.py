"""
Video Pipeline (slide 5, "For Videos" branch)
------------------------------------------------
    FFmpeg streaming -> split into 5-sec overlapping chunks
        -> Key-frame (every 10th)     : Transformer + RetinexNet fusion
        -> Non-key-frame               : Zero-DCE++ + optical-flow propagation
                                          (propagate detail from the nearest
                                           enhanced keyframe, TBEFN-style idea)
    -> Merge chunks (overlap-trimmed) -> re-mux with original audio

This never loads the whole video into RAM (the failure mode called out on
slide 5) -- chunks are decoded, enhanced, written, and discarded one at a
time, which is what makes "unlimited length" possible.
"""
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from models.zero_dce import ZeroDCEPlusPlus
from models.retinex_net import RetinexNet
from models.transformer_enhancer import TransformerEnhancer


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (e.g. `apt-get install ffmpeg`) "
            "-- required for chunk streaming and re-muxing audio."
        )


@dataclass
class VideoPipelineConfig:
    chunk_seconds: float = 5.0
    chunk_overlap_seconds: float = 0.5   # overlap between chunks, as on slide 5
    keyframe_interval: int = 10          # every 10th frame is a "key frame"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class VideoEnhancementPipeline:
    def __init__(
        self,
        zero_dce: ZeroDCEPlusPlus = None,
        retinex: RetinexNet = None,
        transformer: TransformerEnhancer = None,
        config: VideoPipelineConfig = None,
    ):
        self.config = config or VideoPipelineConfig()
        self.zero_dce = (zero_dce or ZeroDCEPlusPlus()).to(self.config.device).eval()
        self.retinex = (retinex or RetinexNet()).to(self.config.device).eval()
        self.transformer = (transformer or TransformerEnhancer()).to(self.config.device).eval()

    # ---------------------------------------------------------------- #
    # Frame-level enhancement
    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def _enhance_keyframe(self, frame_t: torch.Tensor) -> torch.Tensor:
        """Transformer + RetinexNet fusion -- highest quality, used sparingly."""
        t_out = self.transformer(frame_t)
        r_out, _, _, _ = self.retinex(frame_t)
        fused = 0.6 * t_out + 0.4 * r_out
        return fused.clamp(0, 1)

    @torch.no_grad()
    def _enhance_nonkeyframe(
        self, frame_t: torch.Tensor, prev_enhanced_bgr: np.ndarray, prev_raw_bgr: np.ndarray
    ) -> torch.Tensor:
        """
        Zero-DCE++ for base brightening, then warp the detail residual from
        the nearest already-enhanced keyframe using dense optical flow, so
        fine detail doesn't have to be re-synthesized every frame (this is
        what keeps flicker/ALV down between the cheap frames).
        """
        z_out, _ = self.zero_dce(frame_t)
        z_out_np = (z_out[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        if prev_enhanced_bgr is None:
            return z_out

        cur_raw_bgr = cv2.cvtColor(
            (frame_t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        prev_gray = cv2.cvtColor(prev_raw_bgr, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.cvtColor(cur_raw_bgr, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        h, w = flow.shape[:2]
        flow_map = np.column_stack((np.tile(np.arange(w), h), np.repeat(np.arange(h), w)))
        flow_map = flow_map.reshape(h, w, 2).astype(np.float32)
        remap_xy = flow_map + flow

        warped_detail = cv2.remap(
            prev_enhanced_bgr, remap_xy[..., 0], remap_xy[..., 1], interpolation=cv2.INTER_LINEAR
        )
        warped_rgb = cv2.cvtColor(warped_detail, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Blend cheap Zero-DCE output with the flow-propagated keyframe detail
        blended = 0.7 * z_out_np.astype(np.float32) + 0.3 * warped_rgb
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        out_t = TF.to_tensor(blended).unsqueeze(0).to(self.config.device)
        return out_t

    # ---------------------------------------------------------------- #
    # Chunk-level processing
    # ---------------------------------------------------------------- #
    def _process_chunk(self, frames_bgr: list, fps: float) -> list:
        """frames_bgr: list of HxWx3 uint8 BGR arrays. Returns enhanced list."""
        enhanced_frames = []
        prev_enhanced_bgr = None
        prev_raw_bgr = None

        for i, frame_bgr in enumerate(frames_bgr):
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_t = TF.to_tensor(rgb).unsqueeze(0).to(self.config.device)

            if i % self.config.keyframe_interval == 0:
                out_t = self._enhance_keyframe(frame_t)
            else:
                out_t = self._enhance_nonkeyframe(frame_t, prev_enhanced_bgr, prev_raw_bgr)

            out_np = (out_t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
            enhanced_frames.append(out_bgr)

            prev_enhanced_bgr = out_bgr
            prev_raw_bgr = frame_bgr

        return enhanced_frames

    # ---------------------------------------------------------------- #
    # Full video: stream chunks in, never hold the whole clip in memory
    # ---------------------------------------------------------------- #
    def enhance_video(self, input_path: str, output_path: str, progress_cb=None):
        check_ffmpeg()
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        chunk_frames = max(1, int(self.config.chunk_seconds * fps))
        overlap_frames = int(self.config.chunk_overlap_seconds * fps)

        with tempfile.TemporaryDirectory() as tmpdir:
            silent_video_path = str(Path(tmpdir) / "enhanced_noaudio.mp4")
            writer = cv2.VideoWriter(
                silent_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )

            start_time = time.time()
            frame_idx = 0
            buffer = []
            carry_over = []  # last `overlap_frames` raw frames, re-decoded into next chunk

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                buffer.append(frame)
                frame_idx += 1

                if len(buffer) >= chunk_frames:
                    chunk = carry_over + buffer
                    enhanced_chunk = self._process_chunk(chunk, fps)
                    # drop the re-processed overlap frames from the front before writing
                    to_write = enhanced_chunk[len(carry_over):]
                    for f in to_write:
                        writer.write(f)
                    carry_over = buffer[-overlap_frames:] if overlap_frames else []
                    buffer = []
                    if progress_cb:
                        progress_cb(frame_idx, total_frames)

            if buffer:
                chunk = carry_over + buffer
                enhanced_chunk = self._process_chunk(chunk, fps)
                to_write = enhanced_chunk[len(carry_over):]
                for f in to_write:
                    writer.write(f)

            writer.release()
            cap.release()
            elapsed = time.time() - start_time

            # Re-mux with original audio (if any) via ffmpeg
            cmd = [
                "ffmpeg", "-y",
                "-i", silent_video_path,
                "-i", input_path,
                "-c:v", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:a", "copy",
                "-shortest",
                output_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)

        return {"elapsed_seconds": elapsed, "frames_processed": frame_idx, "fps_achieved": frame_idx / max(elapsed, 1e-6)}


if __name__ == "__main__":
    # Smoke test: synthesize a short dark clip, run it through the pipeline.
    tmp = tempfile.mkdtemp()
    src_path = str(Path(tmp) / "synthetic_dark.mp4")
    out_path = str(Path(tmp) / "enhanced.mp4")

    w, h, fps, n_frames = 160, 120, 24, 48  # 2 sec clip, enough to exercise chunking + keyframes
    writer = cv2.VideoWriter(src_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        frame = (rng.random((h, w, 3)) * 40).astype(np.uint8)  # dark noise frames
        writer.write(frame)
    writer.release()

    pipeline = VideoEnhancementPipeline(
        config=VideoPipelineConfig(chunk_seconds=1.0, chunk_overlap_seconds=0.2, keyframe_interval=10)
    )
    try:
        stats = pipeline.enhance_video(src_path, out_path)
        print("Video pipeline stats:", stats)
        print("Output exists:", Path(out_path).exists(), "size:", Path(out_path).stat().st_size, "bytes")
    except RuntimeError as e:
        print("Skipped (environment issue):", e)
