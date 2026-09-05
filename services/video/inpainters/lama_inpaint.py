"""
lama_inpaint.py — Deep Learning Inpainter using LaMa (Large Mask Inpainting) ONNX.
Supports aspect-ratio preserving windowed inference and calibrated output range scaling.
"""

from __future__ import annotations

import os
from pathlib import Path
import cv2
import numpy as np

from services.video.inpainters.base import BaseInpainter
from services.video.inpainters.opencv_inpaint import OpenCVInpainter

MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "lama_fp32.onnx"


class LamaInpainter(BaseInpainter):
    """LaMa (Large Mask Inpainting) ONNX Runtime inpainter with OpenCV fallback."""

    def __init__(self, model_path: str | Path | None = None, providers: list[str] | None = None):
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.session = None
        self._fallback = OpenCVInpainter(method="telea")

        if self.model_path.exists():
            try:
                import onnxruntime as ort

                available = ort.get_available_providers()
                chosen_providers = []
                if providers:
                    chosen_providers = [p for p in providers if p in available]
                if not chosen_providers:
                    if "CUDAExecutionProvider" in available:
                        chosen_providers.append("CUDAExecutionProvider")
                    chosen_providers.append("CPUExecutionProvider")

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = min(os.cpu_count() or 4, 8)
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                self.session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=opts,
                    providers=chosen_providers,
                )
                active = self.session.get_providers()
                print(f"LaMa inpainter loaded on {active[0] if active else 'unknown'}")
            except Exception as exc:
                print(f"LaMa ONNX init failed ({exc}), falling back to OpenCV")
                self.session = None

    def _inpaint_single_patch(self, patch_img: np.ndarray, patch_mask: np.ndarray) -> np.ndarray:
        """Run single 512x512 LaMa inference on patch."""
        h, w = patch_img.shape[:2]
        img_resized = cv2.resize(patch_img, (512, 512), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(patch_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_t = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        mask_t = ((mask_resized > 10).astype(np.float32))[None, None, ...]

        out = self.session.run(None, {"image": img_t, "mask": mask_t})[0]

        # Handle output scale: LaMa outputs [0, 255] float32
        if out.max() > 1.5:
            out_rgb = out[0].transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
        else:
            out_rgb = (out[0].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)

        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        return cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_CUBIC)

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask is None or mask.max() == 0:
            return image.copy()

        if self.session is None:
            return self._fallback.inpaint(image, mask)

        try:
            h, w = image.shape[:2]
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            # If image is wide (e.g. subtitle strip), use windowed 512x512 patches to prevent distortion
            if w > 1.8 * h and w > 400:
                result = image.copy()
                step = int(h * 1.5)
                patch_w = min(w, max(h, 450))

                x = 0
                while x < w:
                    x_end = min(w, x + patch_w)
                    x_start = max(0, x_end - patch_w)

                    p_img = result[:, x_start:x_end]
                    p_mask = mask[:, x_start:x_end]

                    if np.sum(p_mask > 0) > 0:
                        p_inpainted = self._inpaint_single_patch(p_img, p_mask)
                        alpha = cv2.GaussianBlur((p_mask > 0).astype(np.float32), (7, 7), 0)[:, :, None]
                        blended = (p_inpainted.astype(np.float32) * alpha + p_img.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
                        result[:, x_start:x_end] = blended

                    if x_end >= w:
                        break
                    x += step

                return result
            else:
                return self._inpaint_single_patch(image, mask)

        except Exception as exc:
            print(f"LaMa inference error ({exc}), falling back to OpenCV")
            return self._fallback.inpaint(image, mask)