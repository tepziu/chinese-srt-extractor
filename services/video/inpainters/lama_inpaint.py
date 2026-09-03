"""
lama_inpaint.py — Deep Learning Inpainter using LaMa (Large Mask Inpainting) ONNX.
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

                # Prefer CUDA if working, otherwise CPU
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
                self.session = ort.InferenceSession(
                    str(self.model_path), sess_options=opts, providers=chosen_providers
                )
                print(f"LaMa inpainter loaded on {self.session.get_providers()[0]}")
            except Exception as exc:
                print(f"Warning: Failed to initialize LaMa ONNX ({exc}); falling back to OpenCV")
                self.session = None
        else:
            print(f"Notice: LaMa model not found at {self.model_path}; using OpenCV inpainter")

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask is None or mask.max() == 0:
            return image.copy()

        if self.session is None:
            return self._fallback.inpaint(image, mask)

        try:
            h, w = image.shape[:2]
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            # LaMa ONNX expects 512x512 RGB float32 [0, 1]
            img_resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
            mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

            rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_t = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
            mask_t = ((mask_resized > 10).astype(np.float32))[None, None, ...]

            out = self.session.run(None, {"image": img_t, "mask": mask_t})[0]
            out_rgb = (out[0].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

            return cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
        except Exception as exc:
            print(f"LaMa inference error ({exc}), falling back to OpenCV")
            return self._fallback.inpaint(image, mask)
