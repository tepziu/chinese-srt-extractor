"""
opencv_inpaint.py — High-speed inpainter with grain-matching texture synthesis.
Preserves natural background grain (leather, asphalt, fabric, walls) without smooth smudging.
"""

from __future__ import annotations

import cv2
import numpy as np

from services.video.inpainters.base import BaseInpainter


class OpenCVInpainter(BaseInpainter):
    """Fast inpainter using cv2.inpaint with natural grain texture matching."""

    def __init__(self, method: str = "telea", inpaint_radius: int = 7):
        self.method = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
        self.inpaint_radius = inpaint_radius

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask is None or mask.max() == 0:
            return image.copy()

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        binary_mask = (mask > 10).astype(np.uint8) * 255

        # 1. Base inpainting
        inpainted = cv2.inpaint(image, binary_mask, self.inpaint_radius, self.method)

        # 2. Extract natural background grain from unmasked pixels
        unmasked = binary_mask == 0
        if np.sum(unmasked) > 200:
            clean_blur = cv2.GaussianBlur(image, (5, 5), 0)
            grain_diff = image.astype(np.float32) - clean_blur.astype(np.float32)
            grain_std = np.std(grain_diff[unmasked], axis=0)

            # If background has natural texture/grain (std > 1.2)
            if np.mean(grain_std) > 1.2:
                h, w = image.shape[:2]
                noise = np.random.normal(0, grain_std * 0.85, (h, w, 3))
                alpha = cv2.GaussianBlur((binary_mask > 0).astype(np.float32), (5, 5), 0)[:, :, None]
                blended = inpainted.astype(np.float32) + noise * alpha
                return np.clip(blended, 0, 255).astype(np.uint8)

        return inpainted