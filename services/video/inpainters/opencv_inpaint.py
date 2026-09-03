"""
opencv_inpaint.py — High-speed CPU inpainter using OpenCV (Telea / Navier-Stokes).
Zero-VRAM footprint, ~80+ FPS on 1080p cropped strips.
"""

from __future__ import annotations

import cv2
import numpy as np

from services.video.inpainters.base import BaseInpainter


class OpenCVInpainter(BaseInpainter):
    """CPU-based fast inpainter using cv2.inpaint."""

    def __init__(self, method: str = "telea", inpaint_radius: int = 3):
        self.method = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
        self.inpaint_radius = inpaint_radius

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask is None or mask.max() == 0:
            return image.copy()

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Ensure mask is strict binary uint8 (0 or 255)
        binary_mask = (mask > 10).astype(np.uint8) * 255
        return cv2.inpaint(image, binary_mask, self.inpaint_radius, self.method)
