"""
opencv_inpaint.py — High-speed inpainter with grain-matching texture synthesis.
Preserves natural background grain (leather, asphalt, fabric, walls) without smooth smudging.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from services.video.inpainters.base import BaseInpainter


@dataclass
class _BaseCache:
    image: np.ndarray
    mask: np.ndarray
    result: np.ndarray
    parameters: tuple


class OpenCVInpainter(BaseInpainter):
    """Fast inpainter using cv2.inpaint with natural grain texture matching."""

    def __init__(self, method: str = "telea", inpaint_radius: int = 7,
                 cache_max_pixels: int = 1_000_000):
        self.method = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
        self.inpaint_radius = inpaint_radius
        # One exact ROI only: at most ~7 MB for an 8-bit BGR image + mask.
        # No perceptual hash, skipped frame, cross-video cache, or GPU memory.
        self.cache_max_pixels = max(0, int(cache_max_pixels))
        self._base_cache = None
        self.base_cache_hits = 0
        self.base_cache_misses = 0

    def cache_info(self) -> dict:
        entry = self._base_cache
        size = (entry.image.nbytes + entry.mask.nbytes + entry.result.nbytes) if entry else 0
        return {"enabled": self.cache_max_pixels > 0, "scope": "last_exact_roi_and_mask",
                "hits": self.base_cache_hits, "misses": self.base_cache_misses,
                "max_pixels": self.cache_max_pixels, "cached_bytes": size}

    def _inpaint_base(self, image: np.ndarray, binary_mask: np.ndarray):
        parameters = (self.method, self.inpaint_radius)
        entry = self._base_cache
        eligible = 0 < image.shape[0] * image.shape[1] <= self.cache_max_pixels
        if (eligible and entry is not None and entry.parameters == parameters and
                entry.image.dtype == image.dtype and
                np.array_equal(entry.mask, binary_mask) and np.array_equal(entry.image, image)):
            self.base_cache_hits += 1
            return entry.result, True
        self.base_cache_misses += 1
        result = cv2.inpaint(image, binary_mask, self.inpaint_radius, self.method)
        self._base_cache = (_BaseCache(image.copy(), binary_mask.copy(), result.copy(), parameters)
                            if eligible else None)
        return result, False

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask is None or mask.max() == 0:
            return image.copy()

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        binary_mask = (mask > 10).astype(np.uint8) * 255

        # 1. Only deterministic base inpainting is reused for identical inputs.
        # Grain synthesis below must still execute independently for each frame.
        inpainted, cache_hit = self._inpaint_base(image, binary_mask)

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

        return inpainted.copy() if cache_hit else inpainted
