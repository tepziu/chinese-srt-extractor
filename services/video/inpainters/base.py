"""
base.py — Abstract interface for subtitle/video inpainters.
Allows plugging in OpenCV, LaMa, MI-GAN, or future inpainting backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseInpainter(ABC):
    """Abstract base class for all image/video inpainting backends."""

    @abstractmethod
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint a single BGR image using a single-channel binary mask (255=inpaint, 0=keep).

        Args:
            image: uint8 BGR image of shape (H, W, 3).
            mask: uint8 grayscale mask of shape (H, W), non-zero pixels are inpainted.

        Returns:
            uint8 BGR inpainted image of shape (H, W, 3).
        """
        pass

    def inpaint_batch(self, images: list[np.ndarray], masks: list[np.ndarray]) -> list[np.ndarray]:
        """Inpaint a batch of images with corresponding masks."""
        return [self.inpaint(img, m) for img, m in zip(images, masks)]
