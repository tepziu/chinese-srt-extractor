"""
mask_generator.py — High-precision subtitle text mask generation.
Detects subtitle cores (yellow, white, bright text) and their exact black outline/shadow,
without falsely detecting light-colored backgrounds (dashboards, roads, sky, shirts).
"""

from __future__ import annotations

import cv2
import numpy as np


def generate_text_mask(crop_img: np.ndarray, dilation_radius: int = 10) -> np.ndarray:
    """Extract an accurate, clean subtitle text mask covering characters and strokes.

    Specifically avoids false-positive background detection on light-colored surfaces.
    """
    if crop_img is None or crop_img.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    h, w = crop_img.shape[:2]
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

    # 1. Detect yellow / warm subtitle text in HSV
    yellow_mask = cv2.inRange(hsv, np.array([16, 65, 110]), np.array([38, 255, 255]))

    # 2. Detect crisp white subtitle text (low saturation, very high brightness)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 225]), np.array([180, 40, 255]))

    # 3. Detect high-contrast bright text against background
    text_cores = cv2.bitwise_or(yellow_mask, white_mask)

    # If no core text detected, check if there is general high-saturation subtitle text
    if np.sum(text_cores > 0) < 50:
        sat_text = cv2.inRange(hsv, np.array([0, 100, 140]), np.array([180, 255, 255]))
        if np.sum(sat_text > 0) > 80:
            text_cores = sat_text

    # If still no text detected in strip, return zero mask (do not inpaint clean background!)
    if np.sum(text_cores > 0) < 50:
        return np.zeros((h, w), dtype=np.uint8)

    # 4. Extract black stroke/outline directly attached to text characters
    black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))

    k_stroke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilated_cores = cv2.dilate(text_cores, k_stroke, iterations=2)
    attached_stroke = cv2.bitwise_and(black, dilated_cores)

    # Combine text cores and their black stroke
    full_text = cv2.bitwise_or(text_cores, attached_stroke)

    # 5. Seal internal character loops (e.g. inside Chinese characters like 国, 面, 日, 口)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed_text = cv2.morphologyEx(full_text, cv2.MORPH_CLOSE, k_close)

    # 6. Smooth dilation to cleanly swallow anti-aliased edges and drop shadows
    rad = max(5, dilation_radius | 1)
    k_margin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad, rad))
    final_mask = cv2.dilate(closed_text, k_margin, iterations=1)

    return final_mask


def feather_blend(
    original: np.ndarray,
    inpainted: np.ndarray,
    mask: np.ndarray,
    blur_ksize: int = 7,
) -> np.ndarray:
    """Blend inpainted pixels seamlessly into the original strip using Gaussian feathering."""
    if mask is None or mask.max() == 0:
        return original.copy()

    k_size = max(3, blur_ksize | 1)
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (k_size, k_size), 0)
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]

    blended = (inpainted.astype(np.float32) * alpha + original.astype(np.float32) * (1.0 - alpha))
    return np.clip(blended, 0, 255).astype(np.uint8)