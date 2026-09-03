"""
mask_generator.py — Local text mask generation and seamless feather blending.
Employs character bridging, convex hull grouping, and boundary expansion
to completely eliminate black outline / drop shadow ghost text.
"""

from __future__ import annotations

import cv2
import numpy as np


def generate_text_mask(crop_img: np.ndarray, dilation_radius: int = 12) -> np.ndarray:
    """Extract a stable, cohesive subtitle mask from a cropped subtitle strip.

    Detects text strokes (white, yellow, bright colors), bridges adjacent characters,
    groups text contours with convex hulls, and expands boundaries to swallow
    black outlines, strokes, and drop shadows completely.
    """
    if crop_img is None or crop_img.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    h, w = crop_img.shape[:2]
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # 1. Detect bright text (white / light gray)
    _, light_mask = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)

    # 2. Detect yellow / warm subtitle text in HSV
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([15, 60, 130]), np.array([38, 255, 255]))

    # Combine detections
    combined = cv2.bitwise_or(light_mask, yellow_mask)

    # Fallback to Otsu if very low pixel count
    if np.sum(combined > 0) < 40:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        combined = otsu

    # If still negligible text, return blank
    if np.sum(combined > 0) < 40:
        return np.zeros_like(gray)

    # 3. Morphological close to bridge adjacent characters horizontally
    k_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
    connected = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_connect)

    # 4. Group text components using convex hulls to eliminate letter outline gaps
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    for c in contours:
        area = cv2.contourArea(c)
        if area > 80:
            hull = cv2.convexHull(c)
            cv2.drawContours(mask, [hull], -1, 255, -1)

    # 5. Expand hull mask with smooth margin to swallow black outlines & shadows
    k_size = max(5, dilation_radius | 1)
    k_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated_mask = cv2.dilate(mask, k_expand, iterations=1)

    return dilated_mask


def feather_blend(
    original: np.ndarray,
    inpainted: np.ndarray,
    mask: np.ndarray,
    blur_ksize: int = 9,
) -> np.ndarray:
    """Blend inpainted pixels seamlessly into the original strip using Gaussian feathering."""
    if mask is None or mask.max() == 0:
        return original.copy()

    k_size = max(3, blur_ksize | 1)
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (k_size, k_size), 0)
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]

    blended = (inpainted.astype(np.float32) * alpha + original.astype(np.float32) * (1.0 - alpha))
    return np.clip(blended, 0, 255).astype(np.uint8)
