"""An exact, bounded base-inpaint cache must not alter per-frame grain or pixels."""
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from services.video.inpainters.opencv_inpaint import OpenCVInpainter


def sample():
    image = np.full((60, 160, 3), 70, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[22:35, 55:85] = 255
    return image, mask


def test_identical_pixels_and_binary_mask_reuse_only_base_inpainting():
    image, mask = sample()
    engine = OpenCVInpainter()
    real = cv2.inpaint
    with patch("services.video.inpainters.opencv_inpaint.cv2.inpaint", wraps=real) as compute:
        first = engine.inpaint(image, mask)
        expected = first.copy()
        first[:] = 0  # A caller must never be able to mutate the cached result.
        second = engine.inpaint(image.copy(), mask.copy())
        assert np.array_equal(second, expected)
        second[:] = 0  # A cache hit must also return a caller-owned array.
        third = engine.inpaint(image.copy(), mask.copy())
    assert compute.call_count == 1
    assert np.array_equal(third, expected)
    info = engine.cache_info()
    assert info["hits"] == 2 and info["misses"] == 1
    assert info["cached_bytes"] == image.nbytes * 2 + mask.nbytes


def test_pixel_mask_radius_and_method_changes_invalidate_cache():
    image, mask = sample()
    engine = OpenCVInpainter()
    real = cv2.inpaint
    with patch("services.video.inpainters.opencv_inpaint.cv2.inpaint", wraps=real) as compute:
        engine.inpaint(image, mask)
        image[0, 0] = 71
        engine.inpaint(image, mask)
        mask[22, 54] = 255
        engine.inpaint(image, mask)
        engine.inpaint_radius = 3
        engine.inpaint(image, mask)
        engine.method = cv2.INPAINT_NS
        engine.inpaint(image, mask)
    assert compute.call_count == 5
    assert engine.cache_info()["hits"] == 0


@pytest.mark.parametrize("limit", [0, 100])
def test_cache_can_be_disabled_and_never_exceeds_pixel_budget(limit):
    image, mask = sample()
    engine = OpenCVInpainter(cache_max_pixels=limit)
    real = cv2.inpaint
    with patch("services.video.inpainters.opencv_inpaint.cv2.inpaint", wraps=real) as compute:
        engine.inpaint(image, mask)
        engine.inpaint(image, mask)
    assert compute.call_count == 2
    assert engine.cache_info()["cached_bytes"] == 0


def test_cached_base_keeps_seeded_per_frame_grain_identical_to_uncached_path():
    image, mask = sample()
    image = np.random.default_rng(23).integers(0, 256, image.shape, dtype=np.uint8)
    outputs = []
    for limit in (0, 1_000_000):
        random = np.random.RandomState(91)
        engine = OpenCVInpainter(cache_max_pixels=limit)
        with patch("services.video.inpainters.opencv_inpaint.np.random.normal", side_effect=random.normal) as grain:
            pair = [engine.inpaint(image.copy(), mask.copy()) for _ in range(2)]
        assert grain.call_count == 2  # Grain is still sampled per frame, not cached.
        outputs.append(pair)
    assert all(np.array_equal(old, new) for old, new in zip(*outputs))
    assert not np.array_equal(outputs[1][0], outputs[1][1])


def test_normalized_binary_masks_and_shape_changes_are_handled_exactly():
    image, mask = sample()
    engine = OpenCVInpainter()
    real = cv2.inpaint
    softer = mask.copy()
    softer[softer > 0] = 100
    with patch("services.video.inpainters.opencv_inpaint.cv2.inpaint", wraps=real) as compute:
        first = engine.inpaint(image, mask)
        second = engine.inpaint(image, softer)
        assert np.array_equal(first, second)
        engine.inpaint(image[:50], mask[:50])
    assert compute.call_count == 2
    assert engine.cache_info()["hits"] == 1


def test_tests_do_not_inherit_a_live_publish_outbox():
    from services.batch_pipeline import validate_batch_options
    assert validate_batch_options({})["publish_outbox_dir"] is None
    assert validate_batch_options({})["publish_rights_status"] == "review_required"
