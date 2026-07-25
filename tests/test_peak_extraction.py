"""Tests for braggplot.peak_extraction."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from braggplot.peak_extraction import extract_peaks


def _synthetic_diffractogram(width=800, height=300, peak_cols=(150, 350, 550, 700)) -> bytes:
    """Build a white image with dark V-shaped 'peaks' at known columns,
    so the test has ground truth for where peaks should be detected.
    """
    img = np.full((height, width), 255, dtype=np.uint8)
    for pc in peak_cols:
        for dx in range(-30, 31):
            col = pc + dx
            if 0 <= col < width:
                peak_height = max(0, 30 - abs(dx))
                row = height - 20 - peak_height
                img[max(0, row - 2): row + 2, col] = 0
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_detects_peaks_at_expected_positions():
    peak_cols = (150, 350, 550, 700)
    image_bytes = _synthetic_diffractogram(peak_cols=peak_cols)

    peaks = extract_peaks(image_bytes, axis_min=20.0, axis_max=80.0)

    expected = sorted(20 + (c / 799) * 60 for c in peak_cols)
    assert len(peaks) == len(expected)
    for got, want in zip(peaks, expected):
        assert got == pytest.approx(want, abs=0.5)


def test_inverted_axis_range_raises():
    image_bytes = _synthetic_diffractogram()
    with pytest.raises(ValueError):
        extract_peaks(image_bytes, axis_min=80.0, axis_max=20.0)


def test_invalid_image_bytes_raises():
    with pytest.raises(ValueError):
        extract_peaks(b"not an image")
