"""
XRD Crystallography Streamlit App
-----------------------------------
Step 1 of N: Imports and core image-processing function (`extract_peaks`).

NOTE: This file intentionally contains ONLY imports and `extract_peaks`
at this stage. No Streamlit UI, no math/crystallography logic (e.g.
Bragg's law, d-spacing), and no database code has been added yet.
"""

from __future__ import annotations

# --- Standard library ---
from typing import List

# --- Third-party libraries ---
import numpy as np
import cv2  # opencv-python-headless: image decoding + preprocessing
from scipy.signal import find_peaks


def extract_peaks(image_bytes: bytes) -> list[float]:
    """
    Extract simulated 2-Theta peak positions from an XRD pattern image.

    This function treats the input image as a plotted XRD diffractogram
    (a dark curve on a light/white background) and recovers the
    approximate 2-Theta angles at which the curve has local maxima
    (peaks). It does this purely via image processing: there is no
    parsing of axis labels or tick marks, so the mapping from pixel
    columns to 2-Theta degrees is a LINEAR ASSUMPTION across the full
    width of the image, spanning a fixed range of 20.0 to 80.0 degrees.

    Pipeline overview:
        1. Decode raw image bytes into a grayscale OpenCV image.
        2. Preprocess (invert + threshold) so the dark curve becomes
           bright "signal" against a black background, which simplifies
           locating the curve pixel-by-pixel.
        3. Collapse the 2D image into a 1D signal by, for each pixel
           column, finding the lowest (highest row-index) bright pixel.
           In a typical XRD plot (y-axis = intensity, increasing upward,
           plotted top-down in image space), the "lowest dark pixel"
           in a column corresponds to the topmost point of the curve's
           ink in that column -- i.e., the peak/intensity trace itself.
           We use image height minus row-index as a proxy for intensity,
           so taller peaks in the plot become larger values in the 1D
           signal.
        4. Run `scipy.signal.find_peaks` on that 1D intensity-proxy
           signal to locate candidate peak columns.
        5. Linearly map each peak's pixel column (x-coordinate) to a
           2-Theta value in the [20.0, 80.0] degree range.
        6. Return the sorted list of 2-Theta float values.

    Args:
        image_bytes: Raw bytes of an image file (e.g. PNG/JPG) containing
            a plotted XRD pattern -- a single dark curve on a
            predominantly white/light background, with 2-Theta along
            the x-axis and intensity along the y-axis.

    Returns:
        list[float]: Sorted (ascending) list of detected peak positions,
            expressed as simulated 2-Theta values in degrees, constrained
            to the assumed axis range of [20.0, 80.0].

    Raises:
        ValueError: If `image_bytes` cannot be decoded into a valid image
            by OpenCV (e.g. empty or corrupted data).

    Notes / Limitations (by design at this stage):
        - No real-world calibration of the 2-Theta axis is performed;
          the 20.0-80.0 degree range is a simulated/assumed default.
        - No crystallography math (Bragg's law, d-spacing, Miller
          indices, etc.) is performed here -- this function's sole job
          is pixel-to-angle peak extraction.
        - Threshold and `find_peaks` parameters are reasonable generic
          defaults, not tuned for any specific instrument or image style.
    """
    # --- 1. Decode raw bytes into a grayscale OpenCV image ---
    # np.frombuffer creates a 1D array view of the raw bytes without
    # copying; cv2.imdecode then interprets that buffer as an encoded
    # image (PNG/JPG/etc.) and decodes it into pixel data.
    # IMREAD_GRAYSCALE collapses any color channels into a single
    # intensity channel, which is all we need to find a dark curve.
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    gray_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # cv2.imdecode returns None (rather than raising) on failure, so we
    # must check explicitly and convert that into a Pythonic exception.
    if gray_image is None:
        raise ValueError(
            "extract_peaks: could not decode image_bytes into a valid "
            "image. Ensure the bytes represent a supported format "
            "(e.g. PNG, JPG)."
        )

    height, width = gray_image.shape

    # --- 2. Preprocessing: isolate the dark curve on a white background ---
    # cv2.threshold with THRESH_BINARY_INV flips the usual convention:
    # pixels DARKER than `thresh` (the curve/ink) become white (255) in
    # the output, and pixels LIGHTER than `thresh` (the white background)
    # become black (0). THRESH_OTSU automatically computes a good
    # threshold value from the image's histogram instead of us guessing
    # a fixed constant, which makes this robust to varying scan
    # brightness/contrast. The returned threshold value itself is
    # discarded (we only need the binary mask).
    _, binary_mask = cv2.threshold(
        gray_image,
        0,  # ignored because THRESH_OTSU overrides it
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # --- 3. Collapse the 2D binary mask into a 1D intensity-proxy signal ---
    # For each column, we want the row index of the curve (the lowest/
    # bottommost lit pixel in that column is irrelevant here -- what we
    # actually want is the HIGHEST point of the curve's ink, since plot
    # intensity increases upward but image row-index increases downward).
    # We scan each column for white (curve) pixels and take the SMALLEST
    # row index found (i.e. the topmost ink pixel = highest plotted
    # intensity for that column).
    intensity_signal = np.zeros(width, dtype=np.float64)

    for col in range(width):
        # np.where on a single column returns the row indices where the
        # binary mask is non-zero (i.e. where curve ink was detected).
        lit_rows = np.where(binary_mask[:, col] > 0)[0]

        if lit_rows.size > 0:
            # Topmost ink pixel in this column (smallest row index).
            topmost_row = lit_rows.min()
            # Convert "row index from the top" into an intensity-style
            # value where higher = more intense, by inverting against
            # the image height. A pixel near the top (small row index)
            # yields a large value; a pixel near the bottom (large row
            # index) yields a small value.
            intensity_signal[col] = float(height - topmost_row)
        else:
            # No curve ink detected in this column at all -> treat as
            # baseline/zero intensity rather than leaving a gap.
            intensity_signal[col] = 0.0

    # --- 4. Find peaks in the 1D intensity-proxy signal ---
    # `prominence` filters out shallow bumps/noise by requiring a peak
    # to stand out from its surrounding baseline by a minimum amount.
    # `distance` enforces a minimum pixel separation between detected
    # peaks so closely-spaced noisy fluctuations aren't all reported
    # as distinct peaks. Both are reasonable generic defaults, scaled
    # relative to the image's own dimensions so they're not totally
    # arbitrary across different image resolutions.
    min_prominence = (intensity_signal.max() - intensity_signal.min()) * 0.1
    min_distance = max(1, width // 100)  # at least 1% of image width apart

    peak_indices, _ = find_peaks(
        intensity_signal,
        prominence=min_prominence,
        distance=min_distance,
    )

    # --- 5. Map peak pixel columns (x-coordinates) to simulated 2-Theta ---
    # Linear mapping: column 0 -> 20.0 degrees, column (width-1) -> 80.0
    # degrees, with everything in between interpolated proportionally.
    two_theta_min = 20.0
    two_theta_max = 80.0
    two_theta_range = two_theta_max - two_theta_min

    # Guard against division-by-zero on a pathological 1-pixel-wide image.
    denominator = max(width - 1, 1)

    two_theta_values: List[float] = [
        two_theta_min + (float(col) / denominator) * two_theta_range
        for col in peak_indices
    ]

    # --- 6. Return sorted ascending list of 2-Theta peak positions ---
    return sorted(two_theta_values)
