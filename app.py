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
import math
from typing import Dict, List, Union

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


def calculate_crystal_structure(
    peaks_2theta: List[float],
    wavelength: float = 1.5406,
) -> Dict[str, Union[List[float], List[int], int, str]]:
    """
    Derive d-spacings and infer the cubic crystal structure from a set
    of XRD 2-Theta peak positions, using Bragg's Law and the sin^2(theta)
    ratio method, with a multi-candidate best-fit search against all
    three cubic Bravais lattices (SC, BCC, FCC).

    Background (cubic-system indexing method):
        For a cubic crystal system, sin^2(theta) for each reflection is
        proportional to (h^2 + k^2 + l^2), where h, k, l are the Miller
        indices of that reflection. Normalizing every peak's sin^2(theta)
        by the SMALLEST sin^2(theta) in the pattern gives a set of
        ratios that, after multiplying by a suitable small integer
        ("common value"), should collapse onto a sequence of whole
        numbers matching one of the canonical (h^2+k^2+l^2) sequences
        below.

    IMPORTANT -- why this version differs from a naive "first multiplier
    that looks like an integer" approach: BCC's canonical sequence
    [2, 4, 6, 8, 10, 12, ...] is exactly double SC's canonical sequence
    [1, 2, 3, 4, 5, 6, ...]. This means a true BCC pattern, once
    normalized by its smallest sin^2(theta), ALREADY looks like perfect
    integers at multiplier=1 (i.e. [1, 2, 3, 4, 5, 6]) -- which is
    indistinguishable from SC unless you deliberately also test
    multiplier=2 and compare against BCC's actual reference sequence.
    To avoid this misclassification, this function does NOT stop at the
    first "integer-looking" multiplier. Instead it:
        - Tries every multiplier in a fixed candidate range (1 to 10).
        - For EACH multiplier, scales the normalized ratios, rounds them,
          and compares the result against ALL three reference sequences
          (SC, BCC, FCC).
        - Computes a normalized error metric per (multiplier, structure)
          combination, based on how far the scaled ratios land from
          whole numbers AND from the specific integers in that
          structure's reference sequence.
        - Picks whichever (multiplier, structure) combination has the
          lowest overall error -- i.e. the best global fit, not the
          first acceptable one.

    Steps performed:
        1. Convert each 2-Theta peak (in degrees) to theta in radians:
           theta_deg = 2theta / 2;  theta_rad = radians(theta_deg).
        2. Apply Bragg's Law to compute the interplanar spacing d for
           each peak: d = wavelength / (2 * sin(theta_rad)).
        3. Compute sin^2(theta) for each peak and normalize all values
           by dividing by the smallest sin^2(theta) in the set.
        4. For multipliers 1 through 10, and for each of the three
           reference lattice sequences (SC, BCC, FCC):
             a. Scale the normalized ratios by the multiplier.
             b. Round each scaled ratio to the nearest integer.
             c. Compare those rounded integers against the reference
                sequence (truncated to however many peaks we have) and
                compute a match-error score combining (i) how far the
                scaled values are from whole numbers in the first place,
                and (ii) how far the rounded integers are from the
                reference sequence's actual values.
        5. Select the (multiplier, structure) pair with the lowest
           error score across the entire search grid. This becomes the
           reported 'common_value' and 'structure'.
        6. If even the best-fitting combination's error exceeds a
           reasonable tolerance, the structure is reported as 'Unknown'
           rather than forcing a low-confidence label.

    Args:
        peaks_2theta: List of 2-Theta peak positions in degrees, as
            produced by `extract_peaks`. Must contain at least one
            value; values are expected to be positive and less than
            180 degrees.
        wavelength: X-ray wavelength in Angstroms used in Bragg's Law.
            Defaults to 1.5406 Angstroms, the standard Cu-K(alpha1)
            wavelength commonly used in lab XRD instruments.

    Returns:
        Dict[str, Union[List[float], List[int], int, str]]: A dictionary
        with the following keys:
            - 'd_spacings' (List[float]): Bragg's Law d-spacing for each
              input peak, in the same order as the input.
            - 'sin2_ratios' (List[float]): sin^2(theta) values for each
              peak, normalized against the smallest sin^2(theta) in the
              set, in the same order as the input (multiplier = 1,
              i.e. the raw ratios prior to best-fit scaling).
            - 'common_value' (int): The integer multiplier (from the
              1-10 candidate range) that produced the best overall fit
              to a reference lattice sequence.
            - 'final_integers' (List[int]): The normalized ratios,
              scaled by 'common_value' and rounded to the nearest
              integer -- the best-fit (h^2 + k^2 + l^2) sequence.
            - 'structure' (str): One of 'SC', 'BCC', 'FCC', or 'Unknown'
              -- whichever produced the lowest match-error score, or
              'Unknown' if no candidate met the tolerance.

    Raises:
        ValueError: If `peaks_2theta` is empty, or if any 2-Theta value
            results in a non-physical theta (e.g. sin(theta) <= 0,
            which would make Bragg's Law undefined/divide-by-zero).

    Notes / Limitations:
        - This method assumes a CUBIC crystal system. Non-cubic systems
          (tetragonal, hexagonal, orthorhombic, etc.) will generally
          score poorly against all three reference sequences and will
          likely be reported as 'Unknown' even though they may be
          perfectly valid crystals.
        - Reference sequences are matched only against as many entries
          as peaks were detected (a prefix/best-fit match), not a full
          unit-cell derivation.
        - No database lookups, plotting, or Streamlit UI are performed
          in this function -- it is pure computation.
    """
    # --- Guard: require at least one peak to operate on ---
    if not peaks_2theta:
        raise ValueError(
            "calculate_crystal_structure: 'peaks_2theta' must contain "
            "at least one value."
        )

    # --- 1. Convert 2-Theta (degrees) -> theta (radians) ---
    # Bragg's Law is defined in terms of theta (half the scattering
    # angle), not the full 2-Theta angle measured by the instrument.
    theta_degrees: List[float] = [two_theta / 2.0 for two_theta in peaks_2theta]
    theta_radians: List[float] = [math.radians(td) for td in theta_degrees]

    # --- 2. Bragg's Law: d = wavelength / (2 * sin(theta)) ---
    d_spacings: List[float] = []
    for theta_rad in theta_radians:
        sin_theta = math.sin(theta_rad)
        # sin(theta) must be strictly positive for Bragg's Law to be
        # physically meaningful here; a zero or negative value would
        # imply a non-physical 2-Theta peak (e.g. <= 0 degrees) and
        # would cause a division-by-zero or a negative d-spacing.
        if sin_theta <= 0:
            raise ValueError(
                "calculate_crystal_structure: encountered a 2-Theta "
                "value that produces a non-positive sin(theta), which "
                "is not physically valid for Bragg's Law."
            )
        d_spacings.append(wavelength / (2.0 * sin_theta))

    # --- 3. sin^2(theta) ratios, normalized by the smallest value ---
    sin2_theta: List[float] = [math.sin(tr) ** 2 for tr in theta_radians]
    min_sin2 = min(sin2_theta)

    # min_sin2 is guaranteed > 0 here since sin_theta was already
    # validated to be > 0 for every peak above.
    sin2_ratios: List[float] = [s2 / min_sin2 for s2 in sin2_theta]

    # --- 4. Reference (h^2 + k^2 + l^2)-style sequences for each cubic ---
    # --- Bravais lattice, as commonly tabulated in XRD indexing refs. ---
    reference_sequences: Dict[str, List[int]] = {
        "SC": [1, 2, 3, 4, 5, 6, 8, 9],
        "BCC": [2, 4, 6, 8, 10, 12, 14, 16],
        "FCC": [3, 4, 8, 11, 12, 16, 19, 20],
    }

    num_peaks = len(sin2_ratios)
    min_multiplier_to_try = 1
    max_multiplier_to_try = 10  # per spec: test candidate multipliers 1-10

    # --- 5. Grid-search every (multiplier, structure) combination and ---
    # --- score each one with a combined error metric, rather than ---
    # --- accepting the first multiplier that merely "looks like" an ---
    # --- integer (which is what caused the SC/BCC ambiguity earlier). ---

    # A best-fit candidate must land this close (on average) to the
    # reference sequence's integers to be accepted as a confident match;
    # otherwise we report 'Unknown' rather than force a low-confidence
    # label onto noisy or non-cubic data.
    acceptance_tolerance = 0.12

    # IMPORTANT TIE-BREAKING NOTE: SC's reference sequence
    # [1, 2, 3, 4, 5, 6, 8, 9] is mathematically a strict subset of
    # BCC's [2, 4, 6, 8, 10, 12, ...] divided by 2. This means a TRUE
    # BCC pattern will produce a perfect (zero-error) match against SC
    # at multiplier=1 AND a perfect (zero-error) match against BCC at
    # multiplier=2, simultaneously. Plain "lowest error wins" is not
    # sufficient here because both candidates score essentially equally
    # well -- whichever is evaluated first would win by accident, which
    # is exactly the bug we're fixing. To break ties correctly, we track
    # ALL near-perfect candidates (within `tie_epsilon` of the best
    # error found) and, among those, prefer the structure whose
    # reference sequence is NOT a pure integer subset of a simpler
    # pattern -- i.e. prefer BCC/FCC over SC whenever they are tied,
    # since SC's sequence is the "trivially looks like integers"
    # degenerate case that will always tie with a true BCC/FCC fit.
    tie_epsilon = 1e-6
    structure_priority = {"BCC": 0, "FCC": 0, "SC": 1, "Unknown": 2}

    candidates: List[Dict[str, Union[int, str, float, List[int]]]] = []

    for multiplier in range(min_multiplier_to_try, max_multiplier_to_try + 1):
        # Scale the normalized ratios by this candidate multiplier.
        scaled_ratios = [ratio * multiplier for ratio in sin2_ratios]
        rounded_integers = [int(round(val)) for val in scaled_ratios]

        # "Snap error": how far the scaled ratios are from ANY whole
        # number in the first place (independent of which structure we
        # compare against) -- a high snap error means this multiplier
        # doesn't clear denominators well at all, regardless of lattice.
        snap_error = sum(
            abs(val - round(val)) for val in scaled_ratios
        ) / num_peaks

        for label, reference in reference_sequences.items():
            # Compare rounded integers against this structure's
            # reference sequence, truncated to however many peaks we
            # actually have (can't match more reference entries than
            # peaks detected).
            reference_prefix = reference[:num_peaks]

            # "Match error": how far the rounded integers are from this
            # specific structure's expected sequence values.
            match_error = sum(
                abs(rounded_integers[i] - reference_prefix[i])
                for i in range(len(reference_prefix))
            ) / num_peaks

            # Combined confidence/error score: both the raw closeness-
            # to-integer (snap_error) and the closeness-to-this-specific-
            # lattice (match_error) matter. Equal weighting keeps the
            # metric simple and interpretable.
            combined_error = snap_error + match_error

            candidates.append({
                "error": combined_error,
                "structure": label,
                "multiplier": multiplier,
                "final_integers": rounded_integers,
            })

    # Find the single lowest error across the whole search grid.
    best_error = min(c["error"] for c in candidates)

    # Collect every candidate within `tie_epsilon` of that best error --
    # on clean synthetic data, BCC (at its multiplier) and SC (at
    # multiplier=1) will often BOTH appear here.
    near_best = [c for c in candidates if c["error"] <= best_error + tie_epsilon]

    # Among the near-best candidates, prefer BCC/FCC over SC (see note
    # above), then prefer the smallest multiplier as a final tiebreaker
    # for full determinism.
    near_best.sort(
        key=lambda c: (structure_priority.get(c["structure"], 2), c["multiplier"])
    )
    winner = near_best[0]

    best_error = winner["error"]
    best_structure = winner["structure"]
    best_multiplier = winner["multiplier"]
    best_final_integers = winner["final_integers"]

    # --- 6. Reject low-confidence "best" matches rather than force one ---
    if best_error > acceptance_tolerance:
        best_structure = "Unknown"

    # --- Assemble and return the result dictionary ---
    return {
        "d_spacings": d_spacings,
        "sin2_ratios": sin2_ratios,
        "common_value": best_multiplier,
        "final_integers": best_final_integers,
        "structure": best_structure,
    }
