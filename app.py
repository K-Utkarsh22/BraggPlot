"""
XRD Crystallography Streamlit App
-----------------------------------
Step 1: Imports and core image-processing function (`extract_peaks`).
Step 2: Crystallography physics (`calculate_crystal_structure`).
Step 3: SQLite persistence (`init_db`, `insert_calculation`).
Step 4: Streamlit UI (under `if __name__ == "__main__":`).

This is the final, complete single-file application.
"""

from __future__ import annotations

# --- Standard library ---
import json
import math
import sqlite3
from datetime import datetime
from typing import Dict, List, Union

# --- Third-party libraries ---
import cv2  # opencv-python-headless: image decoding + preprocessing
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import find_peaks


def extract_peaks(
    image_bytes: bytes,
    axis_min: float = 20.0,
    axis_max: float = 80.0,
) -> list[float]:
    """
    Extract simulated 2-Theta peak positions from an XRD pattern image.

    This function treats the input image as a plotted XRD diffractogram
    (a dark curve on a light/white background) and recovers the
    approximate 2-Theta angles at which the curve has local maxima
    (peaks). It does this purely via image processing: there is no
    parsing of axis labels or tick marks, so the mapping from pixel
    columns to 2-Theta degrees is a LINEAR ASSUMPTION across the full
    width of the image, spanning the caller-specified range from
    `axis_min` to `axis_max` degrees (defaulting to 20.0-80.0, a common
    XRD scan range, if not otherwise specified).

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
           2-Theta value in the [axis_min, axis_max] degree range.
        6. Return the sorted list of 2-Theta float values.

    Args:
        image_bytes: Raw bytes of an image file (e.g. PNG/JPG) containing
            a plotted XRD pattern -- a single dark curve on a
            predominantly white/light background, with 2-Theta along
            the x-axis and intensity along the y-axis.
        axis_min: The 2-Theta value (in degrees) corresponding to the
            leftmost pixel column of the image (column 0). Defaults to
            20.0 degrees.
        axis_max: The 2-Theta value (in degrees) corresponding to the
            rightmost pixel column of the image. Defaults to 80.0
            degrees. Must be strictly greater than `axis_min`.

    Returns:
        list[float]: Sorted (ascending) list of detected peak positions,
            expressed as simulated 2-Theta values in degrees, constrained
            to the caller-specified axis range of [axis_min, axis_max].

    Raises:
        ValueError: If `image_bytes` cannot be decoded into a valid image
            by OpenCV (e.g. empty or corrupted data), or if `axis_max` is
            not strictly greater than `axis_min`.

    Notes / Limitations (by design at this stage):
        - No automatic real-world calibration of the 2-Theta axis is
          performed (e.g. no OCR of axis tick labels); `axis_min` and
          `axis_max` are caller-supplied assumptions about what the
          image's horizontal extent represents.
        - No crystallography math (Bragg's law, d-spacing, Miller
          indices, etc.) is performed here -- this function's sole job
          is pixel-to-angle peak extraction.
        - Threshold and `find_peaks` parameters are reasonable generic
          defaults, not tuned for any specific instrument or image style.
    """
    # --- 0. Validate the caller-supplied axis range ---
    # Since axis_min/axis_max are now user-controlled (e.g. from a
    # Streamlit number_input), we guard against a degenerate or
    # inverted range before doing any image work.
    if axis_max <= axis_min:
        raise ValueError(
            f"extract_peaks: 'axis_max' ({axis_max}) must be strictly "
            f"greater than 'axis_min' ({axis_min})."
        )

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
    # Linear mapping: column 0 -> axis_min degrees, column (width-1) ->
    # axis_max degrees, with everything in between interpolated
    # proportionally.
    two_theta_min = axis_min
    two_theta_max = axis_max
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
) -> Dict[str, Union[List[float], List[Dict[str, Union[str, int, float, List[float], List[int]]]], str]]:
    """
    Derive d-spacings and evaluate ALL THREE cubic Bravais lattice
    hypotheses (SC, BCC, FCC) from a set of XRD 2-Theta peak positions,
    using Bragg's Law and the sin^2(theta) ratio method.

    THIS IS A MULTI-HYPOTHESIS WORKBENCH, not a single-winner classifier.
    Earlier versions of this function picked one "best" structure label
    and discarded the rest. This version instead evaluates every
    structure independently and returns a comparison table: for each of
    SC, BCC, and FCC, it reports the best-fit integer sequence, an error
    score (lower = better fit), and -- critically -- the theoretical
    lattice parameter `a` computed from EVERY peak under that
    structure's hypothesis. A genuinely correct structure should yield
    a consistent `a` across all peaks; a wrong structure will usually
    produce noisy, inconsistent `a` values (or fail to fit close
    integers at all), even though SOME `a` value is still always
    reported so the user can see *why* a structure fits poorly, not
    just that it does.

    Background (cubic-system indexing method):
        For a cubic crystal system, sin^2(theta) for each reflection is
        proportional to (h^2 + k^2 + l^2), where h, k, l are the Miller
        indices of that reflection. Normalizing every peak's sin^2(theta)
        by the SMALLEST sin^2(theta) in the pattern gives a set of
        ratios that, after multiplying by a suitable small integer
        ("common value" / multiplier), should collapse onto a sequence
        of whole numbers matching one of the canonical (h^2+k^2+l^2)
        sequences below -- IF that structure is the correct one.

    Lattice parameter formula (per peak, per structure hypothesis):
        a = (wavelength * sqrt(h^2 + k^2 + l^2)) / (2 * sin(theta))
    where (h^2 + k^2 + l^2) is taken from the STRUCTURE'S REFERENCE
    SEQUENCE at that peak's position (the theoretical value implied by
    assuming this structure is correct), not from the peak's own
    rounded/observed integer. This is what makes the per-peak `a`
    values meaningful as a consistency check: if the hypothesis is
    right, every peak should independently reconstruct nearly the same
    `a`; if it's wrong, the reconstructed `a` values will disagree.

    Steps performed (per structure hypothesis):
        1. Convert each 2-Theta peak (in degrees) to theta in radians.
        2. Apply Bragg's Law to compute d-spacing for each peak.
        3. Compute sin^2(theta) for each peak, normalized by the
           smallest sin^2(theta) in the set.
        4. For multipliers 1 through 10, scale the normalized ratios,
           round them, and compare against THIS structure's reference
           sequence (truncated to however many peaks were detected),
           scoring the fit with a combined error metric (closeness to
           integers + closeness to the specific reference values).
        5. Pick the multiplier with the lowest error for this
           structure specifically (each structure gets its own
           independently-optimized multiplier -- they are not forced
           to share one).
        6. Using that multiplier's reference integers as the assumed
           (h^2+k^2+l^2) for each peak, compute a lattice parameter `a`
           per peak via the formula above, then average them.

    A single 'best_fit' label is also derived (using the same BCC/FCC-
    over-SC tie-breaking logic from earlier versions of this function)
    purely for simple database logging -- it is NOT the primary output
    of this function anymore.

    Args:
        peaks_2theta: List of 2-Theta peak positions in degrees, as
            produced by `extract_peaks` or typed in manually. Must
            contain at least one value; values are expected to be
            positive and less than 180 degrees.
        wavelength: X-ray wavelength in Angstroms used in Bragg's Law.
            Defaults to 1.5406 Angstroms, the standard Cu-K(alpha1)
            wavelength.

    Returns:
        Dict with the following keys:
            - 'd_spacings' (List[float]): Bragg's Law d-spacing for
              each input peak, in the same order as the input.
            - 'sin2_ratios' (List[float]): sin^2(theta) values for each
              peak, normalized against the smallest sin^2(theta) in
              the set, in the same order as the input.
            - 'hypotheses' (List[Dict]): one dictionary per structure
              (SC, BCC, FCC), each containing:
                - 'structure' (str): 'SC', 'BCC', or 'FCC'.
                - 'common_value' (int): this structure's own best-fit
                  multiplier (independent of the other two structures).
                - 'final_integers' (List[int]): the rounded
                  (h^2+k^2+l^2) sequence implied by this structure's
                  best multiplier.
                - 'error_score' (float): the fit-quality metric for
                  this structure (lower is better). Always present,
                  even for a poor fit.
                - 'lattice_parameters' (List[float]): the per-peak
                  lattice parameter `a` (Angstroms) computed under
                  this structure's hypothesis. Always fully populated
                  -- one value per input peak -- regardless of fit
                  quality, so the user can see the (likely
                  inconsistent) `a` values that make a bad fit bad.
                - 'avg_lattice_parameter' (float): the mean of
                  'lattice_parameters' for this structure.
            - 'best_fit' (str): One of 'SC', 'BCC', 'FCC', or 'Unknown'
              -- a single convenience label (for database logging only)
              indicating whichever structure scored the lowest error,
              with the same BCC/FCC-over-SC tie-breaking applied as in
              earlier versions of this function.

    Raises:
        ValueError: If `peaks_2theta` is empty, or if any 2-Theta value
            results in a non-physical theta (e.g. sin(theta) <= 0,
            which would make Bragg's Law undefined/divide-by-zero).

    Notes / Limitations:
        - This method assumes a CUBIC crystal system throughout.
        - Reference sequences are matched only against as many entries
          as peaks were detected (a prefix/best-fit match), not a full
          unit-cell derivation.
        - `tie_epsilon` (used only for the convenience 'best_fit'
          label) is set to 0.05 rather than a razor-thin tolerance, to
          remain robust against the rounding noise introduced by
          manually-typed 2-Theta values (e.g. via Verification Mode),
          which would otherwise fall just outside a tighter tolerance
          and silently default to the wrong tie-break winner.
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
    sin_thetas: List[float] = []
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
        sin_thetas.append(sin_theta)
        d_spacings.append(wavelength / (2.0 * sin_theta))

    # --- 3. sin^2(theta) ratios, normalized by the smallest value ---
    sin2_theta: List[float] = [st ** 2 for st in sin_thetas]
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
    max_multiplier_to_try = 10  # test candidate multipliers 1-10

    # --- 5. For EACH structure independently, grid-search multipliers ---
    # --- 1-10 and keep that structure's own best (lowest-error) fit. ---
    # Unlike earlier versions, structures are no longer forced to
    # compete for a single global winner here -- each gets its own
    # best multiplier, since the whole point of the workbench is to
    # show all three hypotheses side by side.
    hypotheses: List[Dict[str, Union[str, int, float, List[float], List[int]]]] = []

    # Also track every (multiplier, structure) candidate's error so we
    # can derive the single 'best_fit' convenience label afterward using
    # the same cross-structure tie-breaking logic as before.
    all_candidates: List[Dict[str, Union[int, str, float, List[int]]]] = []

    for label, reference in reference_sequences.items():
        reference_prefix = reference[:num_peaks]

        best_multiplier_for_structure = min_multiplier_to_try
        best_error_for_structure = float("inf")
        best_integers_for_structure: List[int] = []

        for multiplier in range(min_multiplier_to_try, max_multiplier_to_try + 1):
            # Scale the normalized ratios by this candidate multiplier.
            scaled_ratios = [ratio * multiplier for ratio in sin2_ratios]
            rounded_integers = [int(round(val)) for val in scaled_ratios]

            # "Snap error": how far the scaled ratios are from ANY whole
            # number in the first place, independent of which structure
            # we compare against.
            snap_error = sum(
                abs(val - round(val)) for val in scaled_ratios
            ) / num_peaks

            # "Match error": how far the rounded integers are from THIS
            # structure's expected reference values.
            match_error = sum(
                abs(rounded_integers[i] - reference_prefix[i])
                for i in range(len(reference_prefix))
            ) / num_peaks

            combined_error = snap_error + match_error

            all_candidates.append({
                "error": combined_error,
                "structure": label,
                "multiplier": multiplier,
                "final_integers": rounded_integers,
            })

            if combined_error < best_error_for_structure:
                best_error_for_structure = combined_error
                best_multiplier_for_structure = multiplier
                best_integers_for_structure = rounded_integers

        # --- 6. Lattice parameter a, per peak, under this structure's ---
        # --- hypothesis: a = wavelength * sqrt(h^2+k^2+l^2) / (2*sin(theta))
        # We use the STRUCTURE'S REFERENCE integers (the theoretical
        # h^2+k^2+l^2 values for this structure at this best multiplier),
        # not the peak's own observed/rounded ratio -- this is what
        # makes consistency-across-peaks a meaningful signal. If a peak
        # doesn't actually correspond to a valid reflection for this
        # structure, its reference value still exists (we always have
        # `num_peaks` worth of reference entries available here since
        # the sequences are pre-truncated to num_peaks), so a lattice
        # parameter is ALWAYS computed for every peak under every
        # structure, even when the fit is poor.
        lattice_parameters: List[float] = []
        for i in range(num_peaks):
            hkl_sum = reference_prefix[i]
            sin_theta = sin_thetas[i]
            a_value = (wavelength * math.sqrt(hkl_sum)) / (2.0 * sin_theta)
            lattice_parameters.append(a_value)

        avg_lattice_parameter = sum(lattice_parameters) / len(lattice_parameters)

        hypotheses.append({
            "structure": label,
            "common_value": best_multiplier_for_structure,
            "final_integers": best_integers_for_structure,
            "error_score": best_error_for_structure,
            "lattice_parameters": lattice_parameters,
            "avg_lattice_parameter": avg_lattice_parameter,
        })

    # --- 7. Derive a single 'best_fit' convenience label for the DB ---
    # Uses the same cross-structure tie-breaking as earlier versions:
    # SC's reference sequence is a strict /2 subset of BCC's, so a true
    # BCC pattern will ALSO score near-perfectly against SC at a smaller
    # multiplier. We collect every candidate within `tie_epsilon` of the
    # global best error and prefer BCC/FCC over SC among those ties.
    #
    # tie_epsilon is widened to 0.05 (from a much tighter 1e-6 in an
    # earlier version) specifically to stay robust against the rounding
    # noise introduced by manually-typed 2-Theta values (Verification
    # Mode): a human typing "44.67" instead of the exact
    # "44.67163127768449" introduces just enough numerical noise to
    # push a true BCC fit's error slightly above a razor-thin tolerance,
    # which previously caused it to lose the tie-break to SC by
    # accident. 0.05 is generous enough to absorb that rounding noise
    # while still being tight enough not to call two genuinely
    # different-quality fits a "tie".
    tie_epsilon = 0.05
    structure_priority = {"BCC": 0, "FCC": 0, "SC": 1, "Unknown": 2}
    acceptance_tolerance = 0.12

    best_error = min(c["error"] for c in all_candidates)
    near_best = [c for c in all_candidates if c["error"] <= best_error + tie_epsilon]
    near_best.sort(
        key=lambda c: (structure_priority.get(c["structure"], 2), c["multiplier"])
    )
    winner = near_best[0]

    best_fit = winner["structure"] if winner["error"] <= acceptance_tolerance else "Unknown"

    # --- Assemble and return the result dictionary ---
    return {
        "d_spacings": d_spacings,
        "sin2_ratios": sin2_ratios,
        "hypotheses": hypotheses,
        "best_fit": best_fit,
    }


# --- Database configuration ---
# Centralized here so both DB functions stay in sync if the filename
# ever needs to change.
DB_FILENAME = "xrd_history.db"


def init_db() -> None:
    """
    Initialize the local SQLite database used to persist XRD analysis
    history.

    Connects to the SQLite database file named by `DB_FILENAME`
    (created automatically by sqlite3 if it does not already exist on
    disk) and ensures a `history` table exists with the following
    schema:

        id            INTEGER PRIMARY KEY AUTOINCREMENT
        timestamp     DATETIME
        common_value  INTEGER
        ratios        TEXT
        structure     TEXT
        peaks         TEXT

    The table is created only if it does not already exist (via
    `CREATE TABLE IF NOT EXISTS`), so calling this function multiple
    times -- e.g. once per Streamlit app run/rerun -- is safe and will
    never wipe or duplicate existing history rows.

    Note on column types: `ratios` and `peaks` are stored as TEXT
    rather than as native list/array columns, because SQLite has no
    built-in array type. The intent is for callers (e.g.
    `insert_calculation`) to serialize Python lists (such as
    `sin2_ratios`, `final_integers`, or the original `peaks_2theta`
    list) into a TEXT-compatible representation -- for example a
    comma-separated string or a JSON-encoded string -- before storing
    them, and to deserialize them back into lists when reading rows
    back out.

    Args:
        None.

    Returns:
        None.

    Raises:
        sqlite3.Error: If the connection to the database file cannot
            be established, or if the `CREATE TABLE` statement fails
            for any reason (e.g. disk I/O error, permissions issue).
    """
    # sqlite3.connect() will create the .db file on disk automatically
    # if it does not already exist at this path.
    connection = sqlite3.connect(DB_FILENAME)
    try:
        cursor = connection.cursor()

        # IF NOT EXISTS makes this idempotent: safe to call on every
        # app startup without disturbing any rows already saved from
        # previous runs.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                common_value INTEGER,
                ratios TEXT,
                structure TEXT,
                peaks TEXT
            )
            """
        )

        # Persist the schema change (CREATE TABLE) to disk.
        connection.commit()
    finally:
        # Always release the connection, even if an error occurred
        # above, to avoid leaving the database file locked.
        connection.close()


def insert_calculation(
    common_value: int,
    ratios: str,
    structure: str,
    peaks: str,
) -> None:
    """
    Insert a single XRD analysis result as a new row into the `history`
    table of the local SQLite database, stamped with the current
    timestamp.

    This function opens its own short-lived connection to
    `DB_FILENAME`, inserts one row, commits the transaction, and closes
    the connection -- it does not assume `init_db()` has already been
    called within the same process, but it DOES assume the `history`
    table already exists (e.g. because `init_db()` was called earlier
    during app startup). If the table does not exist, this will raise
    an `sqlite3.OperationalError`.

    Args:
        common_value: The integer multiplier (`common_value` from
            `calculate_crystal_structure`'s result dictionary) that
            produced the best-fit lattice match.
        ratios: A string representation of the sin^2(theta) ratios (or
            final integer sequence) for this analysis. Callers are
            responsible for serializing their Python list into a
            string (e.g. via `str(some_list)` or `json.dumps(some_list)`)
            before passing it in, since SQLite's TEXT column stores
            this verbatim.
        structure: The identified crystal structure label, e.g. one of
            `'SC'`, `'BCC'`, `'FCC'`, or `'Unknown'`.
        peaks: A string representation of the original 2-Theta peak
            positions (e.g. the list returned by `extract_peaks`,
            serialized to a string by the caller) associated with this
            analysis.

    Returns:
        None.

    Raises:
        sqlite3.Error: If the connection cannot be established, if the
            `history` table does not exist, or if the `INSERT`
            statement fails for any other reason.
    """
    # Capture the current local timestamp at the moment of insertion.
    # Stored as an ISO-8601 string, which SQLite's DATETIME column type
    # affinity will accept and which sorts/compares correctly as text.
    current_timestamp = datetime.now().isoformat()

    connection = sqlite3.connect(DB_FILENAME)
    try:
        cursor = connection.cursor()

        # Parameterized query (the "?" placeholders) is used instead of
        # an f-string/format() to avoid SQL injection and to let
        # sqlite3 handle type adaptation correctly.
        cursor.execute(
            """
            INSERT INTO history (timestamp, common_value, ratios, structure, peaks)
            VALUES (?, ?, ?, ?, ?)
            """,
            (current_timestamp, common_value, ratios, structure, peaks),
        )

        # Persist the inserted row to disk.
        connection.commit()
    finally:
        # Always release the connection, even if an error occurred
        # above, to avoid leaving the database file locked.
        connection.close()


# =============================================================================
# Streamlit UI (Step 4)
# =============================================================================
# Everything below this point is presentation/orchestration logic only.
# No new image-processing, physics, or database logic is introduced here --
# this section exclusively wires together `extract_peaks`,
# `calculate_crystal_structure`, `init_db`, and `insert_calculation`.
# =============================================================================

if __name__ == "__main__":
    # --- 1. Page configuration -------------------------------------------
    # Must be the first Streamlit call in the script per Streamlit's API
    # rules (it configures the page before any other widget renders).
    # initial_sidebar_state="collapsed" keeps the sidebar tucked away by
    # default, since it now holds nothing but optional history -- the
    # main page is the intended focus of the app.
    st.set_page_config(
        layout="wide",
        page_title="XRD Crystal Analyzer",
        initial_sidebar_state="collapsed",
    )

    # --- 2. Initialize the database at the very start of the app run ----
    # Wrapped in a try/except so a database/filesystem problem surfaces
    # as a clear in-app error rather than crashing the whole script.
    try:
        init_db()
    except sqlite3.Error as db_init_error:
        st.error(f"Failed to initialize the database: {db_init_error}")
        st.stop()

    # --- 3. Session state -------------------------------------------------
    # Streamlit reruns the entire script on every widget interaction, so
    # the analysis result from clicking "Analyze" must be cached in
    # st.session_state to survive subsequent reruns (e.g. if the user
    # later opens the History expander, which is itself a rerun-causing
    # interaction). Without this, the results would vanish the instant
    # any other widget on the page was touched.
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "peaks_2theta" not in st.session_state:
        st.session_state.peaks_2theta = None
    if "image_bytes" not in st.session_state:
        st.session_state.image_bytes = None

    # --- 4. Sidebar ---------------------------------------------------
    # Sidebar is now reserved STRICTLY for Calculation History, per the
    # hero-layout redesign. No uploader, no axis inputs, no title here --
    # everything actionable lives on the main page.
    with st.sidebar:
        with st.expander("View History"):
            try:
                history_connection = sqlite3.connect(DB_FILENAME)
                try:
                    history_df = pd.read_sql_query(
                        "SELECT * FROM history ORDER BY timestamp DESC",
                        history_connection,
                    )
                finally:
                    history_connection.close()

                if history_df.empty:
                    st.write("No calculations have been saved yet.")
                else:
                    st.dataframe(history_df, use_container_width=True)

            except sqlite3.Error as history_error:
                st.error(f"Could not load calculation history: {history_error}")

    # --- 5. Main page: hero header -----------------------------------
    # Centered title/subtitle using a 3-column trick (wide side columns
    # squeeze the title into the middle), giving a "hero" feel rather
    # than a plain left-aligned heading.
    hero_left, hero_center, hero_right = st.columns([1, 2, 1])
    with hero_center:
        st.title("XRD Crystal Analyzer")
        st.write(
            "Upload an X-ray diffraction (XRD) pattern image to detect "
            "peaks, compute d-spacings via Bragg's Law, and identify the "
            "likely cubic crystal structure (SC, BCC, or FCC)."
        )

    st.divider()

    # --- 6. Main page: verification mode toggle --------------------------
    # When checked, the app skips OpenCV peak detection entirely and lets
    # the user type 2-Theta values directly, so the math engine
    # (calculate_crystal_structure) can be sanity-checked in isolation
    # from the image-processing pipeline.
    manual_mode = st.checkbox(
        "Manual Peak Input",
        help=(
            "Skip image-based peak detection and type 2-Theta peak "
            "values directly, to verify the crystal-structure math "
            "engine independent of OpenCV peak detection."
        ),
    )

    # --- 7. Main page: control bar (uploader/axis OR manual text input) -
    # This row acts as a persistent "top bar": once results are shown,
    # this same row stays put at the top of the page rather than being
    # replaced, so the user can immediately re-calibrate and re-analyze
    # without scrolling back up or losing context.
    if manual_mode:
        # --- Verification Mode: a single text input replaces the
        # uploader and both axis-calibration fields. ---
        control_manual, control_button = st.columns([4, 1])

        with control_manual:
            manual_peaks_text = st.text_input(
                "Enter 2-Theta values (comma-separated)",
                placeholder="e.g. 44.67, 65.02, 82.33, 98.94, 116.38, 137.15",
                help=(
                    "Type 2-Theta peak positions in degrees, separated "
                    "by commas. These values bypass image processing "
                    "and are passed directly to the crystal-structure "
                    "calculation."
                ),
            )

        with control_button:
            st.write("")
            analyze_clicked = st.button(
                "Analyze", type="primary", use_container_width=True
            )

        # Uploader and axis fields are hidden in manual mode; defined as
        # None/defaults here so later code that references them (outside
        # this branch) never hits a NameError.
        uploaded_file = None
        axis_start, axis_end = 20.0, 80.0
    else:
        # --- Normal Mode: uploader + Axis Calibration fields, as before.
        control_upload, control_start, control_end, control_button = st.columns(
            [3, 1, 1, 1]
        )

        with control_upload:
            uploaded_file = st.file_uploader(
                "Upload a diffractogram image",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
                help="Upload a PNG/JPG/JPEG image of an XRD diffractogram.",
            )

        with control_start:
            axis_start = st.number_input(
                "Axis Start",
                value=20.0,
                step=1.0,
                help="2-Theta value (degrees) at the left edge of the plot.",
            )

        with control_end:
            axis_end = st.number_input(
                "Axis End",
                value=80.0,
                step=1.0,
                help="2-Theta value (degrees) at the right edge of the plot.",
            )

        with control_button:
            # A little vertical spacing so the button lines up with the
            # number inputs instead of their labels.
            st.write("")
            analyze_clicked = st.button(
                "Analyze", type="primary", use_container_width=True
            )

        # Manual text input is hidden in normal mode; defined as an empty
        # string here so later code that references it never hits a
        # NameError.
        manual_peaks_text = ""

    st.divider()

    # --- 8. Analysis trigger -------------------------------------------
    # Analysis now runs ONLY when the "Analyze" button is explicitly
    # clicked (previously it ran automatically the instant a file was
    # uploaded). This is a deliberate behavior change to match the new
    # explicit-button control bar -- flagging it clearly since it
    # differs from every prior version of this app.
    if analyze_clicked:
        if manual_mode:
            # --- Verification Mode: skip extract_peaks entirely -------
            # Parse the comma-separated text directly into a list of
            # floats and feed it straight to calculate_crystal_structure,
            # so the math engine can be validated independent of OpenCV.
            raw_tokens = [tok.strip() for tok in manual_peaks_text.split(",")]
            raw_tokens = [tok for tok in raw_tokens if tok]  # drop blanks

            if not raw_tokens:
                st.warning(
                    "Please enter at least one 2-Theta value "
                    "(comma-separated) before analyzing."
                )
            else:
                try:
                    peaks_2theta = [float(tok) for tok in raw_tokens]
                except ValueError:
                    st.error(
                        "Could not parse the entered values. Please "
                        "enter numbers only, separated by commas "
                        "(e.g. '44.67, 65.02, 82.33')."
                    )
                    peaks_2theta = None

                if peaks_2theta is not None:
                    try:
                        # Step 2: run the physics -- evaluate all three
                        # structure hypotheses (SC, BCC, FCC) directly
                        # on the user-supplied peaks. No image processing.
                        analysis_result = calculate_crystal_structure(
                            peaks_2theta
                        )

                        # Step 3: persist this run to SQLite. The DB
                        # schema only has room for a single structure
                        # label/common_value, so we log the 'best_fit'
                        # hypothesis (and that hypothesis's own
                        # multiplier) as a simple summary -- the full
                        # multi-hypothesis comparison still lives in the
                        # results table below, not in the DB.
                        best_fit_label = analysis_result["best_fit"]
                        best_fit_hypothesis = next(
                            (
                                h
                                for h in analysis_result["hypotheses"]
                                if h["structure"] == best_fit_label
                            ),
                            None,
                        )
                        best_fit_common_value = (
                            best_fit_hypothesis["common_value"]
                            if best_fit_hypothesis is not None
                            else 0
                        )

                        serialized_peaks = json.dumps(peaks_2theta)
                        serialized_ratios = json.dumps(
                            analysis_result["sin2_ratios"]
                        )

                        insert_calculation(
                            common_value=best_fit_common_value,
                            ratios=serialized_ratios,
                            structure=best_fit_label,
                            peaks=serialized_peaks,
                        )

                        # No image exists in manual mode; cache None so
                        # the results display can detect this and skip
                        # rendering st.image.
                        st.session_state.analysis_result = analysis_result
                        st.session_state.peaks_2theta = peaks_2theta
                        st.session_state.image_bytes = None

                        st.success(
                            "Manual analysis complete and saved to history."
                        )

                    except ValueError as processing_error:
                        # Raised by calculate_crystal_structure on
                        # non-physical input (e.g. sin(theta) <= 0).
                        st.error(
                            f"Could not analyze these values: "
                            f"{processing_error}"
                        )
                    except sqlite3.Error as db_error:
                        st.error(
                            "Analysis succeeded, but saving to history "
                            f"failed: {db_error}"
                        )
                    except Exception as unexpected_error:  # noqa: BLE001
                        st.error(
                            f"An unexpected error occurred: "
                            f"{unexpected_error}"
                        )

        elif uploaded_file is None:
            st.warning("Please upload a diffractogram image before analyzing.")
        elif axis_end <= axis_start:
            st.error(
                "'Axis End' must be greater than 'Axis Start'. Please "
                "adjust the calibration values above."
            )
        else:
            try:
                # Read the uploaded file's raw bytes exactly once; reused
                # both for processing and for the on-screen image preview.
                image_bytes = uploaded_file.getvalue()

                with st.spinner("Analyzing diffractogram..."):
                    # Step 1: locate peaks in the diffractogram image and
                    # map them to simulated 2-Theta positions, using the
                    # user-calibrated axis range from the control bar.
                    peaks_2theta = extract_peaks(
                        image_bytes,
                        axis_min=axis_start,
                        axis_max=axis_end,
                    )

                    if not peaks_2theta:
                        st.warning(
                            "No peaks were detected in this image. Try a "
                            "clearer diffractogram with a visible dark "
                            "curve on a light background."
                        )
                        st.session_state.analysis_result = None
                        st.session_state.peaks_2theta = None
                        st.session_state.image_bytes = None
                        st.stop()

                    # Step 2: run the physics -- evaluate all three
                    # structure hypotheses (SC, BCC, FCC).
                    analysis_result = calculate_crystal_structure(peaks_2theta)

                    # Step 3: persist this run to SQLite. The DB schema
                    # only has room for a single structure label/
                    # common_value, so we log the 'best_fit' hypothesis
                    # (and its own multiplier) as a simple summary -- the
                    # full multi-hypothesis comparison lives in the
                    # results table below, not in the DB.
                    best_fit_label = analysis_result["best_fit"]
                    best_fit_hypothesis = next(
                        (
                            h
                            for h in analysis_result["hypotheses"]
                            if h["structure"] == best_fit_label
                        ),
                        None,
                    )
                    best_fit_common_value = (
                        best_fit_hypothesis["common_value"]
                        if best_fit_hypothesis is not None
                        else 0
                    )

                    serialized_peaks = json.dumps(peaks_2theta)
                    serialized_ratios = json.dumps(analysis_result["sin2_ratios"])

                    insert_calculation(
                        common_value=best_fit_common_value,
                        ratios=serialized_ratios,
                        structure=best_fit_label,
                        peaks=serialized_peaks,
                    )

                # Cache results in session_state so they persist below
                # the control bar across reruns, instead of only
                # existing transiently within this `if` block.
                st.session_state.analysis_result = analysis_result
                st.session_state.peaks_2theta = peaks_2theta
                st.session_state.image_bytes = image_bytes

                st.success("Analysis complete and saved to history.")

            except ValueError as processing_error:
                # Raised by extract_peaks (bad image) or
                # calculate_crystal_structure (non-physical input).
                st.error(f"Could not analyze this image: {processing_error}")
            except sqlite3.Error as db_error:
                # Raised by insert_calculation if the save step fails.
                st.error(
                    f"Analysis succeeded, but saving to history failed: {db_error}"
                )
            except Exception as unexpected_error:  # noqa: BLE001
                # Final safety net so an unanticipated failure still
                # shows a clean message instead of an unhandled
                # Streamlit traceback.
                st.error(f"An unexpected error occurred: {unexpected_error}")

    # --- 9. Results display ---------------------------------------------
    # Rendered from session_state (not from local variables scoped to
    # the button-click branch above), so results remain visible on
    # screen even after the script reruns for an unrelated reason (e.g.
    # the user toggling the History expander).
    if st.session_state.analysis_result is not None:
        analysis_result = st.session_state.analysis_result
        peaks_2theta = st.session_state.peaks_2theta
        cached_image_bytes = st.session_state.image_bytes

        col_image, col_metrics = st.columns([1, 1])

        with col_image:
            if cached_image_bytes is not None:
                # Normal mode: an uploaded diffractogram image exists.
                st.subheader("Uploaded Diffractogram")
                st.image(cached_image_bytes, use_container_width=True)
            else:
                # Verification Mode: peaks were typed in manually, so
                # there is no image to display. Show the raw input
                # values instead, to make clear this run bypassed
                # image processing entirely.
                st.subheader("Manual Peak Input")
                st.caption(
                    "No image was processed -- these 2-Theta values "
                    "were entered directly."
                )
                st.write(peaks_2theta)

        with col_metrics:
            st.subheader("Analysis Results")

            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Best Fit (Logged)", analysis_result["best_fit"])
            with metric_col2:
                st.metric("Peaks Found", len(peaks_2theta))

            # --- Multi-Hypothesis Workbench: comparison table ---------
            # Show every structure hypothesis side by side, rather than
            # collapsing to a single winner. Even a poorly-fitting
            # structure still shows its calculated lattice parameter, so
            # the user can SEE why it's a bad fit (inconsistent `a`)
            # rather than just being told it lost.
            hypothesis_rows = [
                {
                    "Structure Type": h["structure"],
                    "Calculated Lattice Parameter a (Å)": round(
                        h["avg_lattice_parameter"], 4
                    ),
                    "Error Score": round(h["error_score"], 4),
                }
                for h in analysis_result["hypotheses"]
            ]
            hypothesis_df = pd.DataFrame(hypothesis_rows)
            st.dataframe(hypothesis_df, use_container_width=True, hide_index=True)

            st.caption(
                "Note: A structure is valid if its lattice parameter "
                "remains consistent across multiple peaks. Lower error "
                "scores indicate a better fit."
            )

            with st.expander("View detailed numeric results"):
                st.write("**2-Theta peaks (degrees):**", peaks_2theta)
                st.write("**d-spacings:**", analysis_result["d_spacings"])
                st.write("**sin² ratios:**", analysis_result["sin2_ratios"])
                for h in analysis_result["hypotheses"]:
                    st.write(
                        f"**{h['structure']} final integers "
                        f"(common_value={h['common_value']}):**",
                        h["final_integers"],
                    )
                    st.write(
                        f"**{h['structure']} per-peak lattice "
                        f"parameters (Å):**",
                        [round(a, 4) for a in h["lattice_parameters"]],
                    )
    else:
        st.info("Upload a diffractogram image and click **Analyze** to begin.")
