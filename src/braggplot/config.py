"""
Central configuration and constants for the BraggPlot app.

Why this file exists: before this refactor, DB_FILENAME lived as a bare
module-level variable inside app.py, and "magic numbers" like the 0.1
prominence factor or the 20.0/80.0 default axis range were scattered
inline at their point of use. Pulling them here means:
  1. There is exactly one place to change a default.
  2. Every other module can `from braggplot.config import X` instead of
     redefining or hardcoding X.
  3. It signals to a reader (or an interviewer) that you deliberately
     separated "things that vary" from "logic that doesn't".

This is NOT a settings framework (no YAML/env parsing, no Pydantic
Settings) — that would be over-engineering for a single-user Streamlit
app. Plain module-level constants are the right amount of structure
here.
"""

from __future__ import annotations

# --- Database ---------------------------------------------------------------
DB_FILENAME = "xrd_history.db"

# --- Peak extraction defaults ------------------------------------------------
DEFAULT_AXIS_MIN_DEGREES = 20.0
DEFAULT_AXIS_MAX_DEGREES = 80.0

# Fraction of the signal's (max - min) range a bump must exceed to count
# as a real peak rather than noise. Kept as a named constant because
# "0.1" with no name is exactly the kind of magic number an interviewer
# will ask you to justify.
PEAK_PROMINENCE_FRACTION = 0.1

# Minimum peak separation as a fraction of image width (in pixels).
PEAK_MIN_DISTANCE_FRACTION = 0.01

# --- Crystallography defaults -------------------------------------------------
# Cu-K(alpha1) X-ray wavelength in Angstroms — the standard lab source.
DEFAULT_WAVELENGTH_ANGSTROM = 1.5406

MIN_MULTIPLIER = 1
MAX_MULTIPLIER = 10

# How close two candidates' error scores must be to be treated as a
# "tie" when deriving the single best_fit label. Widened from a
# near-zero epsilon specifically to tolerate rounding noise from
# manually-typed 2-Theta values (see crystallography.py docstring).
TIE_EPSILON = 0.05

# A candidate's error score must be below this to be labeled a real
# match at all, rather than "Unknown".
ACCEPTANCE_TOLERANCE = 0.12

# Reference (h^2 + k^2 + l^2) sequences for the three cubic Bravais
# lattices, as commonly tabulated in XRD indexing references.
REFERENCE_SEQUENCES: dict[str, list[int]] = {
    "SC": [1, 2, 3, 4, 5, 6, 8, 9],
    "BCC": [2, 4, 6, 8, 10, 12, 14, 16],
    "FCC": [3, 4, 8, 11, 12, 16, 19, 20],
}

# SC's sequence is a strict subset (at a different multiplier) of BCC's,
# so a genuine BCC/FCC pattern will also score deceptively well against
# SC. When candidates are tied within TIE_EPSILON, prefer BCC/FCC over
# SC, and prefer any real structure over "Unknown".
STRUCTURE_TIE_PRIORITY: dict[str, int] = {"BCC": 0, "FCC": 0, "SC": 1, "Unknown": 2}

# --- Materials Project API ----------------------------------------------------
DEFAULT_MATERIALS_API_TOLERANCE_ANGSTROM = 0.15
