"""
Tests for braggplot.crystallography.

Why these specific test cases (say this in the interview if asked
"what testing strategy did you use?"):
  1. A known-good synthetic BCC pattern -- generated FROM the formula
     itself, not from a real scan -- gives us ground truth. If the
     function can't recover BCC from data we constructed to BE BCC,
     the function is wrong. This is the single most valuable test:
     it validates the actual physics/math, not just "does it run".
  2. Edge cases (empty input, non-physical input) validate the guard
     clauses explicitly, matching the Raises: sections in the docstrings.
  3. A single-peak case checks the code doesn't crash on the smallest
     valid input (num_peaks = 1), which is an easy off-by-one trap in
     code that slices reference[:num_peaks].
"""

from __future__ import annotations

import math

import pytest

from braggplot.crystallography import analyze_crystal_structure
from braggplot.config import REFERENCE_SEQUENCES, DEFAULT_WAVELENGTH_ANGSTROM


def _two_theta_from_hkl_sum(hkl_sum: int, a: float, wavelength: float) -> float:
    """Invert Bragg's Law + the cubic d-spacing formula to build a
    synthetic 2-Theta peak for a KNOWN lattice parameter `a` and a
    KNOWN (h^2+k^2+l^2) value. This lets tests construct peaks that are
    correct-by-construction for a given structure, instead of trusting
    hand-picked numbers.
    """
    d = a / math.sqrt(hkl_sum)
    theta_rad = math.asin(wavelength / (2 * d))
    return 2 * math.degrees(theta_rad)


def test_recovers_bcc_from_synthetic_bcc_pattern():
    a_true = 2.87  # Angstroms, close to real alpha-iron
    wavelength = DEFAULT_WAVELENGTH_ANGSTROM
    peaks = [
        _two_theta_from_hkl_sum(h, a_true, wavelength)
        for h in REFERENCE_SEQUENCES["BCC"][:4]
    ]

    result = analyze_crystal_structure(peaks, wavelength=wavelength)

    assert result.best_fit == "BCC"
    bcc_hypothesis = next(h for h in result.hypotheses if h.structure == "BCC")
    assert bcc_hypothesis.avg_lattice_parameter == pytest.approx(a_true, abs=0.01)
    # All three hypotheses are always present, even the losers.
    assert {h.structure for h in result.hypotheses} == {"SC", "BCC", "FCC"}


def test_single_peak_does_not_crash():
    result = analyze_crystal_structure([44.67])
    assert len(result.hypotheses) == 3
    for h in result.hypotheses:
        assert len(h.lattice_parameters) == 1


def test_empty_peaks_raises_value_error():
    with pytest.raises(ValueError):
        analyze_crystal_structure([])


def test_non_physical_two_theta_raises_value_error():
    with pytest.raises(ValueError):
        analyze_crystal_structure([0.0])  # sin(0) == 0 -> non-positive
