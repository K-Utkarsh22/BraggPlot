"""
Bragg's Law + cubic Bravais lattice indexing ("multi-hypothesis workbench").

Given a list of measured 2-Theta peak positions from an XRD pattern,
this module answers: "assuming this is a cubic crystal, which of the
three cubic Bravais lattices (Simple Cubic / Body-Centered Cubic /
Face-Centered Cubic) best explains this pattern, and what is the
lattice parameter `a`?"

It evaluates all three hypotheses independently rather than picking one
winner and discarding the rest, because a poorly-fitting structure is
still informative: showing its inconsistent lattice parameter is what
lets a user see *why* it's a bad fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from braggplot.config import (
    ACCEPTANCE_TOLERANCE,
    MAX_MULTIPLIER,
    MIN_MULTIPLIER,
    REFERENCE_SEQUENCES,
    STRUCTURE_TIE_PRIORITY,
    TIE_EPSILON,
    DEFAULT_WAVELENGTH_ANGSTROM,
)


@dataclass
class StructureHypothesis:
    """One cubic Bravais lattice's fit against the observed peaks.

    A dataclass instead of a raw dict for three concrete reasons:
      1. `hypothesis.error_score` is checked by your IDE/type-checker;
         `hypothesis["eror_score"]` (typo) is only caught at runtime.
      2. The old return type was
         `Dict[str, Union[List[float], List[Dict[...]], str]]` — a type
         so wide it tells a reader almost nothing about the actual shape.
      3. It's the natural place to explain "why a dataclass and not a
         full class with methods" in an interview: this object has no
         behavior of its own (no methods that mutate or compute from
         its own fields after construction) — it is purely a labeled
         bundle of related values. That's exactly what a dataclass is
         for; giving it methods/inheritance would be unused structure.
    """

    structure: str  # "SC", "BCC", or "FCC"
    common_value: int  # this structure's own best-fit multiplier
    final_integers: list[int]  # rounded (h^2+k^2+l^2) sequence at that multiplier
    error_score: float  # lower is better; always populated
    lattice_parameters: list[float]  # per-peak reconstructed `a`, in Angstroms
    avg_lattice_parameter: float


@dataclass
class CrystalAnalysisResult:
    """Full output of `analyze_crystal_structure` for one set of peaks."""

    d_spacings: list[float]
    sin2_ratios: list[float]
    hypotheses: list[StructureHypothesis] = field(default_factory=list)
    best_fit: str = "Unknown"


def _bragg_d_spacings(
    peaks_2theta: list[float], wavelength: float
) -> tuple[list[float], list[float]]:
    """Convert 2-Theta peaks to (d_spacings, sin_thetas) via Bragg's Law.

    Bragg's Law: n*lambda = 2*d*sin(theta), so d = lambda / (2*sin(theta)).
    It's defined in terms of theta (half the scattering angle), which is
    why we divide the measured 2-Theta by 2 before taking sin().

    Raises:
        ValueError: if any 2-Theta value implies a non-positive
            sin(theta) (i.e. <= 0 degrees), which is not physically valid.
    """
    d_spacings: list[float] = []
    sin_thetas: list[float] = []
    for two_theta in peaks_2theta:
        theta_rad = math.radians(two_theta / 2.0)
        sin_theta = math.sin(theta_rad)
        if sin_theta <= 0:
            raise ValueError(
                "A 2-Theta value produced a non-positive sin(theta), "
                "which is not physically valid for Bragg's Law."
            )
        sin_thetas.append(sin_theta)
        d_spacings.append(wavelength / (2.0 * sin_theta))
    return d_spacings, sin_thetas


def _normalized_sin2_ratios(sin_thetas: list[float]) -> list[float]:
    """sin^2(theta) for each peak, normalized by the smallest value.

    For a cubic system, sin^2(theta) is proportional to (h^2+k^2+l^2).
    Normalizing by the smallest value turns that proportionality into a
    set of ratios that should collapse onto small whole numbers once
    scaled by the right integer multiplier — that scaling/rounding step
    happens in `_fit_structure_hypothesis` below.
    """
    sin2_theta = [s ** 2 for s in sin_thetas]
    min_sin2 = min(sin2_theta)  # > 0 guaranteed: sin_thetas already validated > 0
    return [s2 / min_sin2 for s2 in sin2_theta]


def _fit_structure_hypothesis(
    label: str,
    reference: list[int],
    sin2_ratios: list[float],
) -> tuple[StructureHypothesis, list[dict]]:
    """Grid-search multipliers 1..MAX for ONE structure and score the fit.

    Returns the best hypothesis for this structure, plus every
    (multiplier, error) candidate tried — the candidates are needed
    later to derive the single cross-structure `best_fit` label.
    """
    num_peaks = len(sin2_ratios)
    reference_prefix = reference[:num_peaks]

    best_multiplier = MIN_MULTIPLIER
    best_error = float("inf")
    best_integers: list[int] = []
    candidates: list[dict] = []

    for multiplier in range(MIN_MULTIPLIER, MAX_MULTIPLIER + 1):
        scaled_ratios = [ratio * multiplier for ratio in sin2_ratios]
        rounded_integers = [int(round(v)) for v in scaled_ratios]

        # How far the scaled ratios are from ANY whole number.
        snap_error = sum(abs(v - round(v)) for v in scaled_ratios) / num_peaks
        # How far the rounded integers are from THIS structure's
        # expected reference sequence.
        match_error = sum(
            abs(rounded_integers[i] - reference_prefix[i])
            for i in range(len(reference_prefix))
        ) / num_peaks
        combined_error = snap_error + match_error

        candidates.append(
            {
                "error": combined_error,
                "structure": label,
                "multiplier": multiplier,
                "final_integers": rounded_integers,
            }
        )

        if combined_error < best_error:
            best_error = combined_error
            best_multiplier = multiplier
            best_integers = rounded_integers

    return best_multiplier, best_error, best_integers, reference_prefix, candidates


def _lattice_parameters_for_hypothesis(
    reference_prefix: list[int],
    sin_thetas: list[float],
    wavelength: float,
) -> list[float]:
    """a = wavelength * sqrt(h^2+k^2+l^2) / (2*sin(theta)), per peak.

    Uses the STRUCTURE'S reference integers (the theoretical value if
    this structure hypothesis is correct), not the peak's own rounded
    ratio. That's what makes cross-peak consistency meaningful: a
    correct hypothesis reconstructs nearly the same `a` from every
    peak; a wrong one produces noisy, inconsistent values.
    """
    return [
        (wavelength * math.sqrt(hkl_sum)) / (2.0 * sin_theta)
        for hkl_sum, sin_theta in zip(reference_prefix, sin_thetas)
    ]


def _derive_best_fit_label(all_candidates: list[dict]) -> str:
    """Pick one convenience label across all structures, for DB logging.

    SC's reference sequence is a subset of BCC's at a different
    multiplier, so a genuine BCC/FCC pattern can also score deceptively
    well against SC. Among candidates tied within TIE_EPSILON of the
    global best, BCC/FCC are preferred over SC.
    """
    best_error = min(c["error"] for c in all_candidates)
    near_best = [c for c in all_candidates if c["error"] <= best_error + TIE_EPSILON]
    near_best.sort(
        key=lambda c: (STRUCTURE_TIE_PRIORITY.get(c["structure"], 2), c["multiplier"])
    )
    winner = near_best[0]
    return winner["structure"] if winner["error"] <= ACCEPTANCE_TOLERANCE else "Unknown"


def analyze_crystal_structure(
    peaks_2theta: list[float],
    wavelength: float = DEFAULT_WAVELENGTH_ANGSTROM,
) -> CrystalAnalysisResult:
    """Evaluate SC/BCC/FCC hypotheses for a set of XRD 2-Theta peaks.

    See module docstring for the "multi-hypothesis workbench" framing.
    This is pure computation: no I/O, no Streamlit, no database calls.

    Args:
        peaks_2theta: 2-Theta peak positions in degrees. Must be
            non-empty; values should be positive and less than 180.
        wavelength: X-ray wavelength in Angstroms (defaults to Cu-K-alpha1).

    Raises:
        ValueError: if peaks_2theta is empty, or contains a value that
            implies a non-physical (non-positive) sin(theta).
    """
    if not peaks_2theta:
        raise ValueError("peaks_2theta must contain at least one value.")

    d_spacings, sin_thetas = _bragg_d_spacings(peaks_2theta, wavelength)
    sin2_ratios = _normalized_sin2_ratios(sin_thetas)

    hypotheses: list[StructureHypothesis] = []
    all_candidates: list[dict] = []

    for label, reference in REFERENCE_SEQUENCES.items():
        (
            best_multiplier,
            best_error,
            best_integers,
            reference_prefix,
            candidates,
        ) = _fit_structure_hypothesis(label, reference, sin2_ratios)
        all_candidates.extend(candidates)

        lattice_parameters = _lattice_parameters_for_hypothesis(
            reference_prefix, sin_thetas, wavelength
        )
        avg_lattice_parameter = sum(lattice_parameters) / len(lattice_parameters)

        hypotheses.append(
            StructureHypothesis(
                structure=label,
                common_value=best_multiplier,
                final_integers=best_integers,
                error_score=best_error,
                lattice_parameters=lattice_parameters,
                avg_lattice_parameter=avg_lattice_parameter,
            )
        )

    best_fit = _derive_best_fit_label(all_candidates)

    return CrystalAnalysisResult(
        d_spacings=d_spacings,
        sin2_ratios=sin2_ratios,
        hypotheses=hypotheses,
        best_fit=best_fit,
    )
