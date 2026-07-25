"""
Materials Project live-database lookup for cubic material matches.

Cross-checks a calculated lattice parameter `a` against real materials
in the Materials Project database, as a real-world confirmation step
after the XRD analysis pipeline.
"""

from __future__ import annotations

from braggplot.config import DEFAULT_MATERIALS_API_TOLERANCE_ANGSTROM


def identify_material_api(
    api_key: str,
    a_param: float,
    tolerance: float = DEFAULT_MATERIALS_API_TOLERANCE_ANGSTROM,
) -> list[dict]:
    """Search the Materials Project for cubic materials near a_param.

    The summary `search()` endpoint has no documented server-side
    filter for a specific numeric lattice-parameter range, so this
    queries broadly by `crystal_system="Cubic"` (restricting `fields`
    to keep the request light) and filters by `a_param +/- tolerance`
    CLIENT-SIDE in Python.

    `mp_api.client.MPRester` is imported lazily inside this function
    rather than at module level: it's a heavyweight optional dependency
    (pulls in pymatgen) that also needs network access and an API key.
    Keeping the import local means the rest of the app keeps working
    on a machine where `mp-api` isn't installed — only a call to THIS
    function fails, as a normal caught exception rather than an
    import-time crash of the whole app.

    This function never raises. Every failure mode (missing key,
    network error, package not installed, unexpected error) is caught
    and returned as `[{"error": "<message>"}]` instead, so callers
    (Streamlit UI code) can call it directly without a wrapping
    try/except.

    Args:
        api_key: A Materials Project API key. Must be non-empty.
        a_param: Target lattice parameter `a`, in Angstroms — typically
            the winning hypothesis's `avg_lattice_parameter`.
        tolerance: Allowed +/- deviation from a_param, in Angstroms.

    Returns:
        Match dicts sorted by ascending error (closest match first),
        or `[{"error": "..."}]` on failure, or `[]` if nothing is
        within tolerance.
    """
    if not api_key or not api_key.strip():
        return [{"error": "No Materials Project API key was provided."}]

    try:
        from mp_api.client import MPRester
    except ImportError:
        return [{"error": "mp-api not installed"}]

    a_min = a_param - tolerance
    a_max = a_param + tolerance

    try:
        with MPRester(api_key) as mpr:
            docs = mpr.materials.summary.search(
                crystal_system="Cubic",
                fields=["material_id", "formula_pretty", "structure"],
            )
    except Exception as api_error:  # noqa: BLE001 -- see docstring: must never raise
        return [
            {
                "error": (
                    "Materials Project search failed (check your API "
                    f"key and network connection): {api_error}"
                )
            }
        ]

    matches: list[dict] = []
    for doc in docs:
        try:
            structure = doc.structure
            if structure is None:
                continue
            doc_a = float(structure.lattice.a)
        except (AttributeError, TypeError, ValueError):
            continue  # skip malformed entries rather than aborting the whole search

        if a_min <= doc_a <= a_max:
            matches.append(
                {
                    "Formula": getattr(doc, "formula_pretty", "Unknown"),
                    "Material ID": str(getattr(doc, "material_id", "Unknown")),
                    "Theoretical a": doc_a,
                    "Error": abs(doc_a - a_param),
                }
            )

    matches.sort(key=lambda m: m["Error"])
    return matches
