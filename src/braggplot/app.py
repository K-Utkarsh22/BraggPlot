"""
Streamlit UI for the XRD Crystal Analyzer.

This module ONLY does presentation/orchestration: it renders widgets,
manages st.session_state, and calls into braggplot.peak_extraction,
braggplot.crystallography, braggplot.db, and braggplot.materials_api.
No image processing, physics, or SQL lives here.
"""

from __future__ import annotations

import json
import sqlite3

import pandas as pd
import streamlit as st

from braggplot.config import DEFAULT_AXIS_MAX_DEGREES, DEFAULT_AXIS_MIN_DEGREES
from braggplot.crystallography import CrystalAnalysisResult, analyze_crystal_structure
from braggplot.db import CalculationRecord, get_history, init_db, insert_calculation
from braggplot.materials_api import identify_material_api
from braggplot.peak_extraction import extract_peaks


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_material_search(api_key: str, a_param: float) -> list[dict]:
    """Cache Materials Project results for an hour.

    identify_material_api fetches every cubic material in the database
    (tens of thousands of documents) and filters client-side, since
    there's no server-side lattice-parameter filter -- a single call
    can take well over a minute. st.cache_data keys on the exact
    (api_key, a_param) arguments, so re-running the SAME analysis (e.g.
    Streamlit re-rendering after an unrelated widget interaction) reuses
    the cached result instead of re-fetching ~21k documents each time.
    A 1-hour ttl means a stale cache self-expires rather than serving
    outdated matches forever.
    """
    return identify_material_api(api_key=api_key, a_param=a_param)


def _save_result_to_history(
    analysis_result: CrystalAnalysisResult, peaks_2theta: list[float]
) -> None:
    """Log the winning hypothesis as one row in the history table.

    The DB schema only has room for a single structure label/multiplier,
    so we log the best_fit hypothesis as a summary — the full
    multi-hypothesis comparison stays in the on-screen results table,
    not the database.
    """
    best_fit_hypothesis = next(
        (h for h in analysis_result.hypotheses if h.structure == analysis_result.best_fit),
        None,
    )
    best_fit_common_value = (
        best_fit_hypothesis.common_value if best_fit_hypothesis is not None else 0
    )
    insert_calculation(
        CalculationRecord(
            common_value=best_fit_common_value,
            ratios=json.dumps(analysis_result.sin2_ratios),
            structure=analysis_result.best_fit,
            peaks=json.dumps(peaks_2theta),
        )
    )


def _init_session_state() -> None:
    for key in ("analysis_result", "peaks_2theta", "image_bytes"):
        if key not in st.session_state:
            st.session_state[key] = None


def _render_sidebar() -> str:
    """Render the sidebar; return the active Materials Project API key."""
    with st.sidebar:
        st.subheader("Materials Project Search")

        global_api_key = st.secrets.get("MP_API_KEY", "")
        if global_api_key:
            active_api_key = global_api_key
            st.caption("Using a pre-configured Materials Project API key.")
        else:
            active_api_key = st.text_input(
                "Materials Project API Key",
                type="password",
                help=(
                    "Get a free API key by registering at "
                    "materialsproject.org. Used to search the live "
                    "Materials Project database for materials matching "
                    "your calculated lattice parameter."
                ),
            )

        st.divider()

        with st.expander("View History"):
            try:
                history_df = get_history()
                if history_df.empty:
                    st.write("No calculations have been saved yet.")
                else:
                    st.dataframe(history_df, width='stretch')
            except sqlite3.Error as history_error:
                st.error(f"Could not load calculation history: {history_error}")

    return active_api_key


def _render_control_bar() -> tuple[bool, bool, object, str, float, float]:
    """Render the manual-mode toggle + control bar.

    Returns:
        (manual_mode, analyze_clicked, uploaded_file, manual_peaks_text,
         axis_start, axis_end)
    """
    manual_mode = st.checkbox(
        "Manual Peak Input",
        help=(
            "Skip image-based peak detection and type 2-Theta peak "
            "values directly, to verify the crystal-structure math "
            "engine independent of OpenCV peak detection."
        ),
    )

    if manual_mode:
        control_manual, control_button = st.columns([4, 1])
        with control_manual:
            manual_peaks_text = st.text_input(
                "Enter 2-Theta values (comma-separated)",
                placeholder="e.g. 44.67, 65.02, 82.33, 98.94, 116.38, 137.15",
            )
        with control_button:
            st.write("")
            analyze_clicked = st.button("Analyze", type="primary", width='stretch')
        return manual_mode, analyze_clicked, None, manual_peaks_text, DEFAULT_AXIS_MIN_DEGREES, DEFAULT_AXIS_MAX_DEGREES

    control_upload, control_start, control_end, control_button = st.columns([3, 1, 1, 1])
    with control_upload:
        uploaded_file = st.file_uploader(
            "Upload a diffractogram image",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
    with control_start:
        axis_start = st.number_input("Axis Start", value=DEFAULT_AXIS_MIN_DEGREES, step=1.0)
    with control_end:
        axis_end = st.number_input("Axis End", value=DEFAULT_AXIS_MAX_DEGREES, step=1.0)
    with control_button:
        st.write("")
        analyze_clicked = st.button("Analyze", type="primary", width='stretch')

    return manual_mode, analyze_clicked, uploaded_file, "", axis_start, axis_end


def _run_manual_analysis(manual_peaks_text: str) -> None:
    raw_tokens = [tok.strip() for tok in manual_peaks_text.split(",") if tok.strip()]
    if not raw_tokens:
        st.warning("Please enter at least one 2-Theta value (comma-separated) before analyzing.")
        return

    try:
        peaks_2theta = [float(tok) for tok in raw_tokens]
    except ValueError:
        st.error("Could not parse the entered values. Use numbers only, e.g. '44.67, 65.02'.")
        return

    try:
        analysis_result = analyze_crystal_structure(peaks_2theta)
        _save_result_to_history(analysis_result, peaks_2theta)

        st.session_state.analysis_result = analysis_result
        st.session_state.peaks_2theta = peaks_2theta
        st.session_state.image_bytes = None
        st.success("Manual analysis complete and saved to history.")
    except ValueError as processing_error:
        st.error(f"Could not analyze these values: {processing_error}")
    except sqlite3.Error as db_error:
        st.error(f"Analysis succeeded, but saving to history failed: {db_error}")
    except Exception as unexpected_error:  # noqa: BLE001 -- last-resort UI safety net
        st.error(f"An unexpected error occurred: {unexpected_error}")


def _run_image_analysis(uploaded_file, axis_start: float, axis_end: float) -> None:
    if uploaded_file is None:
        st.warning("Please upload a diffractogram image before analyzing.")
        return
    if axis_end <= axis_start:
        st.error("'Axis End' must be greater than 'Axis Start'. Please adjust the values above.")
        return

    try:
        image_bytes = uploaded_file.getvalue()
        with st.spinner("Analyzing diffractogram..."):
            peaks_2theta = extract_peaks(image_bytes, axis_min=axis_start, axis_max=axis_end)

            if not peaks_2theta:
                st.warning(
                    "No peaks were detected in this image. Try a clearer "
                    "diffractogram with a visible dark curve on a light background."
                )
                st.session_state.analysis_result = None
                st.session_state.peaks_2theta = None
                st.session_state.image_bytes = None
                return

            analysis_result = analyze_crystal_structure(peaks_2theta)
            _save_result_to_history(analysis_result, peaks_2theta)

        st.session_state.analysis_result = analysis_result
        st.session_state.peaks_2theta = peaks_2theta
        st.session_state.image_bytes = image_bytes
        st.success("Analysis complete and saved to history.")
    except ValueError as processing_error:
        st.error(f"Could not analyze this image: {processing_error}")
    except sqlite3.Error as db_error:
        st.error(f"Analysis succeeded, but saving to history failed: {db_error}")
    except Exception as unexpected_error:  # noqa: BLE001 -- last-resort UI safety net
        st.error(f"An unexpected error occurred: {unexpected_error}")


def _render_results(active_api_key: str) -> None:
    analysis_result: CrystalAnalysisResult | None = st.session_state.analysis_result
    if analysis_result is None:
        st.info("Upload a diffractogram image and click **Analyze** to begin.")
        return

    peaks_2theta = st.session_state.peaks_2theta
    cached_image_bytes = st.session_state.image_bytes

    col_image, col_metrics = st.columns([1, 1])

    with col_image:
        if cached_image_bytes is not None:
            st.subheader("Uploaded Diffractogram")
            st.image(cached_image_bytes, width='stretch')
        else:
            st.subheader("Manual Peak Input")
            st.caption("No image was processed -- these 2-Theta values were entered directly.")
            st.write(peaks_2theta)

    with col_metrics:
        st.subheader("Analysis Results")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Best Fit (Logged)", analysis_result.best_fit)
        with metric_col2:
            st.metric("Peaks Found", len(peaks_2theta))

        hypothesis_rows = [
            {
                "Structure Type": h.structure,
                "Calculated Lattice Parameter a (Å)": round(h.avg_lattice_parameter, 4),
                "Error Score": round(h.error_score, 4),
            }
            for h in analysis_result.hypotheses
        ]
        st.dataframe(pd.DataFrame(hypothesis_rows), width='stretch', hide_index=True)
        st.caption(
            "Note: A structure is valid if its lattice parameter remains "
            "consistent across multiple peaks. Lower error scores indicate a better fit."
        )

        _render_materials_project_search(analysis_result, active_api_key)

        with st.expander("View detailed numeric results"):
            st.write("**2-Theta peaks (degrees):**", peaks_2theta)
            st.write("**d-spacings:**", analysis_result.d_spacings)
            st.write("**sin² ratios:**", analysis_result.sin2_ratios)
            for h in analysis_result.hypotheses:
                st.write(f"**{h.structure} final integers (common_value={h.common_value}):**", h.final_integers)
                st.write(f"**{h.structure} per-peak lattice parameters (Å):**", [round(a, 4) for a in h.lattice_parameters])


def _render_materials_project_search(analysis_result: CrystalAnalysisResult, active_api_key: str) -> None:
    st.subheader("Materials Project Database Search")
    best_fit_label = analysis_result.best_fit

    if not active_api_key or not active_api_key.strip():
        st.info("Enter a Materials Project API key in the sidebar to search the live database for material matches.")
        return
    if best_fit_label == "Unknown":
        st.info("No confident structure match was found (best fit: 'Unknown'), so there is no lattice parameter to search with.")
        return

    winning_hypothesis = next((h for h in analysis_result.hypotheses if h.structure == best_fit_label), None)
    if winning_hypothesis is None:
        st.warning("Could not locate the winning hypothesis's lattice parameter for the database search.")
        return

    target_a = winning_hypothesis.avg_lattice_parameter
    st.caption(
        "Note: this searches every cubic material in the Materials "
        "Project database (~20,000+ entries) and can take up to a "
        "minute on the first run. Avoid interrupting the app while it runs."
    )
    with st.spinner("Searching live database (this can take up to a minute)..."):
        material_matches = _cached_material_search(active_api_key, round(target_a, 6))

    if len(material_matches) == 1 and "error" in material_matches[0]:
        st.error(material_matches[0]["error"])
    elif not material_matches:
        st.info(f"No materials found within tolerance of a = {target_a:.4f} Å for the {best_fit_label} hypothesis.")
    else:
        matches_df = pd.DataFrame(
            [
                {
                    "Formula": m["Formula"],
                    "Material ID": m["Material ID"],
                    "Theoretical a (Å)": round(m["Theoretical a"], 4),
                    "Error (Å)": round(m["Error"], 4),
                }
                for m in material_matches
            ]
        )
        st.dataframe(matches_df, width='stretch', hide_index=True)


def main() -> None:
    st.set_page_config(layout="wide", page_title="XRD Crystal Analyzer", initial_sidebar_state="collapsed")

    try:
        init_db()
    except sqlite3.Error as db_init_error:
        st.error(f"Failed to initialize the database: {db_init_error}")
        st.stop()

    _init_session_state()
    active_api_key = _render_sidebar()

    hero_left, hero_center, hero_right = st.columns([1, 2, 1])
    with hero_center:
        st.title("XRD Crystal Analyzer")
        st.write(
            "Upload an X-ray diffraction (XRD) pattern image to detect peaks, "
            "compute d-spacings via Bragg's Law, and identify the likely "
            "cubic crystal structure (SC, BCC, or FCC)."
        )
    st.divider()

    manual_mode, analyze_clicked, uploaded_file, manual_peaks_text, axis_start, axis_end = _render_control_bar()
    st.divider()

    if analyze_clicked:
        if manual_mode:
            _run_manual_analysis(manual_peaks_text)
        else:
            _run_image_analysis(uploaded_file, axis_start, axis_end)

    _render_results(active_api_key)


if __name__ == "__main__":
    main()
