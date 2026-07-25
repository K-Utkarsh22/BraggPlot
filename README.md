# BraggPlot — XRD Crystal Structure Analyzer

Upload an X-ray diffraction (XRD) diffractogram image, extract peak
positions, compute d-spacings via Bragg's Law, and evaluate SC / BCC /
FCC cubic Bravais lattice hypotheses — with an optional live
cross-check against the Materials Project database.

## Project layout

```
src/braggplot/
├── config.py          # all constants/defaults in one place
├── crystallography.py # Bragg's Law + cubic lattice indexing (pure math)
├── peak_extraction.py # image -> 2-Theta peaks (OpenCV + scipy)
├── db.py              # SQLite history persistence
├── materials_api.py   # Materials Project live lookup
└── app.py              # Streamlit UI (orchestration only)
tests/                  # pytest unit tests for each module above
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run

```bash
streamlit run src/braggplot/app.py
```

## Test

```bash
pytest
```
