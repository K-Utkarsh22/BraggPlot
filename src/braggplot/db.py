"""
SQLite persistence for calculation history.

All database access lives here. Previously, `init_db`/`insert_calculation`
were defined in app.py but the History expander in the Streamlit UI ran
its own separate `sqlite3.connect(...)` + `pd.read_sql_query(...)`
directly — meaning connection-handling logic was duplicated across two
places that had to be kept in sync by hand. Everything is centralized
here now, including a new `get_history()` used by the UI.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from braggplot.config import DB_FILENAME


@dataclass
class CalculationRecord:
    """One row to persist to the `history` table."""

    common_value: int
    ratios: str  # JSON-serialized list, e.g. sin2_ratios
    structure: str  # 'SC', 'BCC', 'FCC', or 'Unknown'
    peaks: str  # JSON-serialized list of 2-Theta peaks


def init_db() -> None:
    """Create the `history` table if it doesn't already exist.

    Idempotent (CREATE TABLE IF NOT EXISTS): safe to call on every app
    startup without disturbing rows saved in previous runs.

    Raises:
        sqlite3.Error: if the connection or CREATE TABLE fails.
    """
    connection = sqlite3.connect(DB_FILENAME)
    try:
        connection.execute(
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
        connection.commit()
    finally:
        connection.close()


def insert_calculation(record: CalculationRecord) -> None:
    """Insert one analysis result into the `history` table, timestamped now.

    Assumes `init_db()` has already been called earlier in the process
    (it does not create the table itself). Uses a parameterized query
    (the "?" placeholders) rather than an f-string, to avoid SQL
    injection and let sqlite3 handle type adaptation correctly.

    Raises:
        sqlite3.Error: if the connection fails, the table doesn't
            exist, or the INSERT fails for any other reason.
    """
    current_timestamp = datetime.now().isoformat()

    connection = sqlite3.connect(DB_FILENAME)
    try:
        connection.execute(
            """
            INSERT INTO history (timestamp, common_value, ratios, structure, peaks)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                current_timestamp,
                record.common_value,
                record.ratios,
                record.structure,
                record.peaks,
            ),
        )
        connection.commit()
    finally:
        connection.close()


def get_history() -> pd.DataFrame:
    """Return all saved calculations, most recent first.

    Returns an empty DataFrame (not None, not an exception) if the
    table is empty — callers can check `.empty` uniformly instead of
    branching on None vs. DataFrame.

    Raises:
        sqlite3.Error: if the connection fails or the table is missing.
    """
    connection = sqlite3.connect(DB_FILENAME)
    try:
        return pd.read_sql_query(
            "SELECT * FROM history ORDER BY timestamp DESC", connection
        )
    finally:
        connection.close()
