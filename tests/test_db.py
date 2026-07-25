"""Tests for braggplot.db. Uses a temp DB file so tests never touch
the real xrd_history.db used by the running app."""

from __future__ import annotations

import braggplot.config as config
import braggplot.db as db


def test_init_and_insert_and_read(tmp_path, monkeypatch):
    test_db_path = str(tmp_path / "test_history.db")
    monkeypatch.setattr(db, "DB_FILENAME", test_db_path)

    db.init_db()
    db.init_db()  # must be idempotent -- calling twice should not raise

    assert db.get_history().empty

    db.insert_calculation(
        db.CalculationRecord(
            common_value=2, ratios="[1,2,3]", structure="BCC", peaks="[44.6,65.0]"
        )
    )

    history = db.get_history()
    assert len(history) == 1
    assert history.iloc[0]["structure"] == "BCC"
