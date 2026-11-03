"""End-to-end checks for real ECG SQLite history persistence."""

from pathlib import Path
import sqlite3
import sys
import tempfile
from contextlib import closing

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pulseatlas_history import PulseAtlasHistoryStore  # noqa: E402


def main():
    with tempfile.TemporaryDirectory(prefix="pulseatlas-history-") as temp:
        path = Path(temp) / "history.sqlite3"
        store = PulseAtlasHistoryStore(path=path, queue_size=4)
        session_id = store.start_session(sample_rate=1000, lead_count=12)
        leads = np.arange(240 * 12, dtype=np.int32).reshape(240, 12)
        frames = np.arange(1000, 1240, dtype=np.uint32)
        quality = np.zeros((240, 5), dtype=np.uint8)
        quality[17, 0] = 1
        rr = np.asarray([[1100, 1000], [1200, 1100]], dtype=np.uint32)
        assert store.append_samples(session_id, leads, frames, quality, rr)
        store.finish_session(session_id, status="complete")
        store.record_inference(
            session_id, "ps", "NORM", 0.87,
            {"NORM": 0.87, "MI": 0.05, "STTC": 0.03, "CD": 0.03, "HYP": 0.02},
            elapsed_seconds=0.42)
        store.close()

        reopened = PulseAtlasHistoryStore(path=path, queue_size=4)
        rows = reopened.list_recent(4)
        assert len(rows) == 1 and rows[0]["session_id"] == session_id
        row = rows[0]
        assert row["status"] == "complete"
        assert row["sample_count"] == 240 and row["stored_sample_count"] == 240
        assert row["quality_bad_frames"] == 1
        assert row["result_label"] == "NORM" and row["result_backend"] == "ps"
        detail = reopened.get_detail(session_id, lead_index=1, preview_points=40)
        assert detail["preview"] == leads[np.linspace(0, 239, 40).astype(np.int64), 1].tolist()
        assert detail["stored_sample_count"] == 240
        reopened.close()

        with closing(sqlite3.connect(path)) as db, db:
            result_count = db.execute(
                "SELECT COUNT(*) FROM inference_results WHERE session_id=?",
                (session_id,)).fetchone()[0]
            chunk_count = db.execute(
                "SELECT COUNT(*) FROM session_chunks WHERE session_id=?",
                (session_id,)).fetchone()[0]
        assert result_count == 1 and chunk_count == 1

        crashed = PulseAtlasHistoryStore(path=path, queue_size=4)
        interrupted_id = crashed.start_session(sample_rate=1000, lead_count=12)
        crashed.close()  # simulate process exit before finish_session
        recovered = PulseAtlasHistoryStore(path=path, queue_size=4)
        recovered_rows = {r["session_id"]: r for r in recovered.list_recent(4)}
        assert recovered_rows[interrupted_id]["status"] == "incomplete"
        assert recovered_rows[interrupted_id]["ended_at"]
        recovered.close()
    print("real SQLite history persistence and recovery passed")


if __name__ == "__main__":
    main()
