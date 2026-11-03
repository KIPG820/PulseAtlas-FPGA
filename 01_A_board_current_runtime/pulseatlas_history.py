"""Persistent, bounded-memory history for PulseAtlas ECG sessions.

The DMA receive thread only enqueues small copies of completed packet blocks.
Compression and SQLite writes happen on a dedicated worker so SD-card I/O
cannot delay ECG capture or the HDMI/TFT refresh loops.
"""

from __future__ import annotations

from datetime import datetime
from contextlib import contextmanager
from io import BytesIO
import json
import os
from pathlib import Path
import queue
import sqlite3
import threading
import time

import numpy as np


def default_history_path():
    override = os.environ.get("PULSEATLAS_HISTORY_DB")
    if override:
        return Path(override).expanduser()
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home:
        return Path(data_home).expanduser() / "pulseatlas" / "history.sqlite3"
    pynq_home = Path("/home/xilinx/.local/share/pulseatlas/history.sqlite3")
    if Path("/home/xilinx").is_dir():
        return pynq_home
    return Path.home() / ".local" / "share" / "pulseatlas" / "history.sqlite3"


class PulseAtlasHistoryStore:
    """SQLite session store with compressed per-DMA-block ECG samples."""

    def __init__(self, path=None, queue_size=16):
        self.path = Path(path or default_history_path()).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_sessions = max(1, int(os.environ.get("PULSEATLAS_HISTORY_MAX_SESSIONS", "30")))
        self.max_session_seconds = max(
            10, int(os.environ.get("PULSEATLAS_HISTORY_MAX_SESSION_SECONDS", "60")))
        self._queue = queue.Queue(maxsize=max(4, int(queue_size)))
        self._drops = {}
        self._drops_lock = threading.Lock()
        self._closed = False
        self.last_error = None
        self._initialize()
        self._worker = threading.Thread(
            target=self._writer_loop, name="pulseatlas-history-writer", daemon=True)
        self._worker.start()

    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=5.0)
        db.execute("PRAGMA busy_timeout=5000")
        db.execute("PRAGMA foreign_keys=ON")
        return db

    @contextmanager
    def _db(self):
        """Commit/rollback and always close short-lived SQLite connections."""
        db = self._connect()
        try:
            with db:
                yield db
        finally:
            db.close()

    def _initialize(self):
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    sample_rate REAL NOT NULL,
                    lead_count INTEGER NOT NULL DEFAULT 12,
                    sample_count INTEGER NOT NULL DEFAULT 0,
                    stored_sample_count INTEGER NOT NULL DEFAULT 0,
                    quality_bad_frames INTEGER NOT NULL DEFAULT 0,
                    heart_rate_sum REAL NOT NULL DEFAULT 0,
                    heart_rate_count INTEGER NOT NULL DEFAULT 0,
                    avg_heart_rate REAL,
                    dropped_chunks INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'recording',
                    result_label TEXT,
                    result_confidence REAL,
                    result_backend TEXT,
                    probabilities_json TEXT
                );
                CREATE TABLE IF NOT EXISTS session_chunks (
                    session_id INTEGER NOT NULL REFERENCES sessions(session_id),
                    chunk_index INTEGER NOT NULL,
                    first_frame INTEGER NOT NULL,
                    frame_count INTEGER NOT NULL,
                    quality_bad_frames INTEGER NOT NULL,
                    payload BLOB NOT NULL,
                    PRIMARY KEY (session_id, chunk_index)
                );
                CREATE TABLE IF NOT EXISTS inference_results (
                    inference_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL REFERENCES sessions(session_id),
                    created_at TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    result_label TEXT,
                    confidence REAL,
                    probabilities_json TEXT NOT NULL,
                    elapsed_seconds REAL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_started
                    ON sessions(session_id DESC);
                CREATE INDEX IF NOT EXISTS idx_inference_session
                    ON inference_results(session_id, inference_id DESC);
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(sessions)")}
            if "stored_sample_count" not in columns:
                db.execute("ALTER TABLE sessions ADD COLUMN stored_sample_count "
                           "INTEGER NOT NULL DEFAULT 0")
                db.execute("UPDATE sessions SET stored_sample_count="
                           "(SELECT COALESCE(SUM(frame_count),0) FROM session_chunks "
                           "WHERE session_chunks.session_id=sessions.session_id)")
            # A board reset can interrupt capture before finish_session is
            # queued. Never present such a stale session as currently running.
            db.execute("UPDATE sessions SET ended_at=COALESCE(ended_at,?), status='incomplete' "
                       "WHERE status='recording'", (self._now_text(),))
            self._prune_old_sessions(db)

    @staticmethod
    def _now_text():
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def start_session(self, sample_rate, lead_count=12):
        """Create and return a durable session id before capture begins."""
        with self._db() as db:
            cursor = db.execute(
                "INSERT INTO sessions(started_at,sample_rate,lead_count,status) "
                "VALUES(?,?,?,'recording')",
                (self._now_text(), float(sample_rate), int(lead_count)))
            return int(cursor.lastrowid)

    def append_samples(self, session_id, leads, frames, quality, rr_values=()):
        """Queue one completed packet block without doing disk work inline."""
        if session_id is None or self._closed:
            return False
        leads = np.asarray(leads, dtype=np.int32).reshape(-1, 12).copy()
        frames = np.asarray(frames, dtype=np.uint32).reshape(-1).copy()
        quality = np.asarray(quality, dtype=np.uint8).reshape(-1, 5).copy()
        if not len(leads):
            return True
        if len(leads) != len(frames) or len(frames) != len(quality):
            raise ValueError("history sample/frame/quality counts do not match")
        rr = np.asarray(rr_values, dtype=np.uint32).reshape(-1, 2).copy()
        item = ("chunk", int(session_id), leads, frames, quality, rr)
        try:
            self._queue.put(item, timeout=0.02)
            return True
        except queue.Full:
            with self._drops_lock:
                self._drops[int(session_id)] = self._drops.get(int(session_id), 0) + 1
            return False

    def finish_session(self, session_id, status="complete", timeout=10.0):
        if session_id is None or self._closed:
            return
        event = threading.Event()
        self._queue.put(("finish", int(session_id), str(status), event),
                        timeout=max(0.1, float(timeout)))
        if not event.wait(timeout):
            raise RuntimeError("history writer did not finish session in time")

    def record_inference(self, session_id, backend, label, confidence,
                         probabilities, elapsed_seconds=None, timeout=5.0):
        if session_id is None or self._closed:
            return
        event = threading.Event()
        payload = ("inference", int(session_id), str(backend),
                   None if label is None else str(label),
                   None if confidence is None else float(confidence),
                   dict(probabilities or {}),
                   None if elapsed_seconds is None else float(elapsed_seconds),
                   event)
        self._queue.put(payload, timeout=max(0.1, float(timeout)))
        if not event.wait(timeout):
            raise RuntimeError("history writer did not save inference result in time")

    def list_recent(self, limit=4):
        limit = max(1, min(32, int(limit)))
        with self._db() as db:
            rows = db.execute("""
                SELECT session_id, started_at, ended_at, sample_rate, lead_count,
                       sample_count, stored_sample_count, quality_bad_frames, avg_heart_rate,
                       dropped_chunks, status, result_label, result_confidence,
                       result_backend
                FROM sessions ORDER BY session_id DESC LIMIT ?
            """, (limit,)).fetchall()
        keys = ("session_id", "started_at", "ended_at", "sample_rate",
                "lead_count", "sample_count", "stored_sample_count", "quality_bad_frames",
                "avg_heart_rate", "dropped_chunks", "status", "result_label",
                "result_confidence", "result_backend")
        return [dict(zip(keys, row)) for row in rows]

    def get_detail(self, session_id, lead_index=1, preview_points=80):
        if session_id is None:
            return None
        with self._db() as db:
            row = db.execute("""
                SELECT session_id, started_at, ended_at, sample_rate, lead_count,
                       sample_count, stored_sample_count, quality_bad_frames, avg_heart_rate,
                       dropped_chunks, status, result_label, result_confidence,
                       result_backend, probabilities_json
                FROM sessions WHERE session_id=?
            """, (int(session_id),)).fetchone()
            chunks = db.execute("""
                SELECT payload FROM session_chunks WHERE session_id=?
                ORDER BY chunk_index DESC
            """, (int(session_id),)).fetchall()
        if row is None:
            return None
        keys = ("session_id", "started_at", "ended_at", "sample_rate",
                "lead_count", "sample_count", "stored_sample_count", "quality_bad_frames",
                "avg_heart_rate", "dropped_chunks", "status", "result_label",
                "result_confidence", "result_backend", "probabilities_json")
        detail = dict(zip(keys, row))
        lead_index = max(0, min(11, int(lead_index)))
        arrays = []
        for (blob,) in reversed(chunks):
            try:
                with np.load(BytesIO(blob), allow_pickle=False) as saved:
                    arrays.append(np.asarray(saved["leads"][:, lead_index], dtype=np.int32))
            except (OSError, ValueError, KeyError):
                continue
        signal = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int32)
        count = max(2, int(preview_points))
        if signal.size > count:
            indices = np.linspace(0, signal.size - 1, count).astype(np.int64)
            signal = signal[indices]
        detail["preview"] = signal.astype(int).tolist()
        return detail

    @staticmethod
    def _encode_chunk(leads, frames, quality, rr):
        buffer = BytesIO()
        np.savez_compressed(buffer, leads=leads, frames=frames,
                            quality=quality, rr=rr)
        return buffer.getvalue()

    def _save_chunk(self, db, session_id, leads, frames, quality, rr):
        with db:
            session = db.execute(
                "SELECT sample_rate,stored_sample_count FROM sessions WHERE session_id=?",
                (session_id,)).fetchone()
            if session is None:
                return
            sample_rate, stored_total = float(session[0]), int(session[1])
            chunk_index = db.execute(
                "SELECT COALESCE(MAX(chunk_index),-1)+1 FROM session_chunks "
                "WHERE session_id=?", (session_id,)).fetchone()[0]
            bad = int(np.count_nonzero(np.any(quality != 0, axis=1)))
            payload = self._encode_chunk(leads, frames, quality, rr)
            db.execute("""
                INSERT INTO session_chunks(session_id,chunk_index,first_frame,
                    frame_count,quality_bad_frames,payload) VALUES(?,?,?,?,?,?)
            """, (session_id, int(chunk_index), int(frames[0]),
                  int(frames.size), bad, payload))
            stored_total += int(frames.size)
            keep_frames = max(1, int(sample_rate * self.max_session_seconds))
            # Keep a rolling, bounded waveform window per session. Summary
            # counts below continue to describe the entire capture.
            while stored_total > keep_frames:
                oldest = db.execute(
                    "SELECT chunk_index,frame_count FROM session_chunks "
                    "WHERE session_id=? ORDER BY chunk_index LIMIT 1",
                    (session_id,)).fetchone()
                if oldest is None:
                    break
                db.execute("DELETE FROM session_chunks WHERE session_id=? AND chunk_index=?",
                           (session_id, int(oldest[0])))
                stored_total -= int(oldest[1])
            valid_rr = rr[:, 1] if rr.size else np.empty(0, dtype=np.uint32)
            valid_rr = valid_rr[valid_rr > 0]
            heart_rates = (60.0 * sample_rate / valid_rr.astype(np.float64)
                           if valid_rr.size else np.empty(0, dtype=np.float64))
            heart_rates = heart_rates[(heart_rates >= 20.0) & (heart_rates <= 300.0)]
            hr_sum = float(np.sum(heart_rates))
            hr_count = int(heart_rates.size)
            db.execute("""
                UPDATE sessions SET sample_count=sample_count+?,
                    stored_sample_count=?,
                    quality_bad_frames=quality_bad_frames+?,
                    heart_rate_sum=heart_rate_sum+?,
                    heart_rate_count=heart_rate_count+?,
                    avg_heart_rate=CASE WHEN heart_rate_count+?>0
                        THEN (heart_rate_sum+?)/(heart_rate_count+?)
                        ELSE avg_heart_rate END
                WHERE session_id=?
            """, (int(frames.size), stored_total, bad, hr_sum, hr_count, hr_count,
                  hr_sum, hr_count, session_id))

    def _prune_old_sessions(self, db):
        expired = [int(row[0]) for row in db.execute(
            "SELECT session_id FROM sessions WHERE status!='recording' "
            "ORDER BY session_id DESC LIMIT -1 OFFSET ?", (self.max_sessions,))]
        if not expired:
            return
        placeholders = ",".join("?" for _ in expired)
        db.execute("DELETE FROM inference_results WHERE session_id IN (" + placeholders + ")", expired)
        db.execute("DELETE FROM session_chunks WHERE session_id IN (" + placeholders + ")", expired)
        db.execute("DELETE FROM sessions WHERE session_id IN (" + placeholders + ")", expired)

    def _writer_loop(self):
        db = None
        try:
            db = self._connect()
            while True:
                item = self._queue.get()
                try:
                    kind = item[0]
                    if kind == "close":
                        return
                    if kind == "chunk":
                        _, session_id, leads, frames, quality, rr = item
                        self._save_chunk(db, session_id, leads, frames, quality, rr)
                    elif kind == "finish":
                        _, session_id, status, event = item
                        with self._drops_lock:
                            dropped = self._drops.pop(session_id, 0)
                        final_status = "incomplete" if dropped else status
                        with db:
                            db.execute("""
                                UPDATE sessions SET ended_at=?, status=?, dropped_chunks=?
                                WHERE session_id=?
                            """, (self._now_text(), final_status, dropped, session_id))
                            self._prune_old_sessions(db)
                        event.set()
                    elif kind == "inference":
                        (_, session_id, backend, label, confidence,
                         probabilities, elapsed, event) = item
                        encoded = json.dumps(
                            probabilities, sort_keys=True, separators=(",", ":"))
                        with db:
                            db.execute("""
                                INSERT INTO inference_results(session_id,created_at,
                                    backend,result_label,confidence,probabilities_json,
                                    elapsed_seconds) VALUES(?,?,?,?,?,?,?)
                            """, (session_id, self._now_text(), backend, label,
                                  confidence, encoded, elapsed))
                            db.execute("""
                                UPDATE sessions SET result_label=?,result_confidence=?,
                                    result_backend=?,probabilities_json=? WHERE session_id=?
                            """, (label, confidence, backend, encoded, session_id))
                        event.set()
                except Exception as exc:
                    self.last_error = str(exc)
                    if item[0] == "chunk":
                        with self._drops_lock:
                            session_id = int(item[1])
                            self._drops[session_id] = self._drops.get(session_id, 0) + 1
                    if item[0] in ("finish", "inference"):
                        item[-1].set()
                finally:
                    self._queue.task_done()
        finally:
            if db is not None:
                db.close()

    def close(self, timeout=5.0):
        if self._closed:
            return
        self._closed = True
        self._queue.put(("close",), timeout=max(0.1, float(timeout)))
        self._worker.join(timeout=max(0.1, float(timeout)))


__all__ = ["PulseAtlasHistoryStore", "default_history_path"]
