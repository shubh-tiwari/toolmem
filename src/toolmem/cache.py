"""SQLite cache so embeddings and LLM augmentations are computed once per tool version.

One connection is shared across threads (augmentation and the benchmark run in thread pools), so
every statement runs under a lock: concurrent reads and writes on one sqlite3 connection can crash.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

import numpy as np


def _key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


class SQLiteCache:
    def __init__(self, path: str | Path = ".toolmem_cache.sqlite"):
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS embeddings (k TEXT PRIMARY KEY, v BLOB)")
        self._conn.execute("CREATE TABLE IF NOT EXISTS texts (k TEXT PRIMARY KEY, v TEXT)")
        self._conn.commit()

    def get_embedding(self, model: str, text: str) -> np.ndarray | None:
        with self._lock:
            row = self._conn.execute("SELECT v FROM embeddings WHERE k=?", (_key(model, text),)).fetchone()
        return np.frombuffer(row[0], dtype=np.float32).copy() if row else None

    def set_embedding(self, model: str, text: str, vec: np.ndarray) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
                               (_key(model, text), np.asarray(vec, dtype=np.float32).tobytes()))
            self._conn.commit()

    def get_text(self, *parts: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT v FROM texts WHERE k=?", (_key(*parts),)).fetchone()
        return row[0] if row else None

    def set_text(self, value: str, *parts: str) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO texts VALUES (?, ?)", (_key(*parts), value))
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()
