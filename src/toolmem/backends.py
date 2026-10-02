"""Vector index backends: exact in-memory search, plus Chroma, LanceDB and pgvector adapters.

Implement ``VectorBackend`` to plug in another store. Scores are cosine similarity everywhere."""
from __future__ import annotations

import re
from typing import Protocol, Sequence

import numpy as np


class VectorBackend(Protocol):
    def upsert(self, ids: Sequence[str], vectors: np.ndarray) -> None: ...
    def delete(self, ids: Sequence[str]) -> None: ...
    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]: ...
    def __contains__(self, id: str) -> bool: ...
    def __len__(self) -> int: ...


class InMemoryBackend:
    """Exact cosine search with numpy. Fast enough for tens of thousands of tools."""

    def __init__(self):
        self._vecs: dict[str, np.ndarray] = {}
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None

    def upsert(self, ids, vectors):
        for i, v in zip(ids, np.asarray(vectors, dtype=np.float32)):
            self._vecs[i] = v
        self._matrix = None

    def delete(self, ids):
        for i in ids:
            self._vecs.pop(i, None)
        self._matrix = None

    def search(self, vector, k):
        if not self._vecs:
            return []
        if self._matrix is None:
            self._ids = list(self._vecs)
            self._matrix = np.stack([self._vecs[i] for i in self._ids])
        scores = self._matrix @ np.asarray(vector, dtype=np.float32).ravel()
        k = min(k, len(self._ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[i], float(scores[i])) for i in top]

    def __contains__(self, id):
        return id in self._vecs

    def __len__(self):
        return len(self._vecs)


def _missing(lib: str, extra: str) -> ImportError:
    return ImportError(f"{lib} is not installed; run `pip install -e '.[{extra}]'` in your toolmem checkout")


class ChromaBackend:
    """Chroma collection with cosine distance. Pass a ``client``, a ``path`` for a persistent
    store, or neither for an in-process ephemeral one."""

    def __init__(self, collection: str = "toolmem", path: str | None = None, client=None):
        try:
            import chromadb
        except ImportError as e:
            raise _missing("chromadb", "chroma") from e
        if client is None:
            client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        self._col = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def upsert(self, ids, vectors):
        if len(ids):
            self._col.upsert(ids=list(ids), embeddings=np.asarray(vectors, dtype=np.float32).tolist())

    def delete(self, ids):
        if len(ids):
            self._col.delete(ids=list(ids))

    def search(self, vector, k):
        n = len(self)
        if not n:
            return []
        res = self._col.query(query_embeddings=[np.asarray(vector, dtype=np.float32).ravel().tolist()],
                              n_results=min(k, n), include=["distances"])
        return [(i, 1.0 - float(d)) for i, d in zip(res["ids"][0], res["distances"][0])]

    def __contains__(self, id):
        return bool(self._col.get(ids=[id], include=[])["ids"])

    def __len__(self):
        return self._col.count()


class LanceDBBackend:
    """LanceDB table (local directory or any LanceDB URI), created on the first upsert."""

    def __init__(self, uri: str = ".toolmem_lancedb", table: str = "toolmem", db=None):
        try:
            import lancedb
        except ImportError as e:
            raise _missing("lancedb", "lancedb") from e
        self._db = db or lancedb.connect(uri)
        self._name = table
        try:  # table_names() is deprecated in newer lancedb, so probe by opening
            self._table = self._db.open_table(table)
        except (ValueError, FileNotFoundError):
            self._table = None

    @staticmethod
    def _quote(ids) -> str:
        return ", ".join("'" + str(i).replace("'", "''") + "'" for i in ids)

    def upsert(self, ids, vectors):
        rows = [{"id": i, "vector": v.tolist()} for i, v in zip(ids, np.asarray(vectors, dtype=np.float32))]
        if not rows:
            return
        if self._table is None:
            self._table = self._db.create_table(self._name, data=rows)
        else:
            (self._table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(rows))

    def delete(self, ids):
        if self._table is not None and len(ids):
            self._table.delete(f"id IN ({self._quote(ids)})")

    def search(self, vector, k):
        n = len(self)
        if not n:
            return []
        q = self._table.search(np.asarray(vector, dtype=np.float32).ravel().tolist())
        q = q.distance_type("cosine") if hasattr(q, "distance_type") else q.metric("cosine")  # older lancedb
        rows = q.limit(min(k, n)).to_list()
        return [(r["id"], 1.0 - float(r["_distance"])) for r in rows]

    def __contains__(self, id):
        return self._table is not None and self._table.count_rows(f"id = {self._quote([id])}") > 0

    def __len__(self):
        return self._table.count_rows() if self._table is not None else 0


class PgVectorBackend:
    """Postgres with the pgvector extension. Pass a psycopg 3 ``conn`` or a ``dsn``; the table
    is created on the first upsert, when the vector dimension is known."""

    def __init__(self, dsn: str | None = None, table: str = "toolmem_vectors", conn=None):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", table):
            raise ValueError(f"invalid table name {table!r}")
        if conn is None:
            try:
                import psycopg
            except ImportError as e:
                raise _missing("psycopg", "pgvector") from e
            conn = psycopg.connect(dsn, autocommit=True)
        self._conn, self._table = conn, table
        self._ready = self._exists()

    def _sql(self, sql: str, params=(), fetch: bool = False):
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() if fetch else None
        if not getattr(self._conn, "autocommit", True):
            self._conn.commit()
        return rows

    def _exists(self) -> bool:
        return self._sql("SELECT to_regclass(%s) IS NOT NULL", (self._table,), fetch=True)[0][0]

    @staticmethod
    def _vec(v) -> str:
        return "[" + ",".join(f"{x:.8g}" for x in np.asarray(v, dtype=np.float32).ravel()) + "]"

    def upsert(self, ids, vectors):
        vectors = np.asarray(vectors, dtype=np.float32)
        if not len(ids):
            return
        if not self._ready:
            self._sql("CREATE EXTENSION IF NOT EXISTS vector")
            self._sql(f"CREATE TABLE IF NOT EXISTS {self._table} (id TEXT PRIMARY KEY, embedding vector({vectors.shape[1]}))")
            self._ready = True
        for i, v in zip(ids, vectors):
            self._sql(f"INSERT INTO {self._table} (id, embedding) VALUES (%s, %s::vector) "
                      "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding", (i, self._vec(v)))

    def delete(self, ids):
        if self._ready and len(ids):
            self._sql(f"DELETE FROM {self._table} WHERE id = ANY(%s)", (list(ids),))

    def search(self, vector, k):
        if not self._ready:
            return []
        rows = self._sql(f"SELECT id, 1 - (embedding <=> %s::vector) FROM {self._table} "
                         "ORDER BY embedding <=> %s::vector LIMIT %s",
                         (self._vec(vector), self._vec(vector), int(k)), fetch=True)
        return [(i, float(s)) for i, s in rows]

    def __contains__(self, id):
        return self._ready and bool(self._sql(f"SELECT 1 FROM {self._table} WHERE id = %s", (id,), fetch=True))

    def __len__(self):
        return self._sql(f"SELECT count(*) FROM {self._table}", fetch=True)[0][0] if self._ready else 0
