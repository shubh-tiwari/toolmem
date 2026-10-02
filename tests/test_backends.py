"""Vector backend tests: the same contract for every store, plus retrieval through ToolRegistry.

Chroma and LanceDB run for real when installed. pgvector runs against a recording fake
connection, and for real when TOOLMEM_TEST_PG_DSN points at a Postgres with the extension.
"""
import os
import re

import numpy as np
import pytest

from toolmem import HashingEmbedder, InMemoryBackend, Tool, ToolRegistry
from toolmem.backends import PgVectorBackend


def unit(*xs):
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def check_contract(b):
    assert len(b) == 0 and b.search(unit(1, 0, 0), 5) == []
    b.upsert(["a", "b", "c"], np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(1, 1, 0)]))
    assert len(b) == 3 and "a" in b and "zz" not in b
    hits = b.search(unit(1, 0, 0), 25)  # k larger than the store
    assert [h[0] for h in hits] == ["a", "c", "b"]
    assert hits[0][1] == pytest.approx(1.0, abs=1e-4) and hits[1][1] == pytest.approx(0.7071, abs=1e-3)
    b.upsert(["b"], unit(1, 0, 0)[None])  # update in place
    assert len(b) == 3 and b.search(unit(1, 0, 0), 2)[1][0] in {"a", "b"}
    b.delete(["a", "c"])
    assert len(b) == 1 and "a" not in b and [h[0] for h in b.search(unit(1, 0, 0), 5)] == ["b"]


def check_registry(backend):
    reg = ToolRegistry(embedder=HashingEmbedder(), backend=backend)
    assert reg.backend is backend  # an empty backend has len 0 and must not be swapped for the default
    reg.add_many([Tool("get_weather", "Current weather for a city"),
                  Tool("send_email", "Send an email message to a recipient"),
                  Tool("convert_currency", "Convert an amount between currencies")])
    assert reg.search("what is the weather in Paris", k=1, mode="semantic")[0].name == "get_weather"
    assert len(backend) == 3
    reg.remove("get_weather")
    assert "get_weather" not in [r.name for r in reg.search("weather", k=5, mode="semantic")]


def test_inmemory_contract():
    check_contract(InMemoryBackend())


def test_chroma(tmp_path):
    pytest.importorskip("chromadb")
    from toolmem.backends import ChromaBackend
    check_contract(ChromaBackend(collection="contract"))
    check_registry(ChromaBackend(collection="registry"))
    persistent = ChromaBackend(path=str(tmp_path / "chroma"))
    persistent.upsert(["x"], unit(1, 2, 3)[None])
    assert "x" in ChromaBackend(path=str(tmp_path / "chroma"))  # survives reopening


def test_lancedb(tmp_path):
    pytest.importorskip("lancedb")
    from toolmem.backends import LanceDBBackend
    check_contract(LanceDBBackend(uri=str(tmp_path / "db"), table="contract"))
    check_registry(LanceDBBackend(uri=str(tmp_path / "db"), table="registry"))
    b = LanceDBBackend(uri=str(tmp_path / "db"), table="quotes")
    b.upsert(["it's"], unit(1, 0)[None])
    assert "it's" in b and "x' OR '1'='1" not in b  # ids are quoted, not injected
    assert "it's" in LanceDBBackend(uri=str(tmp_path / "db"), table="quotes")


class FakeCursor:
    def __init__(self, conn):
        self.conn, self.rows = conn, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        self.conn.log.append((sql, params))
        self.rows = self.conn.respond(sql, params)

    def fetchall(self):
        return self.rows


class FakePg:
    """Just enough of Postgres + pgvector to check the SQL the backend sends."""

    autocommit = True

    def __init__(self):
        self.log, self.table, self.data = [], None, {}

    def cursor(self):
        return FakeCursor(self)

    def respond(self, sql, params):
        vec = lambda s: np.array([float(x) for x in s.strip("[]").split(",")])
        if sql.startswith("SELECT to_regclass"):
            return [(self.table is not None,)]
        if sql.startswith("CREATE TABLE"):
            self.table = re.search(r"TABLE IF NOT EXISTS (\w+)", sql).group(1)
        elif sql.startswith("INSERT"):
            self.data[params[0]] = vec(params[1])
        elif sql.startswith("DELETE"):
            for i in params[0]:
                self.data.pop(i, None)
        elif sql.startswith("SELECT id, 1 -"):
            q = vec(params[0])
            sims = [(i, float(v @ q / np.linalg.norm(v) / np.linalg.norm(q))) for i, v in self.data.items()]
            return sorted(sims, key=lambda x: -x[1])[:params[2]]
        elif sql.startswith("SELECT 1"):
            return [(1,)] if params[0] in self.data else []
        elif sql.startswith("SELECT count"):
            return [(len(self.data),)]
        return []


def test_pgvector_with_fake_connection():
    conn = FakePg()
    check_contract(PgVectorBackend(conn=conn, table="tools"))
    sqls = [s for s, _ in conn.log]
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sqls
    assert any("embedding vector(3)" in s for s in sqls)
    assert any("ON CONFLICT (id) DO UPDATE" in s for s in sqls)
    assert any("ORDER BY embedding <=> %s::vector" in s for s in sqls)
    check_registry(PgVectorBackend(conn=FakePg()))


def test_pgvector_rejects_unsafe_table_name():
    with pytest.raises(ValueError):
        PgVectorBackend(conn=FakePg(), table="tools; DROP TABLE users")


@pytest.mark.skipif(not os.environ.get("TOOLMEM_TEST_PG_DSN"), reason="set TOOLMEM_TEST_PG_DSN to test real pgvector")
def test_pgvector_real():
    pytest.importorskip("psycopg")
    b = PgVectorBackend(dsn=os.environ["TOOLMEM_TEST_PG_DSN"], table="toolmem_test_vectors")
    b._sql("DROP TABLE IF EXISTS toolmem_test_vectors")
    b._ready = False
    check_contract(b)
    b._sql("DROP TABLE toolmem_test_vectors")
    b._ready = False
    check_registry(b)
    b._sql("DROP TABLE toolmem_test_vectors")
