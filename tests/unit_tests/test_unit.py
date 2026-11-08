from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_synapcores import (
    SynapCoresClient,
    SynapCoresEmbeddings,
    SynapCoresError,
    SynapCoresVectorStore,
)
from langchain_synapcores._client import safe_identifier, vector_literal
from langchain_synapcores._filters import build_where


# -- filters -----------------------------------------------------------------


def test_filter_equality_and_operators() -> None:
    sql, params = build_where({"topic": "billing", "year": {"$gte": 2024, "$lt": 2027}})
    assert sql == (
        "(json_extract(metadata, '$.topic') = $1 AND "
        "(json_extract(metadata, '$.year') >= $2 AND json_extract(metadata, '$.year') < $3))"
    )
    assert params == ["billing", 2024, 2027]


def test_filter_in_or_and_offsets() -> None:
    sql, params = build_where(
        {"$or": [{"k": {"$in": ["a", "b"]}}, {"n": {"$ne": 3}}]}, params=["existing"]
    )
    assert sql == (
        "(json_extract(metadata, '$.k') IN ($2, $3) OR json_extract(metadata, '$.n') != $4)"
    )
    assert params == ["existing", "a", "b", 3]


def test_filter_nested_key_and_nin() -> None:
    sql, params = build_where({"source.kind": {"$nin": ["x"]}})
    assert sql == "json_extract(metadata, '$.source.kind') NOT IN ($1)"
    assert params == ["x"]


@pytest.mark.parametrize(
    "bad",
    [
        {"a'b": 1},
        {"k": {"$like": "x"}},
        {"$not": {}},
        {"k": {"$in": []}},
        {"k": {"$eq": {"nested": 1}}},
    ],
)
def test_filter_rejects_unsafe_input(bad: Dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        build_where(bad)


def test_empty_filter() -> None:
    assert build_where(None) == ("", [])


# -- literals and identifiers ------------------------------------------------


def test_vector_literal() -> None:
    assert vector_literal([0.5, -1, 2.25]) == "[0.5, -1.0, 2.25]"
    with pytest.raises(ValueError):
        vector_literal([float("nan")])
    with pytest.raises(ValueError):
        vector_literal(["1); DROP TABLE x; --"])  # type: ignore[list-item]


@pytest.mark.parametrize("name", ["docs", "_t1", "Docs_2026"])
def test_identifier_ok(name: str) -> None:
    assert safe_identifier(name) == name


@pytest.mark.parametrize("name", ["", "1abc", "a-b", "a b", "x;drop", "a" * 64])
def test_identifier_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        safe_identifier(name)


# -- fake gateway ------------------------------------------------------------


class FakeGateway:
    """In-memory stand-in for the gateway HTTP API (records SQL, serves canned rows)."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.logins = 0
        self.expire_first = False
        self.rows: List[List[Any]] = []
        self.columns: List[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        if request.url.path == "/v1/auth/login":
            self.logins += 1
            return httpx.Response(200, json={"access_token": f"t{self.logins}"})
        auth = request.headers.get("authorization", "")
        if self.expire_first and auth == "Bearer t1":
            return httpx.Response(401, json={"error": {"code": "expired", "message": "expired"}})
        self.calls.append({"path": request.url.path, "body": body})
        if request.url.path == "/v1/ai/embeddings/batch":
            return httpx.Response(
                200, json={"data": {"embeddings": [[0.1, 0.2, 0.3] for _ in body["texts"]]}}
            )
        sql = body.get("sql", "")
        if sql == "SHOW TABLES":
            return httpx.Response(
                200,
                json={"data": {"columns": [{"name": "table_name"}], "rows": [["docs"]]}},
            )
        if sql.startswith("SELECT"):
            return httpx.Response(
                200,
                json={"data": {"columns": [{"name": c} for c in self.columns], "rows": self.rows}},
            )
        if "bad" in sql:
            return httpx.Response(
                400, json={"error": {"code": "parse_error", "message": "syntax error"}}
            )
        return httpx.Response(200, json={"data": {"columns": [], "rows": [], "rows_affected": 1}})

    def client(self, **kwargs: Any) -> SynapCoresClient:
        http = httpx.Client(base_url="http://gw", transport=httpx.MockTransport(self.handler))
        return SynapCoresClient("http://gw", http_client=http, **kwargs)


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("SYNAPCORES_API_KEY", "SYNAPCORES_TOKEN", "SYNAPCORES_USERNAME", "SYNAPCORES_PASSWORD"):
        monkeypatch.delenv(var, raising=False)


def test_client_login_retry_and_envelope() -> None:
    gw = FakeGateway()
    gw.expire_first = True
    gw.columns, gw.rows = ["x"], [[1]]
    c = gw.client(username="u", password="p")
    assert c.execute("SELECT 1 AS x") == [{"x": 1}]
    assert gw.logins == 2


def test_client_errors_and_missing_credentials() -> None:
    gw = FakeGateway()
    with pytest.raises(SynapCoresError, match="No SynapCores credentials"):
        gw.client().execute("SELECT 1")
    with pytest.raises(SynapCoresError, match="syntax error") as exc:
        gw.client(api_key="ak_test").execute("bad sql")
    assert exc.value.code == "parse_error"


def test_client_sends_database() -> None:
    gw = FakeGateway()
    gw.client(api_key="ak_test").execute("SELECT 1", database="demo")
    assert gw.calls[0]["body"]["database"] == "demo"


def test_embeddings_batches() -> None:
    gw = FakeGateway()
    emb = SynapCoresEmbeddings(client=gw.client(api_key="ak_test"), batch_size=2)
    out = emb.embed_documents(["a", "b", "c"])
    assert len(out) == 3 and len(gw.calls) == 2
    assert emb.embed_query("q") == [0.1, 0.2, 0.3]


def test_vectorstore_writes_parameterized_sql() -> None:
    gw = FakeGateway()
    store = SynapCoresVectorStore(
        DeterministicFakeEmbedding(size=3),
        table_name="docs",
        database="demo",
        client=gw.client(api_key="ak_test"),
    )
    ids = store.add_texts(["it's a test", "two"], metadatas=[{"k": 1}, {}], ids=["1", "2"])
    assert ids == ["1", "2"]
    insert = [c for c in gw.calls if c["body"].get("sql", "").startswith("INSERT")][0]["body"]
    assert "it's a test" not in insert["sql"]  # text is bound, never inlined
    assert insert["parameters"][:3] == ["1", "it's a test", '{"k": 1}']
    assert "ON CONFLICT (id) DO UPDATE" in insert["sql"]
    assert insert["database"] == "demo"
    create = [c for c in gw.calls if "CREATE TABLE" in c["body"].get("sql", "")][0]["body"]["sql"]
    assert "VECTOR(3)" in create


def test_vectorstore_search_parses_rows() -> None:
    gw = FakeGateway()
    gw.columns = ["id", "content", "metadata", "score"]
    gw.rows = [["1", "hello", '{"k": 1}', 0.9], ["2", "bye", None, 0.1]]
    store = SynapCoresVectorStore(
        DeterministicFakeEmbedding(size=3), table_name="docs", client=gw.client(api_key="ak_test")
    )
    res = store.similarity_search_with_score("hi", k=2, filter={"k": 1})
    assert [(d.id, d.page_content, d.metadata, s) for d, s in res] == [
        ("1", "hello", {"k": 1}, 0.9),
        ("2", "bye", {}, 0.1),
    ]
    sql = gw.calls[-1]["body"]["sql"]
    assert "COSINE_SIMILARITY" in sql and "ORDER BY score DESC LIMIT 2" in sql
    assert "json_extract(metadata, '$.k') = $1" in sql
    rel = store.similarity_search_with_relevance_scores("hi", k=2)
    assert rel[0][1] == pytest.approx(0.95)


def test_vectorstore_rejects_bad_names() -> None:
    with pytest.raises(ValueError):
        SynapCoresVectorStore(DeterministicFakeEmbedding(size=3), table_name="x;drop", client=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SynapCoresVectorStore(DeterministicFakeEmbedding(size=3), distance="manhattan", client=object())  # type: ignore[arg-type]
