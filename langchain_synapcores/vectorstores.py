"""SynapCores vector store for LangChain."""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_synapcores._client import (
    SynapCoresClient,
    SynapCoresError,
    safe_identifier,
    vector_literal,
)
from langchain_synapcores._filters import build_where

_DISTANCES = ("cosine", "euclidean")


class SynapCoresVectorStore(VectorStore):
    """Vector store backed by a `SynapCores <https://synapcores.com>`_ table.

    Documents live in an ordinary SQL table with a ``VECTOR(N)`` column, so the
    same rows are available to SQL, joins, graph queries and AutoML in SynapCores.

    Setup:
        Run SynapCores (``curl -fsSL https://get.synapcores.com | sh`` or the
        ``synapcores/community`` Docker image), then install the package:

        .. code-block:: bash

            pip install -U langchain-synapcores

    Key init args:
        embedding: Embeddings
            Any LangChain embeddings model, for example ``SynapCoresEmbeddings``
            to embed inside the database.
        table_name: str
            Table that stores the documents. Created on first write.
        database: Optional[str]
            SynapCores database. Defaults to the gateway's default database.
        distance: str
            ``"cosine"`` (default) or ``"euclidean"``.

    Client args:
        url: Optional[str]
            Gateway URL. Defaults to env ``SYNAPCORES_URL`` or ``http://127.0.0.1:8080``.
        api_key: Optional[str]
            API key. Defaults to env ``SYNAPCORES_API_KEY``.
        username / password: Optional[str]
            Log in with a user instead of an API key.

    Instantiate:
        .. code-block:: python

            from langchain_synapcores import SynapCoresEmbeddings, SynapCoresVectorStore

            vector_store = SynapCoresVectorStore(
                embedding=SynapCoresEmbeddings(),
                table_name="support_articles",
            )

    Add Documents:
        .. code-block:: python

            from langchain_core.documents import Document

            vector_store.add_documents(
                [
                    Document(page_content="Reset your password from Settings.", metadata={"topic": "account"}),
                    Document(page_content="Invoices are emailed monthly.", metadata={"topic": "billing"}),
                ],
                ids=["1", "2"],
            )

    Search:
        .. code-block:: python

            vector_store.similarity_search("forgot my password", k=1, filter={"topic": "account"})

    Use as Retriever:
        .. code-block:: python

            retriever = vector_store.as_retriever(search_kwargs={"k": 4})
            retriever.invoke("how do I get my invoice?")
    """

    def __init__(
        self,
        embedding: Embeddings,
        *,
        table_name: str = "langchain_documents",
        database: Optional[str] = None,
        distance: str = "cosine",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client: Optional[SynapCoresClient] = None,
        batch_size: int = 64,
    ) -> None:
        if distance not in _DISTANCES:
            raise ValueError(f"distance must be one of {_DISTANCES}")
        self._embedding = embedding
        self.table_name = safe_identifier(table_name, "table name")
        self.database = safe_identifier(database, "database name") if database else None
        self.distance = distance
        self.batch_size = max(1, int(batch_size))
        self.client = client or SynapCoresClient(
            url, api_key=api_key, username=username, password=password
        )
        self._table_ready = False

    # -- helpers ------------------------------------------------------------

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def _sql(self, sql: str, params: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
        return self.client.execute(sql, params, database=self.database)

    def _table_exists(self) -> bool:
        if self._table_ready:
            return True
        rows = self._sql("SHOW TABLES")
        names = {str(next(iter(r.values()))) for r in rows if r}
        self._table_ready = self.table_name in names
        return self._table_ready

    def _ensure_table(self, dim: int) -> None:
        if self._table_ready:
            return
        if self.database:
            self.client.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
        self._sql(
            f"CREATE TABLE IF NOT EXISTS {self.table_name} ("
            "id TEXT PRIMARY KEY, content TEXT, metadata TEXT, "
            f"embedding VECTOR({int(dim)}))"
        )
        self._table_ready = True

    @staticmethod
    def _row_to_doc(row: Dict[str, Any]) -> Document:
        raw = row.get("metadata")
        try:
            metadata = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            metadata = {}
        return Document(id=row.get("id"), page_content=row.get("content") or "", metadata=metadata)

    # -- writes -------------------------------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        texts = list(texts)
        if not texts:
            return []
        metadatas = metadatas or [{} for _ in texts]
        ids = [str(i) if i else str(uuid.uuid4()) for i in (ids or [None] * len(texts))]
        if len(metadatas) != len(texts) or len(ids) != len(texts):
            raise ValueError("texts, metadatas and ids must have the same length")

        vectors = self._embedding.embed_documents(texts)
        self._ensure_table(len(vectors[0]))

        for start in range(0, len(texts), self.batch_size):
            end = start + self.batch_size
            values: List[str] = []
            params: List[Any] = []
            for doc_id, text, meta, vec in zip(
                ids[start:end], texts[start:end], metadatas[start:end], vectors[start:end]
            ):
                params.extend([doc_id, text, json.dumps(meta or {}, default=str)])
                n = len(params)
                values.append(f"(${n - 2}, ${n - 1}, ${n}, {vector_literal(vec)})")
            self._sql(
                f"INSERT INTO {self.table_name} (id, content, metadata, embedding) VALUES "
                + ", ".join(values)
                + " ON CONFLICT (id) DO UPDATE SET content = excluded.content, "
                "metadata = excluded.metadata, embedding = excluded.embedding",
                params,
            )
        return ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if not ids:
            return None
        if not self._table_exists():
            return True
        holders = ", ".join(f"${i + 1}" for i in range(len(ids)))
        self._sql(f"DELETE FROM {self.table_name} WHERE id IN ({holders})", [str(i) for i in ids])
        return True

    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        if not ids or not self._table_exists():
            return []
        holders = ", ".join(f"${i + 1}" for i in range(len(ids)))
        rows = self._sql(
            f"SELECT id, content, metadata FROM {self.table_name} WHERE id IN ({holders})",
            [str(i) for i in ids],
        )
        return [self._row_to_doc(r) for r in rows]

    def drop_table(self) -> None:
        """Delete the table and all its documents."""
        self._sql(f"DROP TABLE IF EXISTS {self.table_name}")
        self._table_ready = False

    # -- search -------------------------------------------------------------

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return documents and scores.

        The score is cosine similarity (higher is closer) for ``distance="cosine"``
        and Euclidean distance (lower is closer) for ``distance="euclidean"``.
        """
        if not self._table_exists():
            return []
        where, params = build_where(filter)
        clause = f" WHERE {where}" if where else ""
        vec = vector_literal(embedding)
        if self.distance == "cosine":
            score, order = f"COSINE_SIMILARITY(embedding, {vec})", "DESC"
        else:
            score, order = f"EUCLIDEAN_DISTANCE(embedding, {vec})", "ASC"
        rows = self._sql(
            f"SELECT id, content, metadata, {score} AS score FROM {self.table_name}"
            f"{clause} ORDER BY score {order} LIMIT {max(1, int(k))}",
            params,
        )
        return [(self._row_to_doc(r), float(r.get("score") or 0.0)) for r in rows]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        return self.similarity_search_with_score_by_vector(
            self._embedding.embed_query(query), k=k, filter=filter, **kwargs
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return [
            d
            for d, _ in self.similarity_search_with_score_by_vector(
                embedding, k=k, filter=filter, **kwargs
            )
        ]

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> List[Document]:
        return [d for d, _ in self.similarity_search_with_score(query, k=k, filter=filter, **kwargs)]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        if self.distance == "cosine":
            return lambda s: (s + 1.0) / 2.0
        return lambda d: 1.0 / (1.0 + d)

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "SynapCoresVectorStore":
        store = cls(embedding=embedding, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store


__all__ = ["SynapCoresVectorStore", "SynapCoresError"]
