"""LangChain standard integration tests against a live SynapCores gateway.

Run with SYNAPCORES_URL plus SYNAPCORES_API_KEY (or SYNAPCORES_USERNAME and
SYNAPCORES_PASSWORD) pointing at a running SynapCores engine.
"""

from __future__ import annotations

import os
import uuid
from typing import Generator, Type

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import (
    EmbeddingsIntegrationTests,
    VectorStoreIntegrationTests,
)

from langchain_synapcores import SynapCoresEmbeddings, SynapCoresVectorStore

DATABASE = os.environ.get("SYNAPCORES_TEST_DATABASE", "langchain_tests")

pytestmark = pytest.mark.skipif(
    not (os.environ.get("SYNAPCORES_API_KEY") or os.environ.get("SYNAPCORES_USERNAME")),
    reason="needs a running SynapCores gateway (set SYNAPCORES_URL and credentials)",
)


class TestSynapCoresVectorStore(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore[override]
        store = SynapCoresVectorStore(
            self.get_embeddings(),
            table_name=f"lc_{uuid.uuid4().hex[:12]}",
            database=DATABASE,
        )
        try:
            yield store
        finally:
            store.drop_table()


class TestSynapCoresEmbeddings(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[SynapCoresEmbeddings]:
        return SynapCoresEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {}


def test_end_to_end_with_in_database_embeddings() -> None:
    """Embed inside SynapCores, store, filter and retrieve."""
    store = SynapCoresVectorStore(
        SynapCoresEmbeddings(), table_name=f"lc_{uuid.uuid4().hex[:12]}", database=DATABASE
    )
    try:
        store.add_texts(
            [
                "Reset your password from the account settings page.",
                "Invoices are emailed on the first day of each month.",
                "Our hiking boots are waterproof and built for mountains.",
            ],
            metadatas=[{"topic": "account"}, {"topic": "billing", "year": 2026}, {"topic": "shop"}],
            ids=["a", "b", "c"],
        )
        top = store.similarity_search("I forgot my password", k=1)
        assert top[0].id == "a"
        billing = store.similarity_search("when do I get my bill?", k=3, filter={"topic": "billing"})
        assert [d.id for d in billing] == ["b"]
        ranged = store.similarity_search("bill", k=3, filter={"year": {"$gte": 2026}})
        assert [d.id for d in ranged] == ["b"]
        either = store.similarity_search(
            "anything", k=3, filter={"$or": [{"topic": "account"}, {"topic": "shop"}]}
        )
        assert sorted(d.id for d in either) == ["a", "c"]
        scored = store.similarity_search_with_relevance_scores("password help", k=3)
        assert all(0.0 <= s <= 1.0 for _, s in scored)
        docs = store.as_retriever(search_kwargs={"k": 2}).invoke("mountain footwear")
        assert docs[0].id == "c"
    finally:
        store.drop_table()
