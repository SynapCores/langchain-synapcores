"""Embeddings computed inside SynapCores."""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.embeddings import Embeddings

from langchain_synapcores._client import SynapCoresClient


class SynapCoresEmbeddings(Embeddings):
    """Embed text with the embedding model that runs inside SynapCores.

    No external embedding API or key is needed: SynapCores hosts the model
    (MiniLM by default, 384 dimensions) next to your data.

    Instantiate:
        .. code-block:: python

            from langchain_synapcores import SynapCoresEmbeddings

            embeddings = SynapCoresEmbeddings(url="http://127.0.0.1:8080", api_key="...")
            vector = embeddings.embed_query("wireless headphones")

    Args:
        model: Optional embedding model name configured in SynapCores.
        url, api_key, username, password: Connection settings (see ``SynapCoresClient``).
        batch_size: Texts per request.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client: Optional[SynapCoresClient] = None,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.batch_size = max(1, int(batch_size))
        self.client = client or SynapCoresClient(
            url, api_key=api_key, username=username, password=password
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self.client.embed(texts[i : i + self.batch_size], model=self.model))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self.client.embed([text], model=self.model)[0]
