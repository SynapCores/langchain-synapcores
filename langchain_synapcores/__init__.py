"""LangChain integration for SynapCores, the AI-native database."""

from importlib import metadata

from langchain_synapcores._client import SynapCoresClient, SynapCoresError
from langchain_synapcores.embeddings import SynapCoresEmbeddings
from langchain_synapcores.vectorstores import SynapCoresVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""

__all__ = [
    "SynapCoresClient",
    "SynapCoresEmbeddings",
    "SynapCoresError",
    "SynapCoresVectorStore",
    "__version__",
]
