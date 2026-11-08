# langchain-synapcores

LangChain integration for [SynapCores](https://synapcores.com), the AI-native database. SynapCores combines SQL, vector search, a Cypher graph, in-database AutoML and agent memory in one self-hosted engine.

This package provides:

- **`SynapCoresVectorStore`**: a LangChain vector store on an ordinary SynapCores SQL table with a `VECTOR(N)` column. Your documents stay queryable with SQL, joinable with your business tables, and usable from graph queries and AutoML.
- **`SynapCoresEmbeddings`**: embeddings computed inside SynapCores, next to your data, with no external embedding API or key.

Try SynapCores in the browser: **[synapcores.com/playground](https://synapcores.com/playground)**

## Installation

```bash
pip install -U langchain-synapcores
```

Run SynapCores locally (free Community Edition):

```bash
curl -fsSL https://get.synapcores.com | sh
```

Docker: `synapcores/community`. See [synapcores.com/download](https://synapcores.com/download).

## Configuration

| Argument | Environment variable | Default |
|---|---|---|
| `url` | `SYNAPCORES_URL` | `http://127.0.0.1:8080` |
| `api_key` | `SYNAPCORES_API_KEY` | none |
| `username` / `password` | `SYNAPCORES_USERNAME` / `SYNAPCORES_PASSWORD` | none |

## Vector store

```python
from langchain_core.documents import Document
from langchain_synapcores import SynapCoresEmbeddings, SynapCoresVectorStore

vector_store = SynapCoresVectorStore(
    embedding=SynapCoresEmbeddings(),   # or any LangChain embeddings model
    table_name="support_articles",
)

vector_store.add_documents(
    [
        Document(page_content="Reset your password from the account settings page.", metadata={"topic": "account"}),
        Document(page_content="Invoices are emailed on the first day of each month.", metadata={"topic": "billing", "year": 2026}),
    ],
    ids=["kb-1", "kb-2"],
)

vector_store.similarity_search("I forgot my password", k=1)
vector_store.similarity_search("when is my bill?", k=2, filter={"topic": "billing"})
vector_store.similarity_search_with_score("bill", k=2, filter={"year": {"$gte": 2026}})
```

### Metadata filters

| Filter | Meaning |
|---|---|
| `{"topic": "billing"}` | equals |
| `{"year": {"$gte": 2024}}` | `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte` |
| `{"topic": {"$in": ["a", "b"]}}` | `$in`, `$nin` |
| `{"$or": [{...}, {...}]}` | `$and`, `$or` |
| `{"source.kind": "pdf"}` | nested keys with dot paths |

### Retriever

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
retriever.invoke("how do I get my invoice?")
```

### SQL on the same data

The store is a normal table: `id TEXT PRIMARY KEY, content TEXT, metadata TEXT, embedding VECTOR(N)`. You can query it directly in SynapCores, for example to join retrieved chunks with customer records:

```sql
SELECT a.id, a.content, COSINE_SIMILARITY(a.embedding, EMBED('reset password')) AS score
FROM support_articles a
ORDER BY score DESC
LIMIT 5;
```

Pass `database="..."` to keep LangChain tables in their own SynapCores database. Use `distance="euclidean"` for Euclidean ranking (the default is cosine).

## Embeddings

```python
from langchain_synapcores import SynapCoresEmbeddings

embeddings = SynapCoresEmbeddings()
embeddings.embed_query("wireless headphones")
embeddings.embed_documents(["first text", "second text"])
```

## Development

```bash
pip install -e ".[test]"
pytest tests/unit_tests
# Integration tests (includes the LangChain standard test suite) need a running engine:
SYNAPCORES_URL=http://127.0.0.1:8080 SYNAPCORES_API_KEY=... pytest tests/integration_tests
```

## Links

- [SynapCores](https://synapcores.com)
- [Playground](https://synapcores.com/playground)
- [Documentation](https://docs.synapcores.com)
- [SynapCores MCP server](https://github.com/SynapCores/synapcores-mcp)

## License

MIT
