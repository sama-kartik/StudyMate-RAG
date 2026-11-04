"""Steps 3 & 4: embed chunks with sentence-transformers and index them in ChromaDB."""
from functools import lru_cache
from typing import Dict, List

# config must be imported first: it points the model cache into the project folder.
from .config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K

import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: List[str]) -> List[List[float]]:
    vecs = _embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vecs.tolist()


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR),
                                       settings=Settings(anonymized_telemetry=False))
    # cosine distance: 0 = identical meaning, 2 = opposite
    return client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def add_chunks(chunks: List[Dict], batch_size: int = 64) -> int:
    col = _collection()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        col.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
            embeddings=embed([c["text"] for c in batch]),
        )
    return len(chunks)


def search(query: str, k: int = TOP_K, source: str = None) -> List[Dict]:
    """Semantic search. Returns chunks ordered from most to least similar."""
    col = _collection()
    if col.count() == 0:
        return []
    res = col.query(
        query_embeddings=embed([query]),
        n_results=min(k, col.count()),
        where={"source": source} if source else None,
    )
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "source": meta["source"], "page": meta["page"],
                     "score": round(1 - dist, 4)})  # cosine similarity
    return hits


def get_document_chunks(source: str) -> List[Dict]:
    res = _collection().get(where={"source": source})
    items = [{"text": d, **m} for d, m in zip(res["documents"], res["metadatas"])]
    items.sort(key=lambda x: (x["page"], x["chunk"]))
    return items


def list_documents() -> List[Dict]:
    res = _collection().get(include=["metadatas"])
    docs = {}
    for m in res["metadatas"]:
        d = docs.setdefault(m["source"], {"source": m["source"], "chunks": 0, "pages": set()})
        d["chunks"] += 1
        d["pages"].add(m["page"])
    return [{"source": d["source"], "chunks": d["chunks"], "pages": len(d["pages"])}
            for d in sorted(docs.values(), key=lambda x: x["source"])]


def delete_document(source: str) -> None:
    _collection().delete(where={"source": source})
