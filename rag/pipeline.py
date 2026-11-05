"""The RAG pipeline, written to be stateless so it runs on serverless hosting.

The vector store itself lives in the user's browser (IndexedDB, see public/static/app.js):

  ingest:   browser extracts PDF text (pdf.js) -> index_pages() chunks + embeds it
            -> browser stores chunks and vectors
  question: embed_query() -> browser ranks its stored chunks by cosine similarity
            -> answer() sends the top passages to Gemini
  summary:  browser sends the document's chunks -> summarise()
"""
import time
from typing import Dict, List

from .chunker import chunk_pages
from .llm import embed, generate

ANSWER_SYSTEM = (
    "You are a study assistant for university students. Answer ONLY using the numbered "
    "context passages from the student's course material. Cite the passages you used "
    "inline like [1] or [2][3]. If the context does not contain the answer, say clearly "
    "that the uploaded material does not cover it; do not make up facts. "
    "Explain clearly and concisely, using bullet points or steps where helpful."
)

SUMMARY_SYSTEM = (
    "You summarise academic course material for revision. Be accurate and only use the "
    "provided text."
)

MAX_PASSAGES = 10
MAX_SUMMARY_CHARS = 60000


def index_pages(source: str, pages: List[Dict]) -> List[Dict]:
    """pages: [{"page": 1, "text": "..."}] -> chunks with embeddings."""
    pages = [p for p in pages if (p.get("text") or "").strip()]
    chunks = chunk_pages(pages, source)
    if not chunks:
        return []
    vectors = embed([c["text"] for c in chunks])
    return [{"id": c["id"], "source": source, "page": c["metadata"]["page"],
             "chunk": c["metadata"]["chunk"], "text": c["text"], "embedding": v}
            for c, v in zip(chunks, vectors)]


def embed_query(question: str) -> List[float]:
    return embed([question], query=True)[0]


def _build_prompt(question: str, passages: List[Dict]) -> str:
    ctx = "\n\n".join(
        f"[{i}] (source: {p['source']}, page {p['page']})\n{p['text']}"
        for i, p in enumerate(passages, start=1)
    )
    return f"Context passages:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"


def answer(question: str, passages: List[Dict]) -> Dict:
    """passages: the top-ranked chunks [{source, page, text, score}], best first."""
    passages = passages[:MAX_PASSAGES]
    if not passages:
        return {"answer": "I couldn't find anything relevant to that in the uploaded material. "
                          "Try rephrasing, or upload the relevant notes first.",
                "sources": [], "timings": {"generation": 0.0}}
    t0 = time.perf_counter()
    text = generate(_build_prompt(question, passages), system=ANSWER_SYSTEM)
    sources = [{"n": i, "source": p["source"], "page": p["page"],
                "score": round(float(p.get("score", 0)), 4), "excerpt": p["text"][:300]}
               for i, p in enumerate(passages, start=1)]
    return {"answer": text, "sources": sources,
            "timings": {"generation": round(time.perf_counter() - t0, 3)}}


def summarise(source: str, chunks: List[Dict]) -> Dict:
    """chunks: the document's chunks [{page, text}] in page order."""
    if not chunks:
        raise ValueError(f"Document '{source}' has no text to summarise.")
    # Gemini's large context lets us send most notes whole. Very long books are
    # sampled evenly so every part of the document is still represented.
    total = sum(len(c["text"]) for c in chunks)
    if total > MAX_SUMMARY_CHARS:
        keep = max(1, int(len(chunks) * MAX_SUMMARY_CHARS / total))
        stride = len(chunks) / keep
        chunks = [chunks[int(i * stride)] for i in range(keep)]
    body = "\n".join(f"(p{c['page']}) {c['text']}" for c in chunks)[:MAX_SUMMARY_CHARS]

    prompt = (
        f"Document: {source}\n\n{body}\n\n"
        "Write a revision summary with:\n"
        "1. A 2-3 sentence overview.\n"
        "2. Key topics as headed bullet points (mention page numbers like (p4)).\n"
        "3. Important definitions/formulas, if any.\n"
        "Use Markdown."
    )
    return {"source": source, "summary": generate(prompt, system=SUMMARY_SYSTEM, temperature=0.3)}
