"""The full RAG pipeline: ingest documents, answer questions, summarise documents."""
import time
from pathlib import Path
from typing import Dict, List

from . import vector_store
from .chunker import chunk_pages
from .config import TOP_K
from .llm import generate
from .pdf_loader import extract_pages

# Chunks less similar than this are treated as irrelevant to the question.
# MiniLM gives fairly low cosine scores for short questions vs. long passages
# (correct matches are often 0.2-0.4, unrelated text < 0.1), so keep this low and
# let the LLM decide whether the remaining context actually answers the question.
MIN_SCORE = 0.12

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


def ingest_pdf(path: str) -> Dict:
    source = Path(path).name
    pages = extract_pages(path)
    if not pages:
        raise ValueError("No extractable text found (the PDF may be scanned images).")
    vector_store.delete_document(source)  # re-uploading replaces the old version
    chunks = chunk_pages(pages, source)
    vector_store.add_chunks(chunks)
    return {"source": source, "pages": len(pages), "chunks": len(chunks)}


def _build_prompt(question: str, hits: List[Dict]) -> str:
    ctx = "\n\n".join(
        f"[{i}] (source: {h['source']}, page {h['page']})\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )
    return f"Context passages:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"


def answer(question: str, k: int = TOP_K, source: str = None) -> Dict:
    t0 = time.perf_counter()
    hits = [h for h in vector_store.search(question, k=k, source=source) if h["score"] >= MIN_SCORE]
    t_retrieval = time.perf_counter() - t0

    if not hits:
        return {
            "answer": "I couldn't find anything relevant to that in the uploaded material. "
                      "Try rephrasing, or upload the relevant notes first.",
            "sources": [], "timings": {"retrieval": round(t_retrieval, 3), "generation": 0.0},
        }

    t1 = time.perf_counter()
    text = generate(_build_prompt(question, hits), system=ANSWER_SYSTEM)
    t_gen = time.perf_counter() - t1

    sources = [{"n": i, "source": h["source"], "page": h["page"], "score": h["score"],
                "excerpt": h["text"][:300]} for i, h in enumerate(hits, start=1)]
    return {"answer": text, "sources": sources,
            "timings": {"retrieval": round(t_retrieval, 3), "generation": round(t_gen, 3)}}


def summarise(source: str, max_chars: int = 60000) -> Dict:
    chunks = vector_store.get_document_chunks(source)
    if not chunks:
        raise ValueError(f"Document '{source}' not found.")

    # Gemini's large context lets us send most notes whole. Very long books are
    # sampled evenly so every part of the document is still represented.
    total = sum(len(c["text"]) for c in chunks)
    if total > max_chars:
        keep = max(1, int(len(chunks) * max_chars / total))
        stride = len(chunks) / keep
        chunks = [chunks[int(i * stride)] for i in range(keep)]
    body = "\n".join(f"(p{c['page']}) {c['text']}" for c in chunks)[:max_chars]

    prompt = (
        f"Document: {source}\n\n{body}\n\n"
        "Write a revision summary with:\n"
        "1. A 2-3 sentence overview.\n"
        "2. Key topics as headed bullet points (mention page numbers like (p4)).\n"
        "3. Important definitions/formulas, if any.\n"
        "Use Markdown."
    )
    return {"source": source, "summary": generate(prompt, system=SUMMARY_SYSTEM, temperature=0.3)}
