"""The RAG pipeline, written to be stateless so it runs on serverless hosting.

The vector store itself lives in the user's browser (IndexedDB, see public/static/app.js):

  ingest:   browser extracts PDF text (pdf.js) -> index_pages() chunks + embeds it
            -> browser stores chunks and vectors
  question: embed_query() -> browser ranks its stored chunks by cosine similarity
            -> answer() sends the top passages to Gemini
  summary:  browser splits the document into sections -> summarise_section() for
            each -> summarise_overview() over all section notes
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
MAX_SECTION_CHARS = 16000   # one summary section (the browser sends ~10,000)
MAX_OVERVIEW_CHARS = 80000  # all section notes together


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
    text = generate(_build_prompt(question, passages), system=ANSWER_SYSTEM,
                    timeout_s=40, budget_s=100)
    sources = [{"n": i, "source": p["source"], "page": p["page"],
                "score": round(float(p.get("score", 0)), 4), "excerpt": p["text"][:300]}
               for i, p in enumerate(passages, start=1)]
    return {"answer": text, "sources": sources,
            "timings": {"generation": round(time.perf_counter() - t0, 3)}}


# ---------------------------------------------------------------------------
# Summaries: map-reduce. The browser splits the document into sections of a few
# pages; each section gets detailed notes (map), then one short call writes the
# overview and key takeaways from those notes (reduce). Every call stays small,
# so long PDFs are covered completely and no request runs into a timeout.
# ---------------------------------------------------------------------------

SECTION_PROMPT = """Below is the text of pages {start}-{end} of "{source}".
Each page begins with a marker like [Page 4].

{body}

Write detailed revision notes for THESE PAGES ONLY, in Markdown:
- Follow the order of the material. Give every question, topic or heading its own
  "### " heading that names it clearly (e.g. "### Q.2 Causes of Poverty in India").
- Under each heading, explain the content in bullet points: all key points,
  definitions, causes/effects, types, examples, figures and statistics. Keep
  numbered lists from the source (e.g. "10 causes") complete — do not drop items.
- Bold important terms. Use a small table when the source compares things.
- After each heading's bullets, add the page it comes from in italics, e.g. *(Page 4)*.
- Be thorough: a student should be able to revise from these notes without the PDF.
  Do not add facts that are not in the text. Do not write an introduction or conclusion.
"""

OVERVIEW_PROMPT = """These are section-by-section revision notes for "{source}" ({pages} pages):

{notes}

Using only these notes, write in Markdown:
## Overview
A 4-6 sentence paragraph explaining what the document covers and its main argument.

## Topics covered
A bullet list of every main topic/question in order, each with its pages, e.g.
"- **Q.1 Meaning of economic development** — Pages 1-2".

## Key takeaways
6-10 bullets with the most important points to remember for an exam.
"""


def summarise_section(source: str, start_page: int, end_page: int, pages: List[Dict]) -> Dict:
    """pages: [{page, text}] for one section of the document."""
    body = "\n\n".join(f"[Page {p['page']}]\n{p['text']}" for p in pages if p["text"].strip())
    if not body:
        raise ValueError("This section has no text.")
    prompt = SECTION_PROMPT.format(source=source, start=start_page, end=end_page,
                                   body=body[:MAX_SECTION_CHARS])
    notes = generate(prompt, system=SUMMARY_SYSTEM, temperature=0.3, timeout_s=70, budget_s=130)
    return {"start_page": start_page, "end_page": end_page, "notes": notes}


def summarise_overview(source: str, total_pages: int, sections: List[Dict]) -> Dict:
    """sections: [{start_page, end_page, notes}] from summarise_section()."""
    notes = "\n\n".join(f"## Pages {s['start_page']}-{s['end_page']}\n{s['notes']}" for s in sections)
    prompt = OVERVIEW_PROMPT.format(source=source, pages=total_pages, notes=notes[:MAX_OVERVIEW_CHARS])
    return {"overview": generate(prompt, system=SUMMARY_SYSTEM, temperature=0.3,
                                 timeout_s=60, budget_s=130)}
