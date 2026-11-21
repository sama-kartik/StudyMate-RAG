"""The RAG pipeline, written to be stateless so it runs on serverless hosting.

The vector store itself lives in the user's browser (IndexedDB, see public/static/app.js):

  ingest:   browser extracts PDF text (pdf.js) -> index_pages() chunks + embeds it
            -> browser stores chunks and vectors
  question: embed_query() -> browser ranks its stored chunks by cosine similarity
            -> answer() sends the top passages to Gemini
  summary:  browser splits the document into sections -> summarise_section() for
            each -> summarise_overview() over all section notes
"""
import re
import time
from typing import Dict, List

from .chunker import chunk_pages
from .llm import embed, generate
from .youtube import fmt_ts, parse_ts

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


def _where(p: Dict) -> str:
    # PDFs are cited by page; YouTube lectures by the time in the video.
    return f"at {fmt_ts(p['page'])} in the video" if p.get("kind") == "video" else f"page {p['page']}"


def _build_prompt(question: str, passages: List[Dict]) -> str:
    ctx = "\n\n".join(
        f"[{i}] (source: {p['source']}, {_where(p)})\n{p['text']}"
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
    sources = [{"n": i, "source": p["source"], "page": p["page"], "kind": p.get("kind", "pdf"),
                "video_id": p.get("video_id"), "score": round(float(p.get("score", 0)), 4),
                "excerpt": p["text"][:300]}
               for i, p in enumerate(passages, start=1)]
    return {"answer": text, "sources": sources,
            "timings": {"generation": round(time.perf_counter() - t0, 3)}}


# ---------------------------------------------------------------------------
# Summaries: map-reduce. The browser splits the document into sections of a few
# pages; each section gets detailed notes (map), then one short call writes the
# overview and key takeaways from those notes (reduce). Every call stays small,
# so long PDFs are covered completely and no request runs into a timeout.
# ---------------------------------------------------------------------------

SECTION_PROMPT = """Below is the text of {span} of "{source}".
Each part begins with a marker like {example}.

{body}

Write detailed revision notes for THIS PART ONLY, in Markdown:
- Follow the order of the material. Give every question, topic or heading its own
  "### " heading that names it clearly (e.g. "### Q.2 Causes of Poverty in India").
- Under each heading, explain the content in bullet points: all key points,
  definitions, causes/effects, types, examples, figures and statistics. Keep
  numbered lists from the source (e.g. "10 causes") complete — do not drop items.
- Bold important terms. Use a small table when the source compares things.
- After each heading's bullets, add where it comes from in italics, e.g. {cite}.
- Be thorough: a student should be able to revise from these notes without the original.
  Do not add facts that are not in the text. Do not write an introduction or conclusion.
"""

OVERVIEW_PROMPT = """These are section-by-section revision notes for "{source}" ({length}):

{notes}

Using only these notes, write in Markdown:
## Overview
A 4-6 sentence paragraph explaining what the document covers and its main argument.

## Topics covered
A bullet list of every main topic/question in order, each with {where}, e.g.
"- **{topic_example}** — {where_example}".

## Key takeaways
6-10 bullets with the most important points to remember for an exam.
"""


def summarise_section(source: str, start_page: int, end_page: int, pages: List[Dict],
                      kind: str = "pdf") -> Dict:
    """pages: [{page, text}] for one section. For videos, "page" is the second in the video."""
    video = kind == "video"
    mark = (lambda v: f"[{fmt_ts(v)}]") if video else (lambda v: f"[Page {v}]")
    body = "\n\n".join(f"{mark(p['page'])}\n{p['text']}" for p in pages if p["text"].strip())
    if not body:
        raise ValueError("This section has no text.")
    span = (f"the lecture video from {fmt_ts(start_page)} to {fmt_ts(end_page)}" if video
            else f"pages {start_page}-{end_page}")
    prompt = SECTION_PROMPT.format(source=source, span=span, body=body[:MAX_SECTION_CHARS],
                                   example="[12:34]" if video else "[Page 4]",
                                   cite="*(12:34)*" if video else "*(Page 4)*")
    notes = generate(prompt, system=SUMMARY_SYSTEM, temperature=0.3, timeout_s=70, budget_s=130)
    return {"start_page": start_page, "end_page": end_page, "notes": notes}


def summarise_overview(source: str, total_pages: int, sections: List[Dict], kind: str = "pdf") -> Dict:
    """sections: [{start_page, end_page, notes}] from summarise_section()."""
    video = kind == "video"
    label = (lambda s: f"{fmt_ts(s['start_page'])}-{fmt_ts(s['end_page'])}") if video \
        else (lambda s: f"Pages {s['start_page']}-{s['end_page']}")
    notes = "\n\n".join(f"## {label(s)}\n{s['notes']}" for s in sections)
    prompt = OVERVIEW_PROMPT.format(
        source=source, notes=notes[:MAX_OVERVIEW_CHARS],
        length=f"a lecture video of about {fmt_ts(total_pages)}" if video else f"{total_pages} pages",
        where="its start time" if video else "its pages",
        topic_example="Backpropagation intuition" if video else "Q.1 Meaning of economic development",
        where_example="from 12:34" if video else "Pages 1-2")
    if video:
        topics = _topic_times(sections)
        if topics:
            prompt += ("\nStart times of the topics, taken from the notes (copy these times exactly; "
                       "never write 00:00 unless it is listed):\n"
                       + "\n".join(f"- {name} — from {fmt_ts(t)}" for name, t in topics) + "\n")
    text = generate(prompt, system=SUMMARY_SYSTEM, temperature=0.3, timeout_s=60, budget_s=130)
    if video and topics:
        text = _fix_topic_times(text, topics)
    return {"overview": text}


_TIME = r"\d{1,2}:\d{2}(?::\d{2})?"


def _topic_times(sections: List[Dict]) -> List:
    """(heading, second) for every "### heading" in the section notes of a video.

    The second is the first time cited under the heading, e.g. "*(12:34)*", kept only
    if it falls inside that section; otherwise the section's own start is used.
    """
    out = []
    for s in sections:
        a, b = int(s["start_page"]), int(s["end_page"])
        parts = re.split(r"^#{2,4}\s+(.+)$", s.get("notes") or "", flags=re.M)
        for name, body in zip(parts[1::2], parts[2::2]):
            m = re.search(rf"\(\s*({_TIME})", body) or re.search(rf"\[({_TIME})\]", body)
            t = parse_ts(m.group(1)) if m else None
            if t is None or not (a - 30 <= t <= b + 60):
                t = a
            out.append((re.sub(r"[*_`]|\(.*?\)", "", name).strip(), t))
    return out


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def _fix_topic_times(text: str, topics: List) -> str:
    """Replace each "**Topic** — from MM:SS" time with the time of the best-matching note heading."""
    def fix(m):
        name = _words(m.group(2))
        best, score = None, 0.0
        for heading, t in topics:
            h = _words(heading)
            if name and h:
                sc = len(name & h) / len(name | h)
                if sc > score:
                    best, score = t, sc
        return m.group(1) + fmt_ts(best) if best is not None and score >= 0.25 else m.group(0)
    return re.sub(rf"^(\s*[-*]\s*\*\*(.+?)\*\*.*?from\s+)({_TIME})", fix, text, flags=re.M)
