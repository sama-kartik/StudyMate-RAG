"""Exam question predictor from previous-year question papers.

1. extract_questions(): Gemini reads one paper (already text, or transcribed from scans
   by rag/vision.py) and lists its questions with marks and the exam year.
2. predict(): Gemini groups every question from every paper into topics (mapped to the
   syllabus units when a syllabus is set) and writes likely questions for each topic.
   The probability of a topic is NOT asked from the model: it is computed here from how
   many papers asked it and how recently, so the numbers are reproducible and explainable.
"""
import json
import re
from typing import Dict, List, Optional

from .llm import generate, loads_json

# Grouping questions needs some judgement: a Flash model with thinking turned off (quick
# mode, ~5-10 s) first, lite models next, the larger models last.
PREDICT_MODELS = ["gemini-3-flash-preview", "gemini-3.5-flash-lite", "gemini-flash-lite-latest",
                  "gemini-3.5-flash", "gemini-flash-latest", "gemini-3.8-flash"]
# Reading questions out of a paper is simple copying: fast "lite" models do it in 2-3 s
# (larger models take 10-60 s), so they go first, with short timeouts so an overloaded
# model is skipped quickly instead of eating the request's time.
EXTRACT_MODELS = ["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
                  "gemini-3-flash-preview", "gemini-3.5-flash", "gemini-3.8-flash"]
MATH_RULE = ("Write every mathematical expression in LaTeX between dollar signs, e.g. "
             "$u = r^n \\cos(n\\theta)$ or $\\frac{\\partial u}{\\partial r}$; never leave LaTeX outside $...$.")
MAX_PAPER_CHARS = 40000
MAX_QUESTIONS = 400

EXTRACT_PROMPT = """Below is the text of a university exam question paper, "{source}"
(it may have been transcribed from scanned pages, so ignore small OCR errors).

{text}

List every question and sub-question a student must answer, in order. Reply as JSON:
{{"exam": "short exam name, at most 5 words, e.g. End Sem 2023 or Mid Sem", "year": <4-digit year or null>,
  "questions": [{{"no": "1(a)", "text": "full question text", "marks": <number or null>, "page": <page number>}}]}}
Rules: copy each question's wording faithfully (fix obvious OCR errors only); {math_rule} Split sub-parts
(a), (b), ... into separate entries; skip instructions like "Attempt any five"; use the year
printed on the paper (null if none)."""

GROUP_PROMPT = """You are analysing {n_papers} previous-year exam papers for the course "{subject}".
{syllabus_part}
Every question asked (id, paper year, marks, text):
{questions}

Group the questions into exam TOPICS (a topic is a concept a teacher examines, e.g.
"K-map simplification", "Full adder design" — not a whole unit, not a single question).
Put every question id in exactly one topic. Then, for each topic, write 2 NEW questions
that are likely in the next exam: same style and difficulty as the past ones, not copies.
Reply as JSON:
{{"topics": [{{"topic": "short name", "unit": "{unit_hint}", "question_ids": ["Q3", ...],
              "predicted": ["...", "..."], "tip": "one sentence on how to prepare it"}}],
  "unasked": [{{"topic": "...", "unit": "..."}}]}}
{math_rule}
"unasked" lists important syllabus topics that no paper has asked yet (empty list if there is no syllabus)."""


def _json(raw: str, list_key: str = "questions") -> Dict:
    return loads_json(raw, list_key)


def _year(v) -> Optional[int]:
    try:
        y = int(str(v).strip()[:4])
    except (TypeError, ValueError):
        return None
    return y if 1990 <= y <= 2100 else None


def year_from_name(name: str) -> Optional[int]:
    m = re.search(r"(?<!\d)(19[9]\d|20\d\d)(?!\d)", name or "")
    return int(m.group(1)) if m else None


def extract_questions(source: str, pages: List[Dict]) -> Dict:
    """pages: [{page, text}]. Returns {exam, year, questions: [{no, text, marks, page}]}."""
    text = "\n\n".join(f"[Page {int(p['page'])}]\n{str(p.get('text') or '').strip()}"
                       for p in pages if str(p.get("text") or "").strip())
    if not text:
        raise ValueError("This paper has no readable text.")
    data = _json(generate(EXTRACT_PROMPT.format(source=source, text=text[:MAX_PAPER_CHARS], math_rule=MATH_RULE),
                          temperature=0.1, timeout_s=20, budget_s=130, json_mode=True,
                          models=EXTRACT_MODELS, quick=True))
    qs = []
    for q in data.get("questions") or []:
        if not isinstance(q, dict) or not str(q.get("text") or "").strip():
            continue
        try:
            marks = float(q.get("marks")) if q.get("marks") not in (None, "") else None
        except (TypeError, ValueError):
            marks = None
        try:
            page = int(q.get("page") or 0) or None
        except (TypeError, ValueError):
            page = None
        qs.append({"no": str(q.get("no") or "").strip()[:20], "text": str(q["text"]).strip()[:1500],
                   "marks": marks, "page": page})
    if not qs:
        raise ValueError("No questions were found in this paper.")
    exam = re.sub(r"\s+", " ", str(data.get("exam") or "")).strip()
    return {"exam": exam if len(exam) <= 40 else exam[:38].rsplit(" ", 1)[0] + "…",
            "year": _year(data.get("year")) or year_from_name(source), "questions": qs}


def _short(text: str, n: int = 80) -> str:
    """Shorten a question to use as a topic name without cutting a $...$ formula in half."""
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0]
    if cut.count("$") % 2:
        cut = cut[:cut.rfind("$")].rstrip()
    return cut + "…"


def probability(present: List[bool]) -> int:
    """present[i]: the topic was asked in paper i, papers ordered oldest -> newest.

    Half plain frequency (share of papers that asked it), half recency-weighted
    frequency (paper i has weight i+1, so last year counts most). With one paper
    there is no trend, so the estimate is capped at 60%.
    """
    n = len(present)
    if not n:
        return 0
    freq = sum(present) / n
    w = list(range(1, n + 1))
    recent = sum(wi for wi, p in zip(w, present) if p) / sum(w)
    p = 100 * (0.5 * freq + 0.5 * recent)
    if n == 1:
        p = min(p, 60)
    return int(round(max(5, min(95, p))))


def label(p: int) -> str:
    return "Very likely" if p >= 70 else "Likely" if p >= 45 else "Possible"


def predict(papers: List[Dict], subject: str = "", syllabus: str = "") -> Dict:
    """papers: [{source, year, questions: [{no, text, marks}]}].

    Returns {papers, topics: [...sorted by probability], unasked: [...]}.
    """
    papers = [p for p in papers if p.get("questions")]
    if not papers:
        raise ValueError("Add at least one past paper first.")
    # oldest -> newest; papers without a year keep their upload order at the end
    order = sorted(range(len(papers)), key=lambda i: (papers[i].get("year") is None, papers[i].get("year") or 0, i))
    papers = [papers[i] for i in order]
    qmap, lines = {}, []
    for pi, p in enumerate(papers):
        for q in p["questions"]:
            if len(qmap) >= MAX_QUESTIONS:
                break
            qid = f"Q{len(qmap) + 1}"
            qmap[qid] = {**q, "paper": pi}
            marks = f"{q['marks']:g} marks" if q.get("marks") else "marks ?"
            lines.append(f"{qid} | {p.get('year') or 'year ?'} | {marks} | {q['text'][:400]}")
    syllabus_part = (f"\nCourse syllabus (use its unit names in \"unit\"):\n{syllabus[:4000]}\n"
                     if syllabus.strip() else "\nNo syllabus was provided.\n")
    prompt = GROUP_PROMPT.format(n_papers=len(papers), subject=subject or "this course",
                                 syllabus_part=syllabus_part, questions="\n".join(lines),
                                 unit_hint="syllabus unit name" if syllabus.strip() else "broad area",
                                 math_rule=MATH_RULE)
    data = _json(generate(prompt, temperature=0.3, timeout_s=50, budget_s=135, json_mode=True,
                          models=PREDICT_MODELS, quick=True), list_key="topics")

    topics, used = [], set()
    for t in data.get("topics") or []:
        if not isinstance(t, dict):
            continue
        ids = [i for i in (t.get("question_ids") or []) if i in qmap and i not in used]
        if not ids:
            continue
        used.update(ids)
        asked = [qmap[i] for i in ids]
        present = [any(q["paper"] == pi for q in asked) for pi in range(len(papers))]
        prob = probability(present)
        years = sorted({papers[q["paper"]].get("year") for q in asked if papers[q["paper"]].get("year")})
        topics.append({
            "topic": str(t.get("topic") or "Topic").strip(), "unit": str(t.get("unit") or "").strip(),
            "probability": prob, "label": label(prob),
            "papers_asked": sum(present), "times_asked": len(asked), "years": years,
            "marks": sum(q.get("marks") or 0 for q in asked),
            "asked": [{"text": q["text"], "no": q.get("no"), "marks": q.get("marks"),
                       "year": papers[q["paper"]].get("year"), "source": papers[q["paper"]]["source"]}
                      for q in asked],
            "predicted": [str(x).strip() for x in (t.get("predicted") or []) if str(x).strip()][:3],
            "tip": str(t.get("tip") or "").strip(),
        })
    # questions the model left out still count, as their own topic
    for qid, q in qmap.items():
        if qid in used:
            continue
        present = [q["paper"] == pi for pi in range(len(papers))]
        prob = probability(present)
        y = papers[q["paper"]].get("year")
        topics.append({"topic": _short(q["text"]), "unit": "",
                       "probability": prob, "label": label(prob), "papers_asked": 1, "times_asked": 1,
                       "years": [y] if y else [], "marks": q.get("marks") or 0,
                       "asked": [{"text": q["text"], "no": q.get("no"), "marks": q.get("marks"),
                                  "year": y, "source": papers[q["paper"]]["source"]}],
                       "predicted": [], "tip": ""})
    topics.sort(key=lambda t: (-t["probability"], -t["times_asked"], -t["marks"]))
    unasked = [{"topic": str(u.get("topic") or "").strip(), "unit": str(u.get("unit") or "").strip()}
               for u in (data.get("unasked") or []) if isinstance(u, dict) and str(u.get("topic") or "").strip()]
    if not syllabus.strip():
        unasked = []          # without a syllabus the model can only guess what "should" be asked
    return {"papers": [{"source": p["source"], "year": p.get("year"), "questions": len(p["questions"])} for p in papers],
            "topics": topics, "unasked": unasked[:20]}
