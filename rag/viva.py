"""Viva simulator: an examiner that asks questions from the student's own material,
grades spoken or typed answers against that material, and asks follow-up questions
when an answer is weak."""
import json
import re
from typing import Dict, List, Optional

from .llm import generate
from .youtube import fmt_ts

DIFFICULTY = {
    "easy": "a straightforward recall question (a definition, a list or a single fact)",
    "medium": "a conceptual question that needs an explanation in the student's own words",
    "hard": "a demanding question that asks to compare, apply, justify or connect ideas",
}

EXAMINER = ("You are a strict but fair university viva examiner. You only use the course material "
            "provided to you. You never reveal the answer while asking a question.")


def _context(passages: List[Dict]) -> str:
    out = []
    for i, p in enumerate(passages, start=1):
        where = f"at {fmt_ts(p['page'])}" if p.get("kind") == "video" else f"page {p['page']}"
        out.append(f"[{i}] ({p['source']}, {where})\n{p['text']}")
    return "\n\n".join(out)


def _json(raw: str) -> Dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}


def ask(passages: List[Dict], difficulty: str = "medium", previous: Optional[List[str]] = None,
        follow_up: Optional[Dict] = None) -> Dict:
    """Return {"question", "topic"} for the next viva question."""
    if not passages:
        raise ValueError("Add a PDF or video first, then start the viva.")
    level = DIFFICULTY.get(difficulty, DIFFICULTY["medium"])
    asked = "\n".join(f"- {q}" for q in (previous or [])[-10:]) or "- (none yet)"
    if follow_up:
        task = (f"The student just answered this question weakly:\nQ: {follow_up.get('question')}\n"
                f"Their answer: {follow_up.get('answer') or '(no answer)'}\n"
                f"Points they missed: {'; '.join(follow_up.get('missing') or []) or 'unclear'}\n\n"
                "Ask ONE short follow-up question that probes the most important missed point, "
                "the way an examiner would ('You mentioned X — but why...?').")
    else:
        task = f"Ask ONE new viva question based on the material: {level}."
    prompt = f"""Course material:
{_context(passages)}

Questions already asked (do not repeat them):
{asked}

{task}
The question must be answerable from the material above, be clear and specific, and take
1-3 minutes to answer aloud. Reply as JSON: {{"question": "...", "topic": "2-5 word topic name"}}"""
    data = _json(generate(prompt, system=EXAMINER, temperature=0.7, timeout_s=40, budget_s=100, json_mode=True))
    q = str(data.get("question") or "").strip()
    if not q:
        raise ValueError("Couldn't create a question — please try again.")
    return {"question": q, "topic": str(data.get("topic") or "").strip()}


def grade(question: str, answer: str, passages: List[Dict], difficulty: str = "medium") -> Dict:
    """Grade an answer out of 10 against the material."""
    answer = (answer or "").strip()
    if not answer:
        return {"score": 0, "verdict": "No answer given.", "strengths": [],
                "missing": ["The question was not answered."], "model_answer": "", "follow_up": True}
    prompt = f"""Course material (numbered passages):
{_context(passages)}

Viva question ({difficulty}): {question}

Student's answer (may be transcribed from speech, so ignore small transcription errors):
\"\"\"{answer[:4000]}\"\"\"

Grade the answer ONLY against the course material above. Reply as JSON:
{{"score": <integer 0-10>,
  "verdict": "one sentence an examiner would say",
  "strengths": ["what the student got right", ...],
  "missing": ["an important point they missed or got wrong, citing the passage like [2]", ...],
  "model_answer": "a concise ideal answer (3-6 sentences) citing passages like [1]"}}
Scoring: 9-10 complete and precise; 7-8 mostly right with small gaps; 4-6 partly right;
1-3 mostly wrong or very vague; 0 irrelevant. Be specific; do not invent facts."""
    data = _json(generate(prompt, system=EXAMINER, temperature=0.2, timeout_s=45, budget_s=110, json_mode=True))
    try:
        score = max(0, min(10, int(round(float(data.get("score", 0))))))
    except (TypeError, ValueError):
        score = 0
    as_list = lambda v: [str(x) for x in v] if isinstance(v, list) else ([str(v)] if v else [])
    return {"score": score, "verdict": str(data.get("verdict") or ""),
            "strengths": as_list(data.get("strengths")), "missing": as_list(data.get("missing")),
            "model_answer": str(data.get("model_answer") or ""), "follow_up": score <= 5}
