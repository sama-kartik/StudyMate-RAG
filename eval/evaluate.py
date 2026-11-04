"""Evaluate the RAG chatbot on a set of test questions.

Metrics (matching the synopsis' evaluation plan):
  * Retrieval quality  - Hit@k and MRR: was the expected page among the retrieved chunks?
  * Answer relevance   - LLM judge, 1-5: does the answer address the question?
  * Faithfulness       - LLM judge, 1-5: is every claim supported by the retrieved context?
  * Source attribution - % of answers that cite at least one passage, e.g. [1]
  * Response time      - mean retrieval and generation latency (seconds)

Usage (from the project root, after uploading your PDFs in the app):
    python -m eval.evaluate                      # uses eval/test_questions.json
    python -m eval.evaluate --file my_tests.json
    python -m eval.evaluate --retrieval-only     # no Gemini calls, retrieval metrics only
Results are written to eval/results.csv.
"""
import argparse
import csv
import json
import re
import statistics
import time
from pathlib import Path

from rag import pipeline, vector_store
from rag.config import TOP_K
from rag.llm import generate

HERE = Path(__file__).resolve().parent

JUDGE_PROMPT = """You are grading a study-assistant chatbot.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
{answer}

Rate the answer on two criteria, each an integer from 1 (worst) to 5 (best):
- relevance: does the answer directly address the question?
- faithfulness: is every factual claim in the answer supported by the retrieved context?
  (An answer that correctly says the context does not cover the question is faithful.)

Reply with JSON only, e.g. {{"relevance": 4, "faithfulness": 5, "reason": "..."}}"""


def judge(question, hits, answer_text):
    ctx = "\n\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))
    raw = generate(JUDGE_PROMPT.format(question=question, context=ctx, answer=answer_text),
                   temperature=0.0)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    return data.get("relevance"), data.get("faithfulness"), data.get("reason", raw[:200])


def rank_of_expected(hits, case):
    """1-based rank of the first hit matching the expected source/pages, or None."""
    src, pages = case.get("source"), set(case.get("pages") or [])
    for i, h in enumerate(hits, 1):
        if (not src or h["source"] == src) and (not pages or h["page"] in pages):
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(HERE / "test_questions.json"))
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--delay", type=float, default=4.0,
                    help="seconds between Gemini calls (free tier is rate-limited)")
    args = ap.parse_args()

    cases = json.loads(Path(args.file).read_text())
    indexed = {d["source"] for d in vector_store.list_documents()}
    vector_store.search("warm-up", k=1)  # load the embedding model so it isn't timed
    rows = []

    for n, case in enumerate(cases, 1):
        q = case["question"]
        if case.get("source") and case["source"] not in indexed:
            print(f"[{n}] SKIP - '{case['source']}' is not uploaded")
            continue

        t0 = time.perf_counter()
        hits = vector_store.search(q, k=args.k)
        t_ret = time.perf_counter() - t0
        rank = rank_of_expected(hits, case)
        row = {"question": q, "expected_source": case.get("source", ""),
               "expected_pages": case.get("pages", ""), "rank": rank or "",
               "hit": int(rank is not None), "rr": round(1 / rank, 3) if rank else 0,
               "retrieval_s": round(t_ret, 3)}

        if not args.retrieval_only:
            res = pipeline.answer(q, k=args.k)
            rel, faith, reason = judge(q, hits, res["answer"])
            row.update({"generation_s": res["timings"]["generation"],
                        "cited": int(bool(re.search(r"\[\d+\]", res["answer"]))),
                        "relevance": rel, "faithfulness": faith, "judge_reason": reason,
                        "answer": res["answer"]})
            time.sleep(args.delay)

        rows.append(row)
        print(f"[{n}] hit={row['hit']} rank={row['rank'] or '-'}"
              + ("" if args.retrieval_only else
                 f" rel={row['relevance']} faith={row['faithfulness']} cited={row['cited']}")
              + f"  {q[:60]}")

    if not rows:
        print("No questions evaluated.")
        return

    def mean(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(statistics.mean(vals), 3) if vals else None

    print("\n===== SUMMARY =====")
    print(f"Questions evaluated : {len(rows)}")
    print(f"Hit@{args.k}             : {mean('hit')}")
    print(f"MRR                 : {mean('rr')}")
    print(f"Mean retrieval time : {mean('retrieval_s')} s")
    if not args.retrieval_only:
        print(f"Mean generation time: {mean('generation_s')} s")
        print(f"Answer relevance    : {mean('relevance')} / 5")
        print(f"Faithfulness        : {mean('faithfulness')} / 5")
        print(f"Source attribution  : {mean('cited')}  (fraction of answers with citations)")

    out = HERE / "results.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-question results saved to {out}")


if __name__ == "__main__":
    main()
