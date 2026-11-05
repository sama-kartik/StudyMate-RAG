---
title: StudyMate RAG
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Course-material chatbot using Retrieval-Augmented Generation
---

# StudyMate — AI-Powered LMS Chatbot using RAG

B.Tech Project-I, Department of CSE, Delhi Technological University (AY 2026-27)
Kartik Sama (23/CS/213) · Kartikey (23/CS/215) · Kislay Garg (23/CS/218)

A chatbot that answers students' questions **from their own course material**.
Upload PDFs (notes, syllabus, textbooks); the system retrieves the most relevant
passages and a language model (Google Gemini) writes an answer grounded in them,
citing the file and page each fact came from.

## Architecture

```
            ┌───────────── Ingestion (once per PDF) ─────────────┐
 PDF ──► PyMuPDF text ──► sentence-aware chunks ──► MiniLM embeddings ──► ChromaDB
          (per page)        (800 chars, 150 overlap)   (384-dim vectors)   (+ file, page)

            ┌───────────── Question answering ───────────────────┐
 Question ──► embed ──► top-5 similar chunks ──► prompt with numbered context ──► Gemini
                         (cosine similarity)                                       │
 Answer with [1][2] citations + source list (file, page, excerpt) ◄────────────────┘
```

| Module | Role |
|---|---|
| `rag/pdf_loader.py` | Extract and clean text page by page |
| `rag/chunker.py` | Split into overlapping, sentence-aligned chunks with metadata |
| `rag/vector_store.py` | Embeddings (sentence-transformers) + ChromaDB index/search |
| `rag/llm.py` | Google Gemini API wrapper |
| `rag/pipeline.py` | Ingest, answer (RAG), summarise |
| `app.py` | Flask REST API + web UI |
| `eval/evaluate.py` | Evaluation: Hit@k, MRR, relevance, faithfulness, attribution, latency |

## Setup

1. Python 3.9+ is required.
2. Install:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Get a free Gemini API key at <https://aistudio.google.com/apikey>, then
   copy `.env.example` to `.env` and paste the key into it.
4. Run:
   ```bash
   python app.py
   ```
   Open <http://127.0.0.1:5000>. The first upload downloads the embedding model (~90 MB).

## Using it

- **Upload** PDFs from the sidebar (drag-and-drop works).
- **Ask** any question. Press Enter to send and Shift+Enter for a new line.
  Expand *Sources* under an answer to see the exact passages used.
- **Search in** limits retrieval to one document.
- **Summarise** produces a revision summary of a document.

Scanned PDFs (images only) have no extractable text and are rejected.

## Evaluation

1. Upload your documents in the app.
2. Write test questions in `eval/test_questions.json`. For each one, list the file
   and page(s) where the answer is:
   ```json
   {"question": "Define 3NF", "source": "DBMS_Unit2.pdf", "pages": [14, 15]}
   ```
3. Run:
   ```bash
   python -m eval.evaluate                  # full evaluation (uses Gemini)
   python -m eval.evaluate --retrieval-only # retrieval metrics only, no API calls
   ```

| Metric | Meaning |
|---|---|
| Hit@k | Fraction of questions where a chunk from the correct page was retrieved |
| MRR | Mean reciprocal rank of the first correct chunk (1.0 = always ranked first) |
| Relevance (1-5) | LLM judge: does the answer address the question? |
| Faithfulness (1-5) | LLM judge: is every claim supported by the retrieved context? |
| Source attribution | Fraction of answers containing citations like [1] |
| Latency | Mean retrieval and generation time in seconds |

Per-question results are saved to `eval/results.csv` for your report.

## Configuration (`.env`)

| Variable | Default | |
|---|---|---|
| `GEMINI_MODEL` | `gemini-3.8-flash` | Any Gemini model available to your key |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 150 | Characters |
| `TOP_K` | 5 | Chunks retrieved per question |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Any sentence-transformers model |

To reset everything, stop the app and delete the `data/` folder.
