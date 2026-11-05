# StudyMate — AI-Powered LMS Chatbot using RAG

B.Tech Project-I, Department of CSE, Delhi Technological University (AY 2026-27)
Kartik Sama (23/CS/213) · Kartikey (23/CS/215) · Kislay Garg (23/CS/218)

A chatbot that answers students' questions **from their own course material**.
Upload PDFs (notes, syllabus, textbooks); the system retrieves the most relevant
passages and a language model (Google Gemini) writes an answer grounded in them,
citing the file and page each fact came from.

## Architecture

The server is **stateless**, so it runs on serverless hosting (Vercel). Each
user's documents and vectors are kept in **their own browser** (IndexedDB), and
no external database is used.

```
 Ingestion (once per PDF)
 ─────────────────────────────────────────────────────────────────────────────
 Browser: PDF ──► pdf.js text (per page) ──► POST /api/index (10 pages at a time)
 Server:  sentence-aware chunks (800 chars, 150 overlap) ──► Gemini embeddings (768-d)
 Browser: chunks + vectors (+ file, page) saved in IndexedDB   ◄── vector store

 Question answering
 ─────────────────────────────────────────────────────────────────────────────
 Browser: question ──► POST /api/embed-query ──► query vector
 Browser: cosine similarity against stored vectors ──► top-5 passages
 Server:  POST /api/answer ──► prompt with numbered passages ──► Gemini
 Browser: answer with [1][2] citations + source list (file, page, similarity)
```

| Part | Technology |
|---|---|
| PDF text extraction | pdf.js (browser); PyMuPDF for the local scripts |
| Chunking | `rag/chunker.py`: sentence-aligned, overlapping chunks |
| Embeddings | Google `gemini-embedding-2` (768 dimensions, unit-normalised) |
| Vector store and search | IndexedDB + cosine similarity (`public/static/app.js`; same algorithm in `rag/search.py`) |
| Answer generation | Google Gemini (`gemini-3.8-flash`, with automatic fallback models) |
| Web app | Flask (`app.py`), HTML/CSS/JavaScript (`templates/`, `public/static/`) |
| Evaluation | `eval/evaluate.py`: Hit@k, MRR, relevance, faithfulness, attribution, latency |

**Fallback models:** if Gemini's main model is overloaded, rate-limited or slow,
the next model in `GEMINI_FALLBACK_MODELS` is tried automatically, so the app
keeps answering during demand spikes.

**Free-tier limits:** Google's free Gemini tier allows about 20 requests per day
per model. Spreading requests over several models gives roughly 150 per day; a
summary of a 10-page PDF uses about 6 requests. Models whose daily quota is used
up are skipped until it resets (midnight US Pacific time). For heavier use, enable
billing for the API key in Google AI Studio.

## Run locally

1. Install Python 3.9 or newer.
2. Install the dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```
3. Get a free Gemini API key at <https://aistudio.google.com/apikey>, copy
   `.env.example` to `.env`, and paste the key into it.
4. Run `python app.py` and open <http://127.0.0.1:5000>.

## Deploy on Vercel

1. Import this GitHub repository at <https://vercel.com/new>. Vercel detects Flask
   automatically, and no build settings are needed.
2. Under **Settings → Environment Variables**, add `GEMINI_API_KEY` with your key.
3. Go to **Deployments**, open the **⋯** menu on the latest deployment, and click **Redeploy**.

Every `git push` to `main` redeploys automatically. `vercel.json` allows requests
up to 60 seconds, so slow Gemini responses aren't cut off.

## Using it

- **Upload** PDFs from the sidebar (drag-and-drop works). They are stored only in
  your browser, so another browser or device starts empty.
- **Ask** any question. Press Enter to send and Shift+Enter for a new line.
  Expand *Sources* under an answer to see the exact passages and similarity scores.
- **Search in** limits retrieval to one document.
- **Summarise** writes complete revision notes for a document. The PDF is split into
  sections of a few pages; each section gets detailed notes (every question/topic,
  definitions, lists, examples, with page numbers), then an overview, a list of topics
  and key takeaways are added. The result can be downloaded as a Markdown file.

Scanned PDFs (images only) have no extractable text and are rejected.

### Built-in documents

To make some PDFs available to every visitor, put them in `sample_docs/` and run
`python -m scripts.build_sample_index`, then commit and push. These PDFs become
public, so see `sample_docs/README.md` first.

## Evaluation

1. Put the PDFs your questions refer to in `eval/docs/`. They are git-ignored.
2. Write test questions in `eval/test_questions.json`. For each one, list the file
   and page(s) where the answer is:
   ```json
   {"question": "Define 3NF", "source": "DBMS_Unit2.pdf", "pages": [14, 15]}
   ```
3. Run:
   ```bash
   python -m eval.evaluate                  # full evaluation (answers + LLM judge)
   python -m eval.evaluate --retrieval-only # retrieval metrics only
   ```

| Metric | Meaning |
|---|---|
| Hit@k | Fraction of questions where a chunk from the correct page was retrieved |
| MRR | Mean reciprocal rank of the first correct chunk (1.0 = always ranked first) |
| Relevance (1-5) | LLM judge: does the answer address the question? |
| Faithfulness (1-5) | LLM judge: is every claim supported by the retrieved context? |
| Source attribution | Fraction of answers containing citations like [1] |
| Latency | Mean retrieval (query embedding + search) and generation time in seconds |

Per-question results are saved to `eval/results.csv`. Embeddings are cached in
`.cache/`, so re-runs only call Gemini for new or changed PDFs.

## Configuration

Set these in `.env` locally, or in Vercel's Environment Variables when hosted.

| Variable | Default | |
|---|---|---|
| `GEMINI_API_KEY` | (none) | Required |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Main answer model |
| `GEMINI_FALLBACK_MODELS` | `gemini-3.7-flash,gemini-3.5-flash,…` | Tried in order when the main model is busy |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `gemini-embedding-2` / 768 | Changing these requires re-uploading PDFs |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 150 | Characters |
| `TOP_K` | 5 | Passages retrieved per question |
