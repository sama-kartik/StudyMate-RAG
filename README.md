# StudyMate — AI-Powered LMS Chatbot using RAG

B.Tech Project-I, Department of CSE, Delhi Technological University (AY 2026-27)
Kartik Sama (23/CS/213) · Kartikey (23/CS/215) · Kislay Garg (23/CS/218)

A chatbot that answers students' questions **from their own course material**.
Upload PDFs (notes, syllabus, textbooks) or paste a **YouTube lecture link**; the
system retrieves the most relevant passages and a language model (Google Gemini)
writes an answer grounded in them, citing the file and page — or the exact moment
in the video — that each fact came from.

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

 YouTube lectures (once per video)
 ─────────────────────────────────────────────────────────────────────────────
 Server:  POST /api/youtube/info ──► real title + channel (YouTube oEmbed)
 Server:  POST /api/youtube/window ──► Gemini watches 5 minutes of the video
          (pictures + sound) ──► one timestamped note per 30 s
 Browser: notes indexed like pages whose "page" is the second in the video
 Browser: citations show ▶ 12:34 and open the video at that second
```

| Part | Technology |
|---|---|
| PDF text extraction | pdf.js (browser); PyMuPDF for the local scripts |
| Chunking | `rag/chunker.py`: sentence-aligned, overlapping chunks |
| Embeddings | Google `gemini-embedding-2` (768 dimensions, unit-normalised) |
| Vector store and search | IndexedDB + cosine similarity (`public/static/app.js`; same algorithm in `rag/search.py`) |
| Answer generation | Google Gemini (`gemini-3.8-flash`, with automatic fallback models) |
| Web app | Flask (`app.py`) with single-user login (`rag/auth.py`), HTML/CSS/JavaScript (`templates/`, `public/static/`) |
| YouTube lectures | `rag/youtube.py`: Gemini video understanding on public YouTube links, 5-minute windows, 30-second timestamped notes |
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

## Login

The app has a single login. Credentials are read from environment variables, never
from the code (this repository is public):

| Variable | |
|---|---|
| `APP_USER_EMAIL` | The login email |
| `APP_PASSWORD` | The password (as an encrypted Vercel environment variable), **or** |
| `APP_PASSWORD_HASH` | a password hash, e.g. in a local `.env` (command in `.env.example`) |
| `SECRET_KEY` | Optional; signs the session cookie (derived from the other secrets if unset) |

Everything except the login page requires signing in; sessions last 7 days when
"Keep me signed in" is ticked.

## Deploy on Vercel

1. Import this GitHub repository at <https://vercel.com/new>. Vercel detects Flask
   automatically, and no build settings are needed.
2. Open your **project's** **Settings → Environment Variables** (not the team's
   Shared variables) and add `GEMINI_API_KEY`, `APP_USER_EMAIL` and `APP_PASSWORD`.
3. Go to **Deployments**, open the **⋯** menu on the latest deployment, and click **Redeploy**.

Every `git push` to `main` redeploys automatically. `vercel.json` allows requests
up to 150 seconds, so slow Gemini responses (and video windows) aren't cut off.

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

### PDFs with images and scanned pages

pdf.js detects pages that have no text layer (scans) and pages that contain
pictures. Those pages are rendered to images and read by Gemini vision
(`rag/vision.py`): scans are transcribed in full (tables as tables, formulas as
text), and pictures on normal pages are described as `[Figure: …]` lines. The
result is indexed like ordinary page text, so questions and summaries can use
diagrams and scanned notes. The option **Read images, diagrams & scanned pages**
under the upload box can be turned off for picture-heavy PDFs (scanned pages are
always read).

### Subjects

Documents and videos are organised into **subjects** (e.g. "DBMS", "Signals & Systems"),
each a separate space: chat answers, summaries and viva questions only use the
current subject's material. Create, rename or delete subjects from the bar on the
Home page; move documents between subjects on **My Documents**. Mark one PDF per
subject as its **syllabus** — the viva examiner then keeps questions within it.

### Viva practice

The **Viva Practice** page simulates an oral exam on your own material. StudyMate
picks a random page or video moment, Gemini (as the examiner) asks a question —
read aloud by the browser — and you answer by **speaking** (Web Speech API, live
transcription; Chrome, Edge, Safari) or by typing. The answer is graded out of 10
against the material (`rag/viva.py`): what you got right, what you missed (with
citations to pages or video timestamps) and a model answer. A weak answer (≤ 5)
triggers a **follow-up question** on the missed point, like a real examiner. The next
question is prepared while you answer, so it usually appears instantly. **End viva**
shows the report at any point: score, grade, per-question feedback, topics to
revise and a downloadable report.

### Exam Predictor (past papers)

The **Exam Predictor** page takes previous-year question papers for the current subject —
normal PDFs, scanned PDFs, or phone photos of a paper (select all photos of one paper
together). Scanned pages and photos are read with Gemini vision, then Gemini lists every
question with its marks and the exam year (editable if it guessed wrong).

**Predict my exam** groups all the questions into topics (mapped to syllabus units when a
syllabus is set) and writes two likely new questions per topic. The probability is not a
model guess: it is computed in `rag/predictor.py` as half the share of papers that asked the
topic and half a recency-weighted share (the latest paper counts most), capped at 60% when
there is only one paper. Each topic links to where your notes or lectures cover it (video
links open at the right second), and the report can be downloaded as Markdown.

### YouTube lectures

Paste a YouTube link in the box under the upload area and click **Add video**.
Gemini watches the lecture directly from the link (what is said *and* what is shown
on slides or the board) in five-minute parts, **three parts in parallel**, and writes
one note per 30 seconds. The video length is read from YouTube so every part can
start at once.
The notes are indexed like PDF pages, so you can ask questions across PDFs and
videos together. Answers cite the moment, e.g. **▶ 10:49**: clicking a citation opens
the video inside StudyMate at that second. **Summarise** works for videos too,
with timestamps instead of page numbers.

- Only **public** videos work (not private or unlisted).
- The free Gemini tier allows about **8 hours of YouTube video per day**; a
  20-minute lecture takes four requests and, when Gemini is not busy, under a minute.
- The title comes from YouTube's oEmbed endpoint, not the model (models were seen
  inventing plausible but wrong titles).

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
