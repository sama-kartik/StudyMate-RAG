"""Flask web server for the AI-powered LMS chatbot.

Local:   python app.py   then open http://127.0.0.1:5000
Vercel:  detected automatically (this file exports `app`).

The server is stateless: it chunks, embeds and generates, while each user's
documents and vectors are stored in their own browser (see public/static/app.js).
"""
from flask import Flask, jsonify, render_template, request

from rag import pipeline
from rag.config import EMBEDDING_DIM, EMBEDDING_MODEL, TOP_K
from rag.llm import LLMError
from rag.search import MIN_SCORE

# Static files live in public/static: Vercel serves public/** from its CDN, and
# locally Flask serves the same folder at the same /static URL.
app = Flask(__name__, static_folder="public/static", static_url_path="/static")
# Vercel rejects request bodies over 4.5 MB; the browser sends pages in small batches.
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024
MAX_PAGES_PER_REQUEST = 25


@app.errorhandler(LLMError)
def handle_llm_error(e):
    status = 429 if e.status == 429 else 502
    return jsonify(error=str(e)), status


@app.errorhandler(ValueError)
def handle_value_error(e):
    return jsonify(error=str(e)), 400


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Request too large."), 413


def _body():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON body.")
    return data


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/config")
def config():
    # The browser checks this so vectors from a different model are never mixed.
    return jsonify(embedding_model=EMBEDDING_MODEL, embedding_dim=EMBEDDING_DIM,
                   top_k=TOP_K, min_score=MIN_SCORE)


@app.post("/api/index")
def index_pages():
    body = _body()
    source = (body.get("source") or "").strip()
    pages = body.get("pages") or []
    if not source or not isinstance(pages, list):
        raise ValueError("Send {source, pages: [{page, text}]}.")
    if len(pages) > MAX_PAGES_PER_REQUEST:
        raise ValueError(f"Send at most {MAX_PAGES_PER_REQUEST} pages per request.")
    clean = [{"page": int(p["page"]), "text": str(p.get("text", ""))} for p in pages]
    return jsonify(chunks=pipeline.index_pages(source, clean))


@app.post("/api/embed-query")
def embed_query():
    q = (_body().get("question") or "").strip()
    if not q:
        raise ValueError("Please type a question.")
    return jsonify(embedding=pipeline.embed_query(q))


@app.post("/api/answer")
def answer():
    body = _body()
    q = (body.get("question") or "").strip()
    if not q:
        raise ValueError("Please type a question.")
    passages = [{"source": str(p["source"]), "page": int(p["page"]), "text": str(p["text"]),
                 "score": float(p.get("score", 0))} for p in (body.get("passages") or [])]
    return jsonify(pipeline.answer(q, passages))


@app.post("/api/summary")
def summary():
    body = _body()
    chunks = [{"page": int(c["page"]), "text": str(c["text"])} for c in (body.get("chunks") or [])]
    return jsonify(pipeline.summarise(str(body.get("source", "")), chunks))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
