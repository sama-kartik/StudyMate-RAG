"""Flask web server for the AI-powered LMS chatbot.

Run:  python app.py   then open http://127.0.0.1:5000
"""
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from rag import pipeline, vector_store
from rag.config import UPLOAD_DIR
from rag.llm import LLMError

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB per upload


@app.errorhandler(LLMError)
def handle_llm_error(e):
    return jsonify(error=str(e)), 502


@app.errorhandler(ValueError)
def handle_value_error(e):
    return jsonify(error=str(e)), 400


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/documents")
def documents():
    return jsonify(vector_store.list_documents())


@app.post("/api/upload")
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify(error="No file received."), 400
    results = []
    for f in files:
        name = secure_filename(f.filename or "")
        if not name.lower().endswith(".pdf"):
            results.append({"source": f.filename, "error": "Only PDF files are supported."})
            continue
        path = UPLOAD_DIR / name
        f.save(path)
        try:
            results.append(pipeline.ingest_pdf(str(path)))
        except ValueError as e:
            results.append({"source": name, "error": str(e)})
    return jsonify(results)


@app.delete("/api/documents/<path:source>")
def delete(source):
    vector_store.delete_document(source)
    path = UPLOAD_DIR / secure_filename(source)
    if path.exists():
        path.unlink()
    return jsonify(ok=True)


@app.post("/api/chat")
def chat():
    body = request.get_json(force=True)
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify(error="Please type a question."), 400
    return jsonify(pipeline.answer(question, source=body.get("source") or None))


@app.post("/api/summary")
def summary():
    body = request.get_json(force=True)
    return jsonify(pipeline.summarise(body.get("source", "")))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
