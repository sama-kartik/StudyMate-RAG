"""Build an index of local PDFs (for the evaluation script and built-in documents).

Embeddings are cached in a JSON file per PDF so repeated runs don't call the
Gemini API again unless the PDF, the model or the chunk settings change.
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, List

from .config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_DIM, EMBEDDING_MODEL
from .pdf_loader import extract_pages
from .pipeline import index_pages


def _fingerprint(pdf: Path) -> str:
    h = hashlib.sha256(pdf.read_bytes())
    h.update(f"{EMBEDDING_MODEL}|{EMBEDDING_DIM}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode())
    return h.hexdigest()[:16]


def index_pdf(pdf: Path, cache_dir: Path) -> Dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{pdf.stem}.{_fingerprint(pdf)}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    pages = extract_pages(str(pdf))
    doc = {"source": pdf.name, "pages": len(pages), "chunks": index_pages(pdf.name, pages)}
    cache.write_text(json.dumps(doc))
    return doc


def index_folder(folder: Path, cache_dir: Path) -> List[Dict]:
    docs = []
    for pdf in sorted(folder.glob("*.pdf")):
        print(f"Indexing {pdf.name} …")
        docs.append(index_pdf(pdf, cache_dir))
    return docs
