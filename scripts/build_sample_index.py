"""Pre-index the PDFs in sample_docs/ so the hosted app has them built in.

Usage (from the project root):
    pip install -r requirements-dev.txt
    python -m scripts.build_sample_index
Then commit public/static/sample_index.json and push; Vercel redeploys automatically.
Everything in sample_docs/ becomes public, so only add material you may share.
"""
import json

from rag.config import BASE_DIR, EMBEDDING_DIM, EMBEDDING_MODEL
from rag.local_index import index_folder

SAMPLE_DIR = BASE_DIR / "sample_docs"
OUT = BASE_DIR / "public" / "static" / "sample_index.json"


def main():
    docs = index_folder(SAMPLE_DIR, BASE_DIR / ".cache" / "index")
    OUT.write_text(json.dumps({"model": EMBEDDING_MODEL, "dim": EMBEDDING_DIM, "docs": docs},
                              separators=(",", ":")))
    total = sum(len(d["chunks"]) for d in docs)
    print(f"Wrote {OUT.relative_to(BASE_DIR)}: {len(docs)} document(s), {total} chunks, "
          f"{OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
