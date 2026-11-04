"""Central configuration. Values can be overridden with a .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
MODEL_DIR = DATA_DIR / "models"
for _d in (UPLOAD_DIR, CHROMA_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Keep downloaded embedding models inside the project instead of ~/.cache.
# Must be set before sentence_transformers / huggingface_hub are imported.
os.environ.setdefault("HF_HOME", str(MODEL_DIR))
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = "course_docs"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))        # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # characters shared between chunks
TOP_K = int(os.getenv("TOP_K", "5"))                    # chunks retrieved per question
