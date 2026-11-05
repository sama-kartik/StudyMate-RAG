"""Central configuration. Values can be overridden with a .env file (locally) or
with Environment Variables in the Vercel dashboard (when hosted)."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Answer generation: tried in order; later models are used when earlier ones are busy.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_FALLBACK_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_FALLBACK_MODELS", "gemini-3.5-flash,gemini-flash-lite-latest,gemini-flash-latest,gemini-3.5-flash-lite"
).split(",") if m.strip()]

# Embeddings. Changing either value makes previously stored vectors incompatible,
# so users would have to re-upload their PDFs.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))        # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # characters shared between chunks
TOP_K = int(os.getenv("TOP_K", "5"))                    # chunks retrieved per question
