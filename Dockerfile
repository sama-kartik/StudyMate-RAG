# Container image for Hugging Face Spaces (Docker SDK). Also runs anywhere Docker does:
#   docker build -t studymate . && docker run -p 7860:7860 -e GEMINI_API_KEY=... studymate
FROM python:3.11-slim

# Spaces runs containers as user 1000; give it a writable home and app folder.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /app

# CPU-only PyTorch keeps the image ~1.5 GB smaller than the default CUDA build.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt gunicorn

COPY --chown=user . .

# Download the embedding model at build time so the first question isn't slow.
RUN python -c "from rag.vector_store import embed; embed(['warm-up'])"

EXPOSE 7860
# One worker: ChromaDB and the embedding model live in this process's memory.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "180", "app:app"]
