"""Step 2: split page text into overlapping chunks.

Chunks are built from whole sentences where possible so that a chunk never
cuts a sentence in half. Each chunk remembers which file and page it came
from, which is what lets the chatbot cite its sources.
"""
import re
from typing import Dict, List

from .config import CHUNK_OVERLAP, CHUNK_SIZE

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _split_text(text: str, size: int, overlap: int) -> List[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]
    chunks, current = [], ""
    for sent in sentences:
        # A single very long "sentence" (tables, code) is hard-split.
        while len(sent) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sent[:size])
            sent = sent[size - overlap:]
        if len(current) + len(sent) + 1 <= size:
            current = (current + " " + sent).strip()
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            # start the tail on a word boundary
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            current = (tail + " " + sent).strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_pages(pages: List[Dict], source: str,
                size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Return [{"id", "text", "metadata": {"source", "page", "chunk"}}]."""
    out = []
    for p in pages:
        for j, text in enumerate(_split_text(p["text"], size, overlap)):
            out.append({
                "id": f"{source}::p{p['page']}::c{j}",
                "text": text,
                "metadata": {"source": source, "page": p["page"], "chunk": j},
            })
    return out
