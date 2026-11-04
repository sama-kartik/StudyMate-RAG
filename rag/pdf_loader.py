"""Step 1 of the pipeline: extract text from a PDF, page by page (PyMuPDF)."""
import re
from typing import Dict, List

import fitz  # PyMuPDF


def _clean(text: str) -> str:
    text = text.replace("­", "")              # soft hyphens
    text = re.sub(r"-\n(\w)", r"\1", text)          # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_path: str) -> List[Dict]:
    """Return [{"page": 1, "text": "..."}, ...], skipping empty pages."""
    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = _clean(page.get_text("text"))
            if text:
                pages.append({"page": i, "text": text})
    return pages
