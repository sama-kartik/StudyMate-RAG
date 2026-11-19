"""Reading PDF pages as images with Gemini vision.

The browser renders a page to JPEG when its text layer is missing (a scanned page)
or when it contains pictures. Gemini then either transcribes the whole page or
describes only its visual elements (diagrams, charts, tables, formulas), and the
result is indexed exactly like ordinary page text.
"""
import base64
import json
import re
from typing import Dict, List

from google.genai import types

from .llm import generate

MAX_PAGES_PER_REQUEST = 3
MAX_IMAGE_BYTES = 1_200_000

TRANSCRIBE = """Page {page} of "{source}" is a scanned image. Transcribe it for a study assistant:
- All text in natural reading order, keeping headings, lists and question numbers.
- Tables as Markdown tables; formulas in plain text (e.g. x^2 + y^2 = r^2).
- For each diagram, chart or figure, add a line "[Figure: ...]" describing what it shows,
  including labels, axes and the point it makes.
Do not add anything that is not on the page."""

VISUALS = """Page {page} of "{source}" contains pictures. Its text has already been extracted:
---
{text}
---
Describe ONLY the visual elements that this text does not already contain: diagrams,
charts, graphs, figures, photos of the board, and tables or formulas drawn as images.
For each one write a line "[Figure: ...]" with what it shows (labels, axes, values,
relationships and the point it makes). If the pictures are only decoration (logos,
borders, photos with no information), return an empty string."""

PROMPT_TAIL = """
Reply as JSON: {{"pages": [{{"page": <number>, "text": "..."}}]}} with one entry per page, in order."""


def read_pages(source: str, pages: List[Dict]) -> List[Dict]:
    """pages: [{page, image (data URL or base64 JPEG), mode: "transcribe"|"visuals", text}]."""
    if not pages:
        return []
    if len(pages) > MAX_PAGES_PER_REQUEST:
        raise ValueError(f"Send at most {MAX_PAGES_PER_REQUEST} page images per request.")
    parts: List[types.Part] = []
    for p in pages:
        raw = str(p.get("image") or "")
        raw = raw.split(",", 1)[1] if raw.startswith("data:") else raw
        try:
            data = base64.b64decode(raw, validate=True)
        except Exception:
            raise ValueError(f"Page {p.get('page')}: the image is not valid base64.")
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError(f"Page {p.get('page')}: the image is too large.")
        page = int(p["page"])
        if p.get("mode") == "visuals":
            instr = VISUALS.format(page=page, source=source, text=str(p.get("text") or "")[:6000])
        else:
            instr = TRANSCRIBE.format(page=page, source=source)
        parts.append(types.Part(text=f"=== PAGE {page} ==="))
        parts.append(types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=data)))
        parts.append(types.Part(text=instr))
    parts.append(types.Part(text=PROMPT_TAIL))

    raw = generate([types.Content(role="user", parts=parts)], temperature=0.1,
                   timeout_s=60, budget_s=130, json_mode=True)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {"pages": []}
    by_page = {}
    for item in data.get("pages") or []:
        try:
            by_page[int(item.get("page"))] = str(item.get("text") or "").strip()
        except (TypeError, ValueError):
            continue
    return [{"page": int(p["page"]), "text": by_page.get(int(p["page"]), "")} for p in pages]
