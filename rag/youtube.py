"""YouTube lectures as a knowledge source.

Gemini watches a public YouTube video directly from its link (pictures and sound),
in 10-minute windows, and writes timestamped study notes. Those notes are then
chunked and embedded like PDF pages, except that the "page" of a chunk is the
second of the video it comes from, so answers can cite e.g. "Lecture at 12:34".
"""
import json
import re
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

from google.genai import types

from .llm import LLMError, generate

WINDOW_S = 300          # length of one window Gemini watches per request
SLOT_S = 30            # one note per 30 seconds of video
GAP_S = 75             # notes ending this long before the window's end trigger a top-up
REQUEST_BUDGET_S = 140 # stay under the 150 s Vercel function limit
MAX_VIDEO_S = 3 * 3600  # stop after 3 hours
# Video is heavy: fast "lite" models first (a 10-minute window takes ~15-30 s and they
# are rarely overloaded), larger models as fallbacks.
VIDEO_MODELS = ["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-3-flash-preview",
                "gemini-3.5-flash", "gemini-flash-latest", "gemini-3.8-flash"]

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def parse_video_id(url: str) -> Optional[str]:
    """Accepts watch?v=, youtu.be/, /shorts/, /embed/, /live/ links or a bare id."""
    url = (url or "").strip()
    if _ID_RE.match(url):
        return url
    try:
        u = urllib.parse.urlparse(url if "://" in url else "https://" + url)
    except ValueError:
        return None
    host = (u.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    cand = None
    if host == "youtu.be":
        cand = u.path.lstrip("/").split("/")[0]
    elif host in ("youtube.com", "music.youtube.com", "youtube-nocookie.com"):
        if u.path == "/watch":
            cand = urllib.parse.parse_qs(u.query).get("v", [""])[0]
        else:
            parts = u.path.strip("/").split("/")
            if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
                cand = parts[1]
    return cand if cand and _ID_RE.match(cand) else None


def video_info(video_id: str) -> Dict:
    """Real title and channel from YouTube's public oEmbed endpoint (no API key).

    The model is not asked for the title: in testing it invented a plausible but wrong one.
    """
    watch = f"https://www.youtube.com/watch?v={video_id}"
    q = urllib.parse.urlencode({"url": watch, "format": "json"})
    req = urllib.request.Request(f"https://www.youtube.com/oembed?{q}", headers={"User-Agent": "StudyMate/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.load(r)
    except Exception:
        raise ValueError("That video can't be used. Check the link, and make sure the video is public "
                         "(private and unlisted videos are not supported).")
    return {"video_id": video_id, "url": watch, "title": data.get("title") or "YouTube video",
            "author": data.get("author_name") or ""}


def fmt_ts(seconds: int) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_ts(ts) -> Optional[int]:
    if isinstance(ts, (int, float)):
        return int(ts)
    parts = re.findall(r"\d+", str(ts))
    if not parts or len(parts) > 3:
        return None
    total = 0
    for p in parts:
        total = total * 60 + int(p)
    return total


NOTES_PROMPT = """You are watching part of a lecture video, from {start} to {end}.
Write study notes as JSON {{"segments": [{{"start": "MM:SS", "text": "..."}}]}} with exactly one
segment for EACH of these start times: {slots}.

Rules:
- Each segment covers only what is said AND shown from its start time until the next one:
  definitions, explanations, formulas, worked examples, slide or board text, and diagrams
  described in words. Write 2-4 informative sentences; do not write "the speaker says".
- Use the video's own clock for "start" (H:MM:SS if over an hour).
- Do not repeat content between segments. Only include content that is in the video.
- If the video ends before a listed time, stop there. If there is no content, return {{"segments": []}}.
"""


def _notes(video_id: str, a: int, b: int, budget_s: float = 120) -> List[Dict]:
    """Timestamped notes for the stretch a..b seconds. Raises LLMError (400 = past the end)."""
    video = types.Part(
        file_data=types.FileData(file_uri=f"https://www.youtube.com/watch?v={video_id}", mime_type="video/*"),
        video_metadata=types.VideoMetadata(start_offset=f"{a}s", end_offset=f"{b}s"),
    )
    # Fixed 30-second slots keep timestamps aligned with the video; without them, fast
    # models compressed the timeline and skipped or repeated material.
    slots = ", ".join(fmt_ts(t) for t in range(a, b, SLOT_S))
    prompt = types.Part(text=NOTES_PROMPT.format(start=fmt_ts(a), end=fmt_ts(b), slots=slots))
    raw = generate([types.Content(role="user", parts=[video, prompt])], temperature=0.2,
                   timeout_s=60, budget_s=budget_s, json_mode=True, low_media_resolution=True,
                   models=VIDEO_MODELS)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {"segments": []}
    segs: List[Dict] = []
    for item in data.get("segments") or []:
        t = parse_ts(item.get("start"))
        text = str(item.get("text") or "").strip()
        # keep notes that fall inside this stretch (small tolerance for rounding)
        if t is None or not text or t < a - 30 or t > b + 30:
            continue
        segs.append({"t": t, "text": text})
    return sorted(segs, key=lambda x: x["t"])


def watch_window(video_id: str, start_s: int) -> Dict:
    """Notes for one window. Returns {"segments", "end", "next"}.

    Models sometimes stop writing part-way through a window, so if the notes end
    well before the window does, the missing stretch is requested once more.
    """
    if start_s >= MAX_VIDEO_S:
        return {"segments": [], "end": True, "next": start_s}
    end_s = start_s + WINDOW_S
    t0 = time.monotonic()
    try:
        segs = _notes(video_id, start_s, end_s, budget_s=120)
    except LLMError as e:
        # Gemini rejects a start offset beyond the end of the video with 400 INVALID_ARGUMENT.
        if e.status == 400 and start_s > 0:
            return {"segments": [], "end": True, "next": start_s}
        if e.status == 400:
            raise ValueError("Gemini couldn't open this video. Only public YouTube videos are supported.")
        raise
    if not segs:
        return {"segments": [], "end": True, "next": end_s}
    last = segs[-1]["t"]
    left = REQUEST_BUDGET_S - (time.monotonic() - t0)
    if last < end_s - GAP_S and left >= 40:
        try:
            seen = {x["t"] for x in segs}
            segs += [x for x in _notes(video_id, last + SLOT_S, end_s, budget_s=left - 5)
                     if x["t"] > last and x["t"] not in seen]
        except LLMError:
            pass  # past the real end of the video, or busy: keep what we have
    return {"segments": segs, "end": False, "next": end_s}
