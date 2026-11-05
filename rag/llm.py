"""Thin wrapper around the Google Gemini API: text generation and embeddings.

If the main generation model is overloaded (503), rate-limited (429) or retired
(404), the request is retried on the fallback models so a live demo keeps working.
"""
import logging
import math
import re
import time
from functools import lru_cache
from typing import Dict, List, Optional

from google import genai
from google.genai import errors, types

from .config import (EMBEDDING_DIM, EMBEDDING_MODEL, GEMINI_API_KEY,
                     GEMINI_FALLBACK_MODELS, GEMINI_MODEL)

log = logging.getLogger(__name__)
# Silence the SDK's harmless "non-text parts (thought_signature)" notice.
logging.getLogger("google_genai.types").setLevel(logging.ERROR)
_RETRYABLE = {404, 429, 500, 503, 504}
EMBED_BATCH = 100  # max texts per Gemini embedding request
GENERATE_BUDGET_S = 45  # default total time for one generate() call


class LLMError(RuntimeError):
    def __init__(self, message: str, status: int = 502, daily_quota: bool = False):
        super().__init__(message)
        self.status = status
        self.daily_quota = daily_quota  # retrying today won't help


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise LLMError(
            "GEMINI_API_KEY is not set. On Vercel: Project → Settings → Environment Variables, "
            "add GEMINI_API_KEY, then redeploy. Locally: copy .env.example to .env and add your key.")
    # One attempt per model: when a model is busy we move to the next one
    # instead of letting the SDK wait and retry the same overloaded model.
    return genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
            timeout=20_000,  # ms; a slow or hung model is skipped after 20 s
        ),
    )


def _models() -> List[str]:
    seen, out = set(), []
    for m in [GEMINI_MODEL] + GEMINI_FALLBACK_MODELS:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


# model name -> time.monotonic() until which its daily free quota is used up
_exhausted: Dict[str, float] = {}


def _daily_quota_wait(e: errors.APIError) -> Optional[float]:
    """If this 429 means 'daily quota used up', return seconds until it resets."""
    if e.code != 429:
        return None
    details = (e.details or {}).get("error", {}).get("details", []) if isinstance(e.details, dict) else []
    per_day, delay = False, 3600.0
    for d in details:
        for v in d.get("violations", []) or []:
            if "PerDay" in str(v.get("quotaId", "")):
                per_day = True
        m = re.match(r"([\d.]+)s", str(d.get("retryDelay", "")))
        if m:
            delay = float(m.group(1))
    return delay if per_day else None


def generate(prompt: str, system: str = None, temperature: float = 0.2,
             timeout_s: float = 20, budget_s: float = GENERATE_BUDGET_S) -> str:
    """Generate text, falling back to other models when one is busy, slow or out of quota.

    Models whose free daily quota is used up are skipped until it resets. If every
    remaining model is only temporarily busy, wait briefly and try another round.
    timeout_s: how long one model may take before we skip to the next.
    budget_s:  total time for this call (keep below vercel.json maxDuration).
    """
    last_err, quota_hit = None, False
    start = time.monotonic()
    while True:
        tried_any = False
        for model in _models():
            if _exhausted.get(model, 0) > time.monotonic():
                quota_hit = True
                continue
            remaining = budget_s - (time.monotonic() - start)
            if remaining < 5:
                break  # stay inside the hosting platform's request time limit
            tried_any = True
            config = types.GenerateContentConfig(
                system_instruction=system, temperature=temperature,
                http_options=types.HttpOptions(timeout=int(min(timeout_s, remaining) * 1000)),
            )
            try:
                resp = _client().models.generate_content(model=model, contents=prompt, config=config)
                return (resp.text or "").strip()
            except errors.APIError as e:
                last_err = e
                wait = _daily_quota_wait(e)
                if wait is not None:
                    _exhausted[model] = time.monotonic() + wait
                    quota_hit = True
                    log.warning("Gemini model %s: daily free quota used up, skipping it", model)
                    continue
                if e.code in _RETRYABLE:
                    log.warning("Gemini model %s failed (%s), trying next model", model, e.code)
                    continue
                raise LLMError(f"Gemini request failed: {e.message}", e.code or 502)
            except LLMError:
                raise
            except Exception as e:  # timeouts / network errors: the next model may still work
                log.warning("Gemini model %s failed (%s), trying next model", model, type(e).__name__)
                last_err = e
                continue
        # Every model was tried (or skipped). Another round only helps if some were
        # just busy, and only if there is time left.
        if not tried_any or budget_s - (time.monotonic() - start) < 15:
            break
        time.sleep(4)

    if quota_hit and not tried_any:
        raise LLMError("The free Gemini quota for today is used up on all models. It resets daily "
                       "(midnight US Pacific time); add billing in Google AI Studio to remove the limit.", 429, daily_quota=True)
    raise LLMError("Gemini is very busy right now. Please try again in a minute.",
                   getattr(last_err, "code", None) or 503)


def _normalise(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [round(x / n, 5) for x in v]  # rounding keeps JSON responses small


def embed(texts: List[str], query: bool = False) -> List[List[float]]:
    """Embed texts with Gemini. Vectors are unit length, so cosine similarity
    is just a dot product (which is what the browser computes)."""
    config = types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT",
        output_dimensionality=EMBEDDING_DIM,
    )
    out: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        for attempt in range(3):
            try:
                resp = _client().models.embed_content(model=EMBEDDING_MODEL, contents=batch, config=config)
                out.extend(_normalise(e.values) for e in resp.embeddings)
                break
            except errors.APIError as e:
                # No fallback model here: vectors from different models can't be
                # compared, so we wait briefly and retry the same model instead.
                if e.code in (429, 500, 503) and attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise LLMError(f"Gemini embedding failed: {e.message}", e.code or 502)
            except LLMError:
                raise
            except Exception as e:
                raise LLMError(f"Gemini embedding failed: {e}")
    return out
