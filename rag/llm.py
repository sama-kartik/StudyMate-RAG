"""Thin wrapper around the Google Gemini API: text generation and embeddings.

If the main generation model is overloaded (503), rate-limited (429) or retired
(404), the request is retried on the fallback models so a live demo keeps working.
"""
import logging
import math
import time
from functools import lru_cache
from typing import List

from google import genai
from google.genai import errors, types

from .config import (EMBEDDING_DIM, EMBEDDING_MODEL, GEMINI_API_KEY,
                     GEMINI_FALLBACK_MODELS, GEMINI_MODEL)

log = logging.getLogger(__name__)
# Silence the SDK's harmless "non-text parts (thought_signature)" notice.
logging.getLogger("google_genai.types").setLevel(logging.ERROR)
_RETRYABLE = {404, 429, 500, 503, 504}
EMBED_BATCH = 100  # max texts per Gemini embedding request
GENERATE_BUDGET_S = 40  # stop trying further models after this (Vercel limit is 60 s)


class LLMError(RuntimeError):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
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


def generate(prompt: str, system: str = None, temperature: float = 0.2) -> str:
    config = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
    last_err = None
    start = time.monotonic()
    for model in _models():
        if time.monotonic() - start > GENERATE_BUDGET_S:
            break  # stay well inside the hosting platform's request time limit
        try:
            resp = _client().models.generate_content(model=model, contents=prompt, config=config)
            return (resp.text or "").strip()
        except errors.APIError as e:
            last_err = e
            if e.code in _RETRYABLE:
                log.warning("Gemini model %s failed (%s), trying next model", model, e.code)
                continue
            break  # e.g. 400 bad request / 403 invalid key: other models won't help
        except LLMError:
            raise
        except Exception as e:  # timeouts / network errors: the next model may still work
            log.warning("Gemini model %s failed (%s), trying next model", model, type(e).__name__)
            last_err = e
            continue
    if last_err is None:
        raise LLMError("Gemini is busy right now. Please try again in a minute.", 503)
    msg = getattr(last_err, "message", None) or str(last_err) or type(last_err).__name__
    raise LLMError(f"Gemini is busy or unreachable right now ({msg}). Please try again in a minute.",
                   getattr(last_err, "code", None) or 502)


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
