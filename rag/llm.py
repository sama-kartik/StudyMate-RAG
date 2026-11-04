"""Thin wrapper around the Google Gemini API.

If the main model is overloaded (503), rate-limited (429) or retired (404), the
request is retried on the fallback models so a live demo keeps working.
"""
import logging
from functools import lru_cache
from typing import List

from google import genai
from google.genai import errors, types

from .config import GEMINI_API_KEY, GEMINI_FALLBACK_MODELS, GEMINI_MODEL

log = logging.getLogger(__name__)
# Silence the SDK's harmless "non-text parts (thought_signature)" notice.
logging.getLogger("google_genai.types").setLevel(logging.ERROR)
_RETRYABLE = {404, 429, 500, 503}


class LLMError(RuntimeError):
    pass


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
            timeout=30_000,  # ms; a hung model is skipped after 30 s
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
    for model in _models():
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
        except Exception as e:  # network errors
            last_err = e
            break
    msg = getattr(last_err, "message", None) or str(last_err)
    raise LLMError(f"Gemini request failed: {msg}")
