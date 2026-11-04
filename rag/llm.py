"""Thin wrapper around the Google Gemini API."""
from functools import lru_cache

from google import genai
from google.genai import types

from .config import GEMINI_API_KEY, GEMINI_MODEL


class LLMError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
    return genai.Client(api_key=GEMINI_API_KEY)


def generate(prompt: str, system: str = None, temperature: float = 0.2) -> str:
    try:
        resp = _client().models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=temperature),
        )
    except LLMError:
        raise
    except Exception as e:  # network, quota, invalid key...
        raise LLMError(f"Gemini request failed: {e}") from e
    return (resp.text or "").strip()
