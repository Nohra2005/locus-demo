"""
Public-demo entry point for the attribute tagger.

Same service as main.py, but the LLM call goes through shared/free_llm.py
(Gemini free tier, then Groq) with per-minute and per-day caps, instead of
OpenRouter. When the budget is used up, tagging returns {} and search is
unaffected.

Run with:  uvicorn demo_main:app --port 8004
"""
import free_llm
import tagger
from main import app  # noqa: F401  (the FastAPI app, unchanged)


def _tag_free_tier(image_b64: str, prompt: str, _api_key: str):
    text, provider = free_llm.ask(lambda _max: (prompt, [image_b64]), max_tokens=300)
    if text is None:
        return None
    tagger.logger.info(f"tagger: answered by {provider}")
    return tagger._parse_json(text)


# tag_image() tries OpenRouter, then the "Gemini direct" function. Point the
# second one at the free-tier chain and switch the first one off.
tagger.OPENROUTER_API_KEY = ""
tagger.GOOGLE_API_KEY = "free-tier" if free_llm.enabled() else ""
tagger._tag_gemini_direct = _tag_free_tier
