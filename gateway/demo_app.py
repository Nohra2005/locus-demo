"""
Public-demo entry point for the gateway.

Wraps the normal gateway app (main.py, unchanged) with one extra middleware:
in DEMO_MODE, only read requests and the shopper-facing search endpoints are
allowed. Store admin, catalog editing, golden-dataset editing, reindexing and
retraining are blocked, so nobody can change the shared demo catalog.

Run with:  uvicorn demo_app:app --port 7860
"""
import os

from fastapi import Request
from fastapi.responses import JSONResponse

import free_llm
import judge_batched
import main
from main import app

DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"

# Free-tier keys: swap in the batched judge (one LLM call per search instead
# of one per result). main.py only checks that some key is set.
if free_llm.enabled():
    main.run_judge = judge_batched.run_judge
    main.GOOGLE_API_KEY = main.GOOGLE_API_KEY or "free-tier"
else:
    main.OPENROUTER_API_KEY = ""
    main.GOOGLE_API_KEY = ""


@app.get("/llm-status")
def llm_status():
    """How much of the free LLM budget the judge has used."""
    return {"enabled": free_llm.enabled(), "providers": free_llm.status()}


# main.py ends with a catch-all route that serves the frontend; put this route first.
app.router.routes.insert(0, app.router.routes.pop())

# Write endpoints a shopper uses. Everything else that writes is blocked.
ALLOWED_WRITES = {
    "/search",
    "/refine",
    "/detect",
    "/classify-crop",
    "/feedback",
    "/track-event",
}

# Read endpoints that expose admin or account data.
BLOCKED_READS_PREFIXES = ("/admin", "/trigger-", "/skipped-products", "/golden-dataset")


@app.middleware("http")
async def demo_guard(request: Request, call_next):
    if DEMO_MODE:
        path, method = request.url.path, request.method
        if method in ("GET", "HEAD", "OPTIONS"):
            if path.startswith(BLOCKED_READS_PREFIXES) and not path.startswith("/golden-dataset/images"):
                return JSONResponse({"detail": "Disabled in the public demo."}, status_code=403)
        elif path not in ALLOWED_WRITES:
            return JSONResponse({"detail": "Disabled in the public demo."}, status_code=403)
    return await call_next(request)
