"""
E2E tests — hits the deployed cloud URL end-to-end.
Requires GATEWAY_URL env var pointing to the live deployment.

Run: GATEWAY_URL=http://20.240.203.22:8000 pytest tests/test_e2e.py -v
     (skipped automatically if GATEWAY_URL is not set)
"""
import io
import os

import pytest
import requests

GATEWAY_URL = os.getenv("GATEWAY_URL", "").rstrip("/")
TIMEOUT     = 45

_IMG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "mlops", "golden_images",
    "8cce3f151dbf3b2c5dd9.jpeg",   # pants — reliable category detection
)


def _skip_if_no_url():
    if not GATEWAY_URL:
        pytest.skip("GATEWAY_URL not set — skipping e2e tests")


@pytest.fixture(scope="module")
def image_bytes():
    with open(_IMG_PATH, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def search_response(image_bytes):
    _skip_if_no_url()
    r = requests.post(
        f"{GATEWAY_URL}/search",
        files={"file": ("pants.jpg", io.BytesIO(image_bytes), "image/jpeg")},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


# ── Availability ───────────────────────────────────────────────────────────────

class TestAvailability:
    def test_gateway_online(self):
        _skip_if_no_url()
        r = requests.get(f"{GATEWAY_URL}/health", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_health_endpoint(self):
        _skip_if_no_url()
        r = requests.get(f"{GATEWAY_URL}/health", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_index_stats_reachable(self):
        _skip_if_no_url()
        r = requests.get(f"{GATEWAY_URL}/index-stats", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_index_is_populated(self):
        _skip_if_no_url()
        r = requests.get(f"{GATEWAY_URL}/index-stats", timeout=TIMEOUT)
        data = r.json()
        counts = [v for v in data.values() if isinstance(v, (int, float))]
        assert any(c > 0 for c in counts), f"Deployed index appears empty: {data}"


# ── Search Quality ─────────────────────────────────────────────────────────────

class TestSearchQuality:
    def test_returns_non_empty_results(self, search_response):
        assert len(search_response.get("matches", [])) > 0, (
            "Deployed /search returned no results"
        )

    def test_response_schema(self, search_response):
        assert "search_id"        in search_response
        assert "detected_category" in search_response
        assert "matches"           in search_response

    def test_match_schema(self, search_response):
        match = search_response["matches"][0]
        assert "product_id" in match
        assert "score"       in match
        assert "image_url"   in match
        assert isinstance(match["score"], float)
        assert 0.0 <= match["score"] <= 1.0

    def test_results_have_valid_image_urls(self, search_response):
        for match in search_response["matches"][:3]:
            url = match.get("image_url", "")
            assert url.startswith("http"), f"Invalid image URL: {url}"

    def test_detect_endpoint_works(self, image_bytes):
        _skip_if_no_url()
        r = requests.post(
            f"{GATEWAY_URL}/detect",
            files={"file": ("pants.jpg", io.BytesIO(image_bytes), "image/jpeg")},
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        assert "detections" in r.json()

    def test_feedback_endpoint_accepts_submission(self, search_response):
        product_id = search_response["matches"][0]["product_id"]
        search_id  = search_response["search_id"]
        r = requests.post(
            f"{GATEWAY_URL}/feedback",
            json={"result_product_id": product_id, "rating": 4, "source": "e2e"},
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
