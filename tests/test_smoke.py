"""
tests/test_smoke.py — Smoke tests for Locus services.

These tests verify that all services are reachable and responding correctly.

CI (K3s): runs automatically — gateway and visual engine are reachable via
    K3s NodePort / port-forward; MLflow tests are skipped (not in K3s).

Local dev: requires the full stack running:
    docker compose up -d

Run with:
    pytest tests/test_smoke.py -v
    GATEWAY_URL=http://20.240.203.22:8000 pytest tests/test_smoke.py -v
"""
import os

import pytest
import requests

GATEWAY_URL      = os.getenv("GATEWAY_URL", "http://localhost:8000").rstrip("/")
VISUAL_URL       = os.getenv("VISUAL_URL",  "http://localhost:8001").rstrip("/")
MLFLOW_URL       = os.getenv("MLFLOW_URL",  "http://localhost:5000").rstrip("/")
TIMEOUT          = 10   # fast endpoints
TIMEOUT_SLOW     = 60   # Qdrant-heavy endpoints (scroll full collection)

_mlflow_available = bool(os.getenv("MLFLOW_URL"))


# ── Gateway ───────────────────────────────────────────────────────────────────

class TestGateway:
    def test_gateway_health(self):
        """Gateway health endpoint returns ready status."""
        r = requests.get(f"{GATEWAY_URL}/health", timeout=TIMEOUT)
        assert r.status_code == 200
        data = r.json()
        assert data.get("ready") is True

    def test_gateway_index_stats(self):
        """Index stats endpoint returns collection info (scrolls full Qdrant collection — slow)."""
        r = requests.get(f"{GATEWAY_URL}/index-stats", timeout=TIMEOUT_SLOW)
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)

    def test_gateway_store_catalogue_requires_param(self):
        """Store catalogue endpoint requires store_name param."""
        r = requests.get(f"{GATEWAY_URL}/store-catalogue", timeout=TIMEOUT_SLOW)
        assert r.status_code == 422  # Unprocessable entity — missing required param

    def test_gateway_skipped_products(self):
        """Skipped products endpoint is reachable (queries Qdrant — slow)."""
        r = requests.get(f"{GATEWAY_URL}/skipped-products", timeout=TIMEOUT_SLOW)
        assert r.status_code == 200

    def test_gateway_feedback_get(self):
        """Feedback GET endpoint is reachable (queries Qdrant — slow)."""
        r = requests.get(f"{GATEWAY_URL}/feedback", timeout=TIMEOUT_SLOW)
        assert r.status_code == 200

    def test_gateway_search_requires_file(self):
        """Search endpoint rejects requests with no image file."""
        r = requests.post(f"{GATEWAY_URL}/search", timeout=TIMEOUT)
        assert r.status_code == 422  # Missing required file field


# ── Visual Engine ─────────────────────────────────────────────────────────────

class TestVisualEngine:
    def test_visual_engine_health(self):
        """Visual engine root returns online status."""
        r = requests.get(f"{VISUAL_URL}/", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_vectorize_requires_file(self):
        """Vectorize endpoint rejects requests with no image file."""
        r = requests.post(f"{VISUAL_URL}/vectorize", timeout=TIMEOUT)
        assert r.status_code == 422

    def test_detect_requires_file(self):
        """Detect endpoint rejects requests with no image file."""
        r = requests.post(f"{VISUAL_URL}/detect", timeout=TIMEOUT)
        assert r.status_code == 422

    def test_classify_text_top(self):
        """Text classification correctly identifies a top."""
        r = requests.post(
            f"{VISUAL_URL}/classify-text",
            json={"title": "Women's floral blouse short sleeve top"},
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        data = r.json()
        assert data.get("category") == "top", f"Expected 'top', got {data.get('category')}"

    def test_classify_text_dress(self):
        """Text classification correctly identifies a dress."""
        r = requests.post(
            f"{VISUAL_URL}/classify-text",
            json={"title": "Midi floral summer dress wrap style"},
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        data = r.json()
        assert data.get("category") == "dress", f"Expected 'dress', got {data.get('category')}"

    def test_classify_text_bag(self):
        """Text classification correctly identifies a bag."""
        r = requests.post(
            f"{VISUAL_URL}/classify-text",
            json={"title": "Leather crossbody shoulder bag"},
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        data = r.json()
        assert data.get("category") == "bag", f"Expected 'bag', got {data.get('category')}"


# ── MLflow ────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _mlflow_available, reason="MLFLOW_URL not set — MLflow not deployed in K3s")
class TestMLflow:
    def test_mlflow_reachable(self):
        """MLflow tracking server is reachable."""
        r = requests.get(f"{MLFLOW_URL}/", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_mlflow_experiments_api(self):
        """MLflow experiments API is functional."""
        r = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search",
            json={"max_results": 100},
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        data = r.json()
        assert "experiments" in data


# ── Integration ───────────────────────────────────────────────────────────────

class TestIntegration:
    def test_category_labels_complete(self):
        """Visual engine exposes all fashion categories including not_fashion sentinel."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "visual_engine"))
        from clip_labels import CANONICAL_LABELS
        expected = {
            "top", "sports_bra", "pants", "leggings", "shorts",
            "skirt", "dress", "sweater", "jacket", "shoes",
            "hat", "bag", "jumpsuit", "not_fashion",
        }
        assert set(CANONICAL_LABELS) == expected, (
            f"Label mismatch. Got: {set(CANONICAL_LABELS)}"
        )

    def test_golden_dataset_loadable(self):
        """Golden dataset JSON is valid and has at least 30 entries."""
        import json, os
        path = os.path.join(os.path.dirname(__file__), "..", "mlops", "golden_dataset.json")
        with open(path) as f:
            dataset = json.load(f)
        assert len(dataset) >= 30, f"Expected ≥30 entries, got {len(dataset)}"
        # Each entry should have required fields
        required = {"query_name", "relevant_product_ids", "query_category_tag"}
        for entry in dataset[:5]:
            missing = required - set(entry.keys())
            assert not missing, f"Entry missing fields: {missing}\n{entry}"
