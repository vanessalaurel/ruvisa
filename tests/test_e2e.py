"""
End-to-end tests for the Skincare Advisory Agent.

Starts a real FastAPI server backed by a real Ollama LLM, sends HTTP requests
through the full pipeline (API -> LangGraph agent -> tools -> DB), and verifies
correctness of responses and database persistence.

Run with:
    cd /home/vanessa/project
    test-venv/bin/python -m pytest tests/test_e2e.py -v -s --tb=short

Requires Ollama running locally with a model (default: llama3.2:latest).
"""

import sqlite3

import httpx
import pytest

AGENT_TIMEOUT = 180  # LLM tool-calling loops can be slow
DB_PATH = "data/skincare.db"


# ---------------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------------
class TestHealthCheck:
    def test_health_endpoint(self, server, base_url):
        r = httpx.get(f"{base_url}/health", timeout=10)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "skincare-agent"
        assert "catalog_products" in body and isinstance(body["catalog_products"], int)
        assert "catalog_review_scores" in body and isinstance(body["catalog_review_scores"], int)


# ---------------------------------------------------------------------------
# 2. Chat – ask the agent for product recommendations
# ---------------------------------------------------------------------------
class TestChat:
    def test_chat_product_recommendation(self, server, base_url, user_id):
        """Ask the agent for acne product recs; should invoke recommend_products tool."""
        r = httpx.post(
            f"{base_url}/api/chat",
            json={
                "user_id": user_id,
                "message": (
                    "I have oily skin with moderate acne (score 0.7) and mild wrinkles "
                    "(score 0.3). Can you recommend the top 3 products for me?"
                ),
            },
            timeout=AGENT_TIMEOUT,
        )
        assert r.status_code == 200, f"Chat failed: {r.text}"
        data = r.json()
        assert data["user_id"] == user_id
        assert len(data["response"]) > 50, "Response too short – agent likely didn't call tools"

    def test_chat_product_search(self, server, base_url, user_id):
        """Ask the agent to search for affordable acne products."""
        r = httpx.post(
            f"{base_url}/api/chat",
            json={
                "user_id": user_id,
                "message": "Search for acne products under $30 for oily skin.",
            },
            timeout=AGENT_TIMEOUT,
        )
        assert r.status_code == 200, f"Chat failed: {r.text}"
        data = r.json()
        assert len(data["response"]) > 20

    def test_chat_track_purchase(self, server, base_url, user_id):
        """Ask the agent to record a purchase."""
        r = httpx.post(
            f"{base_url}/api/chat",
            json={
                "user_id": user_id,
                "message": (
                    "I just bought CeraVe Acne Foaming Cream Cleanser for $15. "
                    "Please record this purchase for me."
                ),
            },
            timeout=AGENT_TIMEOUT,
        )
        assert r.status_code == 200, f"Chat failed: {r.text}"
        data = r.json()
        assert len(data["response"]) > 10


# ---------------------------------------------------------------------------
# 3. Analyze – run skin analysis and persist results
# ---------------------------------------------------------------------------
class TestAnalyze:
    def test_analyze_saves_to_db(self, server, base_url, user_id):
        """POST /analyze should return analysis_id and concern_vector, and store in DB."""
        r = httpx.post(
            f"{base_url}/api/analyze",
            json={
                "user_id": user_id,
                "image_path": "/tmp/test_face.jpg",
                "skin_type": "oily",
            },
            timeout=AGENT_TIMEOUT,
        )
        assert r.status_code == 200, f"Analyze failed: {r.text}"
        data = r.json()
        assert data["user_id"] == user_id
        assert isinstance(data["analysis_id"], int)
        assert data["analysis_id"] > 0
        assert len(data["concern_vector"]) == 7
        assert len(data["summary"]) > 20, "Agent summary too short"


# ---------------------------------------------------------------------------
# 4. Profile – verify user and analysis are stored
# ---------------------------------------------------------------------------
class TestProfile:
    def test_profile_exists(self, server, base_url, user_id):
        """After chat + analyze, the user profile should exist with data."""
        r = httpx.get(f"{base_url}/api/profile/{user_id}", timeout=10)
        assert r.status_code == 200, f"Profile fetch failed: {r.text}"
        data = r.json()
        assert data["user_id"] == user_id
        assert data["skin_type"] == "oily"
        assert data["analyses_count"] >= 1, "Expected at least 1 analysis saved"

    def test_profile_404_for_unknown_user(self, server, base_url):
        """Non-existent user returns 404."""
        r = httpx.get(f"{base_url}/api/profile/nonexistent_user_999", timeout=10)
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 5. Product search – direct API endpoint (no LLM)
# ---------------------------------------------------------------------------
class TestProductSearch:
    def test_search_acne_products(self, server, base_url):
        r = httpx.get(
            f"{base_url}/api/products/search",
            params={"concern": "acne", "limit": 5},
            timeout=30,
        )
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert len(data["results"]) > 0

    def test_search_with_filters(self, server, base_url):
        r = httpx.get(
            f"{base_url}/api/products/search",
            params={
                "concern": "wrinkles",
                "skin_type": "dry",
                "max_price": 50.0,
                "sort_by": "price",
                "limit": 3,
            },
            timeout=30,
        )
        assert r.status_code == 200
        data = r.json()
        assert "results" in data

    def test_search_invalid_concern(self, server, base_url):
        r = httpx.get(
            f"{base_url}/api/products/search",
            params={"concern": "not_a_real_concern"},
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert "Invalid concern" in data["results"]


# ---------------------------------------------------------------------------
# 6. History – verify full journey is recorded
# ---------------------------------------------------------------------------
class TestHistory:
    def test_history_has_analyses(self, server, base_url, user_id):
        """History endpoint should return the user's analyses."""
        r = httpx.get(f"{base_url}/api/history/{user_id}", timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert data["user_id"] == user_id
        assert isinstance(data["analyses"], list)
        assert len(data["analyses"]) >= 1

    def test_history_404_for_unknown(self, server, base_url):
        r = httpx.get(f"{base_url}/api/history/nonexistent_user_999", timeout=10)
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 7. Database integrity checks
# ---------------------------------------------------------------------------
class TestDatabaseIntegrity:
    def test_user_row_exists(self, server, user_id):
        """Verify the user row was created in SQLite."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        conn.close()
        assert row is not None, f"User {user_id} not found in DB"
        assert dict(row)["skin_type"] == "oily"

    def test_analysis_row_exists(self, server, user_id):
        """Verify at least one analysis row was stored."""
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT COUNT(*) as cnt FROM analyses WHERE user_id = ?", (user_id,)
        ).fetchone()
        conn.close()
        assert rows[0] >= 1, "No analysis rows found for test user"

    def test_tables_have_correct_schema(self, server):
        """Verify all four tables exist with expected columns."""
        conn = sqlite3.connect(DB_PATH)
        tables = {
            "users": {"user_id", "name", "skin_type", "created_at"},
            "analyses": {"id", "user_id", "image_path", "concern_vector",
                         "acne_summary", "wrinkle_summary", "full_report", "created_at"},
            "recommendations": {"id", "analysis_id", "user_id", "product_url",
                                "product_title", "brand", "category", "similarity",
                                "price", "created_at"},
            "purchases": {"id", "user_id", "product_url", "product_title",
                          "price", "purchased_at"},
        }
        for table, expected_cols in tables.items():
            cursor = conn.execute(f"PRAGMA table_info({table})")
            actual_cols = {row[1] for row in cursor.fetchall()}
            missing = expected_cols - actual_cols
            assert not missing, f"Table '{table}' missing columns: {missing}"
        conn.close()


# ---------------------------------------------------------------------------
# 8. Second analysis + compare (journey tracking)
# ---------------------------------------------------------------------------
class TestJourneyTracking:
    def test_second_analysis_for_comparison(self, server, base_url, user_id):
        """Run a second analysis so compare_analyses has two data points."""
        r = httpx.post(
            f"{base_url}/api/analyze",
            json={
                "user_id": user_id,
                "image_path": "/tmp/test_face_v2.jpg",
                "skin_type": "oily",
            },
            timeout=AGENT_TIMEOUT,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["analysis_id"] > 0

    def test_compare_via_chat(self, server, base_url, user_id):
        """Ask the agent to compare analyses – should invoke compare_analyses tool."""
        r = httpx.post(
            f"{base_url}/api/chat",
            json={
                "user_id": user_id,
                "message": f"Compare my skin analyses to see if there's any progress. My user ID is {user_id}.",
            },
            timeout=AGENT_TIMEOUT,
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["response"]) > 20

    def test_history_has_multiple_analyses(self, server, base_url, user_id):
        """After two analyses, history should reflect both."""
        r = httpx.get(f"{base_url}/api/history/{user_id}", timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert len(data["analyses"]) >= 2, (
            f"Expected >=2 analyses, got {len(data['analyses'])}"
        )
