"""Shared fixtures for end-to-end tests."""

import os
import signal
import sqlite3
import subprocess
import sys
import time
import uuid

import httpx
import pytest

PYTHON = os.path.join(
    os.path.dirname(__file__), os.pardir, "test-venv", "bin", "python"
)
PYTHON = os.path.abspath(PYTHON)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "skincare.db")
BASE_URL = "http://127.0.0.1:8321"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8321

TEST_USER_PREFIX = "e2e_test_"


def _unique_user_id() -> str:
    return f"{TEST_USER_PREFIX}{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def user_id():
    """Unique user id for the entire test session."""
    return _unique_user_id()


@pytest.fixture(scope="session")
def base_url():
    return BASE_URL


@pytest.fixture(scope="session")
def server(user_id):
    """Start a FastAPI server subprocess, wait for healthy, yield, then tear down."""
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT
    env["SKINCARE_LLM_MODEL"] = os.environ.get("SKINCARE_LLM_MODEL", "llama3.2:latest")

    proc = subprocess.Popen(
        [
            PYTHON, "-m", "uvicorn", "api.main:app",
            "--host", SERVER_HOST,
            "--port", str(SERVER_PORT),
            "--log-level", "info",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    healthy = False
    for _ in range(30):
        time.sleep(1)
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=3)
            if r.status_code == 200:
                healthy = True
                break
        except (httpx.ConnectError, httpx.ReadError):
            continue

    if not healthy:
        out = proc.stdout.read().decode() if proc.stdout else ""
        proc.kill()
        pytest.fail(f"Server did not become healthy within 30s.\nOutput:\n{out}")

    yield proc

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()

    _cleanup_test_data()


def _cleanup_test_data():
    """Remove test users and associated data from the database."""
    if not os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("DELETE FROM purchases WHERE user_id LIKE ?", (f"{TEST_USER_PREFIX}%",))
        conn.execute("DELETE FROM recommendations WHERE user_id LIKE ?", (f"{TEST_USER_PREFIX}%",))
        conn.execute("DELETE FROM analyses WHERE user_id LIKE ?", (f"{TEST_USER_PREFIX}%",))
        conn.execute("DELETE FROM users WHERE user_id LIKE ?", (f"{TEST_USER_PREFIX}%",))
        conn.commit()
    finally:
        conn.close()
