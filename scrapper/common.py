from __future__ import annotations

import gzip
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import httpx
import orjson
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as f:
        f.write(orjson.dumps(obj))
        f.write(b"\n")


def save_text_gz(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(text.encode("utf-8", errors="ignore"))


# Realistic browser User-Agents to rotate
USER_AGENTS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


def get_browser_headers(user_agent: str | None = None) -> Dict[str, str]:
    """Return headers that mimic a real browser."""
    ua = user_agent or random.choice(USER_AGENTS)
    return {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,zh-HK;q=0.8,zh;q=0.7",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    }


@dataclass
class FetchConfig:
    timeout_s: float = 60.0  # Increased timeout
    min_delay_s: float = 2.0  # More polite delays
    max_delay_s: float = 5.0
    max_retries: int = 5


class Fetcher:
    def __init__(self, cfg: FetchConfig | None = None):
        self.cfg = cfg or FetchConfig()
        self._user_agent = random.choice(USER_AGENTS)
        self.client = httpx.Client(
            timeout=httpx.Timeout(self.cfg.timeout_s, connect=30.0),
            headers=get_browser_headers(self._user_agent),
            follow_redirects=True,
        )

    def close(self) -> None:
        self.client.close()

    def _sleep_polite(self) -> None:
        time.sleep(random.uniform(self.cfg.min_delay_s, self.cfg.max_delay_s))

    def _rotate_user_agent(self) -> None:
        """Rotate User-Agent for next request."""
        self._user_agent = random.choice(USER_AGENTS)
        self.client.headers.update(get_browser_headers(self._user_agent))

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception_type((
            httpx.TimeoutException,
            httpx.TransportError,
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            httpx.HTTPStatusError,
        )),
    )
    def get_text(self, url: str) -> str:
        self._sleep_polite()
        self._rotate_user_agent()
        r = self.client.get(url)
        r.raise_for_status()
        return r.text