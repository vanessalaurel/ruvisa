from __future__ import annotations
import random
import time
from dataclasses import dataclass
from typing import Optional
from playwright.sync_api import sync_playwright, Playwright, Browser, Error as PlaywrightError

@dataclass
class BrowserConfig:
    headless: bool = True
    timeout_ms: int = 90000
    max_retries: int = 3
    min_delay: float = 2.0
    max_delay: float = 5.0
    wait_after_load_ms: int = 3000

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
]


class BrowserFetcher:
    """
    Reusable browser fetcher that keeps Firefox open between requests.
    This avoids segfaults from repeatedly launching/closing the browser.
    """
    
    def __init__(self, cfg: BrowserConfig | None = None):
        self.cfg = cfg or BrowserConfig()
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
    
    def _ensure_browser(self) -> Browser:
        if self._browser is None:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.firefox.launch(headless=self.cfg.headless)
        return self._browser
    
    def close(self) -> None:
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
    
    def get_html(self, url: str) -> str:
        """Fetch a page and return the fully rendered HTML."""
        browser = self._ensure_browser()
        
        for attempt in range(self.cfg.max_retries):
            try:
                # Polite delay before request
                time.sleep(random.uniform(self.cfg.min_delay, self.cfg.max_delay))
                
                context = browser.new_context(
                    user_agent=random.choice(USER_AGENTS),
                    viewport={"width": 1920, "height": 1080},
                    locale="en-US",
                )
                
                try:
                    page = context.new_page()
                    page.set_default_timeout(self.cfg.timeout_ms)
                    
                    # Navigate and wait for network to be idle
                    page.goto(url, wait_until="networkidle", timeout=self.cfg.timeout_ms)
                    
                    # Additional wait for Vue.js to finish rendering
                    page.wait_for_timeout(self.cfg.wait_after_load_ms)
                    
                    html = page.content()
                    return html
                finally:
                    context.close()
                    
            except PlaywrightError as e:
                if attempt < self.cfg.max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
        
        raise RuntimeError(f"Failed to fetch {url} after {self.cfg.max_retries} attempts")


# Global fetcher instance (lazy initialized)
_fetcher: Optional[BrowserFetcher] = None


def get_rendered_html(url: str, cfg: BrowserConfig | None = None) -> str:
    """
    Fetch a page using Firefox (headless) and return the fully rendered HTML.
    Uses a global browser instance to avoid segfaults from repeated launches.
    """
    global _fetcher
    if _fetcher is None:
        _fetcher = BrowserFetcher(cfg)
    return _fetcher.get_html(url)


def close_browser() -> None:
    """Close the global browser instance."""
    global _fetcher
    if _fetcher:
        _fetcher.close()
        _fetcher = None