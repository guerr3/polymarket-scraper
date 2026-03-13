"""
Playwright-based HTML fallback scraper for polymarketanalytics.com.

Used as a resilience fallback when API endpoints are unavailable.
Respects rate limits and uses randomized user agents.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from typing import Optional

from .config import AnalyticsHtmlConfig, USER_AGENTS, get_config
from .models import Market, MarketSource, MarketStatus, Outcome
from .pagination import HtmlPagePaginator

logger = logging.getLogger(__name__)


class HtmlFallbackScraper:
    """
    Playwright-based HTML scraper for Polymarket Analytics.

    Only used as a fallback when API endpoints fail.
    """

    def __init__(self, config: Optional[AnalyticsHtmlConfig] = None):
        self.config = config or get_config().analytics_html
        self.paginator = HtmlPagePaginator()
        self._browser = None
        self._context = None

    async def __aenter__(self) -> "HtmlFallbackScraper":
        await self._init_browser()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def _init_browser(self) -> None:
        """Initialize Playwright browser with stealth settings."""
        try:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ],
            )
            self._context = await self._browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport={"width": 1920, "height": 1080},
            )
            logger.info("Playwright browser initialized")
        except ImportError:
            logger.error(
                "Playwright not installed. Run: pip install playwright && "
                "playwright install chromium"
            )
            raise
        except Exception as exc:
            logger.error("Failed to init Playwright: %s", exc)
            raise

    async def close(self) -> None:
        """Close browser and playwright."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if hasattr(self, "_playwright") and self._playwright:
            await self._playwright.stop()

    async def fetch_markets_page(
        self, page_number: int = 1
    ) -> tuple[list[Market], Optional[int]]:
        """
        Fetch a single page of markets from the HTML table.

        Returns (markets, total_pages) where total_pages may be None.
        """
        if not self._context:
            await self._init_browser()

        url = (
            f"{self.config.base_url}{self.config.markets_path}"
            f"?sources={self.config.default_sources}"
            f"&marketStatus={self.config.default_status}"
            f"&page={page_number}"
        )

        page = await self._context.new_page()
        markets: list[Market] = []
        total_pages: Optional[int] = None

        try:
            # Humane delay
            await asyncio.sleep(random.uniform(1.0, 3.0))

            await page.goto(url, wait_until="networkidle", timeout=30000)
            await page.wait_for_selector(
                self.config.selectors["rows"], timeout=15000
            )

            # Detect total pages from "Page N of M" text
            page_text = await page.text_content("body") or ""
            match = re.search(r"Page\s+\d+\s+of\s+(\d+)", page_text)
            if match:
                total_pages = int(match.group(1))

            # Extract rows
            rows = await page.query_selector_all(self.config.selectors["rows"])

            for row in rows:
                try:
                    market = await self._parse_row(row)
                    if market:
                        markets.append(market)
                except Exception as exc:
                    logger.warning("Failed to parse HTML row: %s", exc)

            logger.info(
                "Page %d: extracted %d markets (total pages: %s)",
                page_number,
                len(markets),
                total_pages,
            )

        except Exception as exc:
            logger.error("Failed to fetch HTML page %d: %s", page_number, exc)

        finally:
            await page.close()

        return markets, total_pages

    async def _parse_row(self, row) -> Optional[Market]:
        """Parse a single table row into a Market model."""
        selectors = self.config.selectors

        # Market link → ID and title
        link_el = await row.query_selector(selectors["market_link"])
        if not link_el:
            return None

        href = await link_el.get_attribute("href") or ""
        title = (await link_el.text_content() or "").strip()

        # Extract market ID from /markets/{id}
        market_id = None
        id_match = re.search(r"/markets/(\d+)", href)
        if id_match:
            market_id = int(id_match.group(1))

        # Price
        price_el = await row.query_selector(selectors["price"])
        price_text = (await price_el.text_content() or "").strip() if price_el else ""
        price = self._parse_percentage(price_text)

        # Volume
        vol_el = await row.query_selector(selectors["volume"])
        vol_text = (await vol_el.text_content() or "").strip() if vol_el else ""
        volume = self._parse_number(vol_text)

        # End date
        date_el = await row.query_selector(selectors["end_date"])
        end_date_text = (
            (await date_el.text_content() or "").strip() if date_el else ""
        )

        # Tags
        tag_els = await row.query_selector_all(selectors["tags"])
        tags = []
        for tag_el in tag_els:
            tag_text = (await tag_el.text_content() or "").strip()
            if tag_text:
                tags.append(tag_text)

        outcomes = []
        if price is not None:
            outcomes = [
                Outcome(name="Yes", price=price, token_id=""),
                Outcome(name="No", price=round(1.0 - price, 4), token_id=""),
            ]

        return Market(
            id=market_id,
            question=title,
            slug=href.split("/")[-1] if "/" in href else "",
            status=MarketStatus.ACTIVE,
            source=MarketSource.POLYMARKET,
            volume=volume or 0.0,
            tags=tags,
            outcomes=outcomes,
        )

    async def get_all_markets(self, max_pages: int = 50) -> list[Market]:
        """Fetch all markets using HTML page pagination."""
        all_markets: list[Market] = []
        self.paginator.max_pages = max_pages

        async for page_markets in self.paginator.paginate(
            self.fetch_markets_page
        ):
            all_markets.extend(page_markets)

        logger.info("HTML fallback: fetched %d total markets", len(all_markets))
        return all_markets

    # ------------------------------------------------------------------ #
    #  Parsing helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_percentage(text: str) -> Optional[float]:
        """Parse '73.5%' → 0.735, or '0.735' → 0.735."""
        if not text:
            return None
        text = text.replace("%", "").replace("¢", "").strip()
        try:
            val = float(text)
            # If > 1, assume it was a percentage
            if val > 1.0:
                val /= 100.0
            return round(val, 4)
        except ValueError:
            return None

    @staticmethod
    def _parse_number(text: str) -> Optional[float]:
        """Parse volume strings like '$1.2M', '500K', '1,234'."""
        if not text:
            return None
        text = text.replace("$", "").replace(",", "").strip().upper()
        multiplier = 1.0
        if text.endswith("B"):
            multiplier = 1_000_000_000
            text = text[:-1]
        elif text.endswith("M"):
            multiplier = 1_000_000
            text = text[:-1]
        elif text.endswith("K"):
            multiplier = 1_000
            text = text[:-1]
        try:
            return float(text) * multiplier
        except ValueError:
            return None
