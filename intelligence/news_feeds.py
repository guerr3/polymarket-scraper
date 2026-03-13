"""
News feed aggregator for prediction market intelligence.

Collects news from multiple sources:
- RSS feeds (Reuters, AP, BBC, etc.)
- Twitter/X search (via Nitter or public scraping)
- Google News RSS
- Direct web article fetching

Each feed item is scored for relevance to active markets
and fed into the sentiment pipeline.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from urllib.parse import quote_plus

import aiohttp

from polymarket_client.config import USER_AGENTS
from polymarket_client.resilience import ResilientSession

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Data types
# --------------------------------------------------------------------------- #

@dataclass
class NewsItem:
    """A single news item from any source."""
    title: str
    description: str = ""
    url: str = ""
    source: str = ""          # e.g., "reuters", "bbc", "twitter:@user"
    author: str = ""
    published: Optional[datetime] = None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    category: str = ""        # e.g., "politics", "crypto", "sports"
    relevance_keywords: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Combine title + description for sentiment analysis."""
        parts = [self.title]
        if self.description:
            parts.append(self.description)
        return ". ".join(parts)

    @property
    def age_hours(self) -> float:
        """How old is this item in hours."""
        if not self.published:
            return 0.0
        delta = datetime.now(timezone.utc) - self.published
        return delta.total_seconds() / 3600


# --------------------------------------------------------------------------- #
#  RSS Feed definitions
# --------------------------------------------------------------------------- #

# Curated RSS feeds organized by category
RSS_FEEDS: dict[str, list[dict[str, str]]] = {
    "general": [
        {"name": "Reuters World", "url": "https://feeds.reuters.com/reuters/worldNews", "source": "reuters"},
        {"name": "AP Top News", "url": "https://rsshub.app/apnews/topics/apf-topnews", "source": "ap"},
        {"name": "BBC World", "url": "https://feeds.bbci.co.uk/news/world/rss.xml", "source": "bbc"},
        {"name": "Al Jazeera", "url": "https://www.aljazeera.com/xml/rss/all.xml", "source": "aljazeera"},
    ],
    "politics": [
        {"name": "Politico", "url": "https://rss.politico.com/politics-news.xml", "source": "politico"},
        {"name": "The Hill", "url": "https://thehill.com/feed/", "source": "thehill"},
        {"name": "Reuters Politics", "url": "https://feeds.reuters.com/reuters/politicsNews", "source": "reuters"},
    ],
    "crypto": [
        {"name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/", "source": "coindesk"},
        {"name": "CoinTelegraph", "url": "https://cointelegraph.com/rss", "source": "cointelegraph"},
        {"name": "The Block", "url": "https://www.theblock.co/rss.xml", "source": "theblock"},
    ],
    "finance": [
        {"name": "Reuters Business", "url": "https://feeds.reuters.com/reuters/businessNews", "source": "reuters"},
        {"name": "CNBC", "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html", "source": "cnbc"},
    ],
    "middle_east": [
        {"name": "Reuters Middle East", "url": "https://feeds.reuters.com/reuters/middleeastNews", "source": "reuters"},
        {"name": "BBC Middle East", "url": "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml", "source": "bbc"},
    ],
}

# Google News search RSS template
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

# Nitter instances for Twitter/X scraping (public, no auth needed)
NITTER_INSTANCES = [
    "https://nitter.poast.org",
    "https://nitter.privacydev.net",
    "https://nitter.woodland.cafe",
]


# --------------------------------------------------------------------------- #
#  Feed fetcher
# --------------------------------------------------------------------------- #

class NewsFeedAggregator:
    """
    Aggregates news from RSS feeds, Google News, and Twitter/X.

    Usage:
        async with NewsFeedAggregator() as aggregator:
            items = await aggregator.search("Iran military", categories=["general", "middle_east"])
    """

    def __init__(self, session: Optional[ResilientSession] = None):
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self) -> "NewsFeedAggregator":
        if self._owns_session:
            self._session = ResilientSession()
            await self._session.__aenter__()
        return self

    async def __aexit__(self, *args) -> None:
        if self._owns_session and self._session:
            await self._session.__aexit__(*args)

    # ------------------------------------------------------------------ #
    #  RSS feeds
    # ------------------------------------------------------------------ #

    async def fetch_rss(
        self,
        categories: list[str] | None = None,
        max_age_hours: int = 48,
    ) -> list[NewsItem]:
        """Fetch items from curated RSS feeds."""
        cats = categories or list(RSS_FEEDS.keys())
        feeds_to_fetch = []
        for cat in cats:
            feeds_to_fetch.extend(RSS_FEEDS.get(cat, []))

        tasks = [self._fetch_single_rss(f) for f in feeds_to_fetch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        items: list[NewsItem] = []
        for result in results:
            if isinstance(result, list):
                items.extend(result)
            elif isinstance(result, Exception):
                logger.debug("RSS fetch error: %s", result)

        # Filter by age
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        items = [i for i in items if not i.published or i.published >= cutoff]

        logger.info("Fetched %d RSS items from %d feeds", len(items), len(feeds_to_fetch))
        return items

    async def _fetch_single_rss(self, feed: dict) -> list[NewsItem]:
        """Fetch and parse a single RSS feed."""
        try:
            import random
            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "application/rss+xml, application/xml, text/xml",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    feed["url"],
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        return []
                    text = await resp.text()

            return self._parse_rss_xml(text, feed.get("source", "unknown"))

        except Exception as exc:
            logger.debug("Failed to fetch RSS %s: %s", feed.get("name", "?"), exc)
            return []

    def _parse_rss_xml(self, xml_text: str, source: str) -> list[NewsItem]:
        """Parse RSS/Atom XML into NewsItem list."""
        items: list[NewsItem] = []
        try:
            root = ET.fromstring(xml_text)

            # Handle RSS 2.0
            for item in root.iter("item"):
                title = self._xml_text(item, "title")
                desc = self._xml_text(item, "description")
                link = self._xml_text(item, "link")
                pub_date = self._xml_text(item, "pubDate")
                category = self._xml_text(item, "category")

                if not title:
                    continue

                items.append(NewsItem(
                    title=html.unescape(title).strip(),
                    description=self._clean_html(html.unescape(desc)) if desc else "",
                    url=link,
                    source=source,
                    published=self._parse_rss_date(pub_date),
                    category=category,
                ))

            # Handle Atom feeds
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall(".//atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                summary_el = entry.find("atom:summary", ns)
                link_el = entry.find("atom:link", ns)
                updated_el = entry.find("atom:updated", ns)

                title = title_el.text if title_el is not None else ""
                if not title:
                    continue

                items.append(NewsItem(
                    title=html.unescape(title).strip(),
                    description=self._clean_html(summary_el.text) if summary_el is not None and summary_el.text else "",
                    url=link_el.get("href", "") if link_el is not None else "",
                    source=source,
                    published=self._parse_rss_date(updated_el.text if updated_el is not None else ""),
                ))

        except ET.ParseError as exc:
            logger.debug("RSS parse error: %s", exc)

        return items

    # ------------------------------------------------------------------ #
    #  Google News search
    # ------------------------------------------------------------------ #

    async def search_google_news(
        self,
        query: str,
        max_results: int = 20,
    ) -> list[NewsItem]:
        """Search Google News RSS for a query."""
        url = GOOGLE_NEWS_RSS.format(query=quote_plus(query))

        try:
            import random
            headers = {"User-Agent": random.choice(USER_AGENTS)}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        logger.warning("Google News returned %d for query: %s", resp.status, query)
                        return []
                    text = await resp.text()

            items = self._parse_rss_xml(text, "google_news")
            return items[:max_results]

        except Exception as exc:
            logger.warning("Google News search failed: %s", exc)
            return []

    # ------------------------------------------------------------------ #
    #  Twitter/X via Nitter
    # ------------------------------------------------------------------ #

    async def search_twitter(
        self,
        query: str,
        max_results: int = 20,
    ) -> list[NewsItem]:
        """
        Search Twitter/X via Nitter RSS feeds (no auth required).

        Falls back across multiple Nitter instances.
        """
        items: list[NewsItem] = []

        for instance in NITTER_INSTANCES:
            try:
                url = f"{instance}/search/rss?f=tweets&q={quote_plus(query)}"
                import random
                headers = {"User-Agent": random.choice(USER_AGENTS)}

                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status != 200:
                            continue
                        text = await resp.text()

                parsed = self._parse_rss_xml(text, "twitter")

                # Enrich with Twitter-specific metadata
                for item in parsed:
                    # Extract @username from title or description
                    match = re.search(r"@(\w+)", item.title + " " + item.description)
                    if match:
                        item.author = f"@{match.group(1)}"
                        item.source = f"twitter:{item.author}"

                items = parsed[:max_results]
                if items:
                    logger.info("Fetched %d tweets from %s", len(items), instance)
                    break

            except Exception as exc:
                logger.debug("Nitter %s failed: %s", instance, exc)
                continue

        return items

    async def fetch_user_timeline(
        self,
        username: str,
        max_results: int = 10,
    ) -> list[NewsItem]:
        """Fetch recent tweets from a specific user via Nitter."""
        for instance in NITTER_INSTANCES:
            try:
                url = f"{instance}/{username}/rss"
                import random
                headers = {"User-Agent": random.choice(USER_AGENTS)}

                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status != 200:
                            continue
                        text = await resp.text()

                items = self._parse_rss_xml(text, f"twitter:@{username}")
                for item in items:
                    item.author = f"@{username}"
                return items[:max_results]

            except Exception:
                continue

        return []

    # ------------------------------------------------------------------ #
    #  Unified search
    # ------------------------------------------------------------------ #

    async def search(
        self,
        query: str,
        categories: list[str] | None = None,
        include_twitter: bool = True,
        include_google: bool = True,
        include_rss: bool = True,
        max_age_hours: int = 48,
    ) -> list[NewsItem]:
        """
        Search all available sources for news matching a query.

        Returns items sorted by relevance (keyword overlap with query).
        """
        tasks = []

        if include_rss:
            tasks.append(self.fetch_rss(categories=categories, max_age_hours=max_age_hours))
        if include_google:
            tasks.append(self.search_google_news(query))
        if include_twitter:
            tasks.append(self.search_twitter(query))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_items: list[NewsItem] = []
        for result in results:
            if isinstance(result, list):
                all_items.extend(result)

        # Filter for query relevance
        query_words = set(query.lower().split())
        relevant: list[tuple[float, NewsItem]] = []
        for item in all_items:
            text_lower = item.full_text.lower()
            overlap = sum(1 for w in query_words if w in text_lower)
            score = overlap / max(len(query_words), 1)
            if score > 0.2:  # at least 20% keyword overlap
                item.relevance_keywords = [w for w in query_words if w in text_lower]
                relevant.append((score, item))

        # Sort by relevance, then recency
        relevant.sort(key=lambda x: (x[0], -(x[1].age_hours or 0)), reverse=True)

        filtered = [item for _, item in relevant]
        logger.info("Search '%s': %d items total, %d relevant", query, len(all_items), len(filtered))
        return filtered

    # ------------------------------------------------------------------ #
    #  Market-specific feed
    # ------------------------------------------------------------------ #

    async def get_market_news(
        self,
        market_question: str,
        max_age_hours: int = 48,
    ) -> list[NewsItem]:
        """
        Get news relevant to a specific market question.

        Extracts key terms from the question and searches all sources.
        """
        # Extract meaningful search terms from market question
        query = self._extract_search_query(market_question)
        logger.info("Market news search: '%s' → query: '%s'", market_question[:50], query)

        return await self.search(
            query=query,
            max_age_hours=max_age_hours,
        )

    @staticmethod
    def _extract_search_query(question: str) -> str:
        """Extract meaningful search terms from a market question."""
        # Remove common prediction market boilerplate
        boilerplate = [
            r"^will\s+", r"\?$", r"\bby\s+\w+\s+\d+\b", r"\bbefore\s+\w+\s+\d+\b",
            r"\bby\s+end\s+of\b", r"\bin\s+20\d{2}\b", r"\bthis\s+year\b",
        ]
        cleaned = question.lower()
        for pattern in boilerplate:
            cleaned = re.sub(pattern, " ", cleaned)

        # Remove short stop words but keep meaningful terms
        stop_words = {"the", "a", "an", "is", "are", "be", "to", "of", "and", "or", "in", "on", "at"}
        words = [w for w in cleaned.split() if w not in stop_words and len(w) > 2]

        return " ".join(words[:6])  # keep top 6 terms

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _xml_text(element: ET.Element, tag: str) -> str:
        el = element.find(tag)
        return el.text.strip() if el is not None and el.text else ""

    @staticmethod
    def _clean_html(text: str) -> str:
        """Remove HTML tags from text."""
        if not text:
            return ""
        cleaned = re.sub(r"<[^>]+>", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()[:500]

    @staticmethod
    def _parse_rss_date(date_str: str) -> Optional[datetime]:
        """Parse common RSS date formats."""
        if not date_str:
            return None
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S GMT",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        return None
