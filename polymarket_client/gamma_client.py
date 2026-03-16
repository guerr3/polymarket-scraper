"""
GAMMA REST API client for Polymarket markets and events metadata.

Endpoint: https://gamma-api.polymarket.com
Pagination: offset-based (limit + offset)
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import GammaConfig, get_config, DEFAULT_HEADERS
from .models import Market, gamma_market_to_model
from .pagination import OffsetPaginator
from .resilience import ResilientSession, NonRetryableHttpError

logger = logging.getLogger(__name__)


class GammaClient:
    """Client for the Polymarket GAMMA REST API."""

    def __init__(
        self,
        session: ResilientSession,
        config: Optional[GammaConfig] = None,
    ):
        self.session = session
        self.config = config or get_config().gamma
        self.paginator = OffsetPaginator(
            limit=self.config.default_limit,
            max_limit=self.config.max_limit,
        )

    # ------------------------------------------------------------------ #
    #  Markets
    # ------------------------------------------------------------------ #

    async def get_markets(
        self,
        *,
        active: bool = True,
        order: str = "volume",
        ascending: bool = False,
        limit: int = 100,
        offset: int = 0,
        tag: Optional[str] = None,
    ) -> list[dict]:
        """Fetch a single page of markets from GAMMA API."""
        params: dict = {
            "limit": min(limit, self.config.max_limit),
            "offset": offset,
            "active": str(active).lower(),
            "order": order,
            "ascending": str(ascending).lower(),
        }
        if tag:
            params["tag"] = tag

        result = await self.session.request(
            "GET",
            f"{self.config.base_url}/markets",
            endpoint_name="gamma_markets",
            rate_limit_rps=self.config.rate_limit_rps,
            headers=DEFAULT_HEADERS,
            params=params,
            timeout=self.config.timeout_seconds,
        )
        return result if isinstance(result, list) else []

    async def get_all_markets(
        self,
        *,
        active: bool = True,
        order: str = "volume",
        ascending: bool = False,
        max_pages: int = 50,
    ) -> list[Market]:
        """Fetch all markets using offset pagination, return normalized models."""
        all_markets: list[Market] = []

        async def fetch_page(params: dict) -> list[dict]:
            params["active"] = str(active).lower()
            params["order"] = order
            params["ascending"] = str(ascending).lower()
            return await self.session.request(
                "GET",
                f"{self.config.base_url}/markets",
                endpoint_name="gamma_markets",
                rate_limit_rps=self.config.rate_limit_rps,
                headers=DEFAULT_HEADERS,
                params=params,
                timeout=self.config.timeout_seconds,
            ) or []

        page_count = 0
        async for page in self.paginator.paginate(fetch_page):
            for raw in page:
                try:
                    all_markets.append(gamma_market_to_model(raw))
                except Exception as exc:
                    logger.warning("Failed to parse GAMMA market: %s", exc)
            page_count += 1
            if page_count >= max_pages:
                logger.info("Reached max pages (%d) for markets", max_pages)
                break

        logger.info("Fetched %d markets from GAMMA API", len(all_markets))
        return all_markets

    async def get_market(self, market_id: int) -> Optional[Market]:
        """Fetch a single market by ID."""
        try:
            raw = await self.session.request(
                "GET",
                f"{self.config.base_url}/markets/{market_id}",
                endpoint_name="gamma_market_detail",
                rate_limit_rps=self.config.rate_limit_rps,
                headers=DEFAULT_HEADERS,
                timeout=self.config.timeout_seconds,
            )
            if raw:
                return gamma_market_to_model(raw)
        except Exception as exc:
            logger.error("Failed to fetch market %d: %s", market_id, exc)
        return None

    async def get_market_by_slug(self, slug: str) -> Optional[Market]:
        """Fetch a single market by slug using server-side filtering."""
        try:
            raw = await self.session.request(
                "GET",
                f"{self.config.base_url}/markets",
                endpoint_name="gamma_market_by_slug",
                rate_limit_rps=self.config.rate_limit_rps,
                headers=DEFAULT_HEADERS,
                params={
                    "slug": slug,
                    "limit": 1,
                    "offset": 0,
                },
                timeout=self.config.timeout_seconds,
            )

            if isinstance(raw, list) and raw:
                return gamma_market_to_model(raw[0])

        except NonRetryableHttpError as exc:
            if exc.status != 404:
                logger.warning("Failed slug lookup for %s: %s", slug, exc)
        except Exception as exc:
            logger.warning("Failed slug lookup for %s: %s", slug, exc)

        return None

    # ------------------------------------------------------------------ #
    #  Events
    # ------------------------------------------------------------------ #

    async def get_events(
        self,
        *,
        active: bool = True,
        order: str = "volume",
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Fetch a single page of events from GAMMA API."""
        params: dict = {
            "limit": min(limit, self.config.max_limit),
            "offset": offset,
            "active": str(active).lower(),
            "order": order,
        }

        result = await self.session.request(
            "GET",
            f"{self.config.base_url}/events",
            endpoint_name="gamma_events",
            rate_limit_rps=self.config.rate_limit_rps,
            headers=DEFAULT_HEADERS,
            params=params,
            timeout=self.config.timeout_seconds,
        )
        return result if isinstance(result, list) else []

    async def get_all_events(
        self,
        *,
        active: bool = True,
        order: str = "volume",
        max_pages: int = 50,
    ) -> list[dict]:
        """Fetch all events with pagination."""
        all_events: list[dict] = []

        async def fetch_page(params: dict) -> list[dict]:
            params["active"] = str(active).lower()
            params["order"] = order
            return await self.session.request(
                "GET",
                f"{self.config.base_url}/events",
                endpoint_name="gamma_events",
                rate_limit_rps=self.config.rate_limit_rps,
                headers=DEFAULT_HEADERS,
                params=params,
                timeout=self.config.timeout_seconds,
            ) or []

        page_count = 0
        async for page in self.paginator.paginate(fetch_page):
            all_events.extend(page)
            page_count += 1
            if page_count >= max_pages:
                break

        logger.info("Fetched %d events from GAMMA API", len(all_events))
        return all_events
