"""
CLOB REST API client for Polymarket prices, orderbook, and trades.

Endpoint: https://clob.polymarket.com
Pagination: cursor-based (next_cursor, terminal = "LTE=")
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import ClobConfig, get_config, DEFAULT_HEADERS
from .models import (
    Trade,
    PricePoint,
    clob_trade_to_model,
    price_history_to_points,
)
from .pagination import CursorPaginator
from .resilience import ResilientSession

logger = logging.getLogger(__name__)


class ClobClient:
    """Client for the Polymarket CLOB REST API."""

    def __init__(
        self,
        session: ResilientSession,
        config: Optional[ClobConfig] = None,
    ):
        self.session = session
        self.config = config or get_config().clob
        self.paginator = CursorPaginator(
            terminal_cursor=self.config.terminal_cursor,
        )

    # ------------------------------------------------------------------ #
    #  Markets (CLOB)
    # ------------------------------------------------------------------ #

    async def get_markets(self) -> list[dict]:
        """Fetch all CLOB markets with cursor pagination."""
        all_markets: list[dict] = []

        async def fetch_page(params: dict) -> dict:
            return await self.session.request(
                "GET",
                f"{self.config.base_url}/markets",
                endpoint_name="clob_markets",
                rate_limit_rps=self.config.rate_limit_rps,
                headers=DEFAULT_HEADERS,
                params=params,
                timeout=self.config.timeout_seconds,
            )

        async for page in self.paginator.paginate(fetch_page):
            all_markets.extend(page)

        logger.info("Fetched %d CLOB markets", len(all_markets))
        return all_markets

    # ------------------------------------------------------------------ #
    #  Price history
    # ------------------------------------------------------------------ #

    async def get_price_history(
        self,
        condition_id: str,
        *,
        start_ts: int = 0,
        end_ts: int = 0,
        fidelity: int = 60,
        lookback_days: int = 14,
    ) -> list[PricePoint]:
        """Fetch price timeseries for a market outcome."""
        import time as _time

        now = int(_time.time())
        if end_ts <= 0:
            end_ts = now
        if start_ts <= 0:
            start_ts = end_ts - (lookback_days * 86400)

        params = {
            "market": condition_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "fidelity": fidelity,
        }

        raw = await self.session.request(
            "GET",
            f"{self.config.base_url}/prices-history",
            endpoint_name="clob_prices",
            rate_limit_rps=self.config.rate_limit_rps,
            headers=DEFAULT_HEADERS,
            params=params,
            timeout=self.config.timeout_seconds,
        )
        points = price_history_to_points(raw)
        logger.info(
            "Fetched %d price points for %s", len(points), condition_id[:16]
        )
        return points

    # ------------------------------------------------------------------ #
    #  Trades
    # ------------------------------------------------------------------ #

    async def get_trades(
        self,
        condition_id: str,
        *,
        limit: int = 500,
        max_pages: int = 20,
    ) -> list[Trade]:
        """Fetch trade history for a market using cursor pagination."""
        all_trades: list[Trade] = []

        async def fetch_page(params: dict) -> dict:
            params["market"] = condition_id
            params["limit"] = limit
            return await self.session.request(
                "GET",
                f"{self.config.base_url}/trades",
                endpoint_name="clob_trades",
                rate_limit_rps=self.config.rate_limit_rps,
                headers=DEFAULT_HEADERS,
                params=params,
                timeout=self.config.timeout_seconds,
            )

        page_count = 0
        async for page in self.paginator.paginate(fetch_page):
            for raw in page:
                try:
                    all_trades.append(clob_trade_to_model(raw))
                except Exception as exc:
                    logger.warning("Failed to parse CLOB trade: %s", exc)
            page_count += 1
            if page_count >= max_pages:
                break

        logger.info(
            "Fetched %d trades for %s", len(all_trades), condition_id[:16]
        )
        return all_trades

    # ------------------------------------------------------------------ #
    #  Orderbook snapshot
    # ------------------------------------------------------------------ #

    async def get_orderbook(self, token_id: str) -> dict:
        """Fetch current orderbook snapshot for a token."""
        raw = await self.session.request(
            "GET",
            f"{self.config.base_url}/book",
            endpoint_name="clob_book",
            rate_limit_rps=self.config.rate_limit_rps,
            headers=DEFAULT_HEADERS,
            params={"token_id": token_id},
            timeout=self.config.timeout_seconds,
        )
        return raw if isinstance(raw, dict) else {}
