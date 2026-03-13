"""
Goldsky GraphQL subgraph client for on-chain order and OI data.

- Orderbook subgraph: orders (bids/asks on-chain)
- OI subgraph: user positions / open interest

Pagination: GraphQL-style offset (first + skip)
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import GoldskyConfig, get_config
from .models import (
    OrderBookEntry,
    UserPosition,
    goldsky_order_to_model,
    goldsky_position_to_model,
)
from .pagination import GraphQLPaginator
from .resilience import ResilientSession

logger = logging.getLogger(__name__)

GRAPHQL_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}


class GoldskyClient:
    """Client for Goldsky GraphQL subgraphs on Polygon."""

    def __init__(
        self,
        session: ResilientSession,
        config: Optional[GoldskyConfig] = None,
    ):
        self.session = session
        self.config = config or get_config().goldsky
        self.paginator = GraphQLPaginator(
            first=self.config.max_first,
            max_first=self.config.max_first,
        )

    # ------------------------------------------------------------------ #
    #  Orderbook orders
    # ------------------------------------------------------------------ #

    async def get_orders(
        self,
        market_address: str,
        *,
        max_pages: int = 10,
    ) -> list[OrderBookEntry]:
        """Fetch on-chain orders for a market from the orderbook subgraph."""
        all_orders: list[OrderBookEntry] = []

        async def fetch_page(first: int, skip: int) -> list[dict]:
            query = """
            {
              orders(
                first: %d
                skip: %d
                orderBy: timestamp
                orderDirection: desc
                where: { market: "%s" }
              ) {
                id
                market
                user
                price
                size
                side
                timestamp
              }
            }
            """ % (first, skip, market_address)

            result = await self.session.request(
                "POST",
                self.config.orderbook_url,
                endpoint_name="goldsky_orderbook",
                rate_limit_rps=self.config.rate_limit_rps,
                headers=GRAPHQL_HEADERS,
                json_body={"query": query},
                timeout=self.config.timeout_seconds,
            )

            if isinstance(result, dict):
                data = result.get("data", {})
                return data.get("orders", [])
            return []

        page_count = 0
        async for page in self.paginator.paginate(fetch_page):
            for raw in page:
                try:
                    all_orders.append(goldsky_order_to_model(raw))
                except Exception as exc:
                    logger.warning("Failed to parse Goldsky order: %s", exc)
            page_count += 1
            if page_count >= max_pages:
                break

        logger.info(
            "Fetched %d orders for market %s",
            len(all_orders),
            market_address[:16],
        )
        return all_orders

    # ------------------------------------------------------------------ #
    #  Open Interest positions
    # ------------------------------------------------------------------ #

    async def get_positions(
        self,
        market_address: str,
        *,
        max_pages: int = 10,
    ) -> list[UserPosition]:
        """Fetch user positions (OI) for a market from the OI subgraph."""
        all_positions: list[UserPosition] = []

        async def fetch_page(first: int, skip: int) -> list[dict]:
            query = """
            {
              userPositions(
                first: %d
                skip: %d
                where: { market: "%s" }
              ) {
                id
                user
                market
                outcomeIndex
                size
                avgPrice
              }
            }
            """ % (first, skip, market_address)

            result = await self.session.request(
                "POST",
                self.config.oi_url,
                endpoint_name="goldsky_oi",
                rate_limit_rps=self.config.rate_limit_rps,
                headers=GRAPHQL_HEADERS,
                json_body={"query": query},
                timeout=self.config.timeout_seconds,
            )

            if isinstance(result, dict):
                data = result.get("data", {})
                return data.get("userPositions", [])
            return []

        page_count = 0
        async for page in self.paginator.paginate(fetch_page):
            for raw in page:
                try:
                    all_positions.append(goldsky_position_to_model(raw))
                except Exception as exc:
                    logger.warning("Failed to parse Goldsky position: %s", exc)
            page_count += 1
            if page_count >= max_pages:
                break

        logger.info(
            "Fetched %d positions for market %s",
            len(all_positions),
            market_address[:16],
        )
        return all_positions

    # ------------------------------------------------------------------ #
    #  Aggregate OI for a market
    # ------------------------------------------------------------------ #

    async def get_total_open_interest(self, market_address: str) -> float:
        """Calculate total open interest by summing all position sizes."""
        positions = await self.get_positions(market_address)
        total = sum(p.size for p in positions)
        return total
