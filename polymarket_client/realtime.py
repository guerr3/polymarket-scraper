"""
Real-time polling loops and scheduler for continuous data ingestion.

Cadences:
- Markets list & events: every 60s
- CLOB prices & trades: every 15-30s for high-volume markets
- Goldsky subgraphs: every 60-120s
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Optional

from .config import PollingConfig, get_config
from .gamma_client import GammaClient
from .clob_client import ClobClient
from .goldsky_client import GoldskyClient
from .storage import Storage
from .resilience import ResilientSession

logger = logging.getLogger(__name__)


class RealtimeUpdater:
    """
    Orchestrates real-time data ingestion from all sources.

    Runs polling loops in parallel asyncio tasks with configurable intervals.
    """

    def __init__(
        self,
        session: ResilientSession,
        storage: Storage,
        config: Optional[PollingConfig] = None,
    ):
        self.config = config or get_config().polling
        self.session = session
        self.storage = storage
        self.gamma = GammaClient(session)
        self.clob = ClobClient(session)
        self.goldsky = GoldskyClient(session)
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self, top_n: int = 100) -> None:
        """Start all polling loops."""
        self._running = True
        logger.info("Starting real-time updater (top %d markets)", top_n)

        self._tasks = [
            asyncio.create_task(
                self._poll_markets(), name="poll_markets"
            ),
            asyncio.create_task(
                self._poll_clob_data(top_n), name="poll_clob"
            ),
            asyncio.create_task(
                self._poll_goldsky_data(top_n), name="poll_goldsky"
            ),
        ]

        # Handle graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Stop all polling loops."""
        logger.info("Stopping real-time updater...")
        self._running = False
        for task in self._tasks:
            task.cancel()

    # ------------------------------------------------------------------ #
    #  Polling loops
    # ------------------------------------------------------------------ #

    async def _poll_markets(self) -> None:
        """Poll GAMMA API for market list + events."""
        while self._running:
            try:
                logger.info("Polling markets from GAMMA API...")
                markets = await self.gamma.get_all_markets(
                    active=True, max_pages=10
                )
                if markets:
                    await self.storage.upsert_markets(markets)
                    logger.info("Synced %d markets", len(markets))
            except Exception as exc:
                logger.error("Market polling error: %s", exc)

            await asyncio.sleep(self.config.markets_interval_seconds)

    async def _poll_clob_data(self, top_n: int) -> None:
        """Poll CLOB API for prices and trades on high-volume markets."""
        # Wait for initial market sync
        await asyncio.sleep(5)

        while self._running:
            try:
                top_markets = await self.storage.get_top_markets(limit=top_n)

                for mkt in top_markets:
                    if not self._running:
                        break

                    condition_id = mkt.get("condition_id", "")
                    if not condition_id:
                        continue

                    try:
                        # Fetch price history
                        prices = await self.clob.get_price_history(condition_id)
                        if prices:
                            await self.storage.insert_price_history(
                                condition_id, prices
                            )

                        # Fetch recent trades
                        trades = await self.clob.get_trades(
                            condition_id, limit=100, max_pages=1
                        )
                        if trades:
                            await self.storage.insert_trades(trades)

                    except Exception as exc:
                        logger.warning(
                            "CLOB poll error for %s: %s",
                            condition_id[:16],
                            exc,
                        )

                    # Small delay between markets
                    await asyncio.sleep(0.5)

            except Exception as exc:
                logger.error("CLOB polling error: %s", exc)

            await asyncio.sleep(self.config.clob_interval_seconds)

    async def _poll_goldsky_data(self, top_n: int) -> None:
        """Poll Goldsky subgraphs for order and OI data."""
        # Wait for initial sync
        await asyncio.sleep(10)

        while self._running:
            try:
                top_markets = await self.storage.get_top_markets(limit=min(top_n, 20))

                for mkt in top_markets:
                    if not self._running:
                        break

                    condition_id = mkt.get("condition_id", "")
                    if not condition_id:
                        continue

                    try:
                        # This is best-effort; Goldsky may not have data
                        # for every condition_id (needs market address)
                        oi = await self.goldsky.get_total_open_interest(
                            condition_id
                        )
                        if oi > 0:
                            logger.debug(
                                "OI for %s: %.2f", condition_id[:16], oi
                            )
                    except Exception as exc:
                        logger.debug(
                            "Goldsky poll error for %s: %s",
                            condition_id[:16],
                            exc,
                        )

                    await asyncio.sleep(1.0)

            except Exception as exc:
                logger.error("Goldsky polling error: %s", exc)

            await asyncio.sleep(self.config.goldsky_interval_seconds)

    # ------------------------------------------------------------------ #
    #  One-shot sync
    # ------------------------------------------------------------------ #

    async def sync_markets(self) -> list:
        """One-shot sync of all active markets."""
        logger.info("One-shot market sync starting...")
        markets = await self.gamma.get_all_markets(active=True, max_pages=10)
        if markets and self.storage._pool:
            await self.storage.upsert_markets(markets)
        return markets

    async def sync_trades(self, condition_id: str) -> list:
        """One-shot sync of trades for a specific market."""
        logger.info("Syncing trades for %s", condition_id[:16])
        trades = await self.clob.get_trades(condition_id)
        if trades and self.storage._pool:
            await self.storage.insert_trades(trades)
        return trades

    async def sync_prices(self, condition_id: str) -> list:
        """One-shot sync of price history for a specific market."""
        logger.info("Syncing price history for %s", condition_id[:16])
        prices = await self.clob.get_price_history(condition_id)
        if prices and self.storage._pool:
            await self.storage.insert_price_history(condition_id, prices)
        return prices
