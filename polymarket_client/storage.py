"""
Storage layer: Supabase / PostgreSQL interface.

Provides upsert semantics for markets and append-only for trades.
Uses asyncpg for direct PostgreSQL access.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import asyncpg

from .config import StorageConfig, get_config
from .models import Market, Trade, Trader

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  SQL Schema (also used for migrations)
# --------------------------------------------------------------------------- #

SCHEMA_SQL = """
-- Markets table with upsert on (condition_id)
CREATE TABLE IF NOT EXISTS markets (
    id                SERIAL PRIMARY KEY,
    condition_id      TEXT UNIQUE NOT NULL,
    question          TEXT NOT NULL DEFAULT '',
    slug              TEXT DEFAULT '',
    description       TEXT DEFAULT '',
    start_date        TIMESTAMPTZ,
    end_date          TIMESTAMPTZ,
    status            TEXT DEFAULT 'Active',
    source            TEXT DEFAULT 'Polymarket',
    volume            DOUBLE PRECISION DEFAULT 0,
    volume_24h        DOUBLE PRECISION DEFAULT 0,
    open_interest     DOUBLE PRECISION DEFAULT 0,
    liquidity         DOUBLE PRECISION DEFAULT 0,
    image             TEXT DEFAULT '',
    icon              TEXT DEFAULT '',
    tags              JSONB DEFAULT '[]'::jsonb,
    outcomes          JSONB DEFAULT '[]'::jsonb,
    resolved_value    TEXT,
    creator_address   TEXT,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_markets_condition_id ON markets(condition_id);
CREATE INDEX IF NOT EXISTS idx_markets_status ON markets(status);
CREATE INDEX IF NOT EXISTS idx_markets_volume ON markets(volume DESC);

-- Trades table (append-only)
CREATE TABLE IF NOT EXISTS trades (
    id                  TEXT PRIMARY KEY,
    market_condition_id TEXT NOT NULL,
    asset_id            TEXT DEFAULT '',
    side                TEXT NOT NULL,
    size                DOUBLE PRECISION DEFAULT 0,
    price               DOUBLE PRECISION DEFAULT 0,
    timestamp           BIGINT DEFAULT 0,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market_condition_id);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC);

-- Traders table
CREATE TABLE IF NOT EXISTS traders (
    address         TEXT PRIMARY KEY,
    display_name    TEXT,
    pnl_1d          DOUBLE PRECISION DEFAULT 0,
    pnl_1w          DOUBLE PRECISION DEFAULT 0,
    pnl_1m          DOUBLE PRECISION DEFAULT 0,
    positions_count INTEGER DEFAULT 0,
    volume_traded   DOUBLE PRECISION DEFAULT 0,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Price history table (append-only timeseries)
CREATE TABLE IF NOT EXISTS price_history (
    condition_id    TEXT NOT NULL,
    timestamp       BIGINT NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (condition_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_price_history_condition
    ON price_history(condition_id, timestamp DESC);
"""


class Storage:
    """Async PostgreSQL storage layer for the Polymarket pipeline."""

    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or get_config().storage
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Establish connection pool."""
        self._pool = await asyncpg.create_pool(
            self.config.database_url,
            min_size=2,
            max_size=10,
        )
        logger.info("Connected to PostgreSQL")

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            logger.info("PostgreSQL connection closed")

    async def run_migrations(self) -> None:
        """Create tables if they don't exist."""
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)
        logger.info("Database migrations applied")

    # ------------------------------------------------------------------ #
    #  Markets (upsert)
    # ------------------------------------------------------------------ #

    async def upsert_market(self, market: Market) -> None:
        """Insert or update a market by condition_id."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO markets (
                    condition_id, question, slug, description,
                    start_date, end_date, status, source,
                    volume, volume_24h, open_interest, liquidity,
                    image, icon, tags, outcomes,
                    resolved_value, creator_address, updated_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8,
                    $9, $10, $11, $12, $13, $14, $15, $16,
                    $17, $18, NOW()
                )
                ON CONFLICT (condition_id) DO UPDATE SET
                    question = EXCLUDED.question,
                    slug = EXCLUDED.slug,
                    description = EXCLUDED.description,
                    start_date = EXCLUDED.start_date,
                    end_date = EXCLUDED.end_date,
                    status = EXCLUDED.status,
                    volume = EXCLUDED.volume,
                    volume_24h = EXCLUDED.volume_24h,
                    open_interest = EXCLUDED.open_interest,
                    liquidity = EXCLUDED.liquidity,
                    image = EXCLUDED.image,
                    icon = EXCLUDED.icon,
                    tags = EXCLUDED.tags,
                    outcomes = EXCLUDED.outcomes,
                    resolved_value = EXCLUDED.resolved_value,
                    creator_address = EXCLUDED.creator_address,
                    updated_at = NOW()
                """,
                market.condition_id,
                market.question,
                market.slug,
                market.description,
                market.start_date,
                market.end_date,
                market.status.value,
                market.source.value,
                market.volume,
                market.volume_24h,
                market.open_interest,
                market.liquidity,
                market.image,
                market.icon,
                json.dumps(market.tags),
                json.dumps([o.model_dump() for o in market.outcomes]),
                market.resolved_value,
                market.creator_address,
            )

    async def upsert_markets(self, markets: list[Market]) -> int:
        """Bulk upsert markets. Returns count of processed markets."""
        count = 0
        for market in markets:
            try:
                await self.upsert_market(market)
                count += 1
            except Exception as exc:
                logger.warning(
                    "Failed to upsert market %s: %s",
                    market.condition_id[:16],
                    exc,
                )
        logger.info("Upserted %d/%d markets", count, len(markets))
        return count

    # ------------------------------------------------------------------ #
    #  Trades (append-only with conflict ignore)
    # ------------------------------------------------------------------ #

    async def insert_trades(self, trades: list[Trade]) -> int:
        """Insert trades, ignoring duplicates. Returns count inserted."""
        count = 0
        async with self._pool.acquire() as conn:
            for trade in trades:
                try:
                    result = await conn.execute(
                        """
                        INSERT INTO trades (
                            id, market_condition_id, asset_id,
                            side, size, price, timestamp
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        trade.id,
                        trade.market_condition_id,
                        trade.asset_id,
                        trade.side.value,
                        trade.size,
                        trade.price,
                        trade.timestamp,
                    )
                    if "INSERT" in result:
                        count += 1
                except Exception as exc:
                    logger.warning("Failed to insert trade %s: %s", trade.id, exc)
        logger.info("Inserted %d/%d trades", count, len(trades))
        return count

    # ------------------------------------------------------------------ #
    #  Price history (append-only)
    # ------------------------------------------------------------------ #

    async def insert_price_history(
        self, condition_id: str, points: list
    ) -> int:
        """Insert price history points, ignoring duplicates."""
        count = 0
        async with self._pool.acquire() as conn:
            for point in points:
                try:
                    await conn.execute(
                        """
                        INSERT INTO price_history (condition_id, timestamp, price)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (condition_id, timestamp) DO NOTHING
                        """,
                        condition_id,
                        point.timestamp,
                        point.price,
                    )
                    count += 1
                except Exception:
                    pass
        return count

    # ------------------------------------------------------------------ #
    #  Queries
    # ------------------------------------------------------------------ #

    async def get_top_markets(
        self,
        limit: int = 100,
        status: str = "Active",
    ) -> list[dict]:
        """Get top markets by volume."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM markets
                WHERE status = $1
                ORDER BY volume DESC
                LIMIT $2
                """,
                status,
                limit,
            )
            return [dict(r) for r in rows]

    async def get_market_by_condition(self, condition_id: str) -> Optional[dict]:
        """Get a single market by condition_id."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM markets WHERE condition_id = $1",
                condition_id,
            )
            return dict(row) if row else None

    async def get_price_history(
        self,
        condition_id: str,
        limit: int = 1000,
    ) -> list[dict]:
        """Get stored price history for a market."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT timestamp, price FROM price_history
                WHERE condition_id = $1
                ORDER BY timestamp ASC
                LIMIT $2
                """,
                condition_id,
                limit,
            )
            return [dict(r) for r in rows]
