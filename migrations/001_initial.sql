-- Polymarket Analytics Pipeline - Initial Schema
-- Run: psql $DATABASE_URL -f migrations/001_initial.sql

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
