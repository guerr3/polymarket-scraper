# Polymarket Analytics Data Pipeline

> **⚠️ DISCLAIMER: Not financial advice. This tool is for educational and research purposes only.**

A production-grade, real-time data pipeline and intelligence platform for [Polymarket](https://polymarket.com/) prediction markets. Combines multi-source news aggregation, NLP sentiment analysis, cross-market arbitrage detection, historical calibration modeling, and event-driven triggers into a unified intelligence pipeline.

## Architecture

```
                         ┌─────────────┐
                         │   CLI / UI  │
                         └──────┬──────┘
                                │
               ┌────────────────┼────────────────┐
               │                │                │
     ┌─────────▼──┐  ┌─────────▼──────┐  ┌──────▼─────────┐
     │   GAMMA    │  │   CLOB         │  │  Goldsky        │
     │  REST API  │  │  REST API      │  │  GraphQL        │
     └─────────┬──┘  └─────────┬──────┘  └──────┬─────────┘
               │                │                │
               └────────────────┼────────────────┘
                                │
                         ┌──────▼──────┐
                         │  Unified    │
                         │  Models     │
                         └──────┬──────┘
                                │
        ┌───────────┬───────────┼───────────┬───────────┐
        │           │           │           │           │
  ┌─────▼────┐ ┌────▼────┐ ┌───▼────┐ ┌────▼────┐ ┌───▼──────┐
  │Sentiment │ │ News    │ │Arbitr. │ │ Calib.  │ │ Event    │
  │ NLP      │ │ Feeds   │ │Detect. │ │ Model   │ │ Triggers │
  └─────┬────┘ └────┬────┘ └───┬────┘ └────┬────┘ └───┬──────┘
        │           │           │           │           │
        └───────────┴───────────┼───────────┴───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Intelligence Pipeline │
                    │  (Weighted Composite)  │
                    └───────────┬───────────┘
                                │
               ┌────────────────┼────────────────┐
               │                                 │
     ┌─────────▼──────┐              ┌───────────▼─────┐
     │  PostgreSQL    │              │  Signal Report   │
     │  Storage       │              │  (Buy/Sell/Hold) │
     └────────────────┘              └─────────────────┘
```

### Data Sources

| Source            | Type       | Pagination                            | Data                          |
| ----------------- | ---------- | ------------------------------------- | ----------------------------- |
| GAMMA API         | REST       | offset (limit+offset)                 | Markets, events metadata      |
| CLOB API          | REST       | cursor (next_cursor, terminal=`LTE=`) | Prices, trades, orderbook     |
| Goldsky Subgraphs | GraphQL    | first+skip                            | On-chain orders, OI positions |
| Analytics HTML    | Playwright | page-based                            | Fallback table scraping       |

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 14+ (or use Docker)

### Installation

```bash
# Clone and install
cd polymarket-scraper
pip install -r requirements.txt

# Optional: install Playwright for HTML fallback
pip install playwright
playwright install chromium

# Copy env config
cp .env.example .env
# Edit .env with your DATABASE_URL
```

### Database (Docker)

```bash
# Start PostgreSQL
docker compose up -d postgres

# Run migrations
python main.py migrate
```

## Usage

### One-shot sync

```bash
# Sync all active markets
python main.py sync --entity markets

# Sync and store in PostgreSQL
python main.py sync --entity markets --store

# Export to JSON
python main.py sync --entity markets --output markets.json

# Sync trades for a specific market
python main.py sync --entity trades --condition-id 0x1234...abcd

# Sync price history
python main.py sync --entity prices --condition-id 0x1234...abcd

# Sync events
python main.py sync --entity events
```

### Daemon mode (continuous ingestion)

```bash
# Monitor top 100 markets continuously
python main.py daemon --markets-top-n 100

# With verbose logging
python main.py daemon --markets-top-n 50 --verbose
```

Polling cadences:

- Markets & events: every 60 seconds
- CLOB prices & trades: every 15 seconds (top-N markets)
- Goldsky subgraphs: every 120 seconds

### Intelligence pipeline

```bash
# Full intelligence analysis for a market (sentiment + triggers + calibration + arbitrage)
python main.py intel analyze --slug "will-btc-reach-100k"

# Sentiment analysis only
python main.py intel sentiment --slug "will-btc-reach-100k"

# Cross-market arbitrage scan
python main.py intel arbitrage

# Historical calibration report
python main.py intel calibration

# Event trigger scan for a market
python main.py intel triggers --slug "will-btc-reach-100k"
```

### Intelligence components

| Component       | Description                                                                       | Weight |
| --------------- | --------------------------------------------------------------------------------- | ------ |
| **Sentiment**   | Keyword/lexicon NLP with negation detection, source credibility weighting         | 30%    |
| **Triggers**    | Breaking news, official statements, key accounts, news clusters, sentiment shifts | 30%    |
| **Calibration** | Historical accuracy analysis with Brier score, log loss, exploitable price ranges | 20%    |
| **Arbitrage**   | Cross-market correlation, implication/exclusion logic, temporal consistency       | 20%    |

Signals: `STRONG_BUY` > `BUY` > `LEAN_BUY` > `NEUTRAL` > `LEAN_SELL` > `SELL` > `STRONG_SELL`

### HTML fallback

```bash
# Scrape from polymarketanalytics.com (fallback only)
python main.py html-fallback --max-pages 5 --output fallback.json
```

## Project Structure

```
polymarket-scraper/
├── polymarket_client/
│   ├── __init__.py
│   ├── config.py           # Configuration & env vars
│   ├── models.py           # Pydantic models + mapping functions
│   ├── gamma_client.py     # GAMMA REST API client
│   ├── clob_client.py      # CLOB REST API client
│   ├── goldsky_client.py   # Goldsky GraphQL client
│   ├── html_fallback.py    # Playwright HTML scraper
│   ├── pagination.py       # Offset, cursor, GraphQL, HTML paginators
│   ├── resilience.py       # Retries, rate limiting, circuit breaker
│   ├── storage.py          # PostgreSQL/Supabase storage layer
│   └── realtime.py         # Polling loops & scheduler
├── intelligence/
│   ├── __init__.py
│   ├── sentiment.py        # Keyword/lexicon NLP sentiment analyzer
│   ├── news_feeds.py       # RSS/Atom/Google News/Nitter aggregator
│   ├── arbitrage.py        # Cross-market arbitrage detector
│   ├── calibration.py      # Historical calibration model
│   ├── event_triggers.py   # Event-driven trigger detector
│   └── signals.py          # Unified intelligence pipeline
├── advisor/
│   ├── __init__.py
│   ├── features.py         # Feature engineering (legacy)
│   ├── signals.py          # Signal generation (legacy)
│   └── backtest.py         # Backtesting engine (legacy)
├── tests/
│   ├── test_pagination.py
│   ├── test_models.py
│   ├── test_backtest.py
│   ├── test_sentiment.py
│   ├── test_news_feeds.py
│   ├── test_arbitrage.py
│   ├── test_calibration.py
│   ├── test_event_triggers.py
│   └── test_signals.py
├── migrations/
│   └── 001_initial.sql
├── cli.py                  # Typer CLI
├── main.py                 # Entry point
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

All settings are configurable via environment variables:

| Variable              | Default                                                    | Description                                                     |
| --------------------- | ---------------------------------------------------------- | --------------------------------------------------------------- |
| `DATABASE_URL`        | `postgresql://postgres:postgres@localhost:5432/polymarket` | PostgreSQL connection                                           |
| `SUPABASE_URL`        | –                                                          | Supabase project URL                                            |
| `SUPABASE_KEY`        | –                                                          | Supabase anon key                                               |
| `CLOB_API_KEY`        | –                                                          | Optional CLOB API key for authenticated endpoints (e.g. trades) |
| `CLOB_API_KEY_HEADER` | `X-API-Key`                                                | Header name used for `CLOB_API_KEY`                             |
| `CLOB_AUTH_SCHEME`    | `Bearer`                                                   | Authorization scheme used with `CLOB_API_KEY`                   |
| `PROXY_URL`           | –                                                          | HTTP proxy for requests                                         |

## Resilience

The pipeline includes:

- **Exponential backoff** with jitter on retryable HTTP errors (429, 5xx)
- **Per-endpoint rate limiting** (token bucket)
- **Circuit breakers** per endpoint (opens after 5 failures, resets after 5min)
- **Rotating User-Agent pool** (6 browser-like UAs)
- **Optional proxy support** via `PROXY_URL`

## Testing

```bash
# Run all tests
cd polymarket-scraper
python -m pytest tests/ -v

# Run intelligence tests only
python -m pytest tests/test_sentiment.py tests/test_news_feeds.py tests/test_arbitrage.py tests/test_calibration.py tests/test_event_triggers.py tests/test_signals.py -v

# Run specific test
python -m pytest tests/test_models.py -v
```

## Ethics & Legal

- Uses only **public, unauthenticated** endpoints
- Respects rate limits and robots.txt
- HTML scraping is a **fallback only**, with humane delays
- **No trading execution** — advisor outputs are signals only
- For analytics, research, and strategy prototyping

## License

MIT

---

_Not financial advice. For educational and research purposes only._
