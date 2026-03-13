# Polymarket Analytics Data Pipeline

> **⚠️ DISCLAIMER: Not financial advice. This tool is for educational and research purposes only.**

A production-grade, real-time data pipeline and trading advisor for [Polymarket](https://polymarket.com/) prediction markets.

## Architecture

```
                    ┌─────────────┐
                    │   CLI / UI  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
    ┌─────────▼──┐  ┌──────▼─────┐  ┌──▼──────────┐
    │   GAMMA    │  │   CLOB     │  │  Goldsky     │
    │  REST API  │  │  REST API  │  │  GraphQL     │
    └─────────┬──┘  └──────┬─────┘  └──┬──────────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼──────┐
                    │  Unified    │
                    │  Models     │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │                         │
    ┌─────────▼──────┐      ┌──────────▼─────┐
    │  PostgreSQL    │      │   Trading      │
    │  Storage       │      │   Advisor      │
    └────────────────┘      └────────────────┘
```

### Data Sources

| Source | Type | Pagination | Data |
|--------|------|-----------|------|
| GAMMA API | REST | offset (limit+offset) | Markets, events metadata |
| CLOB API | REST | cursor (next_cursor, terminal=`LTE=`) | Prices, trades, orderbook |
| Goldsky Subgraphs | GraphQL | first+skip | On-chain orders, OI positions |
| Analytics HTML | Playwright | page-based | Fallback table scraping |

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

### Trading advisor

```bash
# Generate signal for a market
python main.py advisor --condition-id 0x1234...abcd

# Run backtest
python main.py backtest --condition-id 0x1234...abcd

# Backtest with custom parameters
python main.py backtest --condition-id 0x1234...abcd --window 40 --hold 8

# Export backtest results
python main.py backtest --condition-id 0x1234...abcd --output backtest.json
```

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
├── advisor/
│   ├── __init__.py
│   ├── features.py         # Feature engineering
│   ├── signals.py          # Signal generation
│   └── backtest.py         # Backtesting engine
├── tests/
│   ├── test_pagination.py
│   ├── test_models.py
│   └── test_backtest.py
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

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/polymarket` | PostgreSQL connection |
| `SUPABASE_URL` | – | Supabase project URL |
| `SUPABASE_KEY` | – | Supabase anon key |
| `PROXY_URL` | – | HTTP proxy for requests |

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

# Run specific test
python -m pytest tests/test_models.py -v
python -m pytest tests/test_backtest.py -v
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

*Not financial advice. For educational and research purposes only.*
