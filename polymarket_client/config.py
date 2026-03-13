"""
Configuration module for Polymarket data pipeline.

All settings are configurable via environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class GammaConfig:
    """Polymarket GAMMA REST API configuration."""

    base_url: str = "https://gamma-api.polymarket.com"
    default_limit: int = 100
    max_limit: int = 500
    rate_limit_rps: float = 5.0  # requests per second
    timeout_seconds: int = 30


@dataclass(frozen=True)
class ClobConfig:
    """Polymarket CLOB REST API configuration."""

    base_url: str = "https://clob.polymarket.com"
    terminal_cursor: str = "LTE="
    rate_limit_rps: float = 10.0
    timeout_seconds: int = 30


@dataclass(frozen=True)
class GoldskyConfig:
    """Goldsky GraphQL subgraph configuration."""

    orderbook_url: str = (
        "https://api.goldsky.com/api/public/"
        "project_cl6mb8i9h0003e201j6li0diw/"
        "subgraphs/orderbook-subgraph/0.0.1/gn"
    )
    oi_url: str = (
        "https://api.goldsky.com/api/public/"
        "project_cl6mb8i9h0003e201j6li0diw/"
        "subgraphs/oi-subgraph/0.0.6/gn"
    )
    max_first: int = 1000
    rate_limit_rps: float = 3.0
    timeout_seconds: int = 60


@dataclass(frozen=True)
class AnalyticsHtmlConfig:
    """Polymarket Analytics HTML fallback configuration."""

    base_url: str = "https://polymarketanalytics.com"
    markets_path: str = "/markets"
    default_sources: str = "Polymarket"
    default_status: str = "Active"
    selectors: dict = field(default_factory=lambda: {
        "rows": "table tbody tr",
        "market_link": "td:first-child a",
        "price": "td:nth-child(2) span",
        "volume": "td:nth-child(4)",
        "end_date": "td:nth-child(6)",
        "tags": "td:nth-child(7) a",
    })


@dataclass(frozen=True)
class StorageConfig:
    """Supabase / PostgreSQL storage configuration."""

    supabase_url: str = os.environ.get("SUPABASE_URL", "")
    supabase_key: str = os.environ.get("SUPABASE_KEY", "")
    database_url: str = os.environ.get(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/polymarket",
    )


@dataclass(frozen=True)
class PollingConfig:
    """Real-time polling cadence configuration."""

    markets_interval_seconds: int = 60
    clob_interval_seconds: int = 15
    goldsky_interval_seconds: int = 120
    top_n_markets: int = 100


@dataclass(frozen=True)
class ResilienceConfig:
    """Resilience and anti-bot configuration."""

    max_retries: int = 5
    base_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 60.0
    jitter_factor: float = 0.5
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_seconds: float = 300.0
    proxy_url: Optional[str] = os.environ.get("PROXY_URL")


# User-Agent rotation pool
USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
]

# Default headers
DEFAULT_HEADERS: dict[str, str] = {
    "Accept": "application/json",
}


@dataclass
class AppConfig:
    """Root application configuration."""

    gamma: GammaConfig = field(default_factory=GammaConfig)
    clob: ClobConfig = field(default_factory=ClobConfig)
    goldsky: GoldskyConfig = field(default_factory=GoldskyConfig)
    analytics_html: AnalyticsHtmlConfig = field(default_factory=AnalyticsHtmlConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    polling: PollingConfig = field(default_factory=PollingConfig)
    resilience: ResilienceConfig = field(default_factory=ResilienceConfig)


def get_config() -> AppConfig:
    """Return the application configuration singleton."""
    return AppConfig()
