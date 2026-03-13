"""
Unified data models for the Polymarket data pipeline.

All upstream API responses are normalized into these Pydantic models.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
#  Enums
# --------------------------------------------------------------------------- #

class MarketStatus(str, Enum):
    ACTIVE = "Active"
    CLOSED = "Closed"
    RESOLVED = "Resolved"


class MarketSource(str, Enum):
    POLYMARKET = "Polymarket"
    KALSHI = "Kalshi"
    LIMITLESS = "Limitless"


class TradeSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


# --------------------------------------------------------------------------- #
#  Core domain models
# --------------------------------------------------------------------------- #

class Outcome(BaseModel):
    """A single outcome within a prediction market."""

    name: str
    price: float = Field(ge=0.0, le=1.0, description="Probability 0-1")
    token_id: str = ""


class Market(BaseModel):
    """Unified market representation across all data sources."""

    id: Optional[int] = None
    condition_id: str = ""
    question: str = ""
    slug: str = ""
    description: str = ""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: MarketStatus = MarketStatus.ACTIVE
    source: MarketSource = MarketSource.POLYMARKET
    volume: float = 0.0
    volume_24h: float = 0.0
    open_interest: float = 0.0
    liquidity: float = 0.0
    image: str = ""
    icon: str = ""
    tags: list[str] = Field(default_factory=list)
    outcomes: list[Outcome] = Field(default_factory=list)
    resolved_value: Optional[str] = None
    creator_address: Optional[str] = None

    @field_validator("end_date", "start_date", mode="before")
    @classmethod
    def parse_date(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            # Handle various ISO formats
            for fmt in (
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d",
            ):
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
            return None
        return v


class Trader(BaseModel):
    """Trader / wallet profile."""

    address: str
    display_name: Optional[str] = None
    pnl_1d: float = 0.0
    pnl_1w: float = 0.0
    pnl_1m: float = 0.0
    positions_count: int = 0
    volume_traded: float = 0.0


class Trade(BaseModel):
    """A single trade execution."""

    id: str
    market_condition_id: str = ""
    asset_id: str = ""
    side: TradeSide = TradeSide.BUY
    size: float = 0.0
    price: float = 0.0
    timestamp: int = 0  # unix seconds


class PricePoint(BaseModel):
    """A single point in a price timeseries."""

    timestamp: int  # unix seconds
    price: float


class OrderBookEntry(BaseModel):
    """On-chain order from Goldsky subgraph."""

    id: str
    market: str
    user: str
    price: float
    size: float
    side: str
    timestamp: int


class UserPosition(BaseModel):
    """Open interest position from Goldsky subgraph."""

    id: str
    user: str
    market: str
    outcome_index: int = 0
    size: float = 0.0
    avg_price: float = 0.0


# --------------------------------------------------------------------------- #
#  Trading advisor models
# --------------------------------------------------------------------------- #

class TradingSignal(BaseModel):
    """Output of the trading advisor for a single market."""

    condition_id: str
    signal: SignalDirection = SignalDirection.NEUTRAL
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    reason: str = ""
    features: dict = Field(default_factory=dict)


class BacktestResult(BaseModel):
    """Summary statistics from a backtest run."""

    condition_id: str
    start_ts: int
    end_ts: int
    total_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    signals: list[TradingSignal] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Mapping functions: upstream → internal models
# --------------------------------------------------------------------------- #

def gamma_market_to_model(raw: dict) -> Market:
    """Map a GAMMA API market response to the internal Market model."""
    outcomes_raw = raw.get("outcomes", [])
    prices_raw = raw.get("outcomePrices", [])

    # outcomes may be a JSON string or list
    if isinstance(outcomes_raw, str):
        import json
        try:
            outcomes_raw = json.loads(outcomes_raw)
        except (json.JSONDecodeError, TypeError):
            outcomes_raw = []

    if isinstance(prices_raw, str):
        import json
        try:
            prices_raw = json.loads(prices_raw)
        except (json.JSONDecodeError, TypeError):
            prices_raw = []

    outcomes = []
    for i, name in enumerate(outcomes_raw):
        price = 0.0
        if i < len(prices_raw):
            try:
                price = float(prices_raw[i])
            except (ValueError, TypeError):
                price = 0.0
        outcomes.append(Outcome(name=str(name), price=price))

    tags = raw.get("tags", [])
    if isinstance(tags, str):
        import json
        try:
            tags = json.loads(tags)
        except (json.JSONDecodeError, TypeError):
            tags = [tags] if tags else []

    return Market(
        id=raw.get("id"),
        condition_id=raw.get("conditionId", ""),
        question=raw.get("question", ""),
        slug=raw.get("slug", ""),
        description=raw.get("description", ""),
        start_date=raw.get("startDate"),
        end_date=raw.get("endDate"),
        status=_parse_status(raw.get("active", True), raw.get("closed", False)),
        source=MarketSource.POLYMARKET,
        volume=_safe_float(raw.get("volume", 0)),
        volume_24h=_safe_float(raw.get("volume24hr", 0)),
        open_interest=_safe_float(raw.get("openInterest", 0)),
        liquidity=_safe_float(raw.get("liquidity", 0)),
        image=raw.get("image", "") or "",
        icon=raw.get("icon", "") or "",
        tags=tags,
        outcomes=outcomes,
        resolved_value=raw.get("resolvedValue"),
        creator_address=raw.get("creatorAddress"),
    )


def clob_trade_to_model(raw: dict) -> Trade:
    """Map a CLOB API trade response to the internal Trade model."""
    return Trade(
        id=str(raw.get("id", "")),
        market_condition_id=raw.get("market", ""),
        asset_id=raw.get("asset_id", ""),
        side=TradeSide(raw.get("side", "BUY").upper()),
        size=_safe_float(raw.get("size", 0)),
        price=_safe_float(raw.get("price", 0)),
        timestamp=int(raw.get("match_time", 0) or raw.get("timestamp", 0)),
    )


def goldsky_order_to_model(raw: dict) -> OrderBookEntry:
    """Map a Goldsky orderbook subgraph response to OrderBookEntry."""
    return OrderBookEntry(
        id=raw.get("id", ""),
        market=raw.get("market", ""),
        user=raw.get("user", ""),
        price=_safe_float(raw.get("price", 0)),
        size=_safe_float(raw.get("size", 0)),
        side=raw.get("side", ""),
        timestamp=int(raw.get("timestamp", 0)),
    )


def goldsky_position_to_model(raw: dict) -> UserPosition:
    """Map a Goldsky OI subgraph response to UserPosition."""
    return UserPosition(
        id=raw.get("id", ""),
        user=raw.get("user", ""),
        market=raw.get("market", ""),
        outcome_index=int(raw.get("outcomeIndex", 0)),
        size=_safe_float(raw.get("size", 0)),
        avg_price=_safe_float(raw.get("avgPrice", 0)),
    )


def price_history_to_points(raw: dict) -> list[PricePoint]:
    """Map CLOB price-history response to PricePoint list."""
    history = raw.get("history", [])
    points = []
    for entry in history:
        try:
            points.append(PricePoint(
                timestamp=int(entry.get("t", 0)),
                price=float(entry.get("p", 0.0)),
            ))
        except (ValueError, TypeError):
            continue
    return points


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _safe_float(value) -> float:
    """Safely convert a value to float."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _parse_status(active: bool, closed: bool) -> MarketStatus:
    """Determine market status from boolean flags."""
    if closed:
        return MarketStatus.RESOLVED
    if active:
        return MarketStatus.ACTIVE
    return MarketStatus.CLOSED
