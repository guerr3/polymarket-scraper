"""
Feature engineering for the Polymarket trading advisor.

Computes per-market features from price history, liquidity,
open interest, volume, and orderbook data.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from polymarket_client.models import PricePoint, OrderBookEntry

logger = logging.getLogger(__name__)


@dataclass
class MarketFeatures:
    """Computed features for a single market."""

    condition_id: str = ""

    # Price-based
    current_price: float = 0.0
    price_change_1h: float = 0.0
    price_change_24h: float = 0.0
    price_change_7d: float = 0.0
    rolling_volatility: float = 0.0
    momentum: float = 0.0
    mean_reversion_score: float = 0.0

    # Moving averages
    sma_12: float = 0.0
    sma_26: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0

    # Volume & liquidity
    volume: float = 0.0
    volume_24h: float = 0.0
    open_interest: float = 0.0
    liquidity: float = 0.0

    # Orderbook
    bid_ask_spread: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    order_imbalance: float = 0.0  # -1 to 1 (negative = more asks)

    # Derived
    volume_oi_ratio: float = 0.0
    price_distance_from_50: float = 0.0  # distance from 0.5 (max uncertainty)


def compute_price_features(
    prices: list[PricePoint],
    window_hours: int = 24,
) -> dict:
    """
    Compute price-based features from a timeseries.

    Returns dict of feature values.
    """
    if not prices or len(prices) < 2:
        return {}

    # Sort by timestamp
    sorted_prices = sorted(prices, key=lambda p: p.timestamp)
    values = [p.price for p in sorted_prices]
    timestamps = [p.timestamp for p in sorted_prices]

    current = values[-1]
    features: dict = {"current_price": current}

    # Price changes at different horizons
    now_ts = timestamps[-1]
    for label, hours in [("1h", 1), ("24h", 24), ("7d", 168)]:
        target_ts = now_ts - (hours * 3600)
        past_price = _find_nearest_price(sorted_prices, target_ts)
        if past_price and past_price > 0:
            features[f"price_change_{label}"] = (current - past_price) / past_price
        else:
            features[f"price_change_{label}"] = 0.0

    # Rolling volatility (std of returns over last N points)
    returns = _compute_returns(values)
    if len(returns) >= 10:
        features["rolling_volatility"] = _std(returns[-min(100, len(returns)):])
    else:
        features["rolling_volatility"] = 0.0

    # Simple momentum (rate of price change)
    if len(values) >= 5:
        recent = values[-5:]
        features["momentum"] = (recent[-1] - recent[0]) / max(len(recent), 1)
    else:
        features["momentum"] = 0.0

    # Mean reversion score: how far from rolling mean
    if len(values) >= 20:
        rolling_mean = sum(values[-20:]) / 20
        if rolling_mean > 0:
            features["mean_reversion_score"] = (current - rolling_mean) / rolling_mean
        else:
            features["mean_reversion_score"] = 0.0

    # Moving averages
    features["sma_12"] = _sma(values, 12)
    features["sma_26"] = _sma(values, 26)
    features["ema_12"] = _ema(values, 12)
    features["ema_26"] = _ema(values, 26)

    # Distance from 0.5 (max uncertainty point)
    features["price_distance_from_50"] = abs(current - 0.5)

    return features


def compute_orderbook_features(
    orders: list[OrderBookEntry],
) -> dict:
    """Compute orderbook imbalance and depth features."""
    if not orders:
        return {}

    bids = [o for o in orders if o.side.lower() in ("buy", "bid", "0")]
    asks = [o for o in orders if o.side.lower() in ("sell", "ask", "1")]

    bid_depth = sum(o.size * o.price for o in bids)
    ask_depth = sum(o.size * o.price for o in asks)
    total_depth = bid_depth + ask_depth

    # Order imbalance: positive = more bid pressure
    imbalance = 0.0
    if total_depth > 0:
        imbalance = (bid_depth - ask_depth) / total_depth

    # Bid-ask spread
    spread = 0.0
    if bids and asks:
        best_bid = max(o.price for o in bids) if bids else 0
        best_ask = min(o.price for o in asks) if asks else 0
        if best_ask > 0:
            spread = (best_ask - best_bid) / best_ask

    return {
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "order_imbalance": imbalance,
        "bid_ask_spread": spread,
    }


def build_market_features(
    condition_id: str,
    prices: list[PricePoint],
    orders: list[OrderBookEntry] | None = None,
    volume: float = 0.0,
    volume_24h: float = 0.0,
    open_interest: float = 0.0,
    liquidity: float = 0.0,
) -> MarketFeatures:
    """
    Build the full MarketFeatures object from all available data.
    """
    mf = MarketFeatures(condition_id=condition_id)

    # Price features
    pf = compute_price_features(prices)
    mf.current_price = pf.get("current_price", 0.0)
    mf.price_change_1h = pf.get("price_change_1h", 0.0)
    mf.price_change_24h = pf.get("price_change_24h", 0.0)
    mf.price_change_7d = pf.get("price_change_7d", 0.0)
    mf.rolling_volatility = pf.get("rolling_volatility", 0.0)
    mf.momentum = pf.get("momentum", 0.0)
    mf.mean_reversion_score = pf.get("mean_reversion_score", 0.0)
    mf.sma_12 = pf.get("sma_12", 0.0)
    mf.sma_26 = pf.get("sma_26", 0.0)
    mf.ema_12 = pf.get("ema_12", 0.0)
    mf.ema_26 = pf.get("ema_26", 0.0)
    mf.price_distance_from_50 = pf.get("price_distance_from_50", 0.0)

    # Orderbook features
    if orders:
        of = compute_orderbook_features(orders)
        mf.bid_depth = of.get("bid_depth", 0.0)
        mf.ask_depth = of.get("ask_depth", 0.0)
        mf.order_imbalance = of.get("order_imbalance", 0.0)
        mf.bid_ask_spread = of.get("bid_ask_spread", 0.0)

    # Volume / liquidity
    mf.volume = volume
    mf.volume_24h = volume_24h
    mf.open_interest = open_interest
    mf.liquidity = liquidity

    # Derived
    if open_interest > 0:
        mf.volume_oi_ratio = volume_24h / open_interest

    return mf


# --------------------------------------------------------------------------- #
#  Pure-Python technical indicator helpers
# --------------------------------------------------------------------------- #

def _sma(values: list[float], period: int) -> float:
    """Simple moving average of the last `period` values."""
    if len(values) < period:
        return sum(values) / len(values) if values else 0.0
    return sum(values[-period:]) / period


def _ema(values: list[float], period: int) -> float:
    """Exponential moving average."""
    if not values:
        return 0.0
    k = 2.0 / (period + 1)
    ema_val = values[0]
    for v in values[1:]:
        ema_val = v * k + ema_val * (1 - k)
    return ema_val


def _compute_returns(values: list[float]) -> list[float]:
    """Compute simple returns from a price series."""
    returns = []
    for i in range(1, len(values)):
        if values[i - 1] != 0:
            returns.append((values[i] - values[i - 1]) / values[i - 1])
        else:
            returns.append(0.0)
    return returns


def _std(values: list[float]) -> float:
    """Standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _find_nearest_price(
    prices: list[PricePoint], target_ts: int
) -> Optional[float]:
    """Find the price nearest to the target timestamp."""
    if not prices:
        return None
    best = min(prices, key=lambda p: abs(p.timestamp - target_ts))
    return best.price
