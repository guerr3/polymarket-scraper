"""
Signal generation for the Polymarket trading advisor.

Combines features into actionable trading signals.
This is for analytics and research only - NOT financial advice.
"""

from __future__ import annotations

import logging
from typing import Optional

from polymarket_client.models import (
    TradingSignal,
    SignalDirection,
    PricePoint,
    OrderBookEntry,
)
from .features import MarketFeatures, build_market_features

logger = logging.getLogger(__name__)

# Thresholds (tunable)
MOMENTUM_THRESHOLD = 0.005
MEAN_REVERSION_THRESHOLD = 0.05
VOLATILITY_HIGH = 0.03
SPREAD_MAX = 0.10
IMBALANCE_THRESHOLD = 0.3
VOLUME_MIN = 1000.0


def generate_signal(features: MarketFeatures) -> TradingSignal:
    """
    Generate a trading signal from computed features.

    Strategy combines:
    1. Momentum (EMA crossover direction)
    2. Mean reversion (deviation from rolling mean)
    3. Orderbook imbalance (bid/ask pressure)
    4. Liquidity filter (skip illiquid markets)

    NOT FINANCIAL ADVICE.
    """
    reasons: list[str] = []
    score = 0.0  # positive = long bias, negative = short bias
    confidence = 0.0

    # Skip illiquid or very tight markets
    if features.volume_24h < VOLUME_MIN:
        return TradingSignal(
            condition_id=features.condition_id,
            signal=SignalDirection.NEUTRAL,
            confidence=0.0,
            reason="Insufficient volume (24h < $1000)",
            features=_features_dict(features),
        )

    if features.bid_ask_spread > SPREAD_MAX and features.bid_ask_spread > 0:
        reasons.append(f"Wide spread ({features.bid_ask_spread:.2%})")
        score -= 0.1

    # 1. EMA crossover
    if features.ema_12 > 0 and features.ema_26 > 0:
        ema_diff = features.ema_12 - features.ema_26
        if ema_diff > MOMENTUM_THRESHOLD:
            score += 0.3
            reasons.append("EMA12 > EMA26 (bullish crossover)")
        elif ema_diff < -MOMENTUM_THRESHOLD:
            score -= 0.3
            reasons.append("EMA12 < EMA26 (bearish crossover)")

    # 2. Momentum
    if abs(features.momentum) > MOMENTUM_THRESHOLD:
        if features.momentum > 0:
            score += 0.2
            reasons.append(f"Positive momentum ({features.momentum:.4f})")
        else:
            score -= 0.2
            reasons.append(f"Negative momentum ({features.momentum:.4f})")

    # 3. Mean reversion
    if abs(features.mean_reversion_score) > MEAN_REVERSION_THRESHOLD:
        # Mean reversion acts opposite to trend
        reversion_signal = -features.mean_reversion_score * 0.2
        score += reversion_signal
        direction = "overbought" if features.mean_reversion_score > 0 else "oversold"
        reasons.append(
            f"Mean reversion: {direction} "
            f"({features.mean_reversion_score:.2%} from mean)"
        )

    # 4. Orderbook imbalance
    if abs(features.order_imbalance) > IMBALANCE_THRESHOLD:
        imbalance_score = features.order_imbalance * 0.25
        score += imbalance_score
        side = "bid" if features.order_imbalance > 0 else "ask"
        reasons.append(
            f"Orderbook imbalance: {side} heavy "
            f"({features.order_imbalance:.2f})"
        )

    # 5. Volatility adjustment
    if features.rolling_volatility > VOLATILITY_HIGH:
        reasons.append(f"High volatility ({features.rolling_volatility:.4f})")
        # Reduce confidence in high-vol environments
        confidence_penalty = 0.15
    else:
        confidence_penalty = 0.0

    # 6. Price near extremes (close to 0 or 1)
    if features.current_price > 0.90 or features.current_price < 0.10:
        reasons.append(
            f"Price near extreme ({features.current_price:.2f}) - "
            "limited upside/downside"
        )
        score *= 0.5

    # Determine signal
    if score > 0.15:
        signal = SignalDirection.LONG
    elif score < -0.15:
        signal = SignalDirection.SHORT
    else:
        signal = SignalDirection.NEUTRAL

    # Confidence: normalized score
    raw_confidence = min(abs(score), 1.0)
    confidence = max(0.0, raw_confidence - confidence_penalty)

    reason = "; ".join(reasons) if reasons else "No strong signals"

    return TradingSignal(
        condition_id=features.condition_id,
        signal=signal,
        confidence=round(confidence, 3),
        reason=reason,
        features=_features_dict(features),
    )


def generate_signal_from_data(
    condition_id: str,
    prices: list[PricePoint],
    orders: list[OrderBookEntry] | None = None,
    volume: float = 0.0,
    volume_24h: float = 0.0,
    open_interest: float = 0.0,
    liquidity: float = 0.0,
) -> TradingSignal:
    """
    Convenience function: build features and generate signal in one call.
    """
    features = build_market_features(
        condition_id=condition_id,
        prices=prices,
        orders=orders,
        volume=volume,
        volume_24h=volume_24h,
        open_interest=open_interest,
        liquidity=liquidity,
    )
    return generate_signal(features)


def _features_dict(features: MarketFeatures) -> dict:
    """Extract key features as a dict for the signal output."""
    return {
        "current_price": round(features.current_price, 4),
        "price_change_24h": round(features.price_change_24h, 4),
        "momentum": round(features.momentum, 6),
        "rolling_volatility": round(features.rolling_volatility, 6),
        "mean_reversion": round(features.mean_reversion_score, 4),
        "ema_12": round(features.ema_12, 4),
        "ema_26": round(features.ema_26, 4),
        "spread": round(features.bid_ask_spread, 4),
        "oi_imbalance": round(features.order_imbalance, 4),
        "volume_24h": round(features.volume_24h, 2),
        "open_interest": round(features.open_interest, 2),
        "liquidity": round(features.liquidity, 2),
    }
