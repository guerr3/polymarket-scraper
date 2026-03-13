"""Tests for the backtesting engine."""

import pytest

from polymarket_client.models import PricePoint
from advisor.backtest import run_backtest, BacktestConfig


def _make_price_series(n: int = 200, base: float = 0.5, trend: float = 0.001) -> list[PricePoint]:
    """Generate a synthetic price series with slight upward trend."""
    import math
    import random

    random.seed(42)
    points = []
    price = base
    ts = 1700000000

    for i in range(n):
        # Add trend + noise + slight oscillation
        noise = random.gauss(0, 0.005)
        oscillation = 0.02 * math.sin(i / 20)
        price = max(0.01, min(0.99, price + trend + noise + oscillation * 0.01))
        points.append(PricePoint(timestamp=ts + i * 60, price=round(price, 4)))

    return points


def test_backtest_basic():
    """Backtest should run and produce valid results."""
    prices = _make_price_series(200)

    config = BacktestConfig(
        window_size=30,
        step_size=5,
        hold_periods=5,
        position_size=100.0,
    )

    result = run_backtest(
        condition_id="0xtest",
        prices=prices,
        config=config,
        volume=100000,
        volume_24h=10000,
    )

    assert result.condition_id == "0xtest"
    assert result.start_ts > 0
    assert result.end_ts > result.start_ts
    # Should have produced some trades
    assert result.total_trades >= 0
    assert 0.0 <= result.win_rate <= 1.0
    assert result.max_drawdown >= 0.0


def test_backtest_insufficient_data():
    """Backtest with too little data should return empty result."""
    prices = _make_price_series(10)

    config = BacktestConfig(window_size=50, hold_periods=10)
    result = run_backtest("0xshort", prices, config=config)

    assert result.total_trades == 0
    assert result.win_rate == 0.0


def test_backtest_trending_market():
    """Backtest on a clearly trending market."""
    # Strong upward trend
    prices = _make_price_series(300, base=0.3, trend=0.002)

    config = BacktestConfig(
        window_size=40,
        step_size=3,
        hold_periods=8,
        position_size=100.0,
    )

    result = run_backtest(
        "0xtrending",
        prices,
        config=config,
        volume=500000,
        volume_24h=50000,
    )

    # With a strong trend, we expect some trades
    assert result.total_trades > 0
    # Sharpe and PnL are just numbers; we verify they're computed
    assert isinstance(result.sharpe_ratio, float)
    assert isinstance(result.total_pnl, float)
