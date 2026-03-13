"""
Backtesting engine for the Polymarket trading advisor.

Simulates strategy PnL over historical price data.
Outputs: win rate, max drawdown, Sharpe-like ratio.

NOT FINANCIAL ADVICE - for research and education only.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from polymarket_client.models import (
    PricePoint,
    TradingSignal,
    BacktestResult,
    SignalDirection,
)
from .features import build_market_features
from .signals import generate_signal

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtest parameters."""

    window_size: int = 50  # lookback window for feature computation
    step_size: int = 5  # step forward N points between signals
    position_size: float = 100.0  # notional USD per trade
    hold_periods: int = 10  # how many steps to hold a position
    transaction_cost: float = 0.002  # 0.2% per trade (round-trip)


@dataclass
class TradeRecord:
    """A single backtest trade."""

    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    side: str  # "long" or "short"
    pnl: float
    pnl_pct: float


def run_backtest(
    condition_id: str,
    prices: list[PricePoint],
    config: Optional[BacktestConfig] = None,
    volume: float = 0.0,
    volume_24h: float = 0.0,
    open_interest: float = 0.0,
    liquidity: float = 0.0,
) -> BacktestResult:
    """
    Run a backtest over historical price data.

    Walk-forward approach:
    1. At each step, compute features from the lookback window
    2. Generate a signal
    3. If signal is long/short, enter a position
    4. Hold for `hold_periods` steps, then exit
    5. Record PnL

    Returns BacktestResult with summary statistics.
    """
    cfg = config or BacktestConfig()

    if len(prices) < cfg.window_size + cfg.hold_periods:
        logger.warning(
            "Insufficient price data for backtest: %d points (need %d)",
            len(prices),
            cfg.window_size + cfg.hold_periods,
        )
        return BacktestResult(
            condition_id=condition_id,
            start_ts=prices[0].timestamp if prices else 0,
            end_ts=prices[-1].timestamp if prices else 0,
        )

    # Sort prices
    sorted_prices = sorted(prices, key=lambda p: p.timestamp)

    trades: list[TradeRecord] = []
    signals: list[TradingSignal] = []
    equity_curve: list[float] = [0.0]

    idx = cfg.window_size
    while idx + cfg.hold_periods < len(sorted_prices):
        # Build features from lookback window
        window = sorted_prices[idx - cfg.window_size : idx]
        features = build_market_features(
            condition_id=condition_id,
            prices=window,
            volume=volume,
            volume_24h=volume_24h,
            open_interest=open_interest,
            liquidity=liquidity,
        )

        signal = generate_signal(features)
        signals.append(signal)

        if signal.signal != SignalDirection.NEUTRAL and signal.confidence > 0.1:
            entry_price = sorted_prices[idx].price
            exit_price = sorted_prices[idx + cfg.hold_periods].price

            if signal.signal == SignalDirection.LONG:
                raw_pnl = (exit_price - entry_price) * cfg.position_size
                side = "long"
            else:  # SHORT
                raw_pnl = (entry_price - exit_price) * cfg.position_size
                side = "short"

            # Subtract transaction costs
            cost = cfg.position_size * cfg.transaction_cost
            net_pnl = raw_pnl - cost
            pnl_pct = net_pnl / cfg.position_size if cfg.position_size > 0 else 0

            trades.append(
                TradeRecord(
                    entry_idx=idx,
                    exit_idx=idx + cfg.hold_periods,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    side=side,
                    pnl=net_pnl,
                    pnl_pct=pnl_pct,
                )
            )
            equity_curve.append(equity_curve[-1] + net_pnl)

        idx += cfg.step_size

    # Compute summary statistics
    total_trades = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    total_pnl = sum(t.pnl for t in trades)
    max_drawdown = _max_drawdown(equity_curve)

    # Sharpe-like ratio (annualized, assuming ~365 trading days)
    if total_trades >= 2:
        returns = [t.pnl_pct for t in trades]
        avg_return = sum(returns) / len(returns)
        std_return = _std(returns)
        sharpe = (avg_return / std_return * math.sqrt(365)) if std_return > 0 else 0.0
    else:
        sharpe = 0.0

    result = BacktestResult(
        condition_id=condition_id,
        start_ts=sorted_prices[0].timestamp,
        end_ts=sorted_prices[-1].timestamp,
        total_trades=total_trades,
        win_rate=round(win_rate, 4),
        total_pnl=round(total_pnl, 2),
        max_drawdown=round(max_drawdown, 2),
        sharpe_ratio=round(sharpe, 4),
        signals=signals[-10:],  # Keep last 10 signals
    )

    logger.info(
        "Backtest complete: %d trades, %.1f%% win rate, $%.2f PnL, "
        "%.2f max DD, %.2f Sharpe",
        total_trades,
        win_rate * 100,
        total_pnl,
        max_drawdown,
        sharpe,
    )
    return result


def _max_drawdown(equity_curve: list[float]) -> float:
    """Compute maximum drawdown from equity curve."""
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _std(values: list[float]) -> float:
    """Standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)
