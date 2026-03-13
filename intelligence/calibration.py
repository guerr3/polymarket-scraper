"""
Calibration model for prediction market accuracy analysis.

Answers the question: "When a market is priced at X%, how often does it
actually resolve Yes?"

A well-calibrated market at 30% should resolve Yes roughly 30% of the time.
Deviations reveal systematic biases that can be exploited.

Uses historical resolved market data from the GAMMA API.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from polymarket_client.models import Market, MarketStatus

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Types
# --------------------------------------------------------------------------- #

@dataclass
class CalibrationBucket:
    """A single bucket in the calibration curve."""
    bin_start: float          # e.g., 0.10
    bin_end: float            # e.g., 0.20
    bin_midpoint: float       # e.g., 0.15
    market_count: int = 0     # how many resolved markets fell in this bucket
    resolved_yes_count: int = 0
    actual_rate: float = 0.0  # what fraction actually resolved Yes
    expected_rate: float = 0.0  # midpoint (what a perfect market would show)
    deviation: float = 0.0    # actual - expected
    overconfident: bool = False  # market thinks Yes more than reality

    @property
    def is_exploitable(self) -> bool:
        """Deviation > 5% and enough samples to be meaningful."""
        return abs(self.deviation) > 0.05 and self.market_count >= 5


@dataclass
class CalibrationReport:
    """Full calibration analysis for a set of resolved markets."""
    total_markets: int = 0
    resolved_yes: int = 0
    resolved_no: int = 0
    buckets: list[CalibrationBucket] = field(default_factory=list)
    brier_score: float = 0.0        # lower is better (0 = perfect)
    log_loss: float = 0.0           # lower is better
    mean_abs_deviation: float = 0.0
    bias_direction: str = ""        # "overconfident_yes", "overconfident_no", "well_calibrated"
    exploitable_ranges: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CalibrationSignal:
    """
    A trading signal derived from calibration analysis.

    If markets at 15% historically resolve Yes 25% of the time,
    then current markets at 15% may be underpriced.
    """
    condition_id: str = ""
    market_question: str = ""
    current_price: float = 0.0
    calibrated_probability: float = 0.0
    edge: float = 0.0       # calibrated - current (positive = underpriced Yes)
    bucket_sample_size: int = 0
    confidence: float = 0.0
    recommendation: str = ""


# --------------------------------------------------------------------------- #
#  Calibration engine
# --------------------------------------------------------------------------- #

class CalibrationModel:
    """
    Builds calibration curves from historical resolved markets.

    Usage:
        model = CalibrationModel()
        report = model.build_calibration(resolved_markets)
        signals = model.find_mispriced(active_markets, report)
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def build_calibration(
        self,
        resolved_markets: list[Market],
    ) -> CalibrationReport:
        """
        Build a calibration curve from resolved markets.

        For each probability bucket (0-10%, 10-20%, etc.),
        compute what fraction actually resolved to "Yes".
        """
        # Filter to resolved markets with outcomes
        resolved = [
            m for m in resolved_markets
            if m.status == MarketStatus.RESOLVED and m.outcomes
        ]

        if not resolved:
            logger.warning("No resolved markets to calibrate")
            return CalibrationReport()

        # Determine resolution: did this market resolve Yes?
        records: list[tuple[float, bool]] = []
        for m in resolved:
            yes_price = m.outcomes[0].price if m.outcomes else 0.0
            did_resolve_yes = self._did_resolve_yes(m)
            if did_resolve_yes is not None:
                records.append((yes_price, did_resolve_yes))

        if not records:
            return CalibrationReport(total_markets=len(resolved))

        # Build buckets
        bin_width = 1.0 / self.n_bins
        buckets: list[CalibrationBucket] = []

        for i in range(self.n_bins):
            bin_start = i * bin_width
            bin_end = (i + 1) * bin_width
            bin_mid = (bin_start + bin_end) / 2

            # Markets in this bucket
            in_bucket = [(price, yes) for price, yes in records if bin_start <= price < bin_end]

            if not in_bucket:
                buckets.append(CalibrationBucket(
                    bin_start=bin_start,
                    bin_end=bin_end,
                    bin_midpoint=bin_mid,
                    expected_rate=bin_mid,
                ))
                continue

            yes_count = sum(1 for _, yes in in_bucket if yes)
            actual_rate = yes_count / len(in_bucket)
            deviation = actual_rate - bin_mid

            buckets.append(CalibrationBucket(
                bin_start=round(bin_start, 2),
                bin_end=round(bin_end, 2),
                bin_midpoint=round(bin_mid, 2),
                market_count=len(in_bucket),
                resolved_yes_count=yes_count,
                actual_rate=round(actual_rate, 4),
                expected_rate=round(bin_mid, 4),
                deviation=round(deviation, 4),
                overconfident=deviation < 0,
            ))

        # Aggregate metrics
        brier = self._brier_score(records)
        log_loss_val = self._log_loss(records)
        mean_abs_dev = sum(abs(b.deviation) for b in buckets if b.market_count > 0)
        n_nonempty = sum(1 for b in buckets if b.market_count > 0)
        mean_abs_dev = mean_abs_dev / n_nonempty if n_nonempty > 0 else 0

        # Bias direction
        overconf_yes = sum(1 for b in buckets if b.overconfident and b.market_count > 0)
        overconf_no = sum(1 for b in buckets if not b.overconfident and b.market_count > 0 and b.deviation != 0)
        if overconf_yes > overconf_no + 2:
            bias = "overconfident_yes"
        elif overconf_no > overconf_yes + 2:
            bias = "overconfident_no"
        else:
            bias = "well_calibrated"

        # Exploitable ranges
        exploitable = [
            f"{b.bin_start:.0%}-{b.bin_end:.0%}: market says {b.bin_midpoint:.0%}, "
            f"reality is {b.actual_rate:.0%} ({'+' if b.deviation > 0 else ''}{b.deviation:.0%})"
            for b in buckets if b.is_exploitable
        ]

        total_yes = sum(1 for _, yes in records if yes)
        total_no = len(records) - total_yes

        return CalibrationReport(
            total_markets=len(records),
            resolved_yes=total_yes,
            resolved_no=total_no,
            buckets=buckets,
            brier_score=round(brier, 4),
            log_loss=round(log_loss_val, 4),
            mean_abs_deviation=round(mean_abs_dev, 4),
            bias_direction=bias,
            exploitable_ranges=exploitable,
        )

    def find_mispriced(
        self,
        active_markets: list[Market],
        calibration: CalibrationReport,
    ) -> list[CalibrationSignal]:
        """
        Find currently active markets that may be mispriced
        based on historical calibration data.
        """
        signals: list[CalibrationSignal] = []

        for market in active_markets:
            if not market.outcomes:
                continue

            current_price = market.outcomes[0].price

            # Find the calibration bucket for this price
            bucket = self._find_bucket(current_price, calibration.buckets)
            if not bucket or bucket.market_count < 3:
                continue

            calibrated_prob = bucket.actual_rate
            edge = calibrated_prob - current_price

            # Only signal if edge is meaningful
            if abs(edge) < 0.05:
                continue

            confidence = min(1.0, bucket.market_count / 20.0) * (1.0 - abs(bucket.deviation) * 0.5)

            if edge > 0:
                recommendation = (
                    f"Potentially UNDERPRICED Yes at {current_price:.0%}. "
                    f"Historical markets at this level resolve Yes {calibrated_prob:.0%} of the time."
                )
            else:
                recommendation = (
                    f"Potentially OVERPRICED Yes at {current_price:.0%}. "
                    f"Historical markets at this level resolve Yes only {calibrated_prob:.0%} of the time."
                )

            signals.append(CalibrationSignal(
                condition_id=market.condition_id,
                market_question=market.question,
                current_price=round(current_price, 4),
                calibrated_probability=round(calibrated_prob, 4),
                edge=round(edge, 4),
                bucket_sample_size=bucket.market_count,
                confidence=round(confidence, 4),
                recommendation=recommendation,
            ))

        # Sort by absolute edge
        signals.sort(key=lambda s: abs(s.edge), reverse=True)

        logger.info(
            "Found %d potentially mispriced markets from %d active",
            len(signals),
            len(active_markets),
        )
        return signals

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _did_resolve_yes(market: Market) -> Optional[bool]:
        """Determine if a market resolved to Yes."""
        rv = market.resolved_value
        if rv is None:
            # Try to infer from outcome prices (1.0 = resolved yes)
            if market.outcomes:
                if market.outcomes[0].price >= 0.99:
                    return True
                elif market.outcomes[0].price <= 0.01:
                    return False
            return None

        rv_lower = str(rv).lower().strip()
        if rv_lower in ("yes", "true", "1", "1.0"):
            return True
        if rv_lower in ("no", "false", "0", "0.0"):
            return False

        # If resolved value matches first outcome name
        if market.outcomes and rv_lower == market.outcomes[0].name.lower():
            return True

        return None

    @staticmethod
    def _find_bucket(
        price: float,
        buckets: list[CalibrationBucket],
    ) -> Optional[CalibrationBucket]:
        """Find the calibration bucket for a given price."""
        for b in buckets:
            if b.bin_start <= price < b.bin_end:
                return b
        return None

    @staticmethod
    def _brier_score(records: list[tuple[float, bool]]) -> float:
        """Brier score: mean squared error of probability vs outcome."""
        if not records:
            return 0.0
        total = sum((price - (1.0 if yes else 0.0)) ** 2 for price, yes in records)
        return total / len(records)

    @staticmethod
    def _log_loss(records: list[tuple[float, bool]], eps: float = 1e-7) -> float:
        """Log loss (cross-entropy) of probability predictions."""
        if not records:
            return 0.0
        total = 0.0
        for price, yes in records:
            p = max(eps, min(1 - eps, price))
            if yes:
                total -= math.log(p)
            else:
                total -= math.log(1 - p)
        return total / len(records)
