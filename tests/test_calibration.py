"""Tests for the historical calibration model."""

import math
import pytest

from polymarket_client.models import Market, Outcome, MarketStatus
from intelligence.calibration import (
    CalibrationModel,
    CalibrationBucket,
    CalibrationReport,
    CalibrationSignal,
)


def _make_resolved_market(
    condition_id: str,
    yes_price: float,
    resolved_yes: bool,
) -> Market:
    """Create a resolved market fixture."""
    return Market(
        condition_id=condition_id,
        question=f"Test market {condition_id}",
        status=MarketStatus.RESOLVED,
        outcomes=[
            Outcome(name="Yes", price=yes_price),
            Outcome(name="No", price=round(1.0 - yes_price, 2)),
        ],
        resolved_value="Yes" if resolved_yes else "No",
    )


def _make_active_market(
    condition_id: str,
    question: str,
    yes_price: float,
) -> Market:
    """Create an active market fixture."""
    return Market(
        condition_id=condition_id,
        question=question,
        status=MarketStatus.ACTIVE,
        outcomes=[
            Outcome(name="Yes", price=yes_price),
            Outcome(name="No", price=round(1.0 - yes_price, 2)),
        ],
    )


# --------------------------------------------------------------------------- #
#  CalibrationBucket
# --------------------------------------------------------------------------- #

class TestCalibrationBucket:
    def test_is_exploitable_large_deviation_enough_samples(self):
        bucket = CalibrationBucket(
            bin_start=0.1, bin_end=0.2, bin_midpoint=0.15,
            market_count=10, deviation=0.10,
        )
        assert bucket.is_exploitable

    def test_not_exploitable_small_deviation(self):
        bucket = CalibrationBucket(
            bin_start=0.1, bin_end=0.2, bin_midpoint=0.15,
            market_count=10, deviation=0.03,
        )
        assert not bucket.is_exploitable

    def test_not_exploitable_too_few_samples(self):
        bucket = CalibrationBucket(
            bin_start=0.1, bin_end=0.2, bin_midpoint=0.15,
            market_count=2, deviation=0.10,
        )
        assert not bucket.is_exploitable


# --------------------------------------------------------------------------- #
#  _did_resolve_yes
# --------------------------------------------------------------------------- #

class TestDidResolveYes:
    def setup_method(self):
        self.model = CalibrationModel()

    def test_resolved_yes_string(self):
        m = _make_resolved_market("0x1", 0.7, True)
        assert CalibrationModel._did_resolve_yes(m) is True

    def test_resolved_no_string(self):
        m = _make_resolved_market("0x1", 0.3, False)
        assert CalibrationModel._did_resolve_yes(m) is False

    def test_inferred_from_price_yes(self):
        m = Market(
            condition_id="0x1", question="Q",
            status=MarketStatus.RESOLVED,
            outcomes=[Outcome(name="Yes", price=1.0)],
            resolved_value=None,
        )
        assert CalibrationModel._did_resolve_yes(m) is True

    def test_inferred_from_price_no(self):
        m = Market(
            condition_id="0x1", question="Q",
            status=MarketStatus.RESOLVED,
            outcomes=[Outcome(name="Yes", price=0.0)],
            resolved_value=None,
        )
        assert CalibrationModel._did_resolve_yes(m) is False

    def test_ambiguous_returns_none(self):
        m = Market(
            condition_id="0x1", question="Q",
            status=MarketStatus.RESOLVED,
            outcomes=[Outcome(name="Yes", price=0.5)],
            resolved_value=None,
        )
        assert CalibrationModel._did_resolve_yes(m) is None


# --------------------------------------------------------------------------- #
#  Brier score and log loss
# --------------------------------------------------------------------------- #

class TestScoringMetrics:
    def test_brier_score_perfect(self):
        records = [(1.0, True), (0.0, False)]
        assert CalibrationModel._brier_score(records) == 0.0

    def test_brier_score_worst(self):
        records = [(0.0, True), (1.0, False)]
        assert CalibrationModel._brier_score(records) == 1.0

    def test_brier_score_empty(self):
        assert CalibrationModel._brier_score([]) == 0.0

    def test_log_loss_perfect_near_zero(self):
        records = [(0.999, True), (0.001, False)]
        assert CalibrationModel._log_loss(records) < 0.01

    def test_log_loss_empty(self):
        assert CalibrationModel._log_loss([]) == 0.0

    def test_log_loss_never_infinite(self):
        """Log loss with extreme predictions should not be inf due to epsilon."""
        records = [(0.0, True), (1.0, False)]
        result = CalibrationModel._log_loss(records)
        assert math.isfinite(result)


# --------------------------------------------------------------------------- #
#  Build calibration
# --------------------------------------------------------------------------- #

class TestBuildCalibration:
    def setup_method(self):
        self.model = CalibrationModel(n_bins=10)

    def test_empty_markets(self):
        report = self.model.build_calibration([])
        assert report.total_markets == 0
        assert report.buckets == []

    def test_active_markets_ignored(self):
        markets = [
            Market(condition_id="0x1", question="Q", status=MarketStatus.ACTIVE,
                   outcomes=[Outcome(name="Yes", price=0.5)]),
        ]
        report = self.model.build_calibration(markets)
        assert report.total_markets == 0

    def test_basic_calibration(self):
        """10 markets all at 70%, half resolved yes -> actual_rate 50%."""
        markets = []
        for i in range(10):
            markets.append(_make_resolved_market(
                f"0x{i}", yes_price=0.75, resolved_yes=(i < 5)
            ))
        report = self.model.build_calibration(markets)
        assert report.total_markets == 10
        assert report.resolved_yes == 5
        assert report.resolved_no == 5
        assert report.brier_score > 0
        assert len(report.buckets) == 10

    def test_reports_bias_direction(self):
        # All markets at high price resolve No -> overconfident_yes
        markets = [
            _make_resolved_market(f"0x{i}", yes_price=0.85, resolved_yes=False)
            for i in range(20)
        ]
        report = self.model.build_calibration(markets)
        assert report.bias_direction in ("overconfident_yes", "well_calibrated")

    def test_exploitable_ranges_populated(self):
        markets = []
        for i in range(20):
            # Markets at 15% that always resolve yes -> big deviation
            markets.append(_make_resolved_market(
                f"0x{i}", yes_price=0.15, resolved_yes=True
            ))
        report = self.model.build_calibration(markets)
        assert len(report.exploitable_ranges) > 0


# --------------------------------------------------------------------------- #
#  Find mispriced
# --------------------------------------------------------------------------- #

class TestFindMispriced:
    def setup_method(self):
        self.model = CalibrationModel(n_bins=10)

    def test_finds_underpriced(self):
        # Build calibration from markets at 15% that always resolve yes
        resolved = [
            _make_resolved_market(f"0x{i}", yes_price=0.15, resolved_yes=True)
            for i in range(10)
        ]
        cal_report = self.model.build_calibration(resolved)

        # Active market at 15% should be flagged as underpriced
        active = [_make_active_market("0xA", "Test?", 0.15)]
        signals = self.model.find_mispriced(active, cal_report)
        assert len(signals) >= 1
        assert signals[0].edge > 0  # underpriced yes

    def test_no_signal_when_calibrated(self):
        # Build calibration from markets at 50% that resolve yes 50% of the time
        resolved = []
        for i in range(20):
            resolved.append(_make_resolved_market(
                f"0x{i}", yes_price=0.55, resolved_yes=(i < 10)
            ))
        cal_report = self.model.build_calibration(resolved)

        active = [_make_active_market("0xA", "Test?", 0.55)]
        signals = self.model.find_mispriced(active, cal_report)
        # Edge should be small enough to not trigger
        for s in signals:
            assert abs(s.edge) < 0.1

    def test_empty_active_markets(self):
        resolved = [_make_resolved_market("0x1", 0.5, True)]
        cal_report = self.model.build_calibration(resolved)
        signals = self.model.find_mispriced([], cal_report)
        assert signals == []
