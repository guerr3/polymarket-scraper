"""Tests for the unified intelligence pipeline."""

import pytest

from polymarket_client.models import Market, Outcome, MarketStatus
from intelligence.signals import (
    SignalStrength,
    ComponentScore,
    IntelligenceReport,
    IntelligencePipeline,
    DEFAULT_WEIGHTS,
)


# --------------------------------------------------------------------------- #
#  SignalStrength
# --------------------------------------------------------------------------- #

class TestSignalStrength:
    def test_values(self):
        assert SignalStrength.STRONG_BUY.value == "strong_buy"
        assert SignalStrength.NEUTRAL.value == "neutral"
        assert SignalStrength.STRONG_SELL.value == "strong_sell"


# --------------------------------------------------------------------------- #
#  ComponentScore
# --------------------------------------------------------------------------- #

class TestComponentScore:
    def test_defaults(self):
        c = ComponentScore(name="test")
        assert c.score == 0.0
        assert c.confidence == 0.0
        assert c.weight == 0.0


# --------------------------------------------------------------------------- #
#  IntelligenceReport
# --------------------------------------------------------------------------- #

class TestIntelligenceReport:
    def test_defaults(self):
        r = IntelligenceReport()
        assert r.composite_score == 0.0
        assert r.signal == SignalStrength.NEUTRAL
        assert r.components == []
        assert r.disclaimer != ""


# --------------------------------------------------------------------------- #
#  DEFAULT_WEIGHTS
# --------------------------------------------------------------------------- #

class TestDefaultWeights:
    def test_sum_to_one(self):
        assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)

    def test_all_positive(self):
        for w in DEFAULT_WEIGHTS.values():
            assert w > 0


# --------------------------------------------------------------------------- #
#  _compute_composite
# --------------------------------------------------------------------------- #

class TestComputeComposite:
    def setup_method(self):
        self.pipeline = IntelligencePipeline()

    def test_all_zero(self):
        components = [
            ComponentScore(name="a", score=0.0, confidence=0.0, weight=0.3),
            ComponentScore(name="b", score=0.0, confidence=0.0, weight=0.7),
        ]
        score, confidence = self.pipeline._compute_composite(components)
        assert score == 0.0
        assert confidence == 0.0

    def test_single_bullish_component(self):
        components = [
            ComponentScore(name="sentiment", score=0.8, confidence=0.9, weight=1.0),
        ]
        score, confidence = self.pipeline._compute_composite(components)
        assert score > 0
        assert confidence > 0

    def test_mixed_components(self):
        components = [
            ComponentScore(name="a", score=0.5, confidence=0.8, weight=0.5),
            ComponentScore(name="b", score=-0.5, confidence=0.8, weight=0.5),
        ]
        score, confidence = self.pipeline._compute_composite(components)
        assert -0.1 <= score <= 0.1  # roughly cancel out

    def test_confidence_weighted(self):
        """High-confidence component should dominate low-confidence one."""
        components = [
            ComponentScore(name="strong", score=0.8, confidence=0.9, weight=0.5),
            ComponentScore(name="weak", score=-0.8, confidence=0.1, weight=0.5),
        ]
        score, confidence = self.pipeline._compute_composite(components)
        assert score > 0  # strong component dominates

    def test_score_clamped(self):
        components = [
            ComponentScore(name="x", score=1.0, confidence=1.0, weight=1.0),
        ]
        score, confidence = self.pipeline._compute_composite(components)
        assert -1.0 <= score <= 1.0
        assert 0.0 <= confidence <= 1.0


# --------------------------------------------------------------------------- #
#  _score_to_signal
# --------------------------------------------------------------------------- #

class TestScoreToSignal:
    def test_strong_buy(self):
        assert IntelligencePipeline._score_to_signal(0.8, 0.8) == SignalStrength.STRONG_BUY

    def test_buy(self):
        assert IntelligencePipeline._score_to_signal(0.4, 0.7) == SignalStrength.BUY

    def test_lean_buy(self):
        assert IntelligencePipeline._score_to_signal(0.2, 0.5) == SignalStrength.LEAN_BUY

    def test_neutral(self):
        assert IntelligencePipeline._score_to_signal(0.0, 0.5) == SignalStrength.NEUTRAL

    def test_lean_sell(self):
        assert IntelligencePipeline._score_to_signal(-0.2, 0.5) == SignalStrength.LEAN_SELL

    def test_sell(self):
        assert IntelligencePipeline._score_to_signal(-0.4, 0.7) == SignalStrength.SELL

    def test_strong_sell(self):
        assert IntelligencePipeline._score_to_signal(-0.8, 0.8) == SignalStrength.STRONG_SELL

    def test_low_confidence_neutral(self):
        """Even a high score with very low confidence should be neutral."""
        assert IntelligencePipeline._score_to_signal(0.9, 0.01) == SignalStrength.NEUTRAL

    def test_boundary_values(self):
        # effective = 0.05 is the lean_buy boundary
        assert IntelligencePipeline._score_to_signal(0.1, 0.5) == SignalStrength.NEUTRAL
        assert IntelligencePipeline._score_to_signal(0.5, 0.2) == SignalStrength.LEAN_BUY


# --------------------------------------------------------------------------- #
#  Pipeline component runners (unit-level, no I/O)
# --------------------------------------------------------------------------- #

class TestPipelineComponentRunners:
    def setup_method(self):
        self.pipeline = IntelligencePipeline()

    def test_run_calibration_no_resolved(self):
        market = Market(
            condition_id="0x1", question="Test?", status=MarketStatus.ACTIVE,
            outcomes=[Outcome(name="Yes", price=0.5)],
        )
        report = IntelligenceReport()
        score = self.pipeline._run_calibration(market, None, report)
        assert score.name == "calibration"
        assert score.confidence == 0.0
        assert "No resolved" in score.detail

    def test_run_arbitrage_no_markets(self):
        market = Market(
            condition_id="0x1", question="Test?", status=MarketStatus.ACTIVE,
            outcomes=[Outcome(name="Yes", price=0.5)],
        )
        report = IntelligenceReport()
        score = self.pipeline._run_arbitrage(market, None, report)
        assert score.name == "arbitrage"
        assert score.confidence == 0.0
        assert "No markets" in score.detail

    def test_run_triggers_no_news(self):
        market = Market(
            condition_id="0x1", question="Test?", status=MarketStatus.ACTIVE,
            outcomes=[Outcome(name="Yes", price=0.5)],
        )
        report = IntelligenceReport()
        score = self.pipeline._run_triggers(market, report)
        assert score.name == "triggers"
        assert score.confidence == 0.0
        assert "No triggers" in score.detail
