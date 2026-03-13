"""Tests for the cross-market arbitrage detector."""

import pytest

from polymarket_client.models import Market, Outcome, MarketStatus
from intelligence.arbitrage import (
    ArbitrageDetector,
    ArbitrageOpportunity,
    MarketRef,
    MarketCluster,
    TOPIC_PATTERNS,
    RELATIONSHIP_KEYWORDS,
)


def _make_market(
    condition_id: str,
    question: str,
    yes_price: float = 0.5,
    no_price: float | None = None,
    status: MarketStatus = MarketStatus.ACTIVE,
    end_date: str | None = None,
) -> Market:
    """Helper to create a Market fixture."""
    if no_price is None:
        no_price = round(1.0 - yes_price, 2)
    return Market(
        condition_id=condition_id,
        question=question,
        status=status,
        outcomes=[Outcome(name="Yes", price=yes_price), Outcome(name="No", price=no_price)],
        end_date=end_date,
    )


# --------------------------------------------------------------------------- #
#  ArbitrageOpportunity
# --------------------------------------------------------------------------- #

class TestArbitrageOpportunity:
    def test_severity_high(self):
        opp = ArbitrageOpportunity(type="correlation", divergence=0.25)
        assert opp.severity == "HIGH"

    def test_severity_medium(self):
        opp = ArbitrageOpportunity(type="correlation", divergence=0.15)
        assert opp.severity == "MEDIUM"

    def test_severity_low(self):
        opp = ArbitrageOpportunity(type="correlation", divergence=0.05)
        assert opp.severity == "LOW"


# --------------------------------------------------------------------------- #
#  Topic clustering
# --------------------------------------------------------------------------- #

class TestClustering:
    def setup_method(self):
        self.detector = ArbitrageDetector()

    def test_clusters_by_topic(self):
        markets = [
            _make_market("0x1", "Will Iran attack Israel?"),
            _make_market("0x2", "Will Iranian regime fall?"),
            _make_market("0x3", "Will Bitcoin reach $100k?"),
        ]
        clusters = self.detector._cluster_markets(markets)
        topics = [c.topic for c in clusters]
        assert "iran" in topics
        assert "crypto_btc" in topics

    def test_single_market_cluster(self):
        markets = [_make_market("0x1", "Will Solana reach $500?")]
        clusters = self.detector._cluster_markets(markets)
        assert len(clusters) == 1
        assert clusters[0].topic == "crypto_sol"

    def test_no_matching_topic(self):
        markets = [_make_market("0x1", "Will it rain tomorrow?")]
        clusters = self.detector._cluster_markets(markets)
        assert len(clusters) == 0


# --------------------------------------------------------------------------- #
#  Implication detection
# --------------------------------------------------------------------------- #

class TestImplicationDetection:
    def setup_method(self):
        self.detector = ArbitrageDetector()

    def test_detects_implication(self):
        result = self.detector._detect_implication(
            "Will there be an invasion of Iran?",
            "Will there be regime change in Iran?",
        )
        assert result is not None
        assert result == ("a", "b")

    def test_detects_reverse_implication(self):
        result = self.detector._detect_implication(
            "Will there be regime change in Iran?",
            "Will there be an invasion of Iran?",
        )
        assert result is not None
        assert result == ("b", "a")

    def test_no_implication(self):
        result = self.detector._detect_implication(
            "Will Bitcoin reach $100k?",
            "Will Ethereum reach $10k?",
        )
        assert result is None


# --------------------------------------------------------------------------- #
#  Exclusion detection
# --------------------------------------------------------------------------- #

class TestExclusionDetection:
    def setup_method(self):
        self.detector = ArbitrageDetector()

    def test_detects_exclusion(self):
        assert self.detector._detect_exclusion(
            "Will there be a ceasefire in Gaza?",
            "Will there be an invasion of Gaza?",
        )

    def test_no_exclusion(self):
        assert not self.detector._detect_exclusion(
            "Will Bitcoin reach $100k?",
            "Will Ethereum reach $10k?",
        )


# --------------------------------------------------------------------------- #
#  Question similarity
# --------------------------------------------------------------------------- #

class TestQuestionSimilarity:
    def test_identical_questions(self):
        sim = ArbitrageDetector._question_similarity(
            "Will BTC reach $100k?", "Will BTC reach $100k?"
        )
        assert sim == 1.0

    def test_no_overlap(self):
        sim = ArbitrageDetector._question_similarity(
            "cats dogs fish", "alpha beta gamma"
        )
        assert sim == 0.0

    def test_partial_overlap(self):
        sim = ArbitrageDetector._question_similarity(
            "Will BTC reach $100k by March?",
            "Will BTC reach $100k by December?",
        )
        assert 0.5 < sim < 1.0

    def test_empty_questions(self):
        assert ArbitrageDetector._question_similarity("", "anything") == 0.0


# --------------------------------------------------------------------------- #
#  Full arbitrage analysis
# --------------------------------------------------------------------------- #

class TestArbitrageAnalysis:
    def setup_method(self):
        self.detector = ArbitrageDetector()

    def test_implication_violation(self):
        """If A implies B, P(A) should be <= P(B). Violation detected."""
        markets = [
            _make_market("0x1", "Will there be an invasion of Iran?", yes_price=0.70),
            _make_market("0x2", "Will there be regime change in Iran?", yes_price=0.30),
        ]
        opps = self.detector.analyze(markets)
        assert len(opps) >= 1
        assert any(o.type == "correlation" for o in opps)

    def test_no_arbitrage_consistent_prices(self):
        """Consistent implication prices should produce no opportunities."""
        markets = [
            _make_market("0x1", "Will there be an invasion of Iran?", yes_price=0.30),
            _make_market("0x2", "Will there be regime change in Iran?", yes_price=0.50),
        ]
        opps = self.detector.analyze(markets)
        correlation_opps = [o for o in opps if o.type == "correlation"]
        assert len(correlation_opps) == 0

    def test_exclusion_violation(self):
        """Mutually exclusive events summing to > 1.0."""
        markets = [
            _make_market("0x1", "Will there be a ceasefire in Israel?", yes_price=0.70),
            _make_market("0x2", "Will there be an invasion of Israel?", yes_price=0.60),
        ]
        opps = self.detector.analyze(markets)
        complement_opps = [o for o in opps if o.type == "complement"]
        assert len(complement_opps) >= 1

    def test_sorted_by_divergence(self):
        markets = [
            _make_market("0x1", "Will there be an invasion of Iran?", yes_price=0.80),
            _make_market("0x2", "Will there be regime change in Iran?", yes_price=0.20),
            _make_market("0x3", "Will Iran attack Israel?", yes_price=0.50),
        ]
        opps = self.detector.analyze(markets)
        if len(opps) >= 2:
            for i in range(len(opps) - 1):
                assert opps[i].divergence >= opps[i + 1].divergence

    def test_empty_markets(self):
        opps = self.detector.analyze([])
        assert opps == []


# --------------------------------------------------------------------------- #
#  Temporal analysis
# --------------------------------------------------------------------------- #

class TestTemporalAnalysis:
    def setup_method(self):
        self.detector = ArbitrageDetector()

    def test_temporal_violation(self):
        """Earlier deadline priced higher than later deadline."""
        markets = [
            _make_market(
                "0x1", "Will X happen by March 31, 2026?",
                yes_price=0.60,
                end_date="2026-03-31T00:00:00Z",
            ),
            _make_market(
                "0x2", "Will X happen by June 30, 2026?",
                yes_price=0.30,
                end_date="2026-06-30T00:00:00Z",
            ),
        ]
        opps = self.detector._check_temporal(markets)
        assert len(opps) >= 1
        assert opps[0].type == "temporal"
