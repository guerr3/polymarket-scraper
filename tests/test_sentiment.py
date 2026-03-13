"""Tests for the keyword sentiment analysis engine."""

import asyncio
import pytest

from intelligence.sentiment import (
    KeywordSentimentAnalyzer,
    SentimentLabel,
    SentimentResult,
    AggregatedSentiment,
    aggregate_sentiments,
    _std,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# --------------------------------------------------------------------------- #
#  KeywordSentimentAnalyzer
# --------------------------------------------------------------------------- #

class TestKeywordSentimentAnalyzer:
    def setup_method(self):
        self.analyzer = KeywordSentimentAnalyzer()

    def test_bullish_text(self):
        result = _run(self.analyzer.analyze(
            "Iran confirmed military invasion launched against targets",
            context="Will Iran invade?",
        ))
        assert result.score > 0
        assert result.label in (SentimentLabel.BULLISH, SentimentLabel.VERY_BULLISH)
        assert len(result.keywords_found) > 0

    def test_bearish_text(self):
        result = _run(self.analyzer.analyze(
            "war denied rejected unlikely failed collapse crashed plummeted recession downturn",
            context="Will there be a war?",
            source="reuters",
        ))
        assert result.score < 0
        assert result.label in (SentimentLabel.BEARISH, SentimentLabel.VERY_BEARISH)

    def test_neutral_text(self):
        result = _run(self.analyzer.analyze(
            "The weather today is sunny with clear skies",
            context="Will BTC reach $100k?",
        ))
        assert result.label == SentimentLabel.NEUTRAL

    def test_negation_flips_direction(self):
        result_no_negation = _run(self.analyzer.analyze(
            "The invasion was confirmed",
            context="Will there be an invasion?",
        ))
        result_negation = _run(self.analyzer.analyze(
            "The invasion was not confirmed",
            context="Will there be an invasion?",
        ))
        # Negation should push score in the opposite direction
        assert result_negation.score < result_no_negation.score

    def test_source_weight_reuters(self):
        result_reuters = _run(self.analyzer.analyze(
            "Military attack confirmed", context="Will there be war?",
            source="reuters",
        ))
        result_unknown = _run(self.analyzer.analyze(
            "Military attack confirmed", context="Will there be war?",
            source="random_blog",
        ))
        assert result_reuters.confidence >= result_unknown.confidence

    def test_empty_context_returns_half_relevance(self):
        result = _run(self.analyzer.analyze("Some breaking news text"))
        assert result.market_relevance == 0.5

    def test_score_clamped_to_range(self):
        result = _run(self.analyzer.analyze(
            "confirmed imminent launched invaded attacked approved signed passed enacted breaking won elected victory surge rally bullish moon breakout",
            context="Will something happen?",
        ))
        assert -1.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0


# --------------------------------------------------------------------------- #
#  Score to label
# --------------------------------------------------------------------------- #

class TestScoreToLabel:
    def test_very_bullish(self):
        assert KeywordSentimentAnalyzer._score_to_label(0.6) == SentimentLabel.VERY_BULLISH

    def test_bullish(self):
        assert KeywordSentimentAnalyzer._score_to_label(0.3) == SentimentLabel.BULLISH

    def test_neutral(self):
        assert KeywordSentimentAnalyzer._score_to_label(0.0) == SentimentLabel.NEUTRAL
        assert KeywordSentimentAnalyzer._score_to_label(0.1) == SentimentLabel.NEUTRAL

    def test_bearish(self):
        assert KeywordSentimentAnalyzer._score_to_label(-0.3) == SentimentLabel.BEARISH

    def test_very_bearish(self):
        assert KeywordSentimentAnalyzer._score_to_label(-0.6) == SentimentLabel.VERY_BEARISH


# --------------------------------------------------------------------------- #
#  Aggregation
# --------------------------------------------------------------------------- #

class TestAggregateSentiments:
    def test_empty_list(self):
        agg = aggregate_sentiments([], market_condition_id="0x1", market_question="Q?")
        assert agg.sample_count == 0
        assert agg.overall_score == 0.0
        assert agg.market_condition_id == "0x1"

    def test_single_bullish_result(self):
        results = [SentimentResult(
            text="confirmed invasion", score=0.6, label=SentimentLabel.BULLISH,
            confidence=0.8, source="reuters", market_relevance=0.9,
        )]
        agg = aggregate_sentiments(results, market_question="Will there be war?")
        assert agg.sample_count == 1
        assert agg.overall_score > 0
        assert agg.bullish_count == 1

    def test_mixed_results(self):
        results = [
            SentimentResult(text="confirmed", score=0.5, label=SentimentLabel.BULLISH,
                            confidence=0.7, source="reuters", market_relevance=0.8),
            SentimentResult(text="denied", score=-0.5, label=SentimentLabel.BEARISH,
                            confidence=0.7, source="ap", market_relevance=0.8),
            SentimentResult(text="weather", score=0.0, label=SentimentLabel.NEUTRAL,
                            confidence=0.2, source="unknown", market_relevance=0.1),
        ]
        agg = aggregate_sentiments(results)
        assert agg.sample_count == 3
        assert agg.bullish_count == 1
        assert agg.bearish_count == 1
        assert agg.neutral_count == 1

    def test_source_breakdown(self):
        results = [
            SentimentResult(text="a", score=0.3, source="reuters", confidence=0.5, market_relevance=0.5),
            SentimentResult(text="b", score=0.7, source="reuters", confidence=0.5, market_relevance=0.5),
            SentimentResult(text="c", score=-0.2, source="bbc", confidence=0.5, market_relevance=0.5),
        ]
        agg = aggregate_sentiments(results)
        assert "reuters" in agg.sources_breakdown
        assert "bbc" in agg.sources_breakdown
        assert agg.sources_breakdown["reuters"] == pytest.approx(0.5, abs=0.01)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

class TestStd:
    def test_empty(self):
        assert _std([]) == 0.0

    def test_single(self):
        assert _std([5.0]) == 0.0

    def test_known_values(self):
        # population std of [2, 4, 4, 4, 5, 5, 7, 9] ≈ 2.0
        result = _std([2, 4, 4, 4, 5, 5, 7, 9])
        assert result == pytest.approx(2.0, abs=0.2)
