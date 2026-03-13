"""
Unified intelligence pipeline for prediction market analysis.

Combines all 4 intelligence sources into a single weighted signal:
1. Sentiment analysis (news + social media NLP)
2. Cross-market arbitrage detection
3. Historical calibration models
4. Event-driven triggers

Replaces the old advisor TradingSignal with an information-advantage
approach that combines multiple signal sources with configurable weights.

NOT FINANCIAL ADVICE — for research and education only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from polymarket_client.models import Market, PricePoint

from intelligence.sentiment import (
    KeywordSentimentAnalyzer,
    SentimentResult,
    AggregatedSentiment,
    aggregate_sentiments,
)
from intelligence.news_feeds import NewsFeedAggregator, NewsItem
from intelligence.arbitrage import ArbitrageDetector, ArbitrageOpportunity
from intelligence.calibration import CalibrationModel, CalibrationReport, CalibrationSignal
from intelligence.event_triggers import EventTriggerDetector, TriggerSummary

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Types
# --------------------------------------------------------------------------- #

class SignalStrength(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    LEAN_BUY = "lean_buy"
    NEUTRAL = "neutral"
    LEAN_SELL = "lean_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class ComponentScore:
    """Score from a single intelligence component."""
    name: str
    score: float = 0.0        # -1 (bearish) to +1 (bullish)
    confidence: float = 0.0   # 0 to 1
    weight: float = 0.0       # configured weight
    detail: str = ""


@dataclass
class IntelligenceReport:
    """Full intelligence pipeline output for a market."""
    market_question: str = ""
    market_slug: str = ""
    condition_id: str = ""

    # Composite signal
    composite_score: float = 0.0     # -1 to +1
    signal: SignalStrength = SignalStrength.NEUTRAL
    confidence: float = 0.0

    # Component scores
    components: list[ComponentScore] = field(default_factory=list)

    # Raw data from each source
    sentiment: Optional[AggregatedSentiment] = None
    triggers: Optional[TriggerSummary] = None
    calibration_signal: Optional[CalibrationSignal] = None
    arbitrage_opportunities: list[ArbitrageOpportunity] = field(default_factory=list)

    # Metadata
    news_items_analyzed: int = 0
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    disclaimer: str = "NOT FINANCIAL ADVICE — for research and education only."


# --------------------------------------------------------------------------- #
#  Pipeline
# --------------------------------------------------------------------------- #

# Default component weights (sum to 1.0)
DEFAULT_WEIGHTS = {
    "sentiment": 0.30,
    "triggers": 0.30,
    "calibration": 0.20,
    "arbitrage": 0.20,
}


class IntelligencePipeline:
    """
    Unified intelligence pipeline combining all signal sources.

    Usage:
        async with IntelligencePipeline() as pipeline:
            report = await pipeline.analyze(
                market=market,
                all_markets=markets,
                resolved_markets=resolved,
            )
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        sentiment_analyzer: KeywordSentimentAnalyzer | None = None,
        news_aggregator: NewsFeedAggregator | None = None,
        arbitrage_detector: ArbitrageDetector | None = None,
        calibration_model: CalibrationModel | None = None,
        trigger_detector: EventTriggerDetector | None = None,
    ):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.sentiment = sentiment_analyzer or KeywordSentimentAnalyzer()
        self.news = news_aggregator
        self.arbitrage = arbitrage_detector or ArbitrageDetector()
        self.calibration = calibration_model or CalibrationModel()
        self.triggers = trigger_detector or EventTriggerDetector()

        self._owns_news = news_aggregator is None

    async def __aenter__(self) -> "IntelligencePipeline":
        if self._owns_news:
            self.news = NewsFeedAggregator()
            await self.news.__aenter__()
        return self

    async def __aexit__(self, *args) -> None:
        if self._owns_news and self.news:
            await self.news.__aexit__(*args)

    async def analyze(
        self,
        market: Market,
        all_markets: list[Market] | None = None,
        resolved_markets: list[Market] | None = None,
    ) -> IntelligenceReport:
        """
        Run the full intelligence pipeline on a market.

        Args:
            market: The target market to analyze.
            all_markets: All active markets (for arbitrage detection).
            resolved_markets: Historical resolved markets (for calibration).
        """
        components: list[ComponentScore] = []
        report = IntelligenceReport(
            market_question=market.question,
            market_slug=market.slug,
            condition_id=market.condition_id,
        )

        # 1. Sentiment + News
        sentiment_score = await self._run_sentiment(market, report)
        components.append(sentiment_score)

        # 2. Event triggers (using the news items already fetched)
        trigger_score = self._run_triggers(market, report)
        components.append(trigger_score)

        # 3. Calibration
        cal_score = self._run_calibration(market, resolved_markets, report)
        components.append(cal_score)

        # 4. Arbitrage
        arb_score = self._run_arbitrage(market, all_markets, report)
        components.append(arb_score)

        # Compute composite score
        report.components = components
        report.composite_score, report.confidence = self._compute_composite(components)
        report.signal = self._score_to_signal(report.composite_score, report.confidence)

        logger.info(
            "Intelligence report for '%s': %s (score=%.3f, confidence=%.3f)",
            market.question[:40],
            report.signal.value,
            report.composite_score,
            report.confidence,
        )

        return report

    # ------------------------------------------------------------------ #
    #  Component runners
    # ------------------------------------------------------------------ #

    async def _run_sentiment(
        self,
        market: Market,
        report: IntelligenceReport,
    ) -> ComponentScore:
        """Run sentiment analysis on news for this market."""
        try:
            # Fetch news
            news_items: list[NewsItem] = []
            if self.news:
                news_items = await self.news.get_market_news(
                    market.question,
                    max_age_hours=48,
                )

            report.news_items_analyzed = len(news_items)

            if not news_items:
                return ComponentScore(
                    name="sentiment",
                    score=0.0,
                    confidence=0.0,
                    weight=self.weights.get("sentiment", 0),
                    detail="No news items found",
                )

            # Analyze sentiment for each item
            results: list[SentimentResult] = []
            for item in news_items[:30]:  # cap at 30 items
                result = await self.sentiment.analyze(
                    text=item.full_text,
                    context=market.question,
                    source=item.source,
                )
                results.append(result)

            # Store results for trigger detection
            self._last_sentiment_results = results
            self._last_news_items = news_items

            # Aggregate
            agg = aggregate_sentiments(
                results,
                market_condition_id=market.condition_id,
                market_question=market.question,
            )
            report.sentiment = agg

            return ComponentScore(
                name="sentiment",
                score=agg.overall_score,
                confidence=agg.confidence,
                weight=self.weights.get("sentiment", 0),
                detail=f"{agg.overall_label.value} ({agg.sample_count} sources, "
                       f"{agg.bullish_count}B/{agg.bearish_count}b/{agg.neutral_count}N)",
            )

        except Exception as exc:
            logger.warning("Sentiment analysis failed: %s", exc)
            return ComponentScore(
                name="sentiment", weight=self.weights.get("sentiment", 0),
                detail=f"Error: {exc}",
            )

    def _run_triggers(
        self,
        market: Market,
        report: IntelligenceReport,
    ) -> ComponentScore:
        """Run event trigger detection."""
        try:
            news_items = getattr(self, "_last_news_items", [])
            sentiment_results = getattr(self, "_last_sentiment_results", [])

            summary = self.triggers.scan(
                news_items=news_items,
                sentiment_results=sentiment_results,
                market_question=market.question,
            )
            report.triggers = summary

            if summary.total_triggers == 0:
                return ComponentScore(
                    name="triggers",
                    score=0.0,
                    confidence=0.0,
                    weight=self.weights.get("triggers", 0),
                    detail="No triggers detected",
                )

            # Score based on trigger direction and severity
            severity_weights = {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.2}
            weighted_sum = 0.0
            weight_sum = 0.0

            for trigger in summary.triggers:
                w = severity_weights.get(trigger.severity.value, 0.2)
                if trigger.direction == "bullish":
                    weighted_sum += w * trigger.confidence
                elif trigger.direction == "bearish":
                    weighted_sum -= w * trigger.confidence
                weight_sum += w

            score = weighted_sum / weight_sum if weight_sum > 0 else 0.0
            score = max(-1.0, min(1.0, score))
            confidence = min(1.0, summary.total_triggers / 5.0)

            return ComponentScore(
                name="triggers",
                score=score,
                confidence=confidence,
                weight=self.weights.get("triggers", 0),
                detail=f"{summary.total_triggers} triggers "
                       f"({summary.critical_count}C/{summary.high_count}H/"
                       f"{summary.medium_count}M/{summary.low_count}L) → {summary.net_direction}",
            )

        except Exception as exc:
            logger.warning("Trigger detection failed: %s", exc)
            return ComponentScore(
                name="triggers", weight=self.weights.get("triggers", 0),
                detail=f"Error: {exc}",
            )

    def _run_calibration(
        self,
        market: Market,
        resolved_markets: list[Market] | None,
        report: IntelligenceReport,
    ) -> ComponentScore:
        """Run calibration analysis."""
        try:
            if not resolved_markets:
                return ComponentScore(
                    name="calibration",
                    score=0.0,
                    confidence=0.0,
                    weight=self.weights.get("calibration", 0),
                    detail="No resolved markets for calibration",
                )

            cal_report = self.calibration.build_calibration(resolved_markets)
            signals = self.calibration.find_mispriced([market], cal_report)

            if not signals:
                return ComponentScore(
                    name="calibration",
                    score=0.0,
                    confidence=0.1,
                    weight=self.weights.get("calibration", 0),
                    detail=f"No mispricing detected (Brier={cal_report.brier_score:.3f})",
                )

            sig = signals[0]
            report.calibration_signal = sig

            # Edge > 0 means market underprices Yes (bullish)
            score = max(-1.0, min(1.0, sig.edge * 5.0))  # scale edge to [-1, 1]

            return ComponentScore(
                name="calibration",
                score=score,
                confidence=sig.confidence,
                weight=self.weights.get("calibration", 0),
                detail=f"Edge: {sig.edge:+.1%} (current={sig.current_price:.0%}, "
                       f"calibrated={sig.calibrated_probability:.0%}, n={sig.bucket_sample_size})",
            )

        except Exception as exc:
            logger.warning("Calibration analysis failed: %s", exc)
            return ComponentScore(
                name="calibration", weight=self.weights.get("calibration", 0),
                detail=f"Error: {exc}",
            )

    def _run_arbitrage(
        self,
        market: Market,
        all_markets: list[Market] | None,
        report: IntelligenceReport,
    ) -> ComponentScore:
        """Run arbitrage detection."""
        try:
            if not all_markets:
                return ComponentScore(
                    name="arbitrage",
                    score=0.0,
                    confidence=0.0,
                    weight=self.weights.get("arbitrage", 0),
                    detail="No markets provided for arbitrage scan",
                )

            opportunities = self.arbitrage.analyze(all_markets)

            # Filter to opportunities involving our target market
            relevant = [
                opp for opp in opportunities
                if opp.market_a.condition_id == market.condition_id
                or opp.market_b.condition_id == market.condition_id
            ]

            report.arbitrage_opportunities = relevant

            if not relevant:
                return ComponentScore(
                    name="arbitrage",
                    score=0.0,
                    confidence=0.1,
                    weight=self.weights.get("arbitrage", 0),
                    detail=f"No arbitrage involving this market ({len(opportunities)} total found)",
                )

            # If our market is overpriced in an arb, that's bearish
            opp = relevant[0]
            if opp.market_a.condition_id == market.condition_id:
                # Our market is market_a; if it's the "expensive" side, bearish
                score = -opp.divergence * 3.0
            else:
                score = opp.divergence * 3.0
            score = max(-1.0, min(1.0, score))

            return ComponentScore(
                name="arbitrage",
                score=score,
                confidence=opp.confidence,
                weight=self.weights.get("arbitrage", 0),
                detail=f"{len(relevant)} arb opportunities: {opp.reasoning[:80]}",
            )

        except Exception as exc:
            logger.warning("Arbitrage detection failed: %s", exc)
            return ComponentScore(
                name="arbitrage", weight=self.weights.get("arbitrage", 0),
                detail=f"Error: {exc}",
            )

    # ------------------------------------------------------------------ #
    #  Scoring
    # ------------------------------------------------------------------ #

    def _compute_composite(
        self,
        components: list[ComponentScore],
    ) -> tuple[float, float]:
        """Compute weighted composite score and confidence."""
        weighted_sum = 0.0
        weight_sum = 0.0
        confidence_sum = 0.0

        for c in components:
            effective_weight = c.weight * c.confidence
            weighted_sum += c.score * effective_weight
            weight_sum += effective_weight
            confidence_sum += c.confidence * c.weight

        if weight_sum > 0:
            composite = weighted_sum / weight_sum
        else:
            composite = 0.0

        total_weight = sum(c.weight for c in components)
        confidence = confidence_sum / total_weight if total_weight > 0 else 0.0

        return (
            round(max(-1.0, min(1.0, composite)), 4),
            round(min(1.0, confidence), 4),
        )

    @staticmethod
    def _score_to_signal(score: float, confidence: float) -> SignalStrength:
        """Convert composite score + confidence to a signal strength."""
        effective = score * confidence

        if effective > 0.4:
            return SignalStrength.STRONG_BUY
        elif effective > 0.2:
            return SignalStrength.BUY
        elif effective > 0.05:
            return SignalStrength.LEAN_BUY
        elif effective < -0.4:
            return SignalStrength.STRONG_SELL
        elif effective < -0.2:
            return SignalStrength.SELL
        elif effective < -0.05:
            return SignalStrength.LEAN_SELL
        return SignalStrength.NEUTRAL
