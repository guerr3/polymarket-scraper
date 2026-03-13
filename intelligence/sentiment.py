"""
Sentiment analysis engine for prediction market intelligence.

Uses a keyword/lexicon-based approach with market-context awareness.
No external API keys required — runs locally with pure Python NLP.

For production use, can be extended with transformer models (HuggingFace)
or external APIs (OpenAI, Anthropic) via the SentimentProvider interface.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Core types
# --------------------------------------------------------------------------- #

class SentimentLabel(str, Enum):
    """Directional sentiment relative to a market's 'Yes' outcome."""
    VERY_BULLISH = "very_bullish"   # strong Yes signal
    BULLISH = "bullish"             # mild Yes signal
    NEUTRAL = "neutral"
    BEARISH = "bearish"             # mild No signal
    VERY_BEARISH = "very_bearish"   # strong No signal


@dataclass
class SentimentResult:
    """Sentiment analysis output for a single text."""
    text: str
    score: float = 0.0             # -1.0 (bearish) to +1.0 (bullish)
    label: SentimentLabel = SentimentLabel.NEUTRAL
    confidence: float = 0.0        # 0.0 to 1.0
    keywords_found: list[str] = field(default_factory=list)
    source: str = ""               # e.g., "reuters", "twitter:@user"
    timestamp: Optional[datetime] = None
    market_relevance: float = 0.0  # 0.0 to 1.0 — how relevant to the market


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment across multiple sources for a market."""
    market_condition_id: str = ""
    market_question: str = ""
    overall_score: float = 0.0      # weighted mean of individual scores
    overall_label: SentimentLabel = SentimentLabel.NEUTRAL
    confidence: float = 0.0
    sample_count: int = 0
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    top_bullish_signals: list[str] = field(default_factory=list)
    top_bearish_signals: list[str] = field(default_factory=list)
    sources_breakdown: dict[str, float] = field(default_factory=dict)
    last_updated: Optional[datetime] = None


# --------------------------------------------------------------------------- #
#  Provider interface (for pluggable backends)
# --------------------------------------------------------------------------- #

class SentimentProvider(Protocol):
    """Interface for pluggable sentiment backends."""

    async def analyze(self, text: str, context: str = "") -> SentimentResult:
        """Analyze sentiment of text in the context of a market question."""
        ...


# --------------------------------------------------------------------------- #
#  Keyword / lexicon based sentiment (zero-dependency)
# --------------------------------------------------------------------------- #

# Directional keywords that push toward "Yes" (positive = event likely)
BULLISH_LEXICON: dict[str, float] = {
    # Escalation / action words
    "confirmed": 0.7, "imminent": 0.8, "launched": 0.9, "deployed": 0.7,
    "invaded": 0.9, "attacked": 0.9, "bombed": 0.85, "strikes": 0.7,
    "offensive": 0.6, "escalation": 0.7, "mobilized": 0.65, "ordered": 0.6,
    "approved": 0.7, "signed": 0.6, "passed": 0.6, "enacted": 0.65,
    "announced": 0.5, "revealed": 0.5, "breaking": 0.6, "urgent": 0.5,
    "inevitable": 0.75, "certain": 0.7, "guaranteed": 0.8,
    "surge": 0.6, "spike": 0.5, "soaring": 0.6, "record": 0.4,
    "winning": 0.6, "leading": 0.5, "ahead": 0.4, "favored": 0.5,
    "likely": 0.5, "probable": 0.5, "expected": 0.45,
    # Crypto / financial bullish
    "rally": 0.6, "bullish": 0.7, "moon": 0.5, "breakout": 0.6,
    "adoption": 0.5, "approval": 0.65, "etf": 0.5,
    # Political
    "elected": 0.7, "victory": 0.7, "won": 0.8, "sworn in": 0.8,
    "inaugurated": 0.8, "ratified": 0.7,
}

# Keywords that push toward "No" (negative = event unlikely)
BEARISH_LEXICON: dict[str, float] = {
    # De-escalation / negation
    "denied": 0.7, "rejected": 0.7, "unlikely": 0.6, "impossible": 0.8,
    "withdrawn": 0.7, "retreated": 0.65, "ceasefire": 0.7, "peace": 0.6,
    "negotiations": 0.5, "diplomacy": 0.5, "talks": 0.4, "agreement": 0.5,
    "deal": 0.5, "resolved": 0.6, "postponed": 0.6, "delayed": 0.55,
    "canceled": 0.7, "cancelled": 0.7, "suspended": 0.6, "halted": 0.6,
    "failed": 0.6, "collapsed": 0.65, "stalled": 0.5,
    "debunked": 0.7, "false": 0.6, "hoax": 0.7, "misinformation": 0.6,
    "losing": 0.6, "trailing": 0.5, "behind": 0.4, "underdog": 0.4,
    "doubtful": 0.5, "uncertain": 0.3, "improbable": 0.6,
    # Crypto / financial bearish
    "crash": 0.7, "bearish": 0.7, "dump": 0.6, "sell-off": 0.65,
    "banned": 0.7, "crackdown": 0.6, "regulation": 0.4,
    # Political
    "defeated": 0.7, "lost": 0.7, "impeached": 0.6, "resigned": 0.6,
    "vetoed": 0.7,
}

# Source credibility weights
SOURCE_WEIGHTS: dict[str, float] = {
    "reuters": 1.0, "ap": 1.0, "bbc": 0.95, "nytimes": 0.95,
    "washingtonpost": 0.9, "wsj": 0.95, "bloomberg": 0.95,
    "cnn": 0.8, "foxnews": 0.75, "guardian": 0.85, "ft": 0.95,
    "politico": 0.85, "axios": 0.8, "aljazeera": 0.85,
    "twitter": 0.5, "reddit": 0.3, "telegram": 0.3,
    "official": 1.0, "government": 0.9, "whitehouse": 0.95,
    "unknown": 0.4,
}


class KeywordSentimentAnalyzer:
    """
    Market-context-aware sentiment analyzer using keyword lexicons.

    Analyzes text against bullish/bearish keyword dictionaries,
    weighing matches by their strength and the source's credibility.
    """

    def __init__(
        self,
        bullish_lexicon: dict[str, float] | None = None,
        bearish_lexicon: dict[str, float] | None = None,
        source_weights: dict[str, float] | None = None,
    ):
        self.bullish = bullish_lexicon or BULLISH_LEXICON
        self.bearish = bearish_lexicon or BEARISH_LEXICON
        self.source_weights = source_weights or SOURCE_WEIGHTS

    async def analyze(
        self,
        text: str,
        context: str = "",
        source: str = "unknown",
    ) -> SentimentResult:
        """
        Analyze sentiment of text relative to a market question.

        Args:
            text: The news headline, tweet, or article text.
            context: The market question for relevance scoring.
            source: Source identifier for credibility weighting.
        """
        text_lower = text.lower()
        context_lower = context.lower()

        # Find keyword matches
        bull_matches: list[tuple[str, float]] = []
        bear_matches: list[tuple[str, float]] = []

        for keyword, weight in self.bullish.items():
            if keyword in text_lower:
                bull_matches.append((keyword, weight))

        for keyword, weight in self.bearish.items():
            if keyword in text_lower:
                bear_matches.append((keyword, weight))

        # Check for negation patterns
        negation_patterns = [
            r"\bnot\s+", r"\bno\s+", r"\bnever\s+", r"\bwon'?t\s+",
            r"\bdidn'?t\s+", r"\bdoes\s?n'?t\s+", r"\bwill\s?n'?t\s+",
            r"\bcannot\s+", r"\bcan'?t\s+", r"\bdeny\s+", r"\brunlikely\s+to\s+",
        ]
        has_negation = any(re.search(p, text_lower) for p in negation_patterns)

        # Calculate raw scores
        bull_score = sum(w for _, w in bull_matches)
        bear_score = sum(w for _, w in bear_matches)

        # If negation detected, swap directional bias
        if has_negation:
            bull_score, bear_score = bear_score * 0.7, bull_score * 0.7

        # Net score: positive = bullish, negative = bearish
        total = bull_score + bear_score
        if total > 0:
            raw_score = (bull_score - bear_score) / total
        else:
            raw_score = 0.0

        # Source credibility weighting
        source_weight = self._get_source_weight(source)
        weighted_score = raw_score * source_weight

        # Market relevance
        relevance = self._compute_relevance(text_lower, context_lower)

        # Final score, clamped to [-1, 1]
        final_score = max(-1.0, min(1.0, weighted_score * (0.5 + 0.5 * relevance)))

        # Confidence based on keyword density and source quality
        keyword_count = len(bull_matches) + len(bear_matches)
        confidence = min(1.0, (keyword_count / 5.0) * source_weight * (0.5 + 0.5 * relevance))

        # Label
        label = self._score_to_label(final_score)

        keywords = [k for k, _ in bull_matches] + [k for k, _ in bear_matches]

        return SentimentResult(
            text=text[:500],
            score=round(final_score, 4),
            label=label,
            confidence=round(confidence, 4),
            keywords_found=keywords,
            source=source,
            timestamp=datetime.now(timezone.utc),
            market_relevance=round(relevance, 4),
        )

    def _get_source_weight(self, source: str) -> float:
        """Look up source credibility weight."""
        source_lower = source.lower()
        for key, weight in self.source_weights.items():
            if key in source_lower:
                return weight
        return self.source_weights.get("unknown", 0.4)

    def _compute_relevance(self, text: str, context: str) -> float:
        """
        Compute how relevant a text is to the market question.

        Uses word overlap between text and market question.
        """
        if not context:
            return 0.5  # unknown relevance

        # Extract meaningful words (skip stop words)
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "will", "be",
            "by", "to", "in", "on", "at", "of", "for", "and", "or",
            "this", "that", "it", "its", "with", "from", "as", "has",
            "have", "had", "do", "does", "did", "if", "but", "not",
            "what", "when", "where", "who", "how", "which", "than",
        }

        text_words = set(re.findall(r'\b[a-z]{3,}\b', text)) - stop_words
        context_words = set(re.findall(r'\b[a-z]{3,}\b', context)) - stop_words

        if not context_words:
            return 0.5

        overlap = text_words & context_words
        relevance = len(overlap) / len(context_words)

        return min(1.0, relevance * 1.5)  # slight boost

    @staticmethod
    def _score_to_label(score: float) -> SentimentLabel:
        if score >= 0.5:
            return SentimentLabel.VERY_BULLISH
        elif score >= 0.15:
            return SentimentLabel.BULLISH
        elif score <= -0.5:
            return SentimentLabel.VERY_BEARISH
        elif score <= -0.15:
            return SentimentLabel.BEARISH
        return SentimentLabel.NEUTRAL


# --------------------------------------------------------------------------- #
#  Aggregation
# --------------------------------------------------------------------------- #

def aggregate_sentiments(
    results: list[SentimentResult],
    market_condition_id: str = "",
    market_question: str = "",
) -> AggregatedSentiment:
    """
    Aggregate multiple sentiment results into a single market view.

    Weights by: source credibility, market relevance, and recency.
    """
    if not results:
        return AggregatedSentiment(
            market_condition_id=market_condition_id,
            market_question=market_question,
        )

    weighted_scores: list[float] = []
    weights: list[float] = []
    bullish = 0
    bearish = 0
    neutral = 0
    top_bull: list[str] = []
    top_bear: list[str] = []
    source_scores: dict[str, list[float]] = {}

    for r in results:
        # Weight = confidence * relevance
        w = max(0.01, r.confidence * (0.5 + 0.5 * r.market_relevance))
        weighted_scores.append(r.score * w)
        weights.append(w)

        if r.score > 0.1:
            bullish += 1
            if r.confidence > 0.3:
                top_bull.append(f"[{r.source}] {r.text[:80]}")
        elif r.score < -0.1:
            bearish += 1
            if r.confidence > 0.3:
                top_bear.append(f"[{r.source}] {r.text[:80]}")
        else:
            neutral += 1

        src = r.source or "unknown"
        source_scores.setdefault(src, []).append(r.score)

    total_weight = sum(weights)
    overall = sum(weighted_scores) / total_weight if total_weight > 0 else 0.0
    overall = max(-1.0, min(1.0, overall))

    # Confidence: higher when sources agree
    score_std = _std([r.score for r in results])
    agreement_factor = max(0.0, 1.0 - score_std * 2)
    confidence = min(1.0, agreement_factor * (len(results) / 10.0))

    # Source breakdown
    breakdown = {
        src: round(sum(scores) / len(scores), 3)
        for src, scores in source_scores.items()
    }

    label = KeywordSentimentAnalyzer._score_to_label(overall)

    return AggregatedSentiment(
        market_condition_id=market_condition_id,
        market_question=market_question,
        overall_score=round(overall, 4),
        overall_label=label,
        confidence=round(confidence, 4),
        sample_count=len(results),
        bullish_count=bullish,
        bearish_count=bearish,
        neutral_count=neutral,
        top_bullish_signals=top_bull[:5],
        top_bearish_signals=top_bear[:5],
        sources_breakdown=breakdown,
        last_updated=datetime.now(timezone.utc),
    )


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)
