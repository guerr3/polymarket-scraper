"""
Sentiment analysis engine for prediction market intelligence.

Provides two backends:

  1. FinBERTSentimentAnalyzer  — HuggingFace ProsusAI/finbert transformer model.
     Produces high-accuracy finance-domain sentiment. Requires `transformers`
     and `torch` (or `onnxruntime` for lighter CPU inference).
     Loaded lazily on first use; subsequent calls reuse the in-process pipeline.

  2. KeywordSentimentAnalyzer  — Zero-dependency lexicon/keyword fallback.
     Used automatically when transformers is not installed, or explicitly when
     you pass `use_finbert=False` to `build_sentiment_analyzer()`.

Factory helper:
    analyzer = build_sentiment_analyzer()   # auto-selects FinBERT if available
    analyzer = build_sentiment_analyzer(use_finbert=False)  # force keyword

NOT FINANCIAL ADVICE — for research and education only.
"""

from __future__ import annotations

import asyncio
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
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


@dataclass
class SentimentResult:
    """Sentiment analysis output for a single text."""
    text: str
    score: float = 0.0             # -1.0 (bearish) to +1.0 (bullish)
    label: SentimentLabel = SentimentLabel.NEUTRAL
    confidence: float = 0.0        # 0.0 to 1.0
    keywords_found: list[str] = field(default_factory=list)
    source: str = ""
    timestamp: Optional[datetime] = None
    market_relevance: float = 0.0  # 0.0 to 1.0
    analyzer_backend: str = "keyword"  # "finbert" | "keyword"


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment across multiple sources for a market."""
    market_condition_id: str = ""
    market_question: str = ""
    overall_score: float = 0.0
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
#  Provider interface
# --------------------------------------------------------------------------- #

class SentimentProvider(Protocol):
    """Interface for pluggable sentiment backends."""

    async def analyze(self, text: str, context: str = "") -> SentimentResult:
        ...


# --------------------------------------------------------------------------- #
#  Source credibility weights  (shared by both analyzers)
# --------------------------------------------------------------------------- #

SOURCE_WEIGHTS: dict[str, float] = {
    "reuters": 1.0, "ap": 1.0, "bbc": 0.95, "nytimes": 0.95,
    "washingtonpost": 0.9, "wsj": 0.95, "bloomberg": 0.95,
    "cnn": 0.8, "foxnews": 0.75, "guardian": 0.85, "ft": 0.95,
    "politico": 0.85, "axios": 0.8, "aljazeera": 0.85,
    "twitter": 0.5, "reddit": 0.3, "telegram": 0.3,
    "official": 1.0, "government": 0.9, "whitehouse": 0.95,
    "unknown": 0.4,
}


def _get_source_weight(source: str) -> float:
    source_lower = source.lower()
    for key, weight in SOURCE_WEIGHTS.items():
        if key in source_lower:
            return weight
    return SOURCE_WEIGHTS["unknown"]


def _compute_relevance(text: str, context: str) -> float:
    """Word-overlap relevance between article text and market question."""
    if not context:
        return 0.5
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "will", "be",
        "by", "to", "in", "on", "at", "of", "for", "and", "or",
        "this", "that", "it", "its", "with", "from", "as", "has",
        "have", "had", "do", "does", "did", "if", "but", "not",
        "what", "when", "where", "who", "how", "which", "than",
    }
    text_words = set(re.findall(r'\b[a-z]{3,}\b', text.lower())) - stop_words
    ctx_words = set(re.findall(r'\b[a-z]{3,}\b', context.lower())) - stop_words
    if not ctx_words:
        return 0.5
    overlap = text_words & ctx_words
    return min(1.0, (len(overlap) / len(ctx_words)) * 1.5)


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
#  FinBERT transformer analyzer
# --------------------------------------------------------------------------- #

# FinBERT label mapping → directional score
# ProsusAI/finbert outputs: "positive", "negative", "neutral"
_FINBERT_LABEL_SCORE: dict[str, float] = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}

_FINBERT_MODEL = "ProsusAI/finbert"
_MAX_TOKENS = 512  # FinBERT max sequence length


class FinBERTSentimentAnalyzer:
    """
    Finance-domain sentiment analyzer powered by ProsusAI/finbert.

    The HuggingFace pipeline is loaded once on first use and cached
    for the lifetime of the process (lazy singleton pattern).

    Falls back to keyword scoring for relevance computation (same as
    KeywordSentimentAnalyzer) so that market_relevance is always populated.

    Args:
        model_name: HuggingFace model identifier. Defaults to ProsusAI/finbert.
        device: "cpu" | "cuda" | "mps" | int (device index). None = auto-detect.
        batch_size: Number of texts to process in one forward pass.
    """

    _pipeline = None  # class-level lazy singleton
    _pipeline_lock = asyncio.Lock()

    def __init__(
        self,
        model_name: str = _FINBERT_MODEL,
        device: str | int | None = None,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._keyword_fallback = KeywordSentimentAnalyzer()  # for relevance

    # ------------------------------------------------------------------ #
    #  Pipeline loading
    # ------------------------------------------------------------------ #

    @classmethod
    async def _get_pipeline(cls, model_name: str, device: str | int | None):
        """Lazily load the HuggingFace pipeline (thread-safe)."""
        if cls._pipeline is not None:
            return cls._pipeline

        async with cls._pipeline_lock:
            # Double-checked locking
            if cls._pipeline is not None:
                return cls._pipeline

            try:
                from transformers import pipeline as hf_pipeline
                import torch

                # Auto-select device
                if device is None:
                    if torch.cuda.is_available():
                        device = 0  # first CUDA GPU
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = "cpu"

                logger.info(
                    "Loading FinBERT model '%s' on device '%s' (first-time load, may take ~30s)...",
                    model_name, device,
                )

                # Run blocking model load in a thread pool to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                pipe = await loop.run_in_executor(
                    None,
                    lambda: hf_pipeline(
                        "text-classification",
                        model=model_name,
                        device=device,
                        top_k=None,           # return all 3 label scores
                        truncation=True,
                        max_length=_MAX_TOKENS,
                    ),
                )
                cls._pipeline = pipe
                logger.info("FinBERT pipeline loaded successfully.")
                return cls._pipeline

            except ImportError:
                logger.error(
                    "transformers/torch not installed. "
                    "Run: pip install transformers torch\n"
                    "Falling back to keyword sentiment."
                )
                raise

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    async def analyze(
        self,
        text: str,
        context: str = "",
        source: str = "unknown",
    ) -> SentimentResult:
        """
        Analyze sentiment of a single text using FinBERT.

        Args:
            text: News headline or article body.
            context: Market question (used for relevance scoring).
            source: Source identifier for credibility weighting.
        """
        pipe = await self._get_pipeline(self.model_name, self.device)

        # Truncate to model max length (characters; tokenizer will sub-truncate)
        truncated = text[:2000]

        # Run inference in executor to avoid blocking the async event loop
        loop = asyncio.get_event_loop()
        raw_output = await loop.run_in_executor(
            None, lambda: pipe(truncated)
        )

        # raw_output shape: [[{"label": str, "score": float}, ...]]
        label_scores: dict[str, float] = {
            item["label"].lower(): item["score"]
            for item in raw_output[0]
        }

        # Directional score: positive - negative (both in [0,1])
        pos = label_scores.get("positive", 0.0)
        neg = label_scores.get("negative", 0.0)
        neu = label_scores.get("neutral", 0.0)

        raw_score = pos - neg  # [-1.0, +1.0]

        # Confidence = max probability mass on a non-neutral label,
        # discounted when neutral dominates
        confidence = max(pos, neg)  # how strongly it leans either way
        if neu > 0.6:               # strongly neutral → low confidence
            confidence *= (1.0 - neu)

        # Source credibility & market relevance
        source_weight = _get_source_weight(source)
        relevance = _compute_relevance(text, context)

        # Apply source credibility to score (keeps sign, reduces magnitude)
        final_score = max(-1.0, min(1.0, raw_score * source_weight))

        # Boost confidence when article is highly relevant to the market
        adjusted_confidence = min(1.0, confidence * (0.6 + 0.4 * relevance) * source_weight)

        label = _score_to_label(final_score)

        return SentimentResult(
            text=text[:500],
            score=round(final_score, 4),
            label=label,
            confidence=round(adjusted_confidence, 4),
            keywords_found=[],   # not applicable for transformer model
            source=source,
            timestamp=datetime.now(timezone.utc),
            market_relevance=round(relevance, 4),
            analyzer_backend="finbert",
        )

    async def analyze_batch(
        self,
        texts: list[str],
        context: str = "",
        sources: list[str] | None = None,
    ) -> list[SentimentResult]:
        """
        Batch analyze multiple texts in a single forward pass.

        More efficient than calling `analyze` in a loop when processing
        many news items (e.g., 30 articles per market scan).

        Args:
            texts: List of text strings to analyze.
            context: Market question for all items.
            sources: Source identifier per text (same length as texts).
        """
        if not texts:
            return []

        sources = sources or ["unknown"] * len(texts)
        pipe = await self._get_pipeline(self.model_name, self.device)

        truncated = [t[:2000] for t in texts]

        loop = asyncio.get_event_loop()
        raw_outputs = await loop.run_in_executor(
            None,
            lambda: pipe(truncated, batch_size=self.batch_size),
        )

        results: list[SentimentResult] = []
        for i, output in enumerate(raw_outputs):
            label_scores = {item["label"].lower(): item["score"] for item in output}
            pos = label_scores.get("positive", 0.0)
            neg = label_scores.get("negative", 0.0)
            neu = label_scores.get("neutral", 0.0)

            raw_score = pos - neg
            confidence = max(pos, neg)
            if neu > 0.6:
                confidence *= (1.0 - neu)

            source = sources[i] if i < len(sources) else "unknown"
            source_weight = _get_source_weight(source)
            relevance = _compute_relevance(texts[i], context)

            final_score = max(-1.0, min(1.0, raw_score * source_weight))
            adjusted_confidence = min(1.0, confidence * (0.6 + 0.4 * relevance) * source_weight)
            label = _score_to_label(final_score)

            results.append(SentimentResult(
                text=texts[i][:500],
                score=round(final_score, 4),
                label=label,
                confidence=round(adjusted_confidence, 4),
                keywords_found=[],
                source=source,
                timestamp=datetime.now(timezone.utc),
                market_relevance=round(relevance, 4),
                analyzer_backend="finbert",
            ))

        return results


# --------------------------------------------------------------------------- #
#  Keyword / lexicon based sentiment (zero-dependency fallback)
# --------------------------------------------------------------------------- #

BULLISH_LEXICON: dict[str, float] = {
    "confirmed": 0.7, "imminent": 0.8, "launched": 0.9, "deployed": 0.7,
    "invaded": 0.9, "attacked": 0.9, "bombed": 0.85, "strikes": 0.7,
    "offensive": 0.6, "escalation": 0.7, "mobilized": 0.65, "ordered": 0.6,
    "approved": 0.7, "signed": 0.6, "passed": 0.6, "enacted": 0.65,
    "announced": 0.5, "revealed": 0.5, "breaking": 0.6, "urgent": 0.5,
    "inevitable": 0.75, "certain": 0.7, "guaranteed": 0.8,
    "surge": 0.6, "spike": 0.5, "soaring": 0.6, "record": 0.4,
    "winning": 0.6, "leading": 0.5, "ahead": 0.4, "favored": 0.5,
    "likely": 0.5, "probable": 0.5, "expected": 0.45,
    "rally": 0.6, "bullish": 0.7, "moon": 0.5, "breakout": 0.6,
    "adoption": 0.5, "approval": 0.65, "etf": 0.5,
    "elected": 0.7, "victory": 0.7, "won": 0.8, "sworn in": 0.8,
    "inaugurated": 0.8, "ratified": 0.7,
}

BEARISH_LEXICON: dict[str, float] = {
    "denied": 0.7, "rejected": 0.7, "unlikely": 0.6, "impossible": 0.8,
    "withdrawn": 0.7, "retreated": 0.65, "ceasefire": 0.7, "peace": 0.6,
    "negotiations": 0.5, "diplomacy": 0.5, "talks": 0.4, "agreement": 0.5,
    "deal": 0.5, "resolved": 0.6, "postponed": 0.6, "delayed": 0.55,
    "canceled": 0.7, "cancelled": 0.7, "suspended": 0.6, "halted": 0.6,
    "failed": 0.6, "collapsed": 0.65, "stalled": 0.5,
    "debunked": 0.7, "false": 0.6, "hoax": 0.7, "misinformation": 0.6,
    "losing": 0.6, "trailing": 0.5, "behind": 0.4, "underdog": 0.4,
    "doubtful": 0.5, "uncertain": 0.3, "improbable": 0.6,
    "crash": 0.7, "bearish": 0.7, "dump": 0.6, "sell-off": 0.65,
    "banned": 0.7, "crackdown": 0.6, "regulation": 0.4,
    "defeated": 0.7, "lost": 0.7, "impeached": 0.6, "resigned": 0.6,
    "vetoed": 0.7,
}


class KeywordSentimentAnalyzer:
    """
    Market-context-aware sentiment analyzer using keyword lexicons.

    Zero-dependency fallback. Used when transformers/torch are not installed
    or explicitly requested via build_sentiment_analyzer(use_finbert=False).
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
        text_lower = text.lower()

        bull_matches: list[tuple[str, float]] = []
        bear_matches: list[tuple[str, float]] = []

        for keyword, weight in self.bullish.items():
            if keyword in text_lower:
                bull_matches.append((keyword, weight))

        for keyword, weight in self.bearish.items():
            if keyword in text_lower:
                bear_matches.append((keyword, weight))

        negation_patterns = [
            r"\bnot\s+", r"\bno\s+", r"\bnever\s+", r"\bwon'?t\s+",
            r"\bdidn'?t\s+", r"\bdoes\s?n'?t\s+", r"\bwill\s?n'?t\s+",
            r"\bcannot\s+", r"\bcan'?t\s+", r"\bdeny\s+",
        ]
        has_negation = any(re.search(p, text_lower) for p in negation_patterns)

        bull_score = sum(w for _, w in bull_matches)
        bear_score = sum(w for _, w in bear_matches)

        if has_negation:
            bull_score, bear_score = bear_score * 0.7, bull_score * 0.7

        total = bull_score + bear_score
        raw_score = (bull_score - bear_score) / total if total > 0 else 0.0

        source_weight = _get_source_weight(source)
        relevance = _compute_relevance(text, context)
        final_score = max(-1.0, min(1.0, raw_score * source_weight * (0.5 + 0.5 * relevance)))

        keyword_count = len(bull_matches) + len(bear_matches)
        confidence = min(1.0, (keyword_count / 5.0) * source_weight * (0.5 + 0.5 * relevance))

        label = _score_to_label(final_score)
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
            analyzer_backend="keyword",
        )

    @staticmethod
    def _score_to_label(score: float) -> SentimentLabel:
        return _score_to_label(score)


# --------------------------------------------------------------------------- #
#  Factory helper
# --------------------------------------------------------------------------- #

def build_sentiment_analyzer(
    use_finbert: bool = True,
    model_name: str = _FINBERT_MODEL,
    device: str | int | None = None,
    batch_size: int = 8,
) -> FinBERTSentimentAnalyzer | KeywordSentimentAnalyzer:
    """
    Factory that returns a FinBERTSentimentAnalyzer when transformers/torch
    are available, or falls back to KeywordSentimentAnalyzer.

    Args:
        use_finbert: Set False to force the keyword analyzer.
        model_name: HuggingFace model ID (default: ProsusAI/finbert).
        device: "cpu", "cuda", "mps", or device index. None = auto.
        batch_size: Batch size for FinBERT inference.

    Returns:
        FinBERTSentimentAnalyzer or KeywordSentimentAnalyzer
    """
    if not use_finbert:
        logger.info("Sentiment backend: keyword (forced)")
        return KeywordSentimentAnalyzer()

    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        logger.info("Sentiment backend: FinBERT (%s)", model_name)
        return FinBERTSentimentAnalyzer(model_name=model_name, device=device, batch_size=batch_size)
    except ImportError:
        logger.warning(
            "transformers/torch not installed — falling back to keyword sentiment.\n"
            "To enable FinBERT: pip install transformers torch"
        )
        return KeywordSentimentAnalyzer()


# --------------------------------------------------------------------------- #
#  Aggregation  (unchanged, works with both backends)
# --------------------------------------------------------------------------- #

def aggregate_sentiments(
    results: list[SentimentResult],
    market_condition_id: str = "",
    market_question: str = "",
) -> AggregatedSentiment:
    """
    Aggregate multiple SentimentResult objects into a single market view.
    Weights by source credibility, market relevance, and result confidence.
    """
    if not results:
        return AggregatedSentiment(
            market_condition_id=market_condition_id,
            market_question=market_question,
        )

    weighted_scores: list[float] = []
    weights: list[float] = []
    bullish = bearish = neutral = 0
    top_bull: list[str] = []
    top_bear: list[str] = []
    source_scores: dict[str, list[float]] = {}

    for r in results:
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

    score_std = _std([r.score for r in results])
    agreement_factor = max(0.0, 1.0 - score_std * 2)
    confidence = min(1.0, agreement_factor * (len(results) / 10.0))

    breakdown = {
        src: round(sum(scores) / len(scores), 3)
        for src, scores in source_scores.items()
    }

    label = _score_to_label(overall)

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
