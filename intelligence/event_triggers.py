"""
Event-driven trigger detection for prediction market intelligence.

Identifies actionable events that may move market prices:
- Breaking news detection (sudden spikes in coverage)
- Official government/institutional statements
- Key account activity on Twitter/X
- Sentiment shift detection (rapid change in aggregate sentiment)
- News clustering (multiple sources covering same story)

These triggers are designed to fire *before* the market fully prices in
the information, giving a brief window of edge.

NOT FINANCIAL ADVICE — for research and education only.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional

from intelligence.news_feeds import NewsItem
from intelligence.sentiment import SentimentResult, SentimentLabel

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Types
# --------------------------------------------------------------------------- #

class TriggerType(str, Enum):
    BREAKING_NEWS = "breaking_news"
    OFFICIAL_STATEMENT = "official_statement"
    KEY_ACCOUNT = "key_account"
    SENTIMENT_SHIFT = "sentiment_shift"
    NEWS_CLUSTER = "news_cluster"


class TriggerSeverity(str, Enum):
    CRITICAL = "critical"   # act immediately
    HIGH = "high"           # strong signal
    MEDIUM = "medium"       # worth monitoring
    LOW = "low"             # informational only


@dataclass
class EventTrigger:
    """A detected event that may move market prices."""
    type: TriggerType
    severity: TriggerSeverity
    title: str
    description: str = ""
    source: str = ""
    market_relevance: float = 0.0    # 0-1
    confidence: float = 0.0          # 0-1
    direction: str = ""              # "bullish", "bearish", "unknown"
    related_items: list[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_actionable(self) -> bool:
        return self.severity in (TriggerSeverity.CRITICAL, TriggerSeverity.HIGH) and self.confidence > 0.4


@dataclass
class TriggerSummary:
    """Summary of all triggers detected for a market."""
    market_question: str = ""
    total_triggers: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    net_direction: str = ""     # "bullish", "bearish", "mixed", "neutral"
    triggers: list[EventTrigger] = field(default_factory=list)
    scanned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# --------------------------------------------------------------------------- #
#  Key accounts (influential Twitter/X users by domain)
# --------------------------------------------------------------------------- #

KEY_ACCOUNTS: dict[str, list[str]] = {
    "politics": [
        "POTUS", "WhiteHouse", "VP", "SpeakerJohnson",
        "SenSchumer", "LeaderMcConnell", "realDonaldTrump",
    ],
    "crypto": [
        "caboringdave", "VitalikButerin", "saboringcz",
        "elonmusk", "garygensler", "SECGov",
    ],
    "geopolitics": [
        "IsraeliPM", "ZelenskyyUa", "KremlinRussia_E",
        "SecDef", "StateDept", "NATO",
    ],
    "finance": [
        "federalreserve", "FedChairPowell", "ECB",
        "business", "markets",
    ],
}

# Breaking-news indicator phrases
BREAKING_INDICATORS = [
    "breaking:", "breaking news:", "just in:", "alert:",
    "developing:", "urgent:", "flash:", "exclusive:",
    "confirmed:", "official:", "just now:",
]

# Official statement patterns
OFFICIAL_PATTERNS = [
    r"\b(?:white\s+house|pentagon|kremlin|downing\s+street)\b.*\b(?:says?|announces?|confirms?|states?)\b",
    r"\b(?:president|prime\s+minister|secretary|minister)\b.*\b(?:says?|announces?|confirms?|orders?)\b",
    r"\bofficial\s+statement\b",
    r"\b(?:doj|fbi|cia|nsa|sec|fda|cdc)\b.*\b(?:announces?|confirms?|charges?|approves?)\b",
    r"\b(?:supreme\s+court|congress|senate|house)\b.*\b(?:rules?|passes?|votes?|approves?|blocks?)\b",
]


# --------------------------------------------------------------------------- #
#  Trigger detector
# --------------------------------------------------------------------------- #

class EventTriggerDetector:
    """
    Detects event-driven triggers from news items and sentiment data.

    Usage:
        detector = EventTriggerDetector()
        triggers = detector.scan(
            news_items=items,
            sentiment_results=sentiments,
            market_question="Will US forces enter Iran by March 31?",
        )
    """

    def __init__(
        self,
        breaking_threshold: int = 3,
        cluster_threshold: int = 3,
        sentiment_shift_threshold: float = 0.3,
    ):
        self.breaking_threshold = breaking_threshold
        self.cluster_threshold = cluster_threshold
        self.sentiment_shift_threshold = sentiment_shift_threshold

    def scan(
        self,
        news_items: list[NewsItem],
        sentiment_results: list[SentimentResult] | None = None,
        market_question: str = "",
    ) -> TriggerSummary:
        """
        Scan news and sentiment data for event triggers.

        Returns a TriggerSummary with all detected triggers sorted by severity.
        """
        triggers: list[EventTrigger] = []

        # 1. Breaking news detection
        triggers.extend(self._detect_breaking_news(news_items, market_question))

        # 2. Official statement detection
        triggers.extend(self._detect_official_statements(news_items, market_question))

        # 3. Key account activity
        triggers.extend(self._detect_key_accounts(news_items, market_question))

        # 4. News clustering (multiple sources, same story)
        triggers.extend(self._detect_news_clusters(news_items, market_question))

        # 5. Sentiment shift detection
        if sentiment_results:
            triggers.extend(self._detect_sentiment_shift(sentiment_results, market_question))

        # Sort by severity (critical first)
        severity_order = {
            TriggerSeverity.CRITICAL: 0,
            TriggerSeverity.HIGH: 1,
            TriggerSeverity.MEDIUM: 2,
            TriggerSeverity.LOW: 3,
        }
        triggers.sort(key=lambda t: (severity_order.get(t.severity, 99), -t.confidence))

        # Build summary
        bullish = sum(1 for t in triggers if t.direction == "bullish")
        bearish = sum(1 for t in triggers if t.direction == "bearish")
        if bullish > bearish + 1:
            net = "bullish"
        elif bearish > bullish + 1:
            net = "bearish"
        elif bullish == 0 and bearish == 0:
            net = "neutral"
        else:
            net = "mixed"

        return TriggerSummary(
            market_question=market_question,
            total_triggers=len(triggers),
            critical_count=sum(1 for t in triggers if t.severity == TriggerSeverity.CRITICAL),
            high_count=sum(1 for t in triggers if t.severity == TriggerSeverity.HIGH),
            medium_count=sum(1 for t in triggers if t.severity == TriggerSeverity.MEDIUM),
            low_count=sum(1 for t in triggers if t.severity == TriggerSeverity.LOW),
            net_direction=net,
            triggers=triggers,
        )

    # ------------------------------------------------------------------ #
    #  Breaking news
    # ------------------------------------------------------------------ #

    def _detect_breaking_news(
        self,
        items: list[NewsItem],
        context: str,
    ) -> list[EventTrigger]:
        """Detect breaking news based on indicator phrases and recency."""
        triggers: list[EventTrigger] = []

        for item in items:
            text_lower = item.full_text.lower()
            is_breaking = any(ind in text_lower for ind in BREAKING_INDICATORS)

            if not is_breaking:
                continue

            relevance = self._compute_relevance(item.full_text, context)
            if relevance < 0.15:
                continue

            # Recency boost: more recent = more severe
            age = item.age_hours
            if age < 1:
                severity = TriggerSeverity.CRITICAL
            elif age < 6:
                severity = TriggerSeverity.HIGH
            elif age < 24:
                severity = TriggerSeverity.MEDIUM
            else:
                severity = TriggerSeverity.LOW

            direction = self._infer_direction(text_lower)

            triggers.append(EventTrigger(
                type=TriggerType.BREAKING_NEWS,
                severity=severity,
                title=f"Breaking: {item.title[:100]}",
                description=item.description[:200] if item.description else "",
                source=item.source,
                market_relevance=round(relevance, 3),
                confidence=round(min(1.0, relevance * 1.5) * (1.0 if age < 6 else 0.7), 3),
                direction=direction,
                related_items=[item.url] if item.url else [],
            ))

        return triggers

    # ------------------------------------------------------------------ #
    #  Official statements
    # ------------------------------------------------------------------ #

    def _detect_official_statements(
        self,
        items: list[NewsItem],
        context: str,
    ) -> list[EventTrigger]:
        """Detect official government/institutional statements."""
        triggers: list[EventTrigger] = []

        for item in items:
            text_lower = item.full_text.lower()
            matched_pattern = None

            for pattern in OFFICIAL_PATTERNS:
                if re.search(pattern, text_lower):
                    matched_pattern = pattern
                    break

            if not matched_pattern:
                continue

            relevance = self._compute_relevance(item.full_text, context)
            if relevance < 0.1:
                continue

            direction = self._infer_direction(text_lower)
            age = item.age_hours

            severity = TriggerSeverity.HIGH if age < 12 else TriggerSeverity.MEDIUM

            triggers.append(EventTrigger(
                type=TriggerType.OFFICIAL_STATEMENT,
                severity=severity,
                title=f"Official: {item.title[:100]}",
                description=item.description[:200] if item.description else "",
                source=item.source,
                market_relevance=round(relevance, 3),
                confidence=round(min(1.0, relevance * 1.3 + 0.2), 3),
                direction=direction,
                related_items=[item.url] if item.url else [],
            ))

        return triggers

    # ------------------------------------------------------------------ #
    #  Key accounts
    # ------------------------------------------------------------------ #

    def _detect_key_accounts(
        self,
        items: list[NewsItem],
        context: str,
    ) -> list[EventTrigger]:
        """Detect tweets/posts from key accounts."""
        triggers: list[EventTrigger] = []
        all_key_accounts = set()
        for accounts in KEY_ACCOUNTS.values():
            all_key_accounts.update(a.lower() for a in accounts)

        for item in items:
            # Check if author is a key account
            author = (item.author or "").lstrip("@").lower()
            source_account = ""
            if "twitter:" in item.source:
                source_account = item.source.split("@")[-1].lower() if "@" in item.source else ""

            account = author or source_account
            if not account or account not in all_key_accounts:
                continue

            relevance = self._compute_relevance(item.full_text, context)
            if relevance < 0.1:
                continue

            direction = self._infer_direction(item.full_text.lower())

            triggers.append(EventTrigger(
                type=TriggerType.KEY_ACCOUNT,
                severity=TriggerSeverity.HIGH if item.age_hours < 6 else TriggerSeverity.MEDIUM,
                title=f"@{account}: {item.title[:80]}",
                description=item.description[:200] if item.description else "",
                source=item.source,
                market_relevance=round(relevance, 3),
                confidence=round(min(1.0, relevance + 0.3), 3),
                direction=direction,
                related_items=[item.url] if item.url else [],
            ))

        return triggers

    # ------------------------------------------------------------------ #
    #  News clustering
    # ------------------------------------------------------------------ #

    def _detect_news_clusters(
        self,
        items: list[NewsItem],
        context: str,
    ) -> list[EventTrigger]:
        """Detect when multiple sources cover the same story."""
        if len(items) < self.cluster_threshold:
            return []

        triggers: list[EventTrigger] = []

        # Group items by keyword overlap
        clusters: list[list[NewsItem]] = []
        used = set()

        for i, item_a in enumerate(items):
            if i in used:
                continue
            cluster = [item_a]
            words_a = set(item_a.title.lower().split())

            for j, item_b in enumerate(items[i + 1:], start=i + 1):
                if j in used:
                    continue
                words_b = set(item_b.title.lower().split())
                # Jaccard similarity
                if not words_a or not words_b:
                    continue
                overlap = len(words_a & words_b) / len(words_a | words_b)
                if overlap > 0.4:
                    cluster.append(item_b)
                    used.add(j)

            if len(cluster) >= self.cluster_threshold:
                used.add(i)
                clusters.append(cluster)

        for cluster in clusters:
            relevance = max(
                self._compute_relevance(item.full_text, context)
                for item in cluster
            )
            if relevance < 0.15:
                continue

            sources = list({item.source for item in cluster if item.source})
            representative = cluster[0]  # first item as representative
            direction = self._infer_direction(representative.full_text.lower())

            triggers.append(EventTrigger(
                type=TriggerType.NEWS_CLUSTER,
                severity=TriggerSeverity.HIGH if len(cluster) >= 5 else TriggerSeverity.MEDIUM,
                title=f"News cluster ({len(cluster)} articles): {representative.title[:80]}",
                description=f"Sources: {', '.join(sources[:5])}",
                source=", ".join(sources[:3]),
                market_relevance=round(relevance, 3),
                confidence=round(min(1.0, len(cluster) / 8.0 + relevance * 0.5), 3),
                direction=direction,
                related_items=[item.url for item in cluster if item.url][:5],
            ))

        return triggers

    # ------------------------------------------------------------------ #
    #  Sentiment shift
    # ------------------------------------------------------------------ #

    def _detect_sentiment_shift(
        self,
        results: list[SentimentResult],
        context: str,
    ) -> list[EventTrigger]:
        """Detect rapid shifts in aggregate sentiment."""
        if len(results) < 3:
            return []

        triggers: list[EventTrigger] = []

        # Split into recent vs older (by timestamp)
        sorted_results = sorted(
            [r for r in results if r.timestamp],
            key=lambda r: r.timestamp,
        )
        if len(sorted_results) < 4:
            return []

        midpoint = len(sorted_results) // 2
        older = sorted_results[:midpoint]
        recent = sorted_results[midpoint:]

        older_avg = sum(r.score for r in older) / len(older) if older else 0
        recent_avg = sum(r.score for r in recent) / len(recent) if recent else 0
        shift = recent_avg - older_avg

        if abs(shift) < self.sentiment_shift_threshold:
            return []

        if shift > 0:
            direction = "bullish"
            severity = TriggerSeverity.HIGH if shift > 0.5 else TriggerSeverity.MEDIUM
        else:
            direction = "bearish"
            severity = TriggerSeverity.HIGH if shift < -0.5 else TriggerSeverity.MEDIUM

        triggers.append(EventTrigger(
            type=TriggerType.SENTIMENT_SHIFT,
            severity=severity,
            title=f"Sentiment shift: {older_avg:+.2f} -> {recent_avg:+.2f} ({shift:+.2f})",
            description=(
                f"Aggregate sentiment shifted {'positively' if shift > 0 else 'negatively'} "
                f"across {len(sorted_results)} data points."
            ),
            source="sentiment_pipeline",
            market_relevance=0.8,
            confidence=round(min(1.0, abs(shift) * 1.5 * (len(sorted_results) / 20.0)), 3),
            direction=direction,
        ))

        return triggers

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_relevance(text: str, context: str) -> float:
        """Compute relevance of text to a market question."""
        if not context:
            return 0.5
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "will", "be",
            "by", "to", "in", "on", "at", "of", "for", "and", "or",
            "this", "that", "it", "with", "from", "as", "has", "have",
        }
        text_words = set(re.findall(r'\b[a-z]{3,}\b', text.lower())) - stop_words
        ctx_words = set(re.findall(r'\b[a-z]{3,}\b', context.lower())) - stop_words
        if not ctx_words:
            return 0.5
        overlap = text_words & ctx_words
        return min(1.0, len(overlap) / len(ctx_words) * 1.5)

    @staticmethod
    def _infer_direction(text_lower: str) -> str:
        """Infer bullish/bearish direction from text keywords."""
        bullish_kw = [
            "confirmed", "imminent", "launched", "invaded", "attacked",
            "approved", "signed", "passed", "enacted", "breaking",
            "won", "elected", "victory", "surge", "rally",
        ]
        bearish_kw = [
            "denied", "rejected", "unlikely", "withdrawn", "ceasefire",
            "peace", "canceled", "failed", "collapsed", "postponed",
            "defeated", "lost", "vetoed", "crashed", "banned",
        ]
        bull = sum(1 for kw in bullish_kw if kw in text_lower)
        bear = sum(1 for kw in bearish_kw if kw in text_lower)
        if bull > bear:
            return "bullish"
        elif bear > bull:
            return "bearish"
        return "unknown"
