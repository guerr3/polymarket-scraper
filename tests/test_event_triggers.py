"""Tests for the event-driven trigger detector."""

import pytest
from datetime import datetime, timezone, timedelta

from intelligence.news_feeds import NewsItem
from intelligence.sentiment import SentimentResult, SentimentLabel
from intelligence.event_triggers import (
    EventTriggerDetector,
    EventTrigger,
    TriggerType,
    TriggerSeverity,
    TriggerSummary,
    KEY_ACCOUNTS,
    BREAKING_INDICATORS,
    OFFICIAL_PATTERNS,
)


def _make_news(
    title: str,
    description: str = "",
    source: str = "reuters",
    author: str = "",
    age_hours: float = 1.0,
    url: str = "",
) -> NewsItem:
    """Helper to create a NewsItem with a specific age."""
    published = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    return NewsItem(
        title=title,
        description=description,
        source=source,
        author=author,
        published=published,
        url=url or f"https://example.com/{title[:20].replace(' ', '-')}",
    )


def _make_sentiment(score: float, timestamp_offset_hours: float = 0) -> SentimentResult:
    """Helper to create a SentimentResult with a timestamp."""
    return SentimentResult(
        text="test",
        score=score,
        label=SentimentLabel.BULLISH if score > 0 else SentimentLabel.BEARISH,
        confidence=0.7,
        timestamp=datetime.now(timezone.utc) - timedelta(hours=timestamp_offset_hours),
    )


# --------------------------------------------------------------------------- #
#  EventTrigger
# --------------------------------------------------------------------------- #

class TestEventTrigger:
    def test_is_actionable_critical_high_confidence(self):
        t = EventTrigger(
            type=TriggerType.BREAKING_NEWS,
            severity=TriggerSeverity.CRITICAL,
            title="Test",
            confidence=0.6,
        )
        assert t.is_actionable

    def test_not_actionable_low_severity(self):
        t = EventTrigger(
            type=TriggerType.BREAKING_NEWS,
            severity=TriggerSeverity.LOW,
            title="Test",
            confidence=0.8,
        )
        assert not t.is_actionable

    def test_not_actionable_low_confidence(self):
        t = EventTrigger(
            type=TriggerType.BREAKING_NEWS,
            severity=TriggerSeverity.CRITICAL,
            title="Test",
            confidence=0.2,
        )
        assert not t.is_actionable


# --------------------------------------------------------------------------- #
#  Breaking news detection
# --------------------------------------------------------------------------- #

class TestBreakingNews:
    def setup_method(self):
        self.detector = EventTriggerDetector()

    def test_detects_breaking_indicator(self):
        items = [_make_news("BREAKING: Iran launches military strike", age_hours=0.5)]
        triggers = self.detector._detect_breaking_news(items, "Will Iran attack?")
        assert len(triggers) >= 1
        assert triggers[0].type == TriggerType.BREAKING_NEWS
        assert triggers[0].severity == TriggerSeverity.CRITICAL  # < 1 hour

    def test_severity_based_on_age(self):
        items_fresh = [_make_news("BREAKING: event happened", age_hours=0.5)]
        items_old = [_make_news("BREAKING: event happened", age_hours=10)]

        t_fresh = self.detector._detect_breaking_news(items_fresh, "event")
        t_old = self.detector._detect_breaking_news(items_old, "event")

        assert t_fresh[0].severity == TriggerSeverity.CRITICAL
        assert t_old[0].severity == TriggerSeverity.MEDIUM

    def test_ignores_irrelevant_breaking(self):
        items = [_make_news("BREAKING: Celebrity spotted at restaurant")]
        triggers = self.detector._detect_breaking_news(items, "Will Iran launch missiles?")
        assert len(triggers) == 0  # low relevance

    def test_no_breaking_indicator(self):
        items = [_make_news("Iran holds routine military exercise")]
        triggers = self.detector._detect_breaking_news(items, "Will Iran attack?")
        assert len(triggers) == 0


# --------------------------------------------------------------------------- #
#  Official statement detection
# --------------------------------------------------------------------------- #

class TestOfficialStatements:
    def setup_method(self):
        self.detector = EventTriggerDetector()

    def test_detects_white_house_statement(self):
        items = [_make_news("White House says new sanctions are imminent", age_hours=2)]
        triggers = self.detector._detect_official_statements(items, "sanctions")
        assert len(triggers) >= 1
        assert triggers[0].type == TriggerType.OFFICIAL_STATEMENT

    def test_detects_president_statement(self):
        items = [_make_news("President announces emergency declaration", age_hours=5)]
        triggers = self.detector._detect_official_statements(items, "emergency")
        assert len(triggers) >= 1

    def test_no_match_for_non_official(self):
        items = [_make_news("Local blogger says economy is doing well")]
        triggers = self.detector._detect_official_statements(items, "economy")
        assert len(triggers) == 0


# --------------------------------------------------------------------------- #
#  Key account detection
# --------------------------------------------------------------------------- #

class TestKeyAccounts:
    def setup_method(self):
        self.detector = EventTriggerDetector()

    def test_detects_key_account_by_author(self):
        items = [_make_news(
            "Major announcement about crypto regulation",
            author="@elonmusk",
            source="twitter:@elonmusk",
            age_hours=2,
        )]
        triggers = self.detector._detect_key_accounts(items, "crypto regulation")
        assert len(triggers) >= 1
        assert triggers[0].type == TriggerType.KEY_ACCOUNT

    def test_ignores_non_key_account(self):
        items = [_make_news(
            "My thoughts on crypto",
            author="@randomuser123",
            source="twitter:@randomuser123",
        )]
        triggers = self.detector._detect_key_accounts(items, "crypto")
        assert len(triggers) == 0


# --------------------------------------------------------------------------- #
#  News clustering
# --------------------------------------------------------------------------- #

class TestNewsClustering:
    def setup_method(self):
        self.detector = EventTriggerDetector(cluster_threshold=3)

    def test_detects_cluster(self):
        items = [
            _make_news("Iran military strike confirmed by officials", source="reuters"),
            _make_news("Iran military strike confirmed by defense ministry", source="bbc"),
            _make_news("Iran military strike confirmed today", source="ap"),
        ]
        triggers = self.detector._detect_news_clusters(items, "Iran military strike")
        assert len(triggers) >= 1
        assert triggers[0].type == TriggerType.NEWS_CLUSTER

    def test_no_cluster_below_threshold(self):
        items = [
            _make_news("Iran military strike", source="reuters"),
            _make_news("Iran military strike", source="bbc"),
        ]
        triggers = self.detector._detect_news_clusters(items, "Iran military strike")
        assert len(triggers) == 0

    def test_dissimilar_titles_not_clustered(self):
        items = [
            _make_news("Iran military action", source="reuters"),
            _make_news("Bitcoin price surges today", source="bbc"),
            _make_news("Weather forecast for tomorrow", source="ap"),
        ]
        triggers = self.detector._detect_news_clusters(items, "Iran")
        assert len(triggers) == 0


# --------------------------------------------------------------------------- #
#  Sentiment shift detection
# --------------------------------------------------------------------------- #

class TestSentimentShift:
    def setup_method(self):
        self.detector = EventTriggerDetector(sentiment_shift_threshold=0.3)

    def test_detects_bullish_shift(self):
        results = [
            _make_sentiment(-0.3, timestamp_offset_hours=10),
            _make_sentiment(-0.2, timestamp_offset_hours=8),
            _make_sentiment(0.3, timestamp_offset_hours=2),
            _make_sentiment(0.4, timestamp_offset_hours=1),
        ]
        triggers = self.detector._detect_sentiment_shift(results, "test")
        assert len(triggers) >= 1
        assert triggers[0].direction == "bullish"

    def test_detects_bearish_shift(self):
        results = [
            _make_sentiment(0.4, timestamp_offset_hours=10),
            _make_sentiment(0.3, timestamp_offset_hours=8),
            _make_sentiment(-0.3, timestamp_offset_hours=2),
            _make_sentiment(-0.4, timestamp_offset_hours=1),
        ]
        triggers = self.detector._detect_sentiment_shift(results, "test")
        assert len(triggers) >= 1
        assert triggers[0].direction == "bearish"

    def test_no_shift_below_threshold(self):
        results = [
            _make_sentiment(0.1, timestamp_offset_hours=10),
            _make_sentiment(0.15, timestamp_offset_hours=8),
            _make_sentiment(0.2, timestamp_offset_hours=2),
            _make_sentiment(0.15, timestamp_offset_hours=1),
        ]
        triggers = self.detector._detect_sentiment_shift(results, "test")
        assert len(triggers) == 0

    def test_too_few_results(self):
        results = [_make_sentiment(0.5, 1), _make_sentiment(-0.5, 2)]
        triggers = self.detector._detect_sentiment_shift(results, "test")
        assert len(triggers) == 0


# --------------------------------------------------------------------------- #
#  Full scan
# --------------------------------------------------------------------------- #

class TestFullScan:
    def setup_method(self):
        self.detector = EventTriggerDetector()

    def test_scan_returns_summary(self):
        items = [
            _make_news("BREAKING: Major event confirmed", age_hours=0.5),
            _make_news("President announces new policy", age_hours=2),
        ]
        summary = self.detector.scan(
            news_items=items,
            market_question="Will major event happen?",
        )
        assert isinstance(summary, TriggerSummary)
        assert summary.total_triggers >= 1
        assert summary.market_question == "Will major event happen?"

    def test_scan_sorts_by_severity(self):
        items = [
            _make_news("BREAKING: Critical event now", age_hours=0.3),
            _make_news("BREAKING: Old event happened", age_hours=30),
        ]
        summary = self.detector.scan(
            news_items=items,
            market_question="event",
        )
        if len(summary.triggers) >= 2:
            severity_order = {
                TriggerSeverity.CRITICAL: 0, TriggerSeverity.HIGH: 1,
                TriggerSeverity.MEDIUM: 2, TriggerSeverity.LOW: 3,
            }
            for i in range(len(summary.triggers) - 1):
                assert severity_order[summary.triggers[i].severity] <= \
                       severity_order[summary.triggers[i + 1].severity]

    def test_scan_empty_items(self):
        summary = self.detector.scan(news_items=[], market_question="test")
        assert summary.total_triggers == 0
        assert summary.net_direction == "neutral"

    def test_net_direction_bullish(self):
        items = [
            _make_news("BREAKING: Confirmed invasion launched", age_hours=0.5),
            _make_news("BREAKING: Confirmed attack imminent", age_hours=0.5),
            _make_news("BREAKING: Victory rally surge", age_hours=0.5),
        ]
        summary = self.detector.scan(
            news_items=items,
            market_question="invasion attack victory rally surge confirmed imminent launched",
        )
        # Should have bullish direction triggers
        bullish_count = sum(1 for t in summary.triggers if t.direction == "bullish")
        assert bullish_count >= 1


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

class TestHelpers:
    def test_compute_relevance_no_context(self):
        assert EventTriggerDetector._compute_relevance("some text", "") == 0.5

    def test_compute_relevance_high_overlap(self):
        rel = EventTriggerDetector._compute_relevance(
            "Iran military strike launches", "Iran military strike"
        )
        assert rel > 0.5

    def test_infer_direction_bullish(self):
        assert EventTriggerDetector._infer_direction("invasion confirmed and launched") == "bullish"

    def test_infer_direction_bearish(self):
        assert EventTriggerDetector._infer_direction("peace deal unlikely rejected") == "bearish"

    def test_infer_direction_unknown(self):
        assert EventTriggerDetector._infer_direction("the weather is nice today") == "unknown"


# --------------------------------------------------------------------------- #
#  Config sanity
# --------------------------------------------------------------------------- #

class TestConfig:
    def test_key_accounts_has_categories(self):
        assert "politics" in KEY_ACCOUNTS
        assert "crypto" in KEY_ACCOUNTS
        assert "geopolitics" in KEY_ACCOUNTS

    def test_breaking_indicators_lowercase(self):
        for ind in BREAKING_INDICATORS:
            assert ind == ind.lower()
