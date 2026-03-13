"""Tests for the news feed aggregator (offline / parsing-only tests)."""

import pytest
from datetime import datetime, timezone, timedelta

from intelligence.news_feeds import (
    NewsItem,
    NewsFeedAggregator,
    RSS_FEEDS,
    GOOGLE_NEWS_RSS,
    NITTER_INSTANCES,
)


# --------------------------------------------------------------------------- #
#  NewsItem dataclass
# --------------------------------------------------------------------------- #

class TestNewsItem:
    def test_full_text_title_only(self):
        item = NewsItem(title="Breaking headline")
        assert item.full_text == "Breaking headline"

    def test_full_text_with_description(self):
        item = NewsItem(title="Headline", description="More details here")
        assert item.full_text == "Headline. More details here"

    def test_age_hours_none_published(self):
        item = NewsItem(title="Test")
        assert item.age_hours == 0.0

    def test_age_hours_recent(self):
        item = NewsItem(
            title="Test",
            published=datetime.now(timezone.utc) - timedelta(hours=3),
        )
        assert 2.9 < item.age_hours < 3.2

    def test_age_hours_old(self):
        item = NewsItem(
            title="Test",
            published=datetime.now(timezone.utc) - timedelta(days=2),
        )
        assert 47 < item.age_hours < 49


# --------------------------------------------------------------------------- #
#  RSS XML parsing
# --------------------------------------------------------------------------- #

class TestRSSParsing:
    def setup_method(self):
        self.aggregator = NewsFeedAggregator()

    def test_parse_rss2(self):
        xml = """<?xml version="1.0"?>
        <rss version="2.0">
          <channel>
            <title>Test Feed</title>
            <item>
              <title>Headline One</title>
              <description>Description one</description>
              <link>https://example.com/1</link>
              <pubDate>Mon, 10 Mar 2026 12:00:00 +0000</pubDate>
              <category>politics</category>
            </item>
            <item>
              <title>Headline Two</title>
              <description>&lt;p&gt;HTML description&lt;/p&gt;</description>
              <link>https://example.com/2</link>
            </item>
          </channel>
        </rss>"""
        items = self.aggregator._parse_rss_xml(xml, "test_source")
        assert len(items) == 2
        assert items[0].title == "Headline One"
        assert items[0].source == "test_source"
        assert items[0].url == "https://example.com/1"
        assert items[0].published is not None
        # HTML should be cleaned from description
        assert "<p>" not in items[1].description

    def test_parse_atom(self):
        xml = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <title>Atom Feed</title>
          <entry>
            <title>Atom Entry</title>
            <summary>Atom summary</summary>
            <link href="https://example.com/atom1"/>
            <updated>2026-03-10T12:00:00Z</updated>
          </entry>
        </feed>"""
        items = self.aggregator._parse_rss_xml(xml, "atom_source")
        assert len(items) == 1
        assert items[0].title == "Atom Entry"
        assert items[0].url == "https://example.com/atom1"

    def test_parse_empty_xml(self):
        items = self.aggregator._parse_rss_xml("<rss><channel></channel></rss>", "x")
        assert items == []

    def test_parse_invalid_xml(self):
        items = self.aggregator._parse_rss_xml("not xml at all", "x")
        assert items == []

    def test_items_without_title_skipped(self):
        xml = """<?xml version="1.0"?>
        <rss version="2.0"><channel>
          <item><description>No title here</description></item>
        </channel></rss>"""
        items = self.aggregator._parse_rss_xml(xml, "x")
        assert len(items) == 0


# --------------------------------------------------------------------------- #
#  Date parsing
# --------------------------------------------------------------------------- #

class TestDateParsing:
    def test_rss_date_format(self):
        dt = NewsFeedAggregator._parse_rss_date("Mon, 10 Mar 2026 12:00:00 +0000")
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 3

    def test_iso_format(self):
        dt = NewsFeedAggregator._parse_rss_date("2026-03-10T12:00:00Z")
        assert dt is not None
        assert dt.day == 10

    def test_empty_date(self):
        assert NewsFeedAggregator._parse_rss_date("") is None

    def test_unparseable_date(self):
        assert NewsFeedAggregator._parse_rss_date("not-a-date") is None


# --------------------------------------------------------------------------- #
#  Search query extraction
# --------------------------------------------------------------------------- #

class TestSearchQueryExtraction:
    def test_strips_boilerplate(self):
        q = NewsFeedAggregator._extract_search_query(
            "Will Bitcoin reach $100k by December 2026?"
        )
        assert "will" not in q
        assert "?" not in q

    def test_removes_stop_words(self):
        q = NewsFeedAggregator._extract_search_query(
            "Will the US president sign the new bill?"
        )
        assert "the" not in q.split()

    def test_limits_terms(self):
        q = NewsFeedAggregator._extract_search_query(
            "Will the very long question about multiple topics including a b c d e f g h i reach conclusion?"
        )
        assert len(q.split()) <= 6


# --------------------------------------------------------------------------- #
#  HTML cleaning
# --------------------------------------------------------------------------- #

class TestCleanHTML:
    def test_removes_tags(self):
        assert NewsFeedAggregator._clean_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_empty_string(self):
        assert NewsFeedAggregator._clean_html("") == ""

    def test_truncates_long_text(self):
        result = NewsFeedAggregator._clean_html("x" * 1000)
        assert len(result) <= 500


# --------------------------------------------------------------------------- #
#  Feed config sanity
# --------------------------------------------------------------------------- #

class TestFeedConfig:
    def test_rss_feeds_has_categories(self):
        assert "general" in RSS_FEEDS
        assert "politics" in RSS_FEEDS
        assert "crypto" in RSS_FEEDS

    def test_feeds_have_required_keys(self):
        for category, feeds in RSS_FEEDS.items():
            for feed in feeds:
                assert "name" in feed
                assert "url" in feed
                assert "source" in feed

    def test_nitter_instances_not_empty(self):
        assert len(NITTER_INSTANCES) > 0

    def test_google_news_template(self):
        assert "{query}" in GOOGLE_NEWS_RSS
