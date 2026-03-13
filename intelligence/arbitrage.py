"""
Cross-market arbitrage detector for prediction markets.

Identifies inconsistencies between related markets:
- Correlated markets that should move together (e.g., "Iran regime falls" vs "US enters Iran")
- Complementary markets that should sum to ~100%
- Temporal arbitrage (same event, different timeframes)

Detects when market prices diverge from logical consistency.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from polymarket_client.models import Market

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Types
# --------------------------------------------------------------------------- #

@dataclass
class ArbitrageOpportunity:
    """A detected cross-market inconsistency."""

    type: str  # "correlation", "complement", "temporal"
    market_a: MarketRef = field(default_factory=lambda: MarketRef())
    market_b: MarketRef = field(default_factory=lambda: MarketRef())
    expected_relationship: str = ""   # e.g., "A implies B", "A + B ≈ 1.0"
    price_a: float = 0.0
    price_b: float = 0.0
    divergence: float = 0.0          # how far from expected
    confidence: float = 0.0          # 0-1
    reasoning: str = ""
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def severity(self) -> str:
        if self.divergence > 0.20:
            return "HIGH"
        elif self.divergence > 0.10:
            return "MEDIUM"
        return "LOW"


@dataclass
class MarketRef:
    """Lightweight reference to a market."""
    condition_id: str = ""
    question: str = ""
    yes_price: float = 0.0


@dataclass
class MarketCluster:
    """A group of related markets."""
    topic: str = ""
    keywords: list[str] = field(default_factory=list)
    markets: list[MarketRef] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Relationship extraction
# --------------------------------------------------------------------------- #

# Keywords that indicate causal/correlated relationships between events
RELATIONSHIP_KEYWORDS = {
    "implies": [
        ("invasion", "regime change"), ("attack", "war"),
        ("elected", "sworn in"), ("passed", "enacted"),
        ("indicted", "convicted"), ("default", "recession"),
        ("ceasefire", "peace"), ("launch", "deployment"),
    ],
    "excludes": [
        ("ceasefire", "invasion"), ("peace deal", "war"),
        ("acquitted", "convicted"), ("denied", "confirmed"),
    ],
}

# Topic extraction patterns
TOPIC_PATTERNS = {
    "iran": r"\biran(?:ian)?\b",
    "russia_ukraine": r"\b(?:russia|ukraine|russian|ukrainian)\b",
    "israel_palestine": r"\b(?:israel|palestine|gaza|hamas|hezbollah)\b",
    "us_politics": r"\b(?:trump|biden|congress|senate|supreme court|white house)\b",
    "crypto_btc": r"\b(?:bitcoin|btc)\b",
    "crypto_eth": r"\b(?:ethereum|eth)\b",
    "crypto_sol": r"\b(?:solana|sol)\b",
    "fed": r"\b(?:federal reserve|fed rate|interest rate|fomc)\b",
    "ai": r"\b(?:artificial intelligence|openai|chatgpt|gpt|anthropic)\b",
    "elections": r"\b(?:election|vote|ballot|primary|nominee)\b",
    "sports_nba": r"\b(?:nba|lakers|celtics|bucks|finals)\b",
    "sports_nfl": r"\b(?:nfl|super bowl|touchdown)\b",
}


class ArbitrageDetector:
    """
    Detects cross-market arbitrage and inconsistencies.

    Analyzes a set of active markets for:
    1. Correlated pairs that should move together
    2. Complement pairs that should sum to ~1.0
    3. Temporal inconsistencies (longer timeframe < shorter timeframe)
    """

    def __init__(self, correlation_threshold: float = 0.15):
        self.correlation_threshold = correlation_threshold

    def analyze(self, markets: list[Market]) -> list[ArbitrageOpportunity]:
        """
        Analyze a set of markets for arbitrage opportunities.

        Returns list of detected opportunities sorted by divergence.
        """
        opportunities: list[ArbitrageOpportunity] = []

        # 1. Cluster markets by topic
        clusters = self._cluster_markets(markets)

        # 2. Within each cluster, look for relationships
        for cluster in clusters:
            if len(cluster.markets) < 2:
                continue

            # Check pairwise relationships
            for i, mkt_a in enumerate(cluster.markets):
                for mkt_b in cluster.markets[i + 1:]:
                    opps = self._check_pair(mkt_a, mkt_b, cluster.topic)
                    opportunities.extend(opps)

        # 3. Check temporal relationships across all markets
        temporal_opps = self._check_temporal(markets)
        opportunities.extend(temporal_opps)

        # Sort by divergence (most significant first)
        opportunities.sort(key=lambda o: o.divergence, reverse=True)

        logger.info(
            "Arbitrage scan: %d markets → %d clusters → %d opportunities",
            len(markets),
            len(clusters),
            len(opportunities),
        )
        return opportunities

    # ------------------------------------------------------------------ #
    #  Clustering
    # ------------------------------------------------------------------ #

    def _cluster_markets(self, markets: list[Market]) -> list[MarketCluster]:
        """Group markets by topic using keyword matching."""
        clusters: dict[str, MarketCluster] = {}

        for market in markets:
            q_lower = market.question.lower()
            yes_price = market.outcomes[0].price if market.outcomes else 0.0

            ref = MarketRef(
                condition_id=market.condition_id,
                question=market.question,
                yes_price=yes_price,
            )

            for topic, pattern in TOPIC_PATTERNS.items():
                if re.search(pattern, q_lower):
                    if topic not in clusters:
                        clusters[topic] = MarketCluster(
                            topic=topic,
                            keywords=re.findall(pattern, q_lower),
                        )
                    clusters[topic].markets.append(ref)

        return list(clusters.values())

    # ------------------------------------------------------------------ #
    #  Pair analysis
    # ------------------------------------------------------------------ #

    def _check_pair(
        self,
        a: MarketRef,
        b: MarketRef,
        topic: str,
    ) -> list[ArbitrageOpportunity]:
        """Check a pair of markets for relationship inconsistencies."""
        opportunities: list[ArbitrageOpportunity] = []

        # 1. Implication check: if A implies B, then P(A) ≤ P(B)
        implication = self._detect_implication(a.question, b.question)
        if implication:
            antecedent, consequent = implication
            if antecedent == "a":
                # A implies B → P(A) should be ≤ P(B)
                if a.yes_price > b.yes_price + self.correlation_threshold:
                    opportunities.append(ArbitrageOpportunity(
                        type="correlation",
                        market_a=a,
                        market_b=b,
                        expected_relationship=f"'{a.question[:40]}' implies '{b.question[:40]}' → P(A) ≤ P(B)",
                        price_a=a.yes_price,
                        price_b=b.yes_price,
                        divergence=round(a.yes_price - b.yes_price, 4),
                        confidence=0.7,
                        reasoning=f"If '{a.question[:30]}...' happens, '{b.question[:30]}...' should also happen. "
                                  f"But A is priced at {a.yes_price:.0%} while B is only {b.yes_price:.0%}.",
                    ))
            else:
                # B implies A
                if b.yes_price > a.yes_price + self.correlation_threshold:
                    opportunities.append(ArbitrageOpportunity(
                        type="correlation",
                        market_a=b,
                        market_b=a,
                        expected_relationship=f"'{b.question[:40]}' implies '{a.question[:40]}' → P(B) ≤ P(A)",
                        price_a=b.yes_price,
                        price_b=a.yes_price,
                        divergence=round(b.yes_price - a.yes_price, 4),
                        confidence=0.7,
                        reasoning=f"Logical implication violated: antecedent priced higher than consequent.",
                    ))

        # 2. Exclusion check: if A excludes B, then P(A) + P(B) ≤ 1.0
        if self._detect_exclusion(a.question, b.question):
            total = a.yes_price + b.yes_price
            if total > 1.0 + self.correlation_threshold:
                opportunities.append(ArbitrageOpportunity(
                    type="complement",
                    market_a=a,
                    market_b=b,
                    expected_relationship=f"Mutually exclusive → P(A) + P(B) ≤ 1.0",
                    price_a=a.yes_price,
                    price_b=b.yes_price,
                    divergence=round(total - 1.0, 4),
                    confidence=0.6,
                    reasoning=f"These events seem mutually exclusive but sum to {total:.0%} > 100%.",
                ))

        # 3. Similar question check (near-duplicate at different prices)
        similarity = self._question_similarity(a.question, b.question)
        if similarity > 0.7:
            price_diff = abs(a.yes_price - b.yes_price)
            if price_diff > self.correlation_threshold:
                opportunities.append(ArbitrageOpportunity(
                    type="correlation",
                    market_a=a,
                    market_b=b,
                    expected_relationship=f"Similar questions ({similarity:.0%} overlap) → similar prices",
                    price_a=a.yes_price,
                    price_b=b.yes_price,
                    divergence=round(price_diff, 4),
                    confidence=round(similarity * 0.8, 2),
                    reasoning=f"These questions appear very similar but are priced {price_diff:.0%} apart.",
                ))

        return opportunities

    # ------------------------------------------------------------------ #
    #  Temporal analysis
    # ------------------------------------------------------------------ #

    def _check_temporal(self, markets: list[Market]) -> list[ArbitrageOpportunity]:
        """
        Detect temporal inconsistencies.

        If "X by March 31" is priced at 40%, then "X by April 30" should be ≥ 40%.
        """
        opportunities: list[ArbitrageOpportunity] = []

        # Group by base question (strip date)
        date_pattern = re.compile(
            r"\b(?:by|before|on)\s+"
            r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
            r"\s+\d{1,2}(?:,?\s*\d{4})?\b",
            re.IGNORECASE,
        )

        temporal_groups: dict[str, list[tuple[Market, datetime | None]]] = {}
        for m in markets:
            base = date_pattern.sub("__DATE__", m.question.lower()).strip()
            if "__DATE__" not in base:
                continue

            # Extract the date
            date_match = date_pattern.search(m.question.lower())
            end_date = m.end_date
            temporal_groups.setdefault(base, []).append((m, end_date))

        for base_q, group in temporal_groups.items():
            if len(group) < 2:
                continue

            # Sort by end date
            dated = [(m, d) for m, d in group if d is not None]
            dated.sort(key=lambda x: x[1])

            for i in range(len(dated) - 1):
                earlier_mkt, earlier_date = dated[i]
                later_mkt, later_date = dated[i + 1]

                earlier_price = earlier_mkt.outcomes[0].price if earlier_mkt.outcomes else 0
                later_price = later_mkt.outcomes[0].price if later_mkt.outcomes else 0

                # Earlier deadline should be ≤ later deadline
                if earlier_price > later_price + self.correlation_threshold:
                    opportunities.append(ArbitrageOpportunity(
                        type="temporal",
                        market_a=MarketRef(
                            condition_id=earlier_mkt.condition_id,
                            question=earlier_mkt.question,
                            yes_price=earlier_price,
                        ),
                        market_b=MarketRef(
                            condition_id=later_mkt.condition_id,
                            question=later_mkt.question,
                            yes_price=later_price,
                        ),
                        expected_relationship="Earlier deadline ≤ later deadline in probability",
                        price_a=earlier_price,
                        price_b=later_price,
                        divergence=round(earlier_price - later_price, 4),
                        confidence=0.85,
                        reasoning=f"'{earlier_mkt.question[:40]}...' (earlier) is priced higher "
                                  f"({earlier_price:.0%}) than '{later_mkt.question[:40]}...' "
                                  f"(later, {later_price:.0%}). This is logically impossible.",
                    ))

        return opportunities

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _detect_implication(
        self,
        q_a: str,
        q_b: str,
    ) -> Optional[tuple[str, str]]:
        """
        Detect if one question implies the other.

        Returns ("a", "b") if A implies B, ("b", "a") if B implies A, else None.
        """
        a_lower = q_a.lower()
        b_lower = q_b.lower()

        for antecedent_kw, consequent_kw in RELATIONSHIP_KEYWORDS["implies"]:
            if antecedent_kw in a_lower and consequent_kw in b_lower:
                return ("a", "b")
            if antecedent_kw in b_lower and consequent_kw in a_lower:
                return ("b", "a")

        return None

    def _detect_exclusion(self, q_a: str, q_b: str) -> bool:
        """Detect if two questions are mutually exclusive."""
        a_lower = q_a.lower()
        b_lower = q_b.lower()

        for kw_a, kw_b in RELATIONSHIP_KEYWORDS["excludes"]:
            if (kw_a in a_lower and kw_b in b_lower) or \
               (kw_b in a_lower and kw_a in b_lower):
                return True

        return False

    @staticmethod
    def _question_similarity(q_a: str, q_b: str) -> float:
        """Compute word-level Jaccard similarity between two questions."""
        stop = {"the", "a", "an", "is", "are", "will", "be", "by", "to", "of", "and", "or", "in", "on", "at", "?"}
        words_a = set(q_a.lower().split()) - stop
        words_b = set(q_b.lower().split()) - stop

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)
