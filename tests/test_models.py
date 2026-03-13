"""Tests for data models and mapping functions."""

import pytest

from polymarket_client.models import (
    Market,
    Trade,
    MarketStatus,
    MarketSource,
    TradeSide,
    gamma_market_to_model,
    clob_trade_to_model,
    goldsky_order_to_model,
    goldsky_position_to_model,
    price_history_to_points,
)


def test_gamma_market_mapping():
    """Test GAMMA API response → Market model."""
    raw = {
        "id": 12345,
        "conditionId": "0xabc123",
        "question": "Will BTC reach $100k?",
        "slug": "will-btc-reach-100k",
        "description": "Test market",
        "active": True,
        "closed": False,
        "volume": "5000000",
        "volume24hr": "250000",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.65", "0.35"]',
        "tags": '["crypto", "bitcoin"]',
        "endDate": "2026-12-31T00:00:00Z",
    }

    market = gamma_market_to_model(raw)

    assert market.id == 12345
    assert market.condition_id == "0xabc123"
    assert market.question == "Will BTC reach $100k?"
    assert market.status == MarketStatus.ACTIVE
    assert market.volume == 5_000_000.0
    assert market.volume_24h == 250_000.0
    assert len(market.outcomes) == 2
    assert market.outcomes[0].name == "Yes"
    assert market.outcomes[0].price == 0.65
    assert market.outcomes[1].price == 0.35
    assert "crypto" in market.tags


def test_gamma_market_mapping_list_inputs():
    """Test GAMMA mapping with list (non-string) outcomes."""
    raw = {
        "id": 1,
        "conditionId": "0x1",
        "question": "Test?",
        "active": True,
        "outcomes": ["Yes", "No"],
        "outcomePrices": [0.7, 0.3],
    }

    market = gamma_market_to_model(raw)
    assert len(market.outcomes) == 2
    assert market.outcomes[0].price == 0.7


def test_clob_trade_mapping():
    """Test CLOB trade response → Trade model."""
    raw = {
        "id": "trade_123",
        "market": "0xcondition",
        "asset_id": "0xasset",
        "side": "BUY",
        "size": "10.5",
        "price": "0.65",
        "match_time": 1700000000,
    }

    trade = clob_trade_to_model(raw)

    assert trade.id == "trade_123"
    assert trade.market_condition_id == "0xcondition"
    assert trade.side == TradeSide.BUY
    assert trade.size == 10.5
    assert trade.price == 0.65
    assert trade.timestamp == 1700000000


def test_goldsky_order_mapping():
    """Test Goldsky order → OrderBookEntry."""
    raw = {
        "id": "order_1",
        "market": "0xmarket",
        "user": "0xuser",
        "price": "0.50",
        "size": "100",
        "side": "buy",
        "timestamp": "1700000000",
    }

    order = goldsky_order_to_model(raw)
    assert order.market == "0xmarket"
    assert order.price == 0.50
    assert order.size == 100.0


def test_goldsky_position_mapping():
    """Test Goldsky position → UserPosition."""
    raw = {
        "id": "pos_1",
        "user": "0xuser",
        "market": "0xmarket",
        "outcomeIndex": "1",
        "size": "500",
        "avgPrice": "0.45",
    }

    pos = goldsky_position_to_model(raw)
    assert pos.outcome_index == 1
    assert pos.size == 500.0
    assert pos.avg_price == 0.45


def test_price_history_mapping():
    """Test price history response → PricePoint list."""
    raw = {
        "history": [
            {"t": 1700000000, "p": 0.5},
            {"t": 1700000060, "p": 0.52},
            {"t": 1700000120, "p": 0.51},
        ]
    }

    points = price_history_to_points(raw)
    assert len(points) == 3
    assert points[0].timestamp == 1700000000
    assert points[1].price == 0.52


def test_market_status_parsing():
    """Test status determination from flags."""
    # Active market
    m = gamma_market_to_model({"active": True, "closed": False, "conditionId": "0x1", "question": "Q"})
    assert m.status == MarketStatus.ACTIVE

    # Resolved market
    m = gamma_market_to_model({"active": False, "closed": True, "conditionId": "0x2", "question": "Q"})
    assert m.status == MarketStatus.RESOLVED

    # Closed market
    m = gamma_market_to_model({"active": False, "closed": False, "conditionId": "0x3", "question": "Q"})
    assert m.status == MarketStatus.CLOSED
