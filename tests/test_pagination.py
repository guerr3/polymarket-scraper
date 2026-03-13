"""Tests for pagination strategies."""

import pytest
import pytest_asyncio

from polymarket_client.pagination import (
    OffsetPaginator,
    CursorPaginator,
    GraphQLPaginator,
)


@pytest.mark.asyncio
async def test_offset_paginator_full_pages():
    """Offset paginator should fetch until empty page."""
    pages_data = [
        [{"id": i} for i in range(10)],  # page 0
        [{"id": i + 10} for i in range(10)],  # page 1
        [],  # page 2 (empty → stop)
    ]
    call_count = 0

    async def mock_fetch(params):
        nonlocal call_count
        idx = call_count
        call_count += 1
        return pages_data[idx] if idx < len(pages_data) else []

    paginator = OffsetPaginator(limit=10)
    results = []
    async for page in paginator.paginate(mock_fetch):
        results.extend(page)

    assert len(results) == 20
    assert call_count == 3


@pytest.mark.asyncio
async def test_offset_paginator_partial_page():
    """Offset paginator should stop on partial page."""
    async def mock_fetch(params):
        return [{"id": i} for i in range(5)]  # partial page (5 < limit=10)

    paginator = OffsetPaginator(limit=10)
    results = []
    async for page in paginator.paginate(mock_fetch):
        results.extend(page)

    assert len(results) == 5


@pytest.mark.asyncio
async def test_cursor_paginator_terminal():
    """Cursor paginator should stop at terminal cursor."""
    pages = [
        {"data": [{"id": 1}], "next_cursor": "abc123"},
        {"data": [{"id": 2}], "next_cursor": "LTE="},  # terminal
    ]
    call_count = 0

    async def mock_fetch(params):
        nonlocal call_count
        idx = call_count
        call_count += 1
        return pages[idx]

    paginator = CursorPaginator(terminal_cursor="LTE=")
    results = []
    async for page in paginator.paginate(mock_fetch):
        results.extend(page)

    assert len(results) == 2
    assert call_count == 2


@pytest.mark.asyncio
async def test_cursor_paginator_decode():
    """Cursor decode should handle base64."""
    decoded = CursorPaginator.decode_cursor("LTE=")
    assert decoded == "-1"


@pytest.mark.asyncio
async def test_graphql_paginator():
    """GraphQL paginator should stop when fewer results returned."""
    async def mock_fetch(first, skip):
        if skip == 0:
            return [{"id": i} for i in range(1000)]
        elif skip == 1000:
            return [{"id": i} for i in range(500)]  # partial → stop
        return []

    paginator = GraphQLPaginator(first=1000)
    results = []
    async for page in paginator.paginate(mock_fetch):
        results.extend(page)

    assert len(results) == 1500
