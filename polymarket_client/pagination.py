"""
Pagination strategies for all Polymarket data sources.

- Gamma API: offset-based (limit + offset)
- CLOB API: cursor-based (next_cursor, terminal = "LTE=")
- Goldsky GraphQL: first + skip
- Analytics HTML: page-based (?page=N)
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Awaitable, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Offset pagination (GAMMA)
# --------------------------------------------------------------------------- #

@dataclass
class OffsetPaginator:
    """
    Offset-based pagination for the GAMMA REST API.

    Increments `offset` by `limit` until an empty page is returned.
    """

    limit: int = 100
    max_limit: int = 500

    async def paginate(
        self,
        fetch_fn: Callable[[dict[str, Any]], Awaitable[list[dict]]],
        base_params: dict[str, Any] | None = None,
    ) -> AsyncGenerator[list[dict], None]:
        """Yield pages of results."""
        params = dict(base_params or {})
        params["limit"] = min(self.limit, self.max_limit)
        offset = 0

        while True:
            params["offset"] = offset
            page = await fetch_fn(params)

            if not page:
                logger.debug("Offset paginator: empty page at offset=%d", offset)
                break

            yield page
            offset += len(page)

            if len(page) < params["limit"]:
                logger.debug("Offset paginator: partial page, stopping")
                break


# --------------------------------------------------------------------------- #
#  Cursor pagination (CLOB)
# --------------------------------------------------------------------------- #

@dataclass
class CursorPaginator:
    """
    Cursor-based pagination for the CLOB REST API.

    Stops when `next_cursor` equals the terminal sentinel "LTE=".
    """

    terminal_cursor: str = "LTE="

    async def paginate(
        self,
        fetch_fn: Callable[[dict[str, Any]], Awaitable[dict]],
        base_params: dict[str, Any] | None = None,
    ) -> AsyncGenerator[list[dict], None]:
        """Yield pages of results."""
        params = dict(base_params or {})
        cursor = ""

        while True:
            if cursor:
                params["next_cursor"] = cursor

            response = await fetch_fn(params)
            data = response.get("data", [])

            if data:
                yield data

            next_cursor = response.get("next_cursor", self.terminal_cursor)

            if not next_cursor or next_cursor == self.terminal_cursor:
                logger.debug("Cursor paginator: reached terminal cursor")
                break

            cursor = next_cursor

    @staticmethod
    def decode_cursor(cursor: str) -> str:
        """Decode a base64-encoded cursor for debugging."""
        try:
            return base64.b64decode(cursor).decode("utf-8")
        except Exception:
            return cursor


# --------------------------------------------------------------------------- #
#  GraphQL offset pagination (Goldsky)
# --------------------------------------------------------------------------- #

@dataclass
class GraphQLPaginator:
    """
    GraphQL-style offset pagination using `first` and `skip`.

    Stops when fewer than `first` results are returned.
    """

    first: int = 1000
    max_first: int = 1000

    async def paginate(
        self,
        fetch_fn: Callable[[int, int], Awaitable[list[dict]]],
    ) -> AsyncGenerator[list[dict], None]:
        """
        Yield pages of results.

        `fetch_fn(first, skip)` should return the list of entities.
        """
        first = min(self.first, self.max_first)
        skip = 0

        while True:
            page = await fetch_fn(first, skip)

            if not page:
                logger.debug("GraphQL paginator: empty page at skip=%d", skip)
                break

            yield page
            skip += len(page)

            if len(page) < first:
                logger.debug("GraphQL paginator: partial page, stopping")
                break


# --------------------------------------------------------------------------- #
#  HTML page pagination (Analytics)
# --------------------------------------------------------------------------- #

@dataclass
class HtmlPagePaginator:
    """
    Page-based pagination for HTML scraping.

    Uses `?page=N` and parses "Page N of {total}" text.
    """

    start_page: int = 1
    max_pages: int = 100

    async def paginate(
        self,
        fetch_fn: Callable[[int], Awaitable[tuple[list[dict], Optional[int]]]],
    ) -> AsyncGenerator[list[dict], None]:
        """
        Yield pages of results.

        `fetch_fn(page_number)` should return (rows, total_pages or None).
        """
        page = self.start_page
        total_pages: Optional[int] = None

        while page <= self.max_pages:
            rows, detected_total = await fetch_fn(page)

            if detected_total is not None:
                total_pages = detected_total

            if not rows:
                break

            yield rows
            page += 1

            if total_pages is not None and page > total_pages:
                logger.debug("HTML paginator: reached last page %d", total_pages)
                break
