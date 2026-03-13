"""
Resilience module: retries, rate limiting, circuit breaker, and anti-bot measures.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Optional

import aiohttp

from .config import USER_AGENTS, ResilienceConfig, get_config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Rate limiter (token bucket)
# --------------------------------------------------------------------------- #

class RateLimiter:
    """Async token-bucket rate limiter."""

    def __init__(self, rate: float, burst: int = 1):
        self.rate = rate  # tokens per second
        self.burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_refill = now

            if self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) / self.rate
                logger.debug("Rate limiter: waiting %.2fs", wait_time)
                await asyncio.sleep(wait_time)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


# --------------------------------------------------------------------------- #
#  Circuit breaker
# --------------------------------------------------------------------------- #

class CircuitBreaker:
    """Simple circuit breaker to avoid hammering failing endpoints."""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 300.0,
        name: str = "default",
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.name = name
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state: str = "closed"  # closed, open, half-open

    @property
    def is_open(self) -> bool:
        if self._state == "open":
            elapsed = time.monotonic() - (self._last_failure_time or 0)
            if elapsed >= self.reset_timeout:
                self._state = "half-open"
                logger.info("Circuit breaker [%s]: half-open", self.name)
                return False
            return True
        return False

    def record_success(self) -> None:
        self._failure_count = 0
        if self._state == "half-open":
            self._state = "closed"
            logger.info("Circuit breaker [%s]: closed (recovered)", self.name)

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self._state = "open"
            logger.warning(
                "Circuit breaker [%s]: OPEN after %d failures",
                self.name,
                self._failure_count,
            )


# --------------------------------------------------------------------------- #
#  Retry decorator
# --------------------------------------------------------------------------- #

def retry_with_backoff(
    max_retries: int = 5,
    base_backoff: float = 1.0,
    max_backoff: float = 60.0,
    jitter_factor: float = 0.5,
    retryable_statuses: tuple[int, ...] = (429, 500, 502, 503, 504),
):
    """
    Decorator for async functions: retries with exponential backoff + jitter.

    Works with functions that return aiohttp.ClientResponse or raise exceptions.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    return result
                except (
                    aiohttp.ClientError,
                    asyncio.TimeoutError,
                    RetryableError,
                ) as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        logger.error(
                            "Retry exhausted after %d attempts for %s: %s",
                            max_retries,
                            func.__name__,
                            exc,
                        )
                        raise

                    backoff = min(
                        base_backoff * (2 ** attempt),
                        max_backoff,
                    )
                    jitter = backoff * jitter_factor * random.random()
                    sleep_time = backoff + jitter
                    logger.warning(
                        "Retry %d/%d for %s (sleeping %.2fs): %s",
                        attempt + 1,
                        max_retries,
                        func.__name__,
                        sleep_time,
                        exc,
                    )
                    await asyncio.sleep(sleep_time)

            raise last_exc  # type: ignore

        return wrapper

    return decorator


# --------------------------------------------------------------------------- #
#  Custom exceptions
# --------------------------------------------------------------------------- #

class RetryableError(Exception):
    """Raised when an HTTP response has a retryable status code."""

    def __init__(self, status: int, message: str = ""):
        self.status = status
        super().__init__(f"HTTP {status}: {message}")


class NonRetryableHttpError(Exception):
    """Raised for 4xx client errors (except 429) that should not be retried."""

    def __init__(self, status: int, message: str = ""):
        self.status = status
        super().__init__(f"HTTP {status}: {message}")


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open."""

    def __init__(self, name: str):
        super().__init__(f"Circuit breaker [{name}] is open")


# --------------------------------------------------------------------------- #
#  Resilient HTTP session
# --------------------------------------------------------------------------- #

class ResilientSession:
    """
    An aiohttp session wrapper with:
    - Rotating user agents
    - Per-endpoint rate limiting
    - Circuit breakers per endpoint name
    - Retry with exponential backoff
    - Optional proxy support
    """

    def __init__(self, config: Optional[ResilienceConfig] = None):
        self.config = config or get_config().resilience
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiters: dict[str, RateLimiter] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

    async def __aenter__(self) -> "ResilientSession":
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args) -> None:
        if self._session:
            await self._session.close()

    def get_rate_limiter(self, endpoint: str, rps: float = 5.0) -> RateLimiter:
        if endpoint not in self._rate_limiters:
            self._rate_limiters[endpoint] = RateLimiter(rate=rps, burst=2)
        return self._rate_limiters[endpoint]

    def get_circuit_breaker(self, endpoint: str) -> CircuitBreaker:
        if endpoint not in self._circuit_breakers:
            self._circuit_breakers[endpoint] = CircuitBreaker(
                failure_threshold=self.config.circuit_breaker_threshold,
                reset_timeout=self.config.circuit_breaker_reset_seconds,
                name=endpoint,
            )
        return self._circuit_breakers[endpoint]

    def _random_user_agent(self) -> str:
        return random.choice(USER_AGENTS)

    @retry_with_backoff()
    async def request(
        self,
        method: str,
        url: str,
        *,
        endpoint_name: str = "default",
        rate_limit_rps: float = 5.0,
        headers: Optional[dict] = None,
        params: Optional[dict] = None,
        json_body: Optional[dict] = None,
        timeout: int = 30,
    ) -> dict | list:
        """
        Make a resilient HTTP request with rate limiting, circuit breaking, and retries.

        Returns parsed JSON response.
        """
        if not self._session:
            self._session = aiohttp.ClientSession()

        # Circuit breaker check
        cb = self.get_circuit_breaker(endpoint_name)
        if cb.is_open:
            raise CircuitOpenError(endpoint_name)

        # Rate limiting
        rl = self.get_rate_limiter(endpoint_name, rate_limit_rps)
        await rl.acquire()

        # Build headers
        req_headers = {"User-Agent": self._random_user_agent()}
        if headers:
            req_headers.update(headers)

        # Proxy
        proxy = self.config.proxy_url

        try:
            async with self._session.request(
                method,
                url,
                headers=req_headers,
                params=params,
                json=json_body,
                proxy=proxy,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status in (429, 500, 502, 503, 504):
                    text = await resp.text()
                    cb.record_failure()
                    raise RetryableError(resp.status, text[:200])

                if resp.status >= 400:
                    text = await resp.text()
                    logger.error(
                        "HTTP %d for %s %s: %s",
                        resp.status,
                        method,
                        url,
                        text[:200],
                    )
                    # 4xx (except 429) are client errors - do NOT retry
                    raise NonRetryableHttpError(resp.status, text[:200])

                cb.record_success()
                return await resp.json(content_type=None)

        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            cb.record_failure()
            raise
