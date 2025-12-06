# Copyright 2025 KTTC AI (https://github.com/kttc-ai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Token-aware rate limiting for LLM API calls.

Unlike traditional request-per-second (RPS) rate limiting, LLM APIs require
token-aware limiting because:
1. LLMs are resource-intensive and token-based
2. Computational load varies significantly per request
3. Providers charge and limit by tokens, not just requests

This module implements:
- TokenBucket: Classic token bucket algorithm for RPM limits
- TokenAwareRateLimiter: Combined RPM + TPM limiting with sliding window

References:
    - https://www.truefoundry.com/blog/rate-limiting-in-llm-gateway
    - https://docs.litellm.ai/docs/routing
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RateLimitMetrics:
    """Metrics for rate limiter monitoring.

    Attributes:
        total_requests: Total requests processed
        total_tokens: Total tokens consumed
        throttled_requests: Requests that had to wait
        rejected_requests: Requests rejected due to limits
        average_wait_time: Average wait time in seconds
    """

    total_requests: int = 0
    total_tokens: int = 0
    throttled_requests: int = 0
    rejected_requests: int = 0
    total_wait_time: float = 0.0

    @property
    def average_wait_time(self) -> float:
        """Calculate average wait time."""
        if self.throttled_requests == 0:
            return 0.0
        return self.total_wait_time / self.throttled_requests

    def reset(self) -> None:
        """Reset all metrics."""
        self.total_requests = 0
        self.total_tokens = 0
        self.throttled_requests = 0
        self.rejected_requests = 0
        self.total_wait_time = 0.0


class TokenBucket:
    """Token bucket algorithm for rate limiting.

    Classic token bucket that refills at a constant rate.
    Used for requests-per-minute (RPM) limiting.

    Example:
        >>> bucket = TokenBucket(capacity=60, refill_rate=1.0)  # 60 RPM
        >>> if bucket.consume(1):
        ...     # Make request
        ...     pass
    """

    def __init__(
        self,
        capacity: int,
        refill_rate: float,
    ) -> None:
        """Initialize token bucket.

        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def tokens(self) -> float:
        """Get current token count (without refilling)."""
        return self._tokens

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        tokens_to_add = elapsed * self.refill_rate
        self._tokens = min(self.capacity, self._tokens + tokens_to_add)
        self._last_refill = now

    async def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if not enough
        """
        async with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def wait_and_consume(self, tokens: int = 1, timeout: float | None = None) -> bool:
        """Wait until tokens are available, then consume.

        Args:
            tokens: Number of tokens to consume
            timeout: Maximum wait time in seconds

        Returns:
            True if tokens were consumed, False if timeout
        """
        start = time.monotonic()

        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

                # Calculate wait time
                needed = tokens - self._tokens
                wait_time = needed / self.refill_rate

            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start
                remaining = timeout - elapsed
                if remaining <= 0:
                    return False
                wait_time = min(wait_time, remaining)

            await asyncio.sleep(min(wait_time, 0.1))  # Poll at most every 100ms

    def reset(self) -> None:
        """Reset bucket to full capacity."""
        self._tokens = float(self.capacity)
        self._last_refill = time.monotonic()


@dataclass
class TokenWindow:
    """Sliding window entry for token tracking."""

    timestamp: float
    tokens: int


class TokenAwareRateLimiter:
    """Combined RPM and TPM rate limiter with sliding window.

    Implements dual rate limiting:
    1. Requests per minute (RPM) - using token bucket
    2. Tokens per minute (TPM) - using sliding window

    The limiter also uses semaphore for concurrency control,
    with timeout starting AFTER acquiring the semaphore (critical for aiohttp).

    Example:
        >>> limiter = TokenAwareRateLimiter(
        ...     name="openai",
        ...     rpm_limit=60,
        ...     tpm_limit=90000,
        ...     max_concurrent=10
        ... )
        >>>
        >>> async with limiter.acquire(estimated_tokens=1000):
        ...     response = await provider.complete(prompt)
        ...     limiter.report_actual_tokens(response_tokens)
    """

    def __init__(
        self,
        name: str,
        rpm_limit: int = 60,
        tpm_limit: int = 90000,
        max_concurrent: int = 10,
        window_size: float = 60.0,
    ) -> None:
        """Initialize rate limiter.

        Args:
            name: Identifier for this limiter (e.g., provider name)
            rpm_limit: Maximum requests per minute
            tpm_limit: Maximum tokens per minute
            max_concurrent: Maximum concurrent requests
            window_size: Sliding window size in seconds (default: 60)
        """
        self.name = name
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.max_concurrent = max_concurrent
        self.window_size = window_size

        # RPM limiting via token bucket
        self._rpm_bucket = TokenBucket(
            capacity=rpm_limit,
            refill_rate=rpm_limit / 60.0,  # tokens per second
        )

        # TPM limiting via sliding window
        self._token_window: deque[TokenWindow] = deque()
        self._window_tokens = 0
        self._window_lock = asyncio.Lock()

        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Metrics
        self._metrics = RateLimitMetrics()

        # Pending token updates (estimated vs actual)
        self._pending_tokens: dict[int, int] = {}
        self._request_counter = 0

    @property
    def metrics(self) -> RateLimitMetrics:
        """Get rate limiter metrics."""
        return self._metrics

    @property
    def concurrent_available(self) -> int:
        """Get number of available concurrent request slots."""
        return self._semaphore._value  # pylint: disable=protected-access

    async def _clean_window(self) -> None:
        """Remove expired entries from sliding window."""
        now = time.monotonic()
        cutoff = now - self.window_size

        while self._token_window and self._token_window[0].timestamp < cutoff:
            entry = self._token_window.popleft()
            self._window_tokens -= entry.tokens

    async def _add_to_window(self, tokens: int) -> None:
        """Add tokens to sliding window."""
        now = time.monotonic()
        self._token_window.append(TokenWindow(timestamp=now, tokens=tokens))
        self._window_tokens += tokens

    async def _can_consume_tokens(self, tokens: int) -> bool:
        """Check if tokens can be consumed within TPM limit."""
        async with self._window_lock:
            await self._clean_window()
            return (self._window_tokens + tokens) <= self.tpm_limit

    async def _wait_for_token_capacity(
        self,
        tokens: int,
        timeout: float | None = None,
    ) -> bool:
        """Wait until token capacity is available.

        Args:
            tokens: Estimated tokens for request
            timeout: Maximum wait time

        Returns:
            True if capacity available, False if timeout
        """
        start = time.monotonic()

        while True:
            async with self._window_lock:
                await self._clean_window()
                available = self.tpm_limit - self._window_tokens

                if available >= tokens:
                    await self._add_to_window(tokens)
                    return True

                # Calculate wait time (oldest entry expiry)
                if self._token_window:
                    oldest = self._token_window[0]
                    wait_time = (oldest.timestamp + self.window_size) - time.monotonic()
                    wait_time = max(0.01, wait_time)
                else:
                    wait_time = 0.1

            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start
                remaining = timeout - elapsed
                if remaining <= 0:
                    return False
                wait_time = min(wait_time, remaining)

            self._metrics.throttled_requests += 1
            self._metrics.total_wait_time += wait_time

            await asyncio.sleep(wait_time)

    def acquire(
        self,
        estimated_tokens: int = 1000,
        timeout: float = 120.0,
    ) -> RateLimitContext:
        """Acquire rate limit slot.

        Args:
            estimated_tokens: Estimated tokens for this request
            timeout: Maximum wait time

        Returns:
            Context manager for the request

        Raises:
            RateLimitExceededError: If timeout reached
        """
        return RateLimitContext(self, estimated_tokens, timeout)

    async def _acquire_impl(
        self,
        estimated_tokens: int,
        timeout: float,
    ) -> int:
        """Internal acquire implementation.

        Returns:
            Request ID for token reporting
        """
        start = time.monotonic()

        # 1. Acquire semaphore (concurrency limit)
        # IMPORTANT: Timeout starts AFTER semaphore acquisition
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=timeout,
            )
        except TimeoutError as exc:
            self._metrics.rejected_requests += 1
            raise RateLimitExceededError(
                f"Rate limiter '{self.name}' concurrency timeout after {timeout}s"
            ) from exc

        elapsed = time.monotonic() - start
        remaining_timeout = timeout - elapsed

        try:
            # 2. Wait for RPM capacity
            if not await self._rpm_bucket.wait_and_consume(1, timeout=remaining_timeout):
                self._semaphore.release()
                self._metrics.rejected_requests += 1
                raise RateLimitExceededError(f"Rate limiter '{self.name}' RPM limit timeout")

            elapsed = time.monotonic() - start
            remaining_timeout = timeout - elapsed

            # 3. Wait for TPM capacity
            if not await self._wait_for_token_capacity(estimated_tokens, remaining_timeout):
                self._semaphore.release()
                self._metrics.rejected_requests += 1
                raise RateLimitExceededError(f"Rate limiter '{self.name}' TPM limit timeout")

            # Track this request
            self._request_counter += 1
            request_id = self._request_counter
            self._pending_tokens[request_id] = estimated_tokens
            self._metrics.total_requests += 1

            return request_id

        except Exception:
            self._semaphore.release()
            raise

    async def _release_impl(self, request_id: int, actual_tokens: int | None = None) -> None:
        """Internal release implementation.

        Args:
            request_id: Request ID from acquire
            actual_tokens: Actual tokens used (None to use estimate)
        """
        self._semaphore.release()

        estimated = self._pending_tokens.pop(request_id, 0)
        actual = actual_tokens if actual_tokens is not None else estimated

        self._metrics.total_tokens += actual

        # Adjust window if actual differs significantly from estimate
        if actual_tokens is not None and actual != estimated:
            async with self._window_lock:
                # Simple adjustment: add difference
                diff = actual - estimated
                if diff != 0:
                    await self._add_to_window(diff)

    def get_status(self) -> dict[str, Any]:
        """Get detailed status for monitoring."""
        return {
            "name": self.name,
            "limits": {
                "rpm": self.rpm_limit,
                "tpm": self.tpm_limit,
                "max_concurrent": self.max_concurrent,
            },
            "current": {
                "rpm_tokens_available": self._rpm_bucket.tokens,
                "tpm_tokens_used": self._window_tokens,
                "tpm_tokens_available": self.tpm_limit - self._window_tokens,
                "concurrent_requests": self.max_concurrent - self.concurrent_available,
            },
            "metrics": {
                "total_requests": self._metrics.total_requests,
                "total_tokens": self._metrics.total_tokens,
                "throttled_requests": self._metrics.throttled_requests,
                "rejected_requests": self._metrics.rejected_requests,
                "average_wait_time": f"{self._metrics.average_wait_time:.3f}s",
            },
        }

    def reset(self) -> None:
        """Reset rate limiter state."""
        self._rpm_bucket.reset()
        self._token_window.clear()
        self._window_tokens = 0
        self._pending_tokens.clear()
        self._metrics.reset()
        logger.info(f"Rate limiter '{self.name}' reset")


class RateLimitContext:
    """Context manager for rate-limited requests."""

    def __init__(
        self,
        limiter: TokenAwareRateLimiter,
        estimated_tokens: int,
        timeout: float,
    ) -> None:
        self._limiter = limiter
        self._estimated_tokens = estimated_tokens
        self._timeout = timeout
        self._request_id: int | None = None
        self._actual_tokens: int | None = None

    def report_tokens(self, tokens: int) -> None:
        """Report actual tokens used.

        Call this before exiting context if actual tokens differ from estimate.
        """
        self._actual_tokens = tokens

    async def __aenter__(self) -> RateLimitContext:
        """Enter context and acquire rate limit slot."""
        self._request_id = await self._limiter._acquire_impl(
            self._estimated_tokens,
            self._timeout,
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> bool:
        """Exit context and release rate limit slot."""
        if self._request_id is not None:
            await self._limiter._release_impl(self._request_id, self._actual_tokens)
        return False


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded and timeout reached."""
