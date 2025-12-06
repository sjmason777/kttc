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

"""Circuit Breaker pattern implementation for LLM providers.

Prevents cascading failures by temporarily disabling providers that are
experiencing errors. The circuit breaker has three states:

- CLOSED: Normal operation, requests are allowed
- OPEN: Provider is failing, requests are blocked
- HALF_OPEN: Testing if provider has recovered

State transitions:
    CLOSED → OPEN: After `failure_threshold` consecutive failures
    OPEN → HALF_OPEN: After `recovery_timeout` seconds
    HALF_OPEN → CLOSED: After `success_threshold` successful requests
    HALF_OPEN → OPEN: On any failure

References:
    - https://martinfowler.com/bliki/CircuitBreaker.html
    - https://resilience4j.readme.io/docs/circuitbreaker
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring.

    Attributes:
        total_calls: Total number of calls attempted
        successful_calls: Number of successful calls
        failed_calls: Number of failed calls
        rejected_calls: Number of calls rejected due to open circuit
        state_changes: Number of state transitions
        last_failure_time: Timestamp of last failure
        last_success_time: Timestamp of last success
        consecutive_failures: Current count of consecutive failures
        consecutive_successes: Current count of consecutive successes in HALF_OPEN
    """

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as percentage."""
        total = self.successful_calls + self.failed_calls
        if total == 0:
            return 0.0
        return (self.failed_calls / total) * 100

    def reset(self) -> None:
        """Reset all metrics."""
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.rejected_calls = 0
        self.state_changes = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.consecutive_failures = 0
        self.consecutive_successes = 0


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of failures before opening circuit (default: 5)
        success_threshold: Successes needed in HALF_OPEN to close (default: 2)
        recovery_timeout: Seconds before transitioning OPEN → HALF_OPEN (default: 60)
        half_open_max_calls: Max concurrent calls allowed in HALF_OPEN (default: 3)
        excluded_exceptions: Exception types that don't count as failures
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    excluded_exceptions: tuple[type[Exception], ...] = field(default_factory=tuple)


class CircuitBreaker:
    """Circuit breaker for LLM provider resilience.

    Implements the circuit breaker pattern to prevent cascading failures
    when an LLM provider is experiencing issues.

    Example:
        >>> breaker = CircuitBreaker(name="openai")
        >>>
        >>> async def call_openai(prompt: str) -> str:
        ...     async with breaker:
        ...         return await openai_provider.complete(prompt)
        >>>
        >>> # Or use the decorator
        >>> @breaker.protect
        ... async def call_openai(prompt: str) -> str:
        ...     return await openai_provider.complete(prompt)

    Thread Safety:
        This implementation is async-safe but not thread-safe.
        Use separate instances for different threads.
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            name: Identifier for this circuit breaker (e.g., provider name)
            config: Configuration options (uses defaults if None)
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._state_changed_at = time.monotonic()
        self._metrics = CircuitBreakerMetrics()
        self._half_open_semaphore = asyncio.Semaphore(self.config.half_open_max_calls)
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics."""
        return self._metrics

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._state_changed_at = time.monotonic()
        self._metrics.state_changes += 1

        if new_state == CircuitState.HALF_OPEN:
            self._metrics.consecutive_successes = 0

        logger.info(
            f"Circuit breaker '{self.name}' state change: {old_state.value} → {new_state.value}"
        )

    async def _check_state_transition(self) -> None:
        """Check if state should transition based on current conditions."""
        now = time.monotonic()

        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            time_in_state = now - self._state_changed_at
            if time_in_state >= self.config.recovery_timeout:
                await self._transition_to(CircuitState.HALF_OPEN)

    def can_execute(self) -> bool:
        """Check if a request can be executed.

        Returns:
            True if request is allowed, False otherwise
        """
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if we should transition to HALF_OPEN
            time_in_state = time.monotonic() - self._state_changed_at
            if time_in_state >= self.config.recovery_timeout:
                return True  # Will transition in __aenter__
            return False

        # HALF_OPEN: allow limited requests
        return True

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._metrics.successful_calls += 1
            self._metrics.last_success_time = time.time()
            self._metrics.consecutive_failures = 0

            if self._state == CircuitState.HALF_OPEN:
                self._metrics.consecutive_successes += 1
                if self._metrics.consecutive_successes >= self.config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)
                    logger.info(
                        f"Circuit breaker '{self.name}' recovered after "
                        f"{self._metrics.consecutive_successes} successful calls"
                    )

    async def record_failure(self, exception: Exception | None = None) -> None:
        """Record a failed call.

        Args:
            exception: The exception that caused the failure (for filtering)
        """
        # Check if exception should be excluded
        if exception and isinstance(exception, self.config.excluded_exceptions):
            logger.debug(
                f"Circuit breaker '{self.name}' ignoring excluded exception: "
                f"{type(exception).__name__}"
            )
            return

        async with self._lock:
            self._metrics.failed_calls += 1
            self._metrics.last_failure_time = time.time()
            self._metrics.consecutive_failures += 1
            self._metrics.consecutive_successes = 0

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in HALF_OPEN reopens the circuit
                await self._transition_to(CircuitState.OPEN)
                logger.warning(
                    f"Circuit breaker '{self.name}' reopened due to failure in HALF_OPEN"
                )

            elif self._state == CircuitState.CLOSED:
                if self._metrics.consecutive_failures >= self.config.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)
                    logger.warning(
                        f"Circuit breaker '{self.name}' opened after "
                        f"{self._metrics.consecutive_failures} consecutive failures"
                    )

    async def __aenter__(self) -> CircuitBreaker:
        """Enter async context manager."""
        self._metrics.total_calls += 1

        async with self._lock:
            await self._check_state_transition()

        if self._state == CircuitState.OPEN:
            self._metrics.rejected_calls += 1
            raise CircuitOpenError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Retry after {self.config.recovery_timeout}s"
            )

        if self._state == CircuitState.HALF_OPEN:
            # Limit concurrent calls in HALF_OPEN
            acquired = self._half_open_semaphore.locked()
            if acquired and self._half_open_semaphore._value == 0:
                self._metrics.rejected_calls += 1
                raise CircuitOpenError(
                    f"Circuit breaker '{self.name}' is HALF_OPEN with max calls reached"
                )
            await self._half_open_semaphore.acquire()

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> bool:
        """Exit async context manager."""
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_semaphore.release()

        if exc_type is None:
            await self.record_success()
        else:
            await self.record_failure(exc_val if isinstance(exc_val, Exception) else None)

        return False  # Don't suppress exceptions

    def protect(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to protect a function with circuit breaker.

        Args:
            func: Async function to protect

        Returns:
            Wrapped function with circuit breaker protection
        """

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with self:
                return await func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._state_changed_at = time.monotonic()
        self._metrics.reset()
        self._half_open_semaphore = asyncio.Semaphore(self.config.half_open_max_calls)
        logger.info(f"Circuit breaker '{self.name}' reset to CLOSED state")

    def get_status(self) -> dict[str, Any]:
        """Get detailed status for monitoring.

        Returns:
            Dictionary with current state and metrics
        """
        return {
            "name": self.name,
            "state": self._state.value,
            "metrics": {
                "total_calls": self._metrics.total_calls,
                "successful_calls": self._metrics.successful_calls,
                "failed_calls": self._metrics.failed_calls,
                "rejected_calls": self._metrics.rejected_calls,
                "failure_rate": f"{self._metrics.failure_rate:.1f}%",
                "consecutive_failures": self._metrics.consecutive_failures,
                "consecutive_successes": self._metrics.consecutive_successes,
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "recovery_timeout": self.config.recovery_timeout,
            },
            "time_in_current_state": time.monotonic() - self._state_changed_at,
        }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and request is rejected."""
