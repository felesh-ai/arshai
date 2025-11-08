"""
Circuit breaker plugin for resilient external service calls.
"""

from typing import Optional, Any, Callable, TypeVar, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import threading
from dataclasses import dataclass
import time

from arshai.extensions.base import Plugin
from arshai.observability.metrics import MetricsCollector

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    expected_exception: type = Exception
    name: Optional[str] = None


class CircuitBreakerError(Exception):
    """Raised when circuit is open."""
    pass


class CircuitBreakerPlugin(Plugin):
    """
    Circuit breaker plugin for resilient external service calls.

    This is an OPTIONAL plugin that adds resilience patterns without
    affecting existing Arshai functionality.

    Example:
        # Optional usage - existing code continues to work
        cb = CircuitBreakerPlugin(config)

        # Wrap async calls
        result = await cb.call_async(external_service.fetch_data, param1, param2)

        # Wrap sync calls
        result = cb.call(external_service.fetch_data_sync, param1, param2)

        # Decorator pattern (opt-in)
        @cb.protected
        async def my_external_call():
            return await external_api.request()
    """

    name = "circuit_breaker"
    version = "1.0.0"

    def __init__(self, config: Optional[CircuitBreakerConfig] = None, enable_metrics: bool = True):
        """
        Initialize circuit breaker with optional config.

        Args:
            config: Optional configuration. Defaults to sensible values.
            enable_metrics: Whether to enable performance metrics collection.
        """
        super().__init__()
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = threading.Lock()

        # Performance metrics
        self.enable_metrics = enable_metrics
        self.metrics = MetricsCollector() if enable_metrics else None
        self._total_calls = 0
        self._successful_calls = 0
        self._failed_calls = 0
        self._rejected_calls = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if self._last_failure_time is None:
            return False

        return datetime.now() - self._last_failure_time > timedelta(
            seconds=self.config.recovery_timeout
        )

    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            self._failure_count = 0
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._successful_calls += 1

            # Record metrics
            if self.enable_metrics and self.metrics:
                self.metrics.increment("circuit_breaker.calls.success", tags={"name": self.config.name})
                if old_state == CircuitState.HALF_OPEN:
                    self.metrics.increment("circuit_breaker.recovery.success", tags={"name": self.config.name})

    def _on_failure(self) -> None:
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            self._failed_calls += 1

            if self._failure_count >= self.config.failure_threshold:
                old_state = self._state
                self._state = CircuitState.OPEN

                # Record metrics for state transition
                if self.enable_metrics and self.metrics and old_state != CircuitState.OPEN:
                    self.metrics.increment("circuit_breaker.state.open", tags={"name": self.config.name})

            # Record metrics
            if self.enable_metrics and self.metrics:
                self.metrics.increment("circuit_breaker.calls.failure", tags={"name": self.config.name})

    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute async function with circuit breaker protection.

        Args:
            func: Async function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If function fails
        """
        start_time = time.time() if self.enable_metrics else None

        # Check circuit state
        with self._lock:
            self._total_calls += 1
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    if self.enable_metrics and self.metrics:
                        self.metrics.increment("circuit_breaker.state.half_open", tags={"name": self.config.name})
                else:
                    self._rejected_calls += 1
                    if self.enable_metrics and self.metrics:
                        self.metrics.increment("circuit_breaker.calls.rejected", tags={"name": self.config.name})
                    raise CircuitBreakerError(
                        f"Circuit breaker is open for {self.config.name or func.__name__}"
                    )

        # Attempt call
        try:
            result = await func(*args, **kwargs)
            self._on_success()

            # Record timing metrics
            if self.enable_metrics and self.metrics and start_time:
                elapsed = time.time() - start_time
                self.metrics.timer("circuit_breaker.call.duration", tags={"name": self.config.name, "status": "success"})
                self.metrics.histogram("circuit_breaker.call.latency", elapsed, tags={"name": self.config.name})

            return result

        except self.config.expected_exception as e:
            self._on_failure()

            # Record timing metrics
            if self.enable_metrics and self.metrics and start_time:
                elapsed = time.time() - start_time
                self.metrics.timer("circuit_breaker.call.duration", tags={"name": self.config.name, "status": "failure"})

            raise

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute sync function with circuit breaker protection.

        Args:
            func: Sync function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If function fails
        """
        # Check circuit state
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker is open for {self.config.name or func.__name__}"
                    )

        # Attempt call
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.config.expected_exception as e:
            self._on_failure()
            raise

    def protected(self, func: Callable) -> Callable:
        """
        Decorator to protect functions with circuit breaker.

        Works with both sync and async functions.

        Example:
            @circuit_breaker.protected
            async def external_call():
                return await api.request()
        """
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                return await self.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                return self.call(func, *args, **kwargs)
            return sync_wrapper

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None

    def get_stats(self) -> dict:
        """Get circuit breaker statistics including performance metrics."""
        with self._lock:
            stats = {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "last_failure": self._last_failure_time.isoformat() if self._last_failure_time else None,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                },
                "performance": {
                    "total_calls": self._total_calls,
                    "successful_calls": self._successful_calls,
                    "failed_calls": self._failed_calls,
                    "rejected_calls": self._rejected_calls,
                    "success_rate": self._successful_calls / self._total_calls if self._total_calls > 0 else 0.0,
                    "failure_rate": self._failed_calls / self._total_calls if self._total_calls > 0 else 0.0,
                    "rejection_rate": self._rejected_calls / self._total_calls if self._total_calls > 0 else 0.0,
                }
            }

            # Add detailed metrics if available
            if self.enable_metrics and self.metrics:
                stats["metrics"] = self.metrics.get_stats()

            return stats

    def get_metadata(self) -> dict:
        """Get plugin metadata (required by Plugin base class)."""
        return {
            "name": self.name,
            "version": self.version,
            "description": "Circuit breaker plugin for resilient external service calls",
            "author": "Arshai Team",
            "tags": ["resilience", "circuit-breaker", "fault-tolerance"]
        }

    def initialize(self) -> None:
        """Initialize the plugin (required by Plugin base class)."""
        # No special initialization required
        pass

    def shutdown(self) -> None:
        """Shutdown the plugin (required by Plugin base class)."""
        # Reset state on shutdown
        self.reset()