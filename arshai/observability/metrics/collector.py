"""
Metrics collection for performance monitoring.
"""

from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import time
import threading
from collections import defaultdict, deque
import statistics


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


class MetricsCollector:
    """
    Thread-safe metrics collector for Arshai components.

    Example:
        collector = MetricsCollector()

        # Count events
        collector.increment("api.calls", tags={"endpoint": "/process"})

        # Record timing
        with collector.timer("processing.time"):
            # Do work
            pass

        # Record value
        collector.gauge("queue.size", 42)

        # Get metrics
        stats = collector.get_stats()
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector.

        Args:
            max_history: Maximum number of data points to keep per metric
        """
        self._lock = threading.Lock()
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._active_timers: Dict[str, float] = {}
        self.max_history = max_history

    def increment(self, name: str, value: float = 1, tags: Optional[Dict[str, str]] = None):
        """
        Increment a counter metric.

        Args:
            name: Metric name
            value: Amount to increment
            tags: Optional tags for the metric
        """
        with self._lock:
            key = self._make_key(name, tags)
            self._counters[key] += value

    def decrement(self, name: str, value: float = 1, tags: Optional[Dict[str, str]] = None):
        """
        Decrement a counter metric.

        Args:
            name: Metric name
            value: Amount to decrement
            tags: Optional tags for the metric
        """
        self.increment(name, -value, tags)

    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Set a gauge metric value.

        Args:
            name: Metric name
            value: Current value
            tags: Optional tags for the metric
        """
        with self._lock:
            key = self._make_key(name, tags)
            self._gauges[key] = value

    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Add a value to a histogram metric.

        Args:
            name: Metric name
            value: Value to record
            tags: Optional tags for the metric
        """
        with self._lock:
            key = self._make_key(name, tags)
            self._histograms[key].append(value)

    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """
        Context manager for timing operations.

        Args:
            name: Metric name
            tags: Optional tags for the metric

        Example:
            with collector.timer("db.query", tags={"query": "select"}):
                # Perform query
                pass
        """
        return TimerContext(self, name, tags)

    def start_timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """
        Start a timer manually.

        Args:
            name: Metric name
            tags: Optional tags for the metric
        """
        with self._lock:
            key = self._make_key(name, tags)
            self._active_timers[key] = time.time()

    def stop_timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """
        Stop a manually started timer.

        Args:
            name: Metric name
            tags: Optional tags for the metric

        Returns:
            Elapsed time in seconds
        """
        with self._lock:
            key = self._make_key(name, tags)
            if key in self._active_timers:
                start_time = self._active_timers.pop(key)
                elapsed = time.time() - start_time
                self._timers[key].append(elapsed)
                if len(self._timers[key]) > self.max_history:
                    self._timers[key] = self._timers[key][-self.max_history:]
                return elapsed
        return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics for all metrics.

        Returns:
            Dictionary with metric statistics
        """
        with self._lock:
            stats = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {},
                "timers": {}
            }

            # Calculate histogram statistics
            for key, values in self._histograms.items():
                if values:
                    stats["histograms"][key] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "stddev": statistics.stdev(values) if len(values) > 1 else 0
                    }

            # Calculate timer statistics
            for key, times in self._timers.items():
                if times:
                    stats["timers"][key] = {
                        "count": len(times),
                        "min": min(times),
                        "max": max(times),
                        "mean": statistics.mean(times),
                        "median": statistics.median(times),
                        "p95": self._percentile(times, 0.95),
                        "p99": self._percentile(times, 0.99)
                    }

            return stats

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timers.clear()
            self._active_timers.clear()

    def _make_key(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Create a unique key for a metric with tags."""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name},{tag_str}"

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]


class TimerContext:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, name: str, tags: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed = time.time() - self.start_time
            key = self.collector._make_key(self.name, self.tags)
            with self.collector._lock:
                self.collector._timers[key].append(elapsed)
                if len(self.collector._timers[key]) > self.collector.max_history:
                    self.collector._timers[key] = self.collector._timers[key][-self.collector.max_history:]