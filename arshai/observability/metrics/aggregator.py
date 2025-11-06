"""
Metrics aggregation utilities for combining metrics from multiple collectors.
"""

from typing import Dict, Any, List, Optional
from .collector import MetricsCollector


class MetricsAggregator:
    """
    Aggregates metrics from multiple collectors.

    Example:
        aggregator = MetricsAggregator()

        # Add collectors
        aggregator.add_collector("circuit_breaker", cb.metrics)
        aggregator.add_collector("batch_processor", bp.metrics)

        # Get aggregated stats
        all_stats = aggregator.get_aggregated_stats()
    """

    def __init__(self):
        """Initialize metrics aggregator."""
        self.collectors: Dict[str, MetricsCollector] = {}

    def add_collector(self, name: str, collector: MetricsCollector):
        """
        Add a metrics collector.

        Args:
            name: Name for this collector
            collector: MetricsCollector instance
        """
        self.collectors[name] = collector

    def remove_collector(self, name: str):
        """
        Remove a metrics collector.

        Args:
            name: Name of collector to remove
        """
        if name in self.collectors:
            del self.collectors[name]

    def get_aggregated_stats(self) -> Dict[str, Any]:
        """
        Get aggregated statistics from all collectors.

        Returns:
            Dictionary with stats from all collectors
        """
        aggregated = {}

        for name, collector in self.collectors.items():
            aggregated[name] = collector.get_stats()

        # Add summary statistics
        aggregated["_summary"] = self._calculate_summary(aggregated)

        return aggregated

    def _calculate_summary(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate summary statistics across all collectors.

        Args:
            stats: Individual collector statistics

        Returns:
            Summary statistics
        """
        summary = {
            "total_collectors": len(self.collectors),
            "total_counters": 0,
            "total_gauges": 0,
            "total_histograms": 0,
            "total_timers": 0
        }

        for collector_stats in stats.values():
            if isinstance(collector_stats, dict):
                summary["total_counters"] += len(collector_stats.get("counters", {}))
                summary["total_gauges"] += len(collector_stats.get("gauges", {}))
                summary["total_histograms"] += len(collector_stats.get("histograms", {}))
                summary["total_timers"] += len(collector_stats.get("timers", {}))

        return summary

    def reset_all(self):
        """Reset all collectors."""
        for collector in self.collectors.values():
            collector.reset()

    def get_collector(self, name: str) -> Optional[MetricsCollector]:
        """
        Get a specific collector by name.

        Args:
            name: Collector name

        Returns:
            MetricsCollector or None if not found
        """
        return self.collectors.get(name)