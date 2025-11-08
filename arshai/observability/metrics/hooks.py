"""
OpenTelemetry integration hooks for metrics export.
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class OpenTelemetryHooks:
    """
    Hooks for integrating with OpenTelemetry.

    This class provides integration points for exporting metrics
    to OpenTelemetry-compatible backends.

    Example:
        hooks = OpenTelemetryHooks()

        # Configure exporter (requires opentelemetry packages)
        if hooks.is_available():
            hooks.configure_metrics_exporter(
                endpoint="http://localhost:4317",
                service_name="arshai-service"
            )

            # Export metrics
            hooks.export_metrics(collector.get_stats())
    """

    def __init__(self):
        """Initialize OpenTelemetry hooks."""
        self.otel_available = self._check_opentelemetry()
        self.meter = None
        self.metrics_exporter = None

    def _check_opentelemetry(self) -> bool:
        """Check if OpenTelemetry is available."""
        try:
            import opentelemetry
            return True
        except ImportError:
            logger.debug("OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk")
            return False

    def is_available(self) -> bool:
        """
        Check if OpenTelemetry is available.

        Returns:
            True if OpenTelemetry packages are installed
        """
        return self.otel_available

    def configure_metrics_exporter(
        self,
        endpoint: str = "http://localhost:4317",
        service_name: str = "arshai",
        insecure: bool = True
    ):
        """
        Configure OpenTelemetry metrics exporter.

        Args:
            endpoint: OTLP endpoint URL
            service_name: Service name for metrics
            insecure: Whether to use insecure connection

        Note:
            Requires opentelemetry packages to be installed
        """
        if not self.otel_available:
            logger.warning("OpenTelemetry not available, cannot configure exporter")
            return

        try:
            from opentelemetry import metrics
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

            # Create resource
            resource = Resource.create({
                "service.name": service_name,
                "service.version": "1.3.0",
            })

            # Create exporter
            self.metrics_exporter = OTLPMetricExporter(
                endpoint=endpoint,
                insecure=insecure
            )

            # Create meter provider
            meter_provider = MeterProvider(
                resource=resource,
                metric_exporters=[self.metrics_exporter]
            )

            # Set global meter provider
            metrics.set_meter_provider(meter_provider)

            # Get meter
            self.meter = metrics.get_meter(service_name)

            logger.info(f"OpenTelemetry metrics configured for {service_name} at {endpoint}")

        except Exception as e:
            logger.error(f"Failed to configure OpenTelemetry: {e}")

    def export_metrics(self, metrics_data: Dict[str, Any]):
        """
        Export metrics to OpenTelemetry.

        Args:
            metrics_data: Metrics data from collector

        Note:
            This is a simplified example. In production, you would
            create proper OpenTelemetry instruments (counters, histograms, etc.)
        """
        if not self.meter:
            logger.debug("OpenTelemetry meter not configured")
            return

        try:
            # Example: Export counters
            for name, value in metrics_data.get("counters", {}).items():
                counter = self.meter.create_counter(
                    name=name.replace(",", "_"),  # OpenTelemetry doesn't like commas
                    description=f"Counter for {name}"
                )
                counter.add(int(value))

            # Example: Export gauges
            for name, value in metrics_data.get("gauges", {}).items():
                gauge = self.meter.create_up_down_counter(
                    name=name.replace(",", "_"),
                    description=f"Gauge for {name}"
                )
                gauge.add(int(value))

            logger.debug("Metrics exported to OpenTelemetry")

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

    def create_span_processor(self):
        """
        Create a span processor for tracing.

        Returns:
            Span processor or None if not available

        Note:
            This is for future tracing support
        """
        if not self.otel_available:
            return None

        try:
            from opentelemetry.sdk.trace import export
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter()
            return export.BatchSpanProcessor(exporter)

        except Exception as e:
            logger.error(f"Failed to create span processor: {e}")
            return None