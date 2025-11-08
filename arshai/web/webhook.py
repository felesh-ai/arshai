"""
Webhook handling utilities for Arshai.
"""

import hmac
import hashlib
import json
from typing import Optional, Dict, Any, Callable, Union
from dataclasses import dataclass
import logging
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class WebhookConfig:
    """Configuration for webhook handler."""
    secret: Optional[str] = None
    signature_header: str = "X-Signature"
    signature_prefix: str = ""  # e.g., "sha256=" for GitHub
    hash_algorithm: str = "sha256"
    max_body_size: int = 10 * 1024 * 1024  # 10MB default


class WebhookValidationError(Exception):
    """Raised when webhook validation fails."""
    pass


class WebhookHandler:
    """
    Generic webhook handler with signature verification.

    This is an OPTIONAL utility for handling webhooks. Existing
    Arshai applications are not affected by this addition.

    Example:
        # Create handler with secret
        handler = WebhookHandler(WebhookConfig(secret="my-secret"))

        # In FastAPI/Flask route
        @app.post("/webhook")
        async def handle_webhook(request: Request):
            # Verify and process
            data = await handler.process_request(request)

            # Use with Arshai workflow
            workflow = MyWorkflow()
            result = await workflow.execute(IWorkflowState(data=data))
            return result

    Extending for specific platforms (in your application code):
        class GitHubWebhookHandler(WebhookHandler):
            def __init__(self, secret: str):
                super().__init__(WebhookConfig(
                    secret=secret,
                    signature_header="X-Hub-Signature-256",
                    signature_prefix="sha256=",
                    hash_algorithm="sha256"
                ))

            def is_pull_request_event(self, data: Dict[str, Any]) -> bool:
                return "pull_request" in data
    """

    def __init__(self, config: Optional[WebhookConfig] = None):
        """
        Initialize webhook handler.

        Args:
            config: Optional configuration
        """
        self.config = config or WebhookConfig()

    def verify_signature(self, body: bytes, signature: str) -> bool:
        """
        Verify webhook signature.

        Args:
            body: Raw request body
            signature: Signature from header

        Returns:
            True if signature is valid
        """
        if not self.config.secret:
            logger.warning("No secret configured, skipping signature verification")
            return True

        # Remove prefix if present
        if self.config.signature_prefix and signature.startswith(self.config.signature_prefix):
            signature = signature[len(self.config.signature_prefix):]

        # Calculate expected signature
        hash_algorithm = getattr(hashlib, self.config.hash_algorithm)
        expected = hmac.new(
            self.config.secret.encode(),
            body,
            hash_algorithm
        ).hexdigest()

        # Compare signatures
        return hmac.compare_digest(expected, signature)

    async def process_request(self, request: Any) -> Dict[str, Any]:
        """
        Process incoming webhook request.

        Args:
            request: HTTP request object (FastAPI Request, Flask request, etc.)

        Returns:
            Parsed webhook data

        Raises:
            WebhookValidationError: If validation fails
        """
        # Get body and signature
        if hasattr(request, 'body'):
            # FastAPI
            body = await request.body() if asyncio.iscoroutinefunction(request.body) else request.body()
        elif hasattr(request, 'get_data'):
            # Flask
            body = request.get_data()
        else:
            raise WebhookValidationError("Unsupported request type")

        # Check body size
        if len(body) > self.config.max_body_size:
            raise WebhookValidationError(f"Body size exceeds limit: {len(body)} > {self.config.max_body_size}")

        # Get signature from headers
        signature = None
        if hasattr(request, 'headers'):
            signature = request.headers.get(self.config.signature_header)

        # Verify signature if configured
        if self.config.secret and signature:
            if not self.verify_signature(body, signature):
                raise WebhookValidationError("Invalid webhook signature")

        # Parse body
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            raise WebhookValidationError(f"Invalid JSON body: {e}")

        # Add metadata
        data["_webhook_metadata"] = {
            "received_at": datetime.now().isoformat(),
            "signature_verified": bool(self.config.secret and signature),
            "body_size": len(body)
        }

        return data

    def create_signature(self, body: Union[str, bytes, dict]) -> str:
        """
        Create signature for outgoing webhooks.

        Args:
            body: Webhook body (string, bytes, or dict)

        Returns:
            Signature string
        """
        if not self.config.secret:
            raise ValueError("Cannot create signature without secret")

        # Convert to bytes
        if isinstance(body, dict):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()

        # Create signature
        hash_algorithm = getattr(hashlib, self.config.hash_algorithm)
        signature = hmac.new(
            self.config.secret.encode(),
            body,
            hash_algorithm
        ).hexdigest()

        # Add prefix if configured
        if self.config.signature_prefix:
            signature = f"{self.config.signature_prefix}{signature}"

        return signature


