"""
Cloudflare AI Gateway LLM implementation.

**DEPRECATED**: This client is deprecated and will be removed in a future version.
Please use `AIGatewayLLM` instead, which supports Cloudflare and other gateways.

This client routes LLM requests through Cloudflare AI Gateway's unified
/compat/chat/completions endpoint, enabling multi-provider support with
a single gateway for caching, rate limiting, and observability.

Supported providers through the unified endpoint:
- OpenAI (gpt-4o, gpt-4-turbo, etc.)
- Anthropic (claude-sonnet-4-5, claude-opus-4, etc.)
- Google AI Studio (gemini-2.0-flash, gemini-1.5-pro, etc.)
- Groq (llama-3-70b, mixtral-8x7b, etc.)
- Mistral (mistral-large, mistral-medium, etc.)
- Cohere (command-r-plus, command-r, etc.)
- xAI (grok-2, etc.)
- DeepSeek (deepseek-chat, deepseek-coder, etc.)
- OpenRouter (any model via openrouter provider)
- And more...

Migration Guide:
    # Old (deprecated)
    from arshai.llms.cloudflare_gateway import CloudflareGatewayLLM, CloudflareGatewayLLMConfig
    config = CloudflareGatewayLLMConfig(
        account_id="xxx",
        gateway_id="my-gateway",
        gateway_token="token",
        provider="anthropic",
        model="claude-sonnet-4-5"
    )

    # New (recommended)
    from arshai.llms.openai_compatible_gateway import AIGatewayLLM, AIGatewayConfig
    config = AIGatewayConfig(
        base_url="https://gateway.ai.cloudflare.com/v1/xxx/my-gateway/compat",
        api_key="token",
        model="anthropic/claude-sonnet-4-5"
    )
"""

import os
import warnings
from typing import Optional
from pydantic import Field, field_validator

from arshai.llms.ai_gateway import (
    AIGatewayLLM,
    AIGatewayConfig,
)


class CloudflareGatewayLLMConfig(AIGatewayConfig):
    """
    Configuration for Cloudflare AI Gateway LLM client (BYOK mode).

    **DEPRECATED**: This config is deprecated. Use `AIGatewayConfig` instead.

    This client uses Cloudflare AI Gateway with stored provider API keys (BYOK).
    Only the gateway token is required - provider keys are managed in the gateway.

    The gateway routes requests to multiple providers through a unified endpoint.
    Users specify provider and model separately for clarity.

    Examples:
        # Old (deprecated) - still works
        config = CloudflareGatewayLLMConfig(
            account_id="your-account-id",
            gateway_id="your-gateway-id",
            gateway_token="your-gateway-token",  # or set CLOUDFLARE_GATEWAY_TOKEN env var
            provider="anthropic",
            model="claude-sonnet-4-5",
        )

        # New (recommended) - use this instead
        from arshai.llms.openai_compatible_gateway import AIGatewayConfig
        config = AIGatewayConfig(
            base_url="https://gateway.ai.cloudflare.com/v1/your-account-id/your-gateway-id/compat",
            api_key="your-gateway-token",
            model="anthropic/claude-sonnet-4-5"
        )
    """

    # Cloudflare-specific fields (for backward compatibility)
    account_id: str = Field(
        description="Cloudflare account ID"
    )
    gateway_id: str = Field(
        description="Cloudflare AI Gateway ID"
    )
    provider: str = Field(
        description="LLM provider (openai, anthropic, google-ai-studio, groq, openrouter, etc.)"
    )

    # Override model field to remove provider prefix requirement
    model: str = Field(
        description="Model name WITHOUT provider prefix (e.g., 'claude-sonnet-4-5', 'gpt-4o')"
    )

    # Gateway authentication (BYOK mode - provider keys stored in gateway)
    gateway_token: Optional[str] = Field(
        default=None,
        description="Cloudflare AI Gateway token. Falls back to CLOUDFLARE_GATEWAY_TOKEN env var."
    )

    # Cloudflare Gateway base URL (configurable for custom/regional endpoints)
    cloudflare_base_url: str = Field(
        default="https://gateway.ai.cloudflare.com",
        description="Cloudflare AI Gateway base URL. Override for custom or regional endpoints."
    )

    def __init__(self, **data):
        """Initialize with deprecation warning."""
        warnings.warn(
            "CloudflareGatewayLLMConfig is deprecated and will be removed in a future version. "
            "Please migrate to AIGatewayConfig. "
            "See migration guide in the class docstring.",
            DeprecationWarning,
            stacklevel=2
        )

        # Extract Cloudflare-specific fields
        account_id = data.get('account_id')
        gateway_id = data.get('gateway_id')
        provider = data.get('provider')
        model = data.get('model')
        gateway_token = data.get('gateway_token') or os.environ.get("CLOUDFLARE_GATEWAY_TOKEN")
        cloudflare_base_url = data.get('cloudflare_base_url', 'https://gateway.ai.cloudflare.com')

        # Validate required fields
        if not account_id:
            raise ValueError("account_id is required for CloudflareGatewayLLMConfig")
        if not gateway_id:
            raise ValueError("gateway_id is required for CloudflareGatewayLLMConfig")
        if not provider:
            raise ValueError("provider is required for CloudflareGatewayLLMConfig")
        if not model:
            raise ValueError("model is required for CloudflareGatewayLLMConfig")
        if not gateway_token:
            raise ValueError(
                "gateway_token is required. Set it directly or via CLOUDFLARE_GATEWAY_TOKEN environment variable."
            )

        # Build base_url and full model name for the generic config
        base_url = f"{cloudflare_base_url}/v1/{account_id}/{gateway_id}/compat"
        full_model_name = f"{provider}/{model}"

        # Build headers
        headers = data.get('headers', {})
        default_headers = {
            "HTTP-Referer": "https://github.com/ArshaAI/arshai",
            "X-Title": "Arshai Framework",
        }
        merged_headers = {**default_headers, **headers}

        # Update data for parent class
        data['base_url'] = base_url
        data['gateway_token'] = gateway_token
        data['model'] = full_model_name
        data['headers'] = merged_headers

        # Call parent constructor
        super().__init__(**data)

        # Store Cloudflare-specific fields for backward compatibility
        self.account_id = account_id
        self.gateway_id = gateway_id
        self.provider = provider
        self.gateway_token = gateway_token
        self.cloudflare_base_url = cloudflare_base_url

    @property
    def compat_base_url(self) -> str:
        """Get the unified /compat base URL for OpenAI client (backward compatibility)."""
        return self.base_url

    @property
    def provider_base_url(self) -> str:
        """Get the provider-specific base URL for OpenAI client (backward compatibility)."""
        return f"{self.cloudflare_base_url}/v1/{self.account_id}/{self.gateway_id}/{self.provider}/v1"

    @property
    def full_model_name(self) -> str:
        """Get the full model name in {provider}/{model} format (backward compatibility)."""
        return self.model


class CloudflareGatewayLLM(AIGatewayLLM):
    """
    Cloudflare AI Gateway LLM client using the unified /compat endpoint (BYOK mode).

    **DEPRECATED**: This client is deprecated. Use `AIGatewayLLM` instead.

    This client uses Cloudflare AI Gateway with stored provider API keys (BYOK).
    Only the gateway token is required - provider keys are managed in the gateway.

    Benefits:
    - Centralized caching and rate limiting
    - Unified logging and analytics
    - Easy provider switching by changing provider/model
    - Secure API key management (keys stored in gateway, not in code)

    Migration Guide:
        # Old (deprecated) - still works
        from arshai.llms.cloudflare_gateway import CloudflareGatewayLLM, CloudflareGatewayLLMConfig
        config = CloudflareGatewayLLMConfig(
            account_id="xxx",
            gateway_id="my-gateway",
            gateway_token="your-gateway-token",
            provider="anthropic",
            model="claude-sonnet-4-5",
        )
        llm = CloudflareGatewayLLM(config)

        # New (recommended) - use this instead
        from arshai.llms.openai_compatible_gateway import AIGatewayLLM, AIGatewayConfig
        config = AIGatewayConfig(
            base_url="https://gateway.ai.cloudflare.com/v1/xxx/my-gateway/compat",
            api_key="your-gateway-token",
            model="anthropic/claude-sonnet-4-5"
        )
        llm = AIGatewayLLM(config)
    """

    config: CloudflareGatewayLLMConfig

    def __init__(self, config: CloudflareGatewayLLMConfig):
        """Initialize with deprecation warning."""
        warnings.warn(
            "CloudflareGatewayLLM is deprecated and will be removed in a future version. "
            "Please migrate to AIGatewayLLM. "
            "See migration guide in the class docstring.",
            DeprecationWarning,
            stacklevel=2
        )

        # Call parent constructor - it will use the converted config
        super().__init__(config)
