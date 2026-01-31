"""
LLM (Large Language Model) clients for the Arshai framework.

This module provides LLM client implementations for various providers,
enabling AI-powered conversations and text generation.

Available clients:
- OpenAIClient: OpenAI GPT models (gpt-4o, gpt-4-turbo, etc.)
- AzureClient: Azure OpenAI Service
- GeminiClient: Google AI (Gemini models)
- OpenRouterClient: OpenRouter multi-provider gateway
- AIGatewayLLM: Generic AI gateway/proxy for OpenAI-compatible endpoints (recommended)
- CloudflareGatewayLLM: Cloudflare AI Gateway (deprecated, use AIGatewayLLM)
"""

from arshai.llms.base_llm_client import BaseLLMClient
from arshai.llms.openai import OpenAIClient
from arshai.llms.azure import AzureClient
from arshai.llms.google_genai import GeminiClient
from arshai.llms.openrouter import OpenRouterClient
from arshai.llms.ai_gateway import (
    AIGatewayLLM,
    AIGatewayConfig,
)
from arshai.llms.cloudflare_gateway import (
    CloudflareGatewayLLM,
    CloudflareGatewayLLMConfig,
)

__all__ = [
    # Base class
    "BaseLLMClient",
    # Provider clients
    "OpenAIClient",
    "AzureClient",
    "GeminiClient",
    "OpenRouterClient",
    # Generic AI gateway (recommended)
    "AIGatewayLLM",
    "AIGatewayConfig",
    # Cloudflare Gateway (deprecated)
    "CloudflareGatewayLLM",
    "CloudflareGatewayLLMConfig",
]
