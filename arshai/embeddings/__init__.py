"""
Embedding services for vector representations.

This module provides embedding capabilities for the Arshai framework,
converting text into vector representations for semantic search and retrieval.

Available embedding clients:
- OpenAIEmbedding: OpenAI text embedding models
- CloudflareGatewayEmbedding: Cloudflare AI Gateway (multi-provider)
"""

from .openai_embeddings import OpenAIEmbedding
from .cloudflare_gateway_embedding import (
    CloudflareGatewayEmbedding,
    CloudflareGatewayEmbeddingConfig,
)

__all__ = [
    "OpenAIEmbedding",
    # Cloudflare Gateway
    "CloudflareGatewayEmbedding",
    "CloudflareGatewayEmbeddingConfig",
] 