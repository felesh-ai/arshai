"""
Embedding services for vector representations.

This module provides embedding capabilities for the Arshai framework,
converting text into vector representations for semantic search and retrieval.

Available embedding clients:
- OpenAIEmbedding: OpenAI text embedding models
- CloudflareGatewayEmbedding: Cloudflare AI Gateway (multi-provider)
- TongyiVisionEmbedding: Gateway-fronted multimodal embeddings
  (tongyi-embedding-vision-plus / qwen3-vl-embedding via the AI Gateway's
  /v1/multimodal-embeddings route). Implements IMultimodalEmbedding.
"""

from .openai_embeddings import OpenAIEmbedding
from .cloudflare_gateway_embedding import (
    CloudflareGatewayEmbedding,
    CloudflareGatewayEmbeddingConfig,
)
from .qwen3_vl_embedding import (
    TongyiVisionEmbedding,
    TongyiVisionEmbeddingConfig,
)

__all__ = [
    "OpenAIEmbedding",
    # Cloudflare Gateway
    "CloudflareGatewayEmbedding",
    "CloudflareGatewayEmbeddingConfig",
    # Tongyi vision (multimodal)
    "TongyiVisionEmbedding",
    "TongyiVisionEmbeddingConfig",
]
