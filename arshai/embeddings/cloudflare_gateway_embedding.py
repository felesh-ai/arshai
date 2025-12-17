"""
Cloudflare AI Gateway Embedding implementation.

This client routes embedding requests through Cloudflare AI Gateway using
provider-specific endpoints, enabling multi-provider support with a single
gateway for caching, rate limiting, and observability.

Supported providers:
- OpenAI (text-embedding-3-small, text-embedding-3-large, ada-002)
- Azure OpenAI (custom deployments)
- Cohere (embed-english-v3.0, embed-multilingual-v3.0)
- AWS Bedrock (amazon.titan-embed-text-v1, cohere.embed-*)
- Workers AI (bge-base-en-v1.5, bge-large-en-v1.5)
- OpenRouter (openai/text-embedding-3-small, etc.)
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from pydantic import Field

from arshai.core.interfaces.iembedding import EmbeddingConfig, IEmbedding

try:
    from openai import OpenAI, AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


logger = logging.getLogger(__name__)


# Provider-specific endpoint templates
EMBEDDING_ENDPOINTS = {
    "openai": "{base}/openai/v1/embeddings",
    "azure": "{base}/azure-openai/{resource}/{deployment}/embeddings?api-version={api_version}",
    "cohere": "{base}/cohere/v1/embed",
    "bedrock": "{base}/aws-bedrock/bedrock-runtime/{region}/model/{model}/invoke",
    "workers-ai": "{base}/workers-ai/v1/embeddings",
    "openrouter": "{base}/openrouter/v1/embeddings",
    "google-ai-studio": "{base}/google-ai-studio/v1/embeddings",
    "mistral": "{base}/mistral/v1/embeddings",
}

# Default embedding dimensions for known models
EMBEDDING_DIMENSIONS = {
    # OpenAI
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Cohere
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
    # Bedrock Titan
    "amazon.titan-embed-text-v1": 1536,
    "amazon.titan-embed-text-v2": 1024,
    # Workers AI
    "bge-base-en-v1.5": 768,
    "bge-large-en-v1.5": 1024,
    "bge-small-en-v1.5": 384,
    # OpenRouter models (using provider/model format)
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-3-large": 3072,
    "openai/text-embedding-ada-002": 1536,
}

class CloudflareGatewayEmbeddingConfig(EmbeddingConfig):
    """
    Configuration for Cloudflare AI Gateway Embedding client (BYOK mode).

    This client uses Cloudflare AI Gateway with stored provider API keys (BYOK).
    Only the gateway token is required - provider keys are managed in the gateway.

    Example:
        # OpenAI embeddings through Cloudflare
        config = CloudflareGatewayEmbeddingConfig(
            account_id="your-account-id",
            gateway_id="your-gateway-id",
            gateway_token="your-gateway-token",
            provider="openai",
            model_name="text-embedding-3-small",
        )

        # OpenRouter embeddings (multi-provider)
        config = CloudflareGatewayEmbeddingConfig(
            account_id="your-account-id",
            gateway_id="your-gateway-id",
            gateway_token="your-gateway-token",
            provider="openrouter",
            model_name="openai/text-embedding-3-small",
        )
    """

    # Cloudflare Gateway settings
    account_id: str = Field(description="Cloudflare account ID")
    gateway_id: str = Field(description="Cloudflare AI Gateway ID")

    # Provider settings
    provider: str = Field(
        description="Embedding provider (openai, azure, cohere, bedrock, workers-ai, openrouter)"
    )
    model_name: str = Field(
        description="Embedding model name"
    )

    # Gateway authentication (BYOK mode - provider keys stored in gateway)
    gateway_token: Optional[str] = Field(
        default=None,
        description="Cloudflare AI Gateway token. Falls back to CLOUDFLARE_GATEWAY_TOKEN env var."
    )

    # Azure-specific settings
    azure_resource: Optional[str] = Field(
        default=None,
        description="Azure OpenAI resource name (required for azure provider)"
    )
    azure_deployment: Optional[str] = Field(
        default=None,
        description="Azure OpenAI deployment name (required for azure provider)"
    )
    azure_api_version: str = Field(
        default="2024-02-01",
        description="Azure OpenAI API version"
    )

    # Bedrock-specific settings
    bedrock_region: str = Field(
        default="us-east-1",
        description="AWS region for Bedrock"
    )

    # Request settings
    batch_size: int = Field(default=32, description="Batch size for embedding requests")
    timeout: float = Field(default=60.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

    # Dimension override
    dimensions: Optional[int] = Field(
        default=None,
        description="Override embedding dimensions (auto-detected if not provided)"
    )

    @property
    def base_url(self) -> str:
        """Get the Cloudflare AI Gateway base URL."""
        return f"https://gateway.ai.cloudflare.com/v1/{self.account_id}/{self.gateway_id}"

    def get_endpoint_url(self) -> str:
        """Get the provider-specific embedding endpoint URL."""
        template = EMBEDDING_ENDPOINTS.get(self.provider)
        if not template:
            raise ValueError(f"Unsupported provider: {self.provider}")

        url = template.format(
            base=self.base_url,
            resource=self.azure_resource or "",
            deployment=self.azure_deployment or "",
            api_version=self.azure_api_version,
            region=self.bedrock_region,
            model=self.model_name,
        )

        return url


class CloudflareGatewayEmbedding(IEmbedding):
    """
    Cloudflare AI Gateway Embedding client (BYOK mode).

    This client uses Cloudflare AI Gateway with stored provider API keys (BYOK).
    Only the gateway token is required - provider keys are managed in the gateway.

    Benefits:
    - Centralized caching and rate limiting
    - Unified logging and analytics
    - Easy provider switching
    - Secure API key management (keys stored in gateway, not in code)

    Note: Uses provider-specific endpoints (e.g., /openrouter/v1/embeddings)
    since the unified /compat/embeddings endpoint is not yet available.

    Example:
        config = CloudflareGatewayEmbeddingConfig(
            account_id="xxx",
            gateway_id="my-gateway",
            gateway_token="your-gateway-token",
            provider="openrouter",
            model_name="openai/text-embedding-3-small",
        )

        embedding = CloudflareGatewayEmbedding(config)
        result = embedding.embed_documents(["Hello world"])
        vectors = result["dense"]  # List of embedding vectors
    """

    def __init__(self, config: CloudflareGatewayEmbeddingConfig):
        """
        Initialize the Cloudflare Gateway embedding client.

        Args:
            config: Configuration for the embedding service
        """
        if not HAS_OPENAI:
            raise ImportError(
                "The openai package is required for CloudflareGatewayEmbedding. "
                "Install it with 'pip install openai'."
            )

        self.config = config
        self._dimension = self._determine_dimension()
        self._initialized = False
        self._client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None
        self.logger = logger

        self._initialize()

    def _determine_dimension(self) -> int:
        """Determine embedding dimension from config or model defaults."""
        if self.config.dimensions:
            return self.config.dimensions

        # Try to find dimension from known models
        model = self.config.model_name
        if model in EMBEDDING_DIMENSIONS:
            return EMBEDDING_DIMENSIONS[model]

        # For OpenRouter format (provider/model)
        if "/" in model:
            base_model = model.split("/")[-1]
            if base_model in EMBEDDING_DIMENSIONS:
                return EMBEDDING_DIMENSIONS[base_model]

        # Default fallback
        self.logger.warning(
            f"Unknown model '{model}', using default dimension 1536. "
            f"Consider setting 'dimensions' in config for accuracy."
        )
        return 1536

    def _initialize(self) -> None:
        """Initialize the embedding client (BYOK mode)."""
        if self._initialized:
            return

        # Get gateway token (required for BYOK mode)
        gateway_token = self.config.gateway_token or os.environ.get("CLOUDFLARE_GATEWAY_TOKEN")
        if not gateway_token:
            raise ValueError(
                "Gateway token is required for Cloudflare AI Gateway (BYOK mode). "
                "Set gateway_token in config or CLOUDFLARE_GATEWAY_TOKEN environment variable."
            )

        # Use gateway token as API key - Cloudflare recognizes it and uses stored provider keys
        api_key = gateway_token

        # Determine base URL for the client
        # For most providers, we use the provider-specific path
        base_url = self._get_client_base_url()

        # Build headers (no cf-aig-authorization needed when using gateway token as api_key)
        default_headers = {
            "HTTP-Referer": "https://github.com/ArshaAI/arshai",
            "X-Title": "Arshai Framework",
        }

        try:
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                default_headers=default_headers,
            )

            self._async_client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                default_headers=default_headers,
            )

            self._initialized = True
            self.logger.info(
                f"Cloudflare Gateway embedding initialized (BYOK mode) - "
                f"Provider: {self.config.provider}, Model: {self.config.model_name}, "
                f"Dimension: {self._dimension}"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize embedding client: {e}")
            raise

    def _get_client_base_url(self) -> str:
        """Get the base URL for the OpenAI client based on provider."""
        base = self.config.base_url

        if self.config.provider == "openai":
            return f"{base}/openai/v1"
        elif self.config.provider == "azure":
            resource = self.config.azure_resource
            deployment = self.config.azure_deployment
            if not resource or not deployment:
                raise ValueError(
                    "azure_resource and azure_deployment are required for Azure provider"
                )
            return f"{base}/azure-openai/{resource}/{deployment}"
        elif self.config.provider == "cohere":
            return f"{base}/cohere/v1"
        elif self.config.provider == "bedrock":
            # Bedrock has a different API structure
            return f"{base}/aws-bedrock/bedrock-runtime/{self.config.bedrock_region}"
        elif self.config.provider == "workers-ai":
            return f"{base}/workers-ai/v1"
        elif self.config.provider == "openrouter":
            return f"{base}/openrouter/v1"
        elif self.config.provider == "google-ai-studio":
            return f"{base}/google-ai-studio/v1"
        elif self.config.provider == "mistral":
            return f"{base}/mistral/v1"
        else:
            # Fallback to provider name as path
            return f"{base}/{self.config.provider}/v1"

    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings produced by this service."""
        return self._dimension

    def embed_documents(self, texts: List[str]) -> Dict[str, Any]:
        """
        Generate embeddings for multiple documents.

        Args:
            texts: List of text documents to embed

        Returns:
            Dictionary containing embeddings with 'dense' key
        """
        if not texts:
            return {"dense": []}

        if not self._initialized:
            self._initialize()

        if self._client is None:
            raise RuntimeError("Embedding client not initialized")

        start_time = time.time()
        dense_vectors = []

        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]

            try:
                if self.config.provider == "bedrock":
                    # Bedrock has different API structure
                    batch_embeddings = self._embed_bedrock(batch)
                elif self.config.provider == "cohere":
                    # Cohere uses different endpoint
                    batch_embeddings = self._embed_cohere(batch)
                else:
                    # OpenAI-compatible providers
                    batch_embeddings = self._embed_openai_compatible(batch)

                dense_vectors.extend(batch_embeddings)

            except Exception as e:
                self.logger.error(f"Error generating embeddings: {e}")
                raise

        duration = time.time() - start_time
        self.logger.debug(
            f"Generated {len(texts)} embeddings in {duration:.2f}s "
            f"({len(texts) / duration:.1f} texts/sec)"
        )

        return {"dense": dense_vectors}

    def _embed_openai_compatible(self, texts: List[str]) -> List[List[float]]:
        """Embed using OpenAI-compatible API."""
        response = self._client.embeddings.create(
            model=self.config.model_name,
            input=texts,
            encoding_format="float"
        )

        # Sort by index to ensure correct order
        embeddings_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in embeddings_data]

    def _embed_cohere(self, texts: List[str]) -> List[List[float]]:
        """Embed using Cohere API through gateway."""
        # Cohere uses a slightly different structure
        # The OpenAI client should handle this through the gateway
        response = self._client.embeddings.create(
            model=self.config.model_name,
            input=texts,
        )

        embeddings_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in embeddings_data]

    def _embed_bedrock(self, texts: List[str]) -> List[List[float]]:
        """Embed using AWS Bedrock through gateway."""
        # Bedrock embedding via gateway
        # Note: Bedrock may require AWS signing - this is a simplified version
        embeddings = []

        for text in texts:
            response = self._client.embeddings.create(
                model=self.config.model_name,
                input=[text],
            )

            if response.data:
                embeddings.append(response.data[0].embedding)

        return embeddings

    def embed_document(self, text: str) -> Dict[str, Any]:
        """
        Generate embeddings for a single document.

        Args:
            text: Text document to embed

        Returns:
            Dictionary containing embeddings with 'dense' key
        """
        result = self.embed_documents([text])
        if result["dense"]:
            return {"dense": result["dense"][0]}
        return {"dense": []}

    async def aembed_documents(self, texts: List[str]) -> Dict[str, Any]:
        """
        Asynchronously generate embeddings for multiple documents.

        Args:
            texts: List of text documents to embed

        Returns:
            Dictionary containing embeddings with 'dense' key
        """
        if not texts:
            return {"dense": []}

        if not self._initialized:
            self._initialize()

        if self._async_client is None:
            raise RuntimeError("Async embedding client not initialized")

        start_time = time.time()
        dense_vectors = []

        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]

            try:
                response = await self._async_client.embeddings.create(
                    model=self.config.model_name,
                    input=batch,
                    encoding_format="float"
                )

                embeddings_data = sorted(response.data, key=lambda x: x.index)
                batch_embeddings = [item.embedding for item in embeddings_data]
                dense_vectors.extend(batch_embeddings)

            except Exception as e:
                self.logger.error(f"Error generating async embeddings: {e}")
                raise

        duration = time.time() - start_time
        self.logger.debug(
            f"Generated {len(texts)} embeddings async in {duration:.2f}s"
        )

        return {"dense": dense_vectors}

    async def aembed_document(self, text: str) -> Dict[str, Any]:
        """
        Asynchronously generate embeddings for a single document.

        Args:
            text: Text document to embed

        Returns:
            Dictionary containing embeddings with 'dense' key
        """
        result = await self.aembed_documents([text])
        if result["dense"]:
            return {"dense": result["dense"][0]}
        return {"dense": []}

    def health_check(self) -> Dict[str, Any]:
        """
        Check embedding service health.

        Returns:
            Health status dictionary
        """
        status = {
            "initialized": self._initialized,
            "provider": self.config.provider,
            "model": self.config.model_name,
            "dimension": self._dimension,
            "healthy": False,
        }

        if not self._initialized or self._client is None:
            return status

        try:
            start = time.time()
            self._client.embeddings.create(
                model=self.config.model_name,
                input=["health check"],
            )
            latency_ms = (time.time() - start) * 1000

            status["healthy"] = True
            status["latency_ms"] = latency_ms

        except Exception as e:
            status["error"] = str(e)

        return status

    def close(self) -> None:
        """Clean up resources."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

        if self._async_client:
            # Note: async client should be closed in async context
            self._async_client = None

        self._initialized = False
        self.logger.debug("Cloudflare Gateway embedding resources released")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
