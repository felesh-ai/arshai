"""
Pytest test suite for Cloudflare AI Gateway Embedding client (BYOK mode).
Tests embedding generation, batch processing, and health checks.

This test requires CLOUDFLARE_GATEWAY_TOKEN environment variable or .env.cloudflare_gateway file.
"""

import pytest
import logging
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import functools
import os
import numpy as np

from arshai.embeddings import CloudflareGatewayEmbedding, CloudflareGatewayEmbeddingConfig

# Load test environment variables
test_env_path = Path(__file__).parent.parent / "llms" / ".env.cloudflare_gateway"
load_dotenv(test_env_path)

# Setup logging for pytest
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Rate limit handling decorator
def retry_on_rate_limit(max_retries=3, wait_seconds=30):
    """Decorator to retry API calls when rate limited (429 error)"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    is_rate_limit = (
                        "429" in error_str or
                        "rate_limit" in error_str.lower() or
                        "too many requests" in error_str.lower()
                    )

                    if is_rate_limit:
                        if attempt < max_retries:
                            logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). Waiting {wait_seconds} seconds...")
                            import time
                            time.sleep(wait_seconds)
                            continue
                        else:
                            logger.error(f"Rate limit exceeded after {max_retries + 1} attempts")
                            raise
                    else:
                        raise
            return None
        return wrapper
    return decorator


def retry_on_rate_limit_async(max_retries=3, wait_seconds=30):
    """Async decorator to retry API calls when rate limited"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    is_rate_limit = (
                        "429" in error_str or
                        "rate_limit" in error_str.lower() or
                        "too many requests" in error_str.lower()
                    )

                    if is_rate_limit:
                        if attempt < max_retries:
                            logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). Waiting {wait_seconds} seconds...")
                            await asyncio.sleep(wait_seconds)
                            continue
                        else:
                            logger.error(f"Rate limit exceeded after {max_retries + 1} attempts")
                            raise
                    else:
                        raise
            return None
        return wrapper
    return decorator


# Gateway credentials from environment
ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "a138914859c77ab7fe9fb4665b424e07")
GATEWAY_ID = os.environ.get("CLOUDFLARE_GATEWAY_ID", "green-bank")
GATEWAY_TOKEN = os.environ.get("CLOUDFLARE_GATEWAY_TOKEN", "NDdttbHj__CQgwBgrlLKQbhhk93ea9iX09l08qbk")


# Skip all tests if gateway token is not available
pytestmark = pytest.mark.skipif(
    not GATEWAY_TOKEN,
    reason="CLOUDFLARE_GATEWAY_TOKEN not set"
)


# Test Configuration and Fixtures
@pytest.fixture(scope="session")
def cloudflare_gateway_embedding_config():
    """Create test configuration for Cloudflare Gateway Embedding"""
    return CloudflareGatewayEmbeddingConfig(
        account_id=ACCOUNT_ID,
        gateway_id=GATEWAY_ID,
        gateway_token=GATEWAY_TOKEN,
        provider="openrouter",
        model_name="openai/text-embedding-3-small",
    )


@pytest.fixture(scope="session")
def cloudflare_gateway_embedding_client(cloudflare_gateway_embedding_config):
    """Create Cloudflare Gateway Embedding client for testing"""
    client = CloudflareGatewayEmbedding(cloudflare_gateway_embedding_config)
    yield client
    client.close()


# Test Data
TEST_DOCUMENTS = [
    "Artificial intelligence is transforming technology",
    "Machine learning powers modern AI systems",
    "Deep learning uses neural networks for complex tasks",
    "Natural language processing enables text understanding",
]

SIMILAR_DOCUMENTS = [
    "The weather is sunny today",
    "The forecast predicts rain tomorrow",
]

DISSIMILAR_DOCUMENTS = [
    "AI transforms the technology industry",
    "The weather outside is beautiful",
]


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


class TestCloudflareGatewayEmbeddingClient:
    """Test class for Cloudflare Gateway Embedding client"""

    @pytest.fixture(scope="class")
    def client(self, cloudflare_gateway_embedding_client):
        """Provide shared client for all tests"""
        return cloudflare_gateway_embedding_client

    def test_client_initialization(self, client):
        """Test that the Cloudflare Gateway Embedding client initializes correctly"""
        assert client is not None
        assert client.config.provider == "openrouter"
        assert client.config.model_name == "openai/text-embedding-3-small"
        assert client._client is not None
        assert client._initialized is True
        logger.info("✅ Client initialization test passed")

    def test_config_properties(self, cloudflare_gateway_embedding_config):
        """Test that config properties are correct"""
        config = cloudflare_gateway_embedding_config
        assert config.base_url == f"https://gateway.ai.cloudflare.com/v1/{ACCOUNT_ID}/{GATEWAY_ID}"
        assert config.provider == "openrouter"
        assert config.model_name == "openai/text-embedding-3-small"
        logger.info("✅ Config properties test passed")

    def test_dimension(self, client):
        """Test that dimension is correctly detected"""
        assert client.dimension > 0
        # text-embedding-3-small has 1536 dimensions
        assert client.dimension == 1536
        logger.info(f"✅ Dimension test passed - Dimension: {client.dimension}")

    @retry_on_rate_limit(max_retries=2, wait_seconds=30)
    def test_embed_single_document(self, client):
        """Test embedding a single document"""
        text = "This is a test document for embedding"

        result = client.embed_document(text)

        assert isinstance(result, dict)
        assert "dense" in result
        assert len(result["dense"]) == client.dimension

        # Verify vector values are reasonable
        vector = result["dense"]
        assert all(isinstance(v, float) for v in vector)
        assert not all(v == 0 for v in vector)  # Not all zeros

        logger.info(f"✅ Single document embedding test passed - Vector dim: {len(vector)}")

    @retry_on_rate_limit(max_retries=2, wait_seconds=30)
    def test_embed_documents_batch(self, client):
        """Test embedding multiple documents"""
        result = client.embed_documents(TEST_DOCUMENTS)

        assert isinstance(result, dict)
        assert "dense" in result

        vectors = result["dense"]
        assert len(vectors) == len(TEST_DOCUMENTS)

        for i, vector in enumerate(vectors):
            assert len(vector) == client.dimension
            assert all(isinstance(v, float) for v in vector)

        logger.info(f"✅ Batch embedding test passed - {len(vectors)} vectors generated")

    @retry_on_rate_limit(max_retries=2, wait_seconds=30)
    def test_embedding_similarity(self, client):
        """Test that similar documents have higher similarity"""
        # Embed similar documents (both about weather)
        similar_result = client.embed_documents(SIMILAR_DOCUMENTS)
        similar_vectors = similar_result["dense"]

        # Embed dissimilar documents (AI vs weather)
        dissimilar_result = client.embed_documents(DISSIMILAR_DOCUMENTS)
        dissimilar_vectors = dissimilar_result["dense"]

        # Calculate similarities
        similar_sim = cosine_similarity(similar_vectors[0], similar_vectors[1])
        dissimilar_sim = cosine_similarity(dissimilar_vectors[0], dissimilar_vectors[1])

        logger.info(f"Similar documents similarity: {similar_sim:.4f}")
        logger.info(f"Dissimilar documents similarity: {dissimilar_sim:.4f}")

        # Similar documents should have higher similarity
        assert similar_sim > dissimilar_sim, (
            f"Expected similar documents to have higher similarity. "
            f"Similar: {similar_sim:.4f}, Dissimilar: {dissimilar_sim:.4f}"
        )

        logger.info(f"✅ Embedding similarity test passed")

    @retry_on_rate_limit(max_retries=2, wait_seconds=30)
    def test_health_check(self, client):
        """Test health check functionality"""
        health = client.health_check()

        assert isinstance(health, dict)
        assert health["initialized"] is True
        assert health["provider"] == "openrouter"
        assert health["model"] == "openai/text-embedding-3-small"
        assert health["dimension"] == client.dimension
        assert health["healthy"] is True
        assert "latency_ms" in health

        logger.info(f"✅ Health check test passed - Latency: {health.get('latency_ms', 'N/A')}ms")

    @pytest.mark.asyncio
    @retry_on_rate_limit_async(max_retries=2, wait_seconds=30)
    async def test_async_embed_documents(self, client):
        """Test async embedding of documents"""
        result = await client.aembed_documents(TEST_DOCUMENTS[:2])

        assert isinstance(result, dict)
        assert "dense" in result

        vectors = result["dense"]
        assert len(vectors) == 2

        for vector in vectors:
            assert len(vector) == client.dimension

        logger.info(f"✅ Async embedding test passed - {len(vectors)} vectors generated")

    @retry_on_rate_limit(max_retries=2, wait_seconds=30)
    def test_empty_input_handling(self, client):
        """Test handling of empty input"""
        result = client.embed_documents([])

        assert isinstance(result, dict)
        assert "dense" in result
        assert len(result["dense"]) == 0

        logger.info("✅ Empty input handling test passed")

    @retry_on_rate_limit(max_retries=2, wait_seconds=30)
    def test_special_characters(self, client):
        """Test embedding text with special characters"""
        special_texts = [
            "Hello! @#$% world?",
            "日本語テスト (Japanese test)",
            "Emoji test: 🚀 🌟 💻",
        ]

        result = client.embed_documents(special_texts)

        assert isinstance(result, dict)
        assert "dense" in result
        assert len(result["dense"]) == len(special_texts)

        logger.info("✅ Special characters test passed")

    @retry_on_rate_limit(max_retries=2, wait_seconds=30)
    def test_long_text(self, client):
        """Test embedding long text"""
        long_text = " ".join(["This is a test sentence."] * 100)

        result = client.embed_document(long_text)

        assert isinstance(result, dict)
        assert "dense" in result
        assert len(result["dense"]) == client.dimension

        logger.info("✅ Long text test passed")


class TestCloudflareGatewayEmbeddingProviders:
    """Test different embedding providers through Cloudflare Gateway"""

    @retry_on_rate_limit(max_retries=2, wait_seconds=30)
    def test_openrouter_openai_embedding(self):
        """Test OpenAI embeddings through OpenRouter"""
        config = CloudflareGatewayEmbeddingConfig(
            account_id=ACCOUNT_ID,
            gateway_id=GATEWAY_ID,
            gateway_token=GATEWAY_TOKEN,
            provider="openrouter",
            model_name="openai/text-embedding-3-small",
        )

        client = CloudflareGatewayEmbedding(config)

        try:
            result = client.embed_document("Test embedding")
            assert "dense" in result
            assert len(result["dense"]) == 1536  # text-embedding-3-small dimension
            logger.info("✅ OpenRouter OpenAI embedding test passed")
        finally:
            client.close()

    def test_missing_gateway_token_raises_error(self):
        """Test that missing gateway token raises an error"""
        with pytest.raises(ValueError) as exc_info:
            config = CloudflareGatewayEmbeddingConfig(
                account_id=ACCOUNT_ID,
                gateway_id=GATEWAY_ID,
                gateway_token=None,  # No token
                provider="openrouter",
                model_name="openai/text-embedding-3-small",
            )
            # Clear env var temporarily
            original_token = os.environ.get("CLOUDFLARE_GATEWAY_TOKEN")
            if original_token:
                del os.environ["CLOUDFLARE_GATEWAY_TOKEN"]

            try:
                CloudflareGatewayEmbedding(config)
            finally:
                if original_token:
                    os.environ["CLOUDFLARE_GATEWAY_TOKEN"] = original_token

        assert "gateway_token" in str(exc_info.value).lower() or "required" in str(exc_info.value).lower()
        logger.info("✅ Missing gateway token error test passed")


# Run tests with: pytest tests/unit/embeddings/test_cloudflare_gateway_embedding.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
