"""
Integration tests for Cloudflare AI Gateway clients.

These tests verify end-to-end functionality with the real Cloudflare AI Gateway.
They test both LLM and Embedding clients together in realistic scenarios.

Requirements:
- CLOUDFLARE_GATEWAY_TOKEN environment variable set
- Valid Cloudflare AI Gateway with provider keys configured (BYOK mode)
"""

import pytest
import logging
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
import numpy as np

from arshai.llms import CloudflareGatewayLLM, CloudflareGatewayLLMConfig
from arshai.embeddings import CloudflareGatewayEmbedding, CloudflareGatewayEmbeddingConfig
from arshai.core.interfaces.illm import ILLMInput

# Load environment variables
test_env_path = Path(__file__).parent.parent.parent / "unit" / "llms" / ".env.cloudflare_gateway"
load_dotenv(test_env_path)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gateway credentials
ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "a138914859c77ab7fe9fb4665b424e07")
GATEWAY_ID = os.environ.get("CLOUDFLARE_GATEWAY_ID", "green-bank")
GATEWAY_TOKEN = os.environ.get("CLOUDFLARE_GATEWAY_TOKEN", "NDdttbHj__CQgwBgrlLKQbhhk93ea9iX09l08qbk")


# Skip all tests if gateway token is not available
pytestmark = pytest.mark.skipif(
    not GATEWAY_TOKEN,
    reason="CLOUDFLARE_GATEWAY_TOKEN not set"
)


# Pydantic models for structured output
class DocumentSummary(BaseModel):
    """Summary of a document"""
    title: str = Field(description="A title for the document")
    summary: str = Field(description="A brief summary of the content")
    key_topics: List[str] = Field(description="Key topics discussed")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")


class SearchResult(BaseModel):
    """Search result with relevance score"""
    document: str
    relevance_score: float
    explanation: str


def cosine_similarity(vec1, vec2) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


class TestCloudflareGatewayIntegration:
    """Integration tests for Cloudflare AI Gateway"""

    @pytest.fixture(scope="class")
    def llm_client(self):
        """Create LLM client for integration tests"""
        config = CloudflareGatewayLLMConfig(
            account_id=ACCOUNT_ID,
            gateway_id=GATEWAY_ID,
            gateway_token=GATEWAY_TOKEN,
            provider="openrouter",
            model="openai/gpt-4o-mini",
            temperature=0.3,
            max_tokens=500,
        )
        client = CloudflareGatewayLLM(config)
        yield client
        client.close()

    @pytest.fixture(scope="class")
    def embedding_client(self):
        """Create Embedding client for integration tests"""
        config = CloudflareGatewayEmbeddingConfig(
            account_id=ACCOUNT_ID,
            gateway_id=GATEWAY_ID,
            gateway_token=GATEWAY_TOKEN,
            provider="openrouter",
            model_name="openai/text-embedding-3-small",
        )
        client = CloudflareGatewayEmbedding(config)
        yield client
        client.close()

    @pytest.mark.asyncio
    async def test_llm_basic_functionality(self, llm_client):
        """Test basic LLM chat functionality"""
        input_data = ILLMInput(
            system_prompt="You are a helpful assistant.",
            user_message="What is 2 + 2? Answer in one word."
        )

        response = await llm_client.chat(input_data)

        assert "llm_response" in response
        assert "4" in response["llm_response"].lower() or "four" in response["llm_response"].lower()
        assert "usage" in response

        logger.info(f"✅ LLM basic functionality test passed")

    @pytest.mark.asyncio
    async def test_llm_streaming(self, llm_client):
        """Test LLM streaming functionality"""
        input_data = ILLMInput(
            system_prompt="You are a helpful assistant.",
            user_message="Count from 1 to 5, one number per line."
        )

        chunks = []
        final_text = ""

        async for chunk in llm_client.stream(input_data):
            chunks.append(chunk)
            if chunk.get("llm_response"):
                final_text = chunk["llm_response"]

        assert len(chunks) > 0
        assert len(final_text) > 0
        # Should contain at least some numbers
        assert any(str(i) in final_text for i in range(1, 6))

        logger.info(f"✅ LLM streaming test passed - {len(chunks)} chunks received")

    @pytest.mark.asyncio
    async def test_llm_structured_output(self, llm_client):
        """Test LLM structured output with Pydantic model"""
        document = """
        The new solar panel installation project has exceeded expectations.
        The system is generating 30% more electricity than predicted, and
        the community response has been overwhelmingly positive. Local jobs
        have increased and carbon emissions are down significantly.
        """

        input_data = ILLMInput(
            system_prompt="Analyze the following document and provide a structured summary.",
            user_message=f"Analyze this document:\n\n{document}",
            structure_type=DocumentSummary
        )

        response = await llm_client.chat(input_data)

        assert "llm_response" in response
        result = response["llm_response"]
        assert isinstance(result, DocumentSummary)
        assert len(result.title) > 0
        assert len(result.summary) > 0
        assert len(result.key_topics) > 0
        assert result.sentiment.lower() in ["positive", "negative", "neutral", "mixed"]

        logger.info(f"✅ LLM structured output test passed - Title: {result.title}")

    def test_embedding_basic_functionality(self, embedding_client):
        """Test basic embedding generation"""
        text = "Artificial intelligence is transforming the world."

        result = embedding_client.embed_document(text)

        assert "dense" in result
        assert len(result["dense"]) == embedding_client.dimension
        assert all(isinstance(v, float) for v in result["dense"])

        logger.info(f"✅ Embedding basic functionality test passed - Dim: {len(result['dense'])}")

    def test_embedding_batch(self, embedding_client):
        """Test batch embedding generation"""
        texts = [
            "Machine learning enables computers to learn from data.",
            "Deep learning uses neural networks with many layers.",
            "Natural language processing helps computers understand text.",
        ]

        result = embedding_client.embed_documents(texts)

        assert "dense" in result
        assert len(result["dense"]) == len(texts)

        for vector in result["dense"]:
            assert len(vector) == embedding_client.dimension

        logger.info(f"✅ Embedding batch test passed - {len(result['dense'])} vectors generated")

    def test_embedding_similarity_search(self, embedding_client):
        """Test semantic similarity using embeddings"""
        documents = [
            "Python is a popular programming language.",
            "JavaScript is used for web development.",
            "Machine learning is a subset of artificial intelligence.",
            "The weather today is sunny and warm.",
        ]

        query = "What programming languages are commonly used?"

        # Embed all documents and query
        doc_result = embedding_client.embed_documents(documents)
        query_result = embedding_client.embed_document(query)

        doc_vectors = doc_result["dense"]
        query_vector = query_result["dense"]

        # Calculate similarities
        similarities = [
            cosine_similarity(query_vector, doc_vec)
            for doc_vec in doc_vectors
        ]

        # Create ranked results
        ranked = sorted(
            zip(documents, similarities),
            key=lambda x: x[1],
            reverse=True
        )

        logger.info("Similarity search results:")
        for doc, sim in ranked:
            logger.info(f"  {sim:.4f}: {doc[:50]}...")

        # The programming-related documents should rank higher than weather
        assert ranked[-1][0] == documents[3], "Weather document should rank lowest"

        logger.info("✅ Embedding similarity search test passed")

    @pytest.mark.asyncio
    async def test_llm_with_tools(self, llm_client):
        """Test LLM function calling"""
        def calculate(expression: str) -> str:
            """Calculate a mathematical expression."""
            try:
                result = eval(expression)
                return f"The result of {expression} is {result}"
            except Exception as e:
                return f"Error calculating: {e}"

        input_data = ILLMInput(
            system_prompt="You are a helpful math assistant. Use the calculate tool for math operations.",
            user_message="What is 15 * 7?",
            regular_functions={"calculate": calculate},
            max_turns=5
        )

        response = await llm_client.chat(input_data)

        assert "llm_response" in response
        result_text = str(response["llm_response"])
        # Should contain 105 (15 * 7)
        assert "105" in result_text

        logger.info("✅ LLM with tools test passed")

    @pytest.mark.asyncio
    async def test_combined_rag_workflow(self, llm_client, embedding_client):
        """Test a RAG-like workflow combining embeddings and LLM"""
        # Knowledge base
        knowledge_base = [
            "Python was created by Guido van Rossum and released in 1991.",
            "JavaScript was created by Brendan Eich in 1995 for Netscape Navigator.",
            "Java was developed by James Gosling at Sun Microsystems in 1995.",
            "The Eiffel Tower is located in Paris, France.",
            "The Great Wall of China is over 13,000 miles long.",
        ]

        query = "Who created Python and when?"

        # Step 1: Embed documents and query
        doc_result = embedding_client.embed_documents(knowledge_base)
        query_result = embedding_client.embed_document(query)

        doc_vectors = doc_result["dense"]
        query_vector = query_result["dense"]

        # Step 2: Find most relevant documents
        similarities = [
            cosine_similarity(query_vector, doc_vec)
            for doc_vec in doc_vectors
        ]

        # Get top 2 documents
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:2]
        relevant_docs = [knowledge_base[i] for i in top_indices]

        logger.info(f"Retrieved documents: {relevant_docs}")

        # Step 3: Use LLM to answer based on retrieved context
        context = "\n".join(relevant_docs)
        input_data = ILLMInput(
            system_prompt="Answer the user's question based ONLY on the provided context. Be concise.",
            user_message=f"Context:\n{context}\n\nQuestion: {query}"
        )

        response = await llm_client.chat(input_data)
        answer = str(response["llm_response"])

        logger.info(f"RAG Answer: {answer}")

        # Validate answer contains expected information
        assert "guido" in answer.lower() or "van rossum" in answer.lower()
        assert "1991" in answer

        logger.info("✅ Combined RAG workflow test passed")

    @pytest.mark.asyncio
    async def test_provider_info_in_usage(self, llm_client):
        """Test that provider information is included in usage data"""
        input_data = ILLMInput(
            system_prompt="You are a helpful assistant.",
            user_message="Hello!"
        )

        response = await llm_client.chat(input_data)

        assert "usage" in response
        usage = response["usage"]

        assert "provider" in usage
        assert usage["provider"] == "openrouter"
        assert "model" in usage
        assert "openrouter" in usage["model"]
        assert "input_tokens" in usage
        assert "output_tokens" in usage
        assert "total_tokens" in usage

        logger.info(f"✅ Provider info test passed - Usage: {usage}")

    def test_health_checks(self, llm_client, embedding_client):
        """Test health check for both clients"""
        # Embedding health check
        embed_health = embedding_client.health_check()

        assert embed_health["initialized"] is True
        assert embed_health["healthy"] is True
        assert embed_health["provider"] == "openrouter"

        logger.info(f"✅ Health checks passed - Embedding latency: {embed_health.get('latency_ms', 'N/A')}ms")


class TestCloudflareGatewayErrorHandling:
    """Test error handling for Cloudflare Gateway clients"""

    @pytest.mark.asyncio
    async def test_invalid_gateway_token_llm(self):
        """Test that invalid gateway token results in error or empty response"""
        config = CloudflareGatewayLLMConfig(
            account_id=ACCOUNT_ID,
            gateway_id=GATEWAY_ID,
            gateway_token="invalid-token-12345",
            provider="openrouter",
            model="openai/gpt-4o-mini",
        )

        client = CloudflareGatewayLLM(config)

        try:
            response = await client.chat(ILLMInput(
                system_prompt="Test",
                user_message="Test"
            ))
            # If no exception, check response indicates failure
            # Either llm_response is None/empty or contains error
            llm_response = response.get("llm_response")
            assert llm_response is None or llm_response == "" or "error" in str(llm_response).lower(), \
                f"Expected error or empty response with invalid token, got: {response}"
            logger.info("✅ Invalid token returned empty/error response (handled gracefully)")
        except Exception as e:
            # Exception is also acceptable - verify it's auth-related
            error_str = str(e).lower()
            assert "401" in error_str or "auth" in error_str or "unauthorized" in error_str, \
                f"Expected auth error, got: {e}"
            logger.info("✅ Invalid token raised auth exception as expected")
        finally:
            client.close()

    def test_missing_gateway_token(self):
        """Test that missing gateway token raises ValueError"""
        # Temporarily clear the env var
        original_token = os.environ.pop("CLOUDFLARE_GATEWAY_TOKEN", None)

        try:
            with pytest.raises(ValueError) as exc_info:
                config = CloudflareGatewayLLMConfig(
                    account_id=ACCOUNT_ID,
                    gateway_id=GATEWAY_ID,
                    gateway_token=None,
                    provider="openrouter",
                    model="openai/gpt-4o-mini",
                )
                CloudflareGatewayLLM(config)

            assert "gateway_token" in str(exc_info.value).lower() or "required" in str(exc_info.value).lower()
        finally:
            # Restore the env var
            if original_token:
                os.environ["CLOUDFLARE_GATEWAY_TOKEN"] = original_token

        logger.info("✅ Missing token error handling test passed")


# Run tests with: pytest tests/integration/cloudflare_gateway/test_cloudflare_gateway_integration.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
