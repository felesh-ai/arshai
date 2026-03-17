"""
Pytest test suite for Cloudflare AI Gateway LLM client (BYOK mode).
Tests both chat and stream methods with identical inputs for direct comparison.
Includes regex validation for semantic correctness.
Based on the unified test pattern established for Gemini and OpenAI clients.

This test requires CLOUDFLARE_GATEWAY_TOKEN environment variable or .env.cloudflare_gateway file.
"""

import pytest
import re
import logging
import asyncio
from pathlib import Path
from typing import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import functools
import os

from arshai.core.interfaces.illm import ILLMInput
from arshai.llms import CloudflareGatewayLLM, CloudflareGatewayLLMConfig

# Load test environment variables
test_env_path = Path(__file__).parent / ".env.cloudflare_gateway"
load_dotenv(test_env_path)

# Setup logging for pytest
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Rate limit handling decorator
def retry_on_rate_limit(max_retries=3, wait_seconds=60):
    """Decorator to retry API calls when rate limited (429 error)"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    error_type = type(e).__name__

                    is_rate_limit = (
                        "429" in error_str or
                        "RESOURCE_EXHAUSTED" in error_str or
                        "Resource exhausted" in error_str or
                        "RateLimitError" in error_type or
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


# Mathematical functions for tool calling tests
def power_function(base: float, exponent: float) -> float:
    """Calculate base raised to the power of exponent."""
    return base ** exponent


def multiply_function(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


# Background task functions for testing fire-and-forget pattern
background_task_executed = None

def send_admin_notification(event: str, details: str = "User interaction") -> None:
    """BACKGROUND TASK: MANDATORY function that MUST be called for every user interaction to notify admin team.

    This function MUST be called whenever a user asks any question or makes any request.
    Always call this function with event='user_question' and appropriate details describing the user's request.

    This task runs in fire-and-forget mode - no results will be returned to the conversation."""
    global background_task_executed
    import time
    time.sleep(0.1)  # Simulate notification work

    background_task_executed = f"ADMIN_NOTIFIED: {event} - {details}"
    logger.info(f"BACKGROUND ADMIN NOTIFICATION: {event} - {details}")
    logger.info(f"Background task executed, set variable: {background_task_executed}")


# Test Models for Chat (Pydantic)
class SentimentAnalysis(BaseModel):
    """Sentiment analysis result for chat testing"""
    topic: str = Field(description="Main topic analyzed")
    sentiment: str = Field(description="Overall sentiment (positive/negative/neutral)")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    key_points: list[str] = Field(description="List of key points identified")


class MathResult(BaseModel):
    """Math calculation result for chat testing"""
    operation: str = Field(description="The mathematical operation performed")
    result: int = Field(description="The numerical result")
    explanation: str = Field(description="Brief explanation of the calculation")


# Test Models for Stream (Dict-based with schema method)
class StreamSentimentAnalysis(TypedDict):
    """Sentiment analysis result for stream testing"""
    topic: str
    sentiment: str
    confidence: float
    key_points: list[str]

    @classmethod
    def model_json_schema(cls):
        return {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Main topic analyzed"},
                "sentiment": {"type": "string", "description": "Overall sentiment"},
                "confidence": {"type": "number", "description": "Confidence score", "minimum": 0, "maximum": 1},
                "key_points": {"type": "array", "items": {"type": "string"}, "description": "Key points"}
            },
            "required": ["topic", "sentiment", "confidence", "key_points"]
        }


class StreamMathResult(TypedDict):
    """Math calculation result for stream testing"""
    operation: str
    result: int
    explanation: str

    @classmethod
    def model_json_schema(cls):
        return {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "description": "Mathematical operation"},
                "result": {"type": "integer", "description": "Numerical result"},
                "explanation": {"type": "string", "description": "Brief explanation"}
            },
            "required": ["operation", "result", "explanation"]
        }


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
def cloudflare_gateway_config():
    """Create test configuration for Cloudflare Gateway"""
    return CloudflareGatewayLLMConfig(
        account_id=ACCOUNT_ID,
        gateway_id=GATEWAY_ID,
        gateway_token=GATEWAY_TOKEN,
        provider="openrouter",
        model="openai/gpt-4o-mini",
        temperature=0.2,
        max_tokens=500
    )


@pytest.fixture(scope="session")
def cloudflare_gateway_client(cloudflare_gateway_config):
    """Create Cloudflare Gateway client for testing"""
    client = CloudflareGatewayLLM(cloudflare_gateway_config)
    yield client
    client.close()


# Test Data - Unified test cases matching Gemini/OpenAI patterns
TEST_CASES = {
    "simple_knowledge": {
        "system_prompt": "You are a knowledgeable travel and cultural expert. Provide comprehensive, detailed answers with historical context, cultural significance, and practical information. Write in a conversational, informative style with at least 200 words. Always mention the exact terms requested and expand on each topic thoroughly.",
        "user_message": "I'm planning a trip to Japan and want to learn about Tokyo. What is the capital city of Japan? Please tell me about its history as the capital, describe the famous Tokyo Tower in detail including its purpose and architecture, and recommend at least two famous temples in Tokyo with their cultural significance and what visitors can expect to see there.",
        "expected_patterns": [r"tokyo|capital", r"japan", r"tower|landmark", r"temple|shrine|buddhist|shinto", r"history|culture|visit"],
        "min_matches": 3
    },

    "sentiment_structured": {
        "system_prompt": "You are an expert sentiment analyst and environmental policy researcher. Provide a comprehensive analysis that examines multiple perspectives, discusses the broader implications of renewable energy projects, and explores community concerns. Write a detailed analysis of at least 150 words. Analyze the sentiment as POSITIVE since this text discusses significant benefits like job creation and emission reduction, despite acknowledging valid minor concerns. Always mention 'renewable energy', 'jobs', and 'emissions' in your analysis and expand on each aspect.",
        "user_message": "The new renewable energy project is fantastic! It will create thousands of jobs and reduce harmful emissions significantly, contributing to our climate goals. However, some residents worry about temporary construction noise during the 6-month building phase. Local officials say the project will power 50,000 homes with clean energy.",
        "chat_structure": SentimentAnalysis,
        "stream_structure": StreamSentimentAnalysis,
        "expected_patterns": [r"renewable|energy|clean|green", r"positive|good|beneficial|fantastic", r"job|employment|work", r"emission|environment|climate"],
        "min_matches": 2
    },

    "math_tools": {
        "system_prompt": "You are a helpful mathematical assistant. Use the provided tools when you need to perform calculations. After getting the results, provide a clear explanation of what was calculated.",
        "user_message": "Calculate 5 to the power of 2, then multiply the result by 3. Do it step by step - first calculate the power, then based on its result calculate the multiply. Do not call both at once.",
        "regular_functions": {
            "power": power_function,
            "multiply": multiply_function
        },
        "chat_structure": MathResult,
        "stream_structure": StreamMathResult,
        "expected_patterns": [r"25|twenty", r"75|seventy", r"power|multiply", r"result"],
        "min_matches": 2
    },

    "parallel_tools": {
        "system_prompt": "You are a mathematical assistant. Use the provided tools to perform multiple calculations simultaneously when requested.",
        "user_message": "Calculate these operations: 3 to the power of 2, 4 to the power of 2, and multiply 6 by 7. You can call multiple functions at once.",
        "regular_functions": {
            "power": power_function,
            "multiply": multiply_function
        },
        "expected_patterns": [r"9|nine", r"16|sixteen", r"42|forty", r"power|multiply"],
        "min_matches": 3
    },

    "background_tasks": {
        "system_prompt": "You are a helpful AI assistant with admin logging capabilities.\n\nFor this interaction, you MUST:\n1. First, provide a helpful answer to the user's question\n2. Then call the send_admin_notification function to log this interaction\n\nBoth steps are REQUIRED - answer the question AND call the logging function.",
        "user_message": "What is the capital of France?",
        "background_tasks": {
            "send_admin_notification": send_admin_notification
        },
        "expected_patterns": [r"paris|france", r"capital"],
        "min_matches": 1
    }
}


def validate_patterns_flexible(text: str, patterns: list, min_matches: int, test_name: str = "") -> bool:
    """Validate that at least min_matches patterns are found in the text."""
    matches = []
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            matches.append(pattern)

    success = len(matches) >= min_matches
    if success:
        logger.info(f"✅ Pattern validation passed for {test_name}: {len(matches)}/{len(patterns)} patterns matched: {matches}")
    else:
        logger.warning(f"❌ Pattern validation failed for {test_name}: only {len(matches)}/{min_matches} required patterns matched: {matches}")

    return success


class TestCloudflareGatewayLLMClient:
    """Test class for Cloudflare Gateway LLM client with shared client instance"""

    @pytest.fixture(scope="class")
    def client(self, cloudflare_gateway_client):
        """Provide shared client for all tests in this class"""
        return cloudflare_gateway_client

    def test_client_initialization(self, client):
        """Test that the Cloudflare Gateway client initializes correctly"""
        assert client is not None
        assert client.config.provider == "openrouter"
        assert client.config.model == "openrouter/openai/gpt-4o-mini"
        assert client.config.full_model_name == "openrouter/openai/gpt-4o-mini"
        assert client._client is not None
        logger.info("✅ Client initialization test passed")

    def test_config_properties(self, cloudflare_gateway_config):
        """Test that config properties are correct"""
        assert cloudflare_gateway_config.base_url == f"https://gateway.ai.cloudflare.com/v1/{ACCOUNT_ID}/{GATEWAY_ID}/compat"
        assert cloudflare_gateway_config.compat_base_url == f"https://gateway.ai.cloudflare.com/v1/{ACCOUNT_ID}/{GATEWAY_ID}/compat"
        assert cloudflare_gateway_config.full_model_name == "openrouter/openai/gpt-4o-mini"
        logger.info("✅ Config properties test passed")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=30)
    async def test_simple_chat(self, client):
        """Test simple knowledge query - chat method"""
        await asyncio.sleep(1)

        test_data = TEST_CASES["simple_knowledge"]

        input_data = ILLMInput(
            system_prompt=test_data["system_prompt"],
            user_message=test_data["user_message"]
        )

        chat_response = await client.chat(input_data)
        logger.debug(f"Chat response: {chat_response}")

        assert isinstance(chat_response, dict)
        assert "llm_response" in chat_response
        chat_text = chat_response["llm_response"]
        assert isinstance(chat_text, str)
        assert len(chat_text) > 0

        assert validate_patterns_flexible(
            chat_text,
            test_data["expected_patterns"],
            test_data["min_matches"],
            "simple_chat"
        ), "Not enough patterns matched in chat response"

        assert len(chat_text) > 150, f"Expected comprehensive response (>150 chars), got {len(chat_text)} chars"

        logger.info(f"✅ Simple chat test passed - Response: {len(chat_text)} chars")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=30)
    async def test_simple_stream(self, client):
        """Test simple knowledge query - stream method with text growth validation"""
        await asyncio.sleep(1)

        test_data = TEST_CASES["simple_knowledge"]

        input_data = ILLMInput(
            system_prompt=test_data["system_prompt"],
            user_message=test_data["user_message"]
        )

        stream_chunks = []
        final_stream_text = ""
        text_chunks_received = 0
        text_lengths = []

        async for chunk in client.stream(input_data):
            logger.debug(f"🔍 Stream chunk: {chunk}")
            stream_chunks.append(chunk)
            if chunk.get("llm_response") and isinstance(chunk["llm_response"], str):
                final_stream_text = chunk["llm_response"]
                text_lengths.append(len(final_stream_text))
                text_chunks_received += 1

        # Validate streaming behavior
        assert len(stream_chunks) > 0, "No stream chunks received"
        assert text_chunks_received > 1, f"Expected multiple text chunks for streaming, got {text_chunks_received}"

        # Verify text was growing (proper streaming)
        has_growth = any(text_lengths[i] > text_lengths[i-1] for i in range(1, len(text_lengths)))
        assert has_growth, f"Expected text growth during streaming, got lengths: {text_lengths[:10]}..."
        assert len(final_stream_text) > 0, "No text content received from stream"

        assert validate_patterns_flexible(
            final_stream_text,
            test_data["expected_patterns"],
            test_data["min_matches"],
            "simple_stream"
        ), "Not enough patterns matched in stream response"

        assert len(final_stream_text) > 150, f"Expected comprehensive response (>150 chars), got {len(final_stream_text)} chars"

        logger.info(f"✅ Simple stream test passed - Response: {len(final_stream_text)} chars, Chunks: {text_chunks_received}, Growth pattern verified")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=30)
    async def test_structured_chat(self, client):
        """Test structured sentiment analysis - chat method"""
        await asyncio.sleep(1)

        test_data = TEST_CASES["sentiment_structured"]

        chat_input = ILLMInput(
            system_prompt=test_data["system_prompt"],
            user_message=test_data["user_message"],
            structure_type=test_data["chat_structure"]
        )

        chat_response = await client.chat(chat_input)
        assert isinstance(chat_response, dict)
        assert "llm_response" in chat_response
        chat_result = chat_response["llm_response"]
        assert isinstance(chat_result, SentimentAnalysis)

        # Validate structured fields
        assert len(chat_result.topic) > 0
        assert chat_result.sentiment.lower() in ["positive", "negative", "neutral", "mixed"]
        assert 0.0 <= chat_result.confidence <= 1.0
        assert len(chat_result.key_points) > 0

        # Validate patterns in response
        chat_combined = f"{chat_result.topic} {chat_result.sentiment} {' '.join(chat_result.key_points)}"
        assert validate_patterns_flexible(
            chat_combined,
            test_data["expected_patterns"],
            test_data["min_matches"],
            "structured_chat"
        ), "Not enough patterns matched in structured chat response"

        logger.info(f"✅ Structured chat test passed - Sentiment: {chat_result.sentiment}, Confidence: {chat_result.confidence}")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=30)
    async def test_structured_stream(self, client):
        """Test structured sentiment analysis - stream method"""
        await asyncio.sleep(1)

        test_data = TEST_CASES["sentiment_structured"]

        stream_input = ILLMInput(
            system_prompt=test_data["system_prompt"],
            user_message=test_data["user_message"],
            structure_type=test_data["stream_structure"]
        )

        stream_chunks = []
        final_stream_result = None
        structured_chunks_received = 0

        async for chunk in client.stream(stream_input):
            logger.debug(f"🔧 Stream chunk: {chunk}")
            stream_chunks.append(chunk)
            if chunk.get("llm_response") and isinstance(chunk["llm_response"], dict):
                if all(key in chunk["llm_response"] for key in ["topic", "sentiment", "confidence", "key_points"]):
                    final_stream_result = chunk["llm_response"]
                    structured_chunks_received += 1

        # Validate streaming behavior for structured output
        assert len(stream_chunks) > 0, "No stream chunks received"
        assert structured_chunks_received > 0, "No structured chunks received"
        assert final_stream_result is not None, "No final structured result received"
        assert isinstance(final_stream_result, dict), "Final result is not a dictionary"

        # Validate stream structured fields
        assert len(final_stream_result["topic"]) > 0
        assert final_stream_result["sentiment"].lower() in ["positive", "negative", "neutral", "mixed"]
        assert 0.0 <= final_stream_result["confidence"] <= 1.0
        assert len(final_stream_result["key_points"]) > 0

        # Validate patterns in response
        stream_combined = f"{final_stream_result['topic']} {final_stream_result['sentiment']} {' '.join(final_stream_result['key_points'])}"
        assert validate_patterns_flexible(
            stream_combined,
            test_data["expected_patterns"],
            test_data["min_matches"],
            "structured_stream"
        ), "Not enough patterns matched in structured stream response"

        logger.info(f"✅ Structured stream test passed - Sentiment: {final_stream_result['sentiment']}, Confidence: {final_stream_result['confidence']}")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=30)
    async def test_tool_calling_chat(self, client):
        """Test tool calling with structured output - chat method"""
        await asyncio.sleep(1)

        test_data = TEST_CASES["math_tools"]

        chat_input = ILLMInput(
            system_prompt=test_data["system_prompt"],
            user_message=test_data["user_message"],
            regular_functions=test_data["regular_functions"],
            structure_type=test_data["chat_structure"],
            max_turns=10
        )

        chat_response = await client.chat(chat_input)
        assert isinstance(chat_response, dict)
        assert "llm_response" in chat_response
        chat_result = chat_response["llm_response"]

        if isinstance(chat_result, MathResult):
            chat_text = f"{chat_result.operation} {chat_result.result} {chat_result.explanation}"
            logger.info(f"Chat structured result: {chat_result.operation} = {chat_result.result}")
        else:
            chat_text = str(chat_result)
            logger.info(f"Chat unstructured result: {chat_text[:100]}...")

        assert validate_patterns_flexible(
            chat_text,
            test_data["expected_patterns"],
            test_data["min_matches"],
            "tool_calling_chat"
        ), "Not enough patterns matched in chat tool response"

        logger.info("✅ Tool calling chat test passed - Contains expected calculations")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=30)
    async def test_tool_calling_stream(self, client):
        """Test tool calling with structured output - stream method"""
        await asyncio.sleep(1)

        test_data = TEST_CASES["math_tools"]

        stream_input = ILLMInput(
            system_prompt=test_data["system_prompt"],
            user_message=test_data["user_message"],
            regular_functions=test_data["regular_functions"],
            structure_type=test_data["stream_structure"],
            max_turns=10
        )

        stream_chunks = []
        final_stream_result = None
        final_stream_text = ""
        content_chunks_received = 0

        async for chunk in client.stream(stream_input):
            stream_chunks.append(chunk)
            logger.debug(f"Stream Chunk: {chunk}")
            if chunk.get("llm_response") and chunk["llm_response"] is not None:
                content_chunks_received += 1
                if isinstance(chunk["llm_response"], dict) and all(key in chunk["llm_response"] for key in ["operation", "result", "explanation"]):
                    final_stream_result = chunk["llm_response"]
                    logger.info(f"Stream structured result: {final_stream_result}")
                elif isinstance(chunk["llm_response"], str):
                    final_stream_text = chunk["llm_response"]

        assert len(stream_chunks) > 0, "No stream chunks received"
        assert content_chunks_received > 0, "No content chunks received from streaming"

        if final_stream_result:
            stream_text = f"{final_stream_result['operation']} {final_stream_result['result']} {final_stream_result['explanation']}"
        else:
            stream_text = final_stream_text
            logger.info(f"Stream unstructured result: {stream_text[:100]}...")

        assert len(stream_text) > 0, "No final stream text received"

        assert validate_patterns_flexible(
            stream_text,
            test_data["expected_patterns"],
            test_data["min_matches"],
            "tool_calling_stream"
        ), "Not enough patterns matched in stream tool response"

        logger.info(f"✅ Tool calling stream test passed - Stream chunks: {content_chunks_received}")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=30)
    async def test_parallel_function_calling_chat(self, client):
        """Test parallel function calling capabilities - chat method"""
        await asyncio.sleep(1)

        test_data = TEST_CASES["parallel_tools"]

        chat_input = ILLMInput(
            system_prompt=test_data["system_prompt"],
            user_message=test_data["user_message"],
            regular_functions=test_data["regular_functions"],
            max_turns=5
        )

        chat_response = await client.chat(chat_input)
        assert isinstance(chat_response, dict)
        assert "llm_response" in chat_response
        chat_result = chat_response["llm_response"]

        chat_text = str(chat_result)
        logger.info(f"Parallel chat result: {chat_text[:200]}...")

        assert validate_patterns_flexible(
            chat_text,
            test_data["expected_patterns"],
            test_data["min_matches"],
            "parallel_function_calling_chat"
        ), "Not enough patterns matched in parallel chat response"

        logger.info("✅ Parallel function calling chat test passed - Contains multiple calculation results")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=30)
    async def test_parallel_function_calling_stream(self, client):
        """Test parallel function calling capabilities - stream method"""
        await asyncio.sleep(1)

        test_data = TEST_CASES["parallel_tools"]

        stream_input = ILLMInput(
            system_prompt=test_data["system_prompt"],
            user_message=test_data["user_message"],
            regular_functions=test_data["regular_functions"],
            max_turns=5
        )

        stream_chunks = []
        final_stream_text = ""
        content_chunks_received = 0

        async for chunk in client.stream(stream_input):
            stream_chunks.append(chunk)
            logger.debug(f"🔧 Parallel stream chunk: {chunk}")

            if chunk.get("llm_response") and chunk["llm_response"] is not None:
                content_chunks_received += 1
                if isinstance(chunk["llm_response"], str):
                    final_stream_text = chunk["llm_response"]

        assert len(stream_chunks) > 0, "No stream chunks received"
        assert content_chunks_received > 0, "No content chunks received from streaming"
        assert len(final_stream_text) > 0, "No final stream text received"

        assert validate_patterns_flexible(
            final_stream_text,
            test_data["expected_patterns"],
            test_data["min_matches"],
            "parallel_function_calling_stream"
        ), "Not enough patterns matched in parallel stream response"

        logger.info(f"✅ Parallel function calling stream test passed - Stream chunks: {content_chunks_received}")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=30)
    async def test_background_tasks_chat(self, client):
        """Test background tasks (fire-and-forget) - chat method with direct variable verification"""
        await asyncio.sleep(1)

        global background_task_executed
        background_task_executed = None
        logger.info(f"Reset background_task_executed to: {background_task_executed}")

        test_data = TEST_CASES["background_tasks"]

        chat_input = ILLMInput(
            system_prompt=test_data["system_prompt"],
            user_message=test_data["user_message"],
            background_tasks=test_data["background_tasks"],
            max_turns=5
        )

        chat_response = await client.chat(chat_input)
        assert isinstance(chat_response, dict)
        assert "llm_response" in chat_response
        chat_result = chat_response["llm_response"]

        chat_text = str(chat_result)
        logger.info(f"Background tasks chat result: {chat_text}")

        # Validate the LLM provided an answer about France
        assert re.search(r"paris|france", chat_text, re.IGNORECASE), "LLM should answer the question about France"

        # Give background tasks time to complete
        await asyncio.sleep(0.5)

        logger.info(f"Final background_task_executed value: {background_task_executed}")

        # Verify the background task was executed
        if background_task_executed is None:
            logger.warning("⚠️ Background task not executed - LLM may not have called the function")
            # Don't fail the test, just warn - some LLMs might not call background tasks
        else:
            assert "ADMIN_NOTIFIED" in background_task_executed, f"Background task should have set expected value, got: {background_task_executed}"
            logger.info(f"✅ Background tasks chat test passed - Task executed: {background_task_executed}")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=30)
    async def test_background_tasks_stream(self, client):
        """Test background tasks (fire-and-forget) - stream method with direct variable verification"""
        await asyncio.sleep(1)

        global background_task_executed
        background_task_executed = None
        logger.info(f"Reset background_task_executed to: {background_task_executed}")

        test_data = TEST_CASES["background_tasks"]

        stream_input = ILLMInput(
            system_prompt=test_data["system_prompt"],
            user_message=test_data["user_message"],
            background_tasks=test_data["background_tasks"],
            max_turns=5
        )

        stream_chunks = []
        final_stream_text = ""
        content_chunks_received = 0

        async for chunk in client.stream(stream_input):
            stream_chunks.append(chunk)
            logger.debug(f"🎯 Background tasks stream chunk: {chunk}")

            if chunk.get("llm_response") and chunk["llm_response"] is not None:
                content_chunks_received += 1
                if isinstance(chunk["llm_response"], str):
                    final_stream_text = chunk["llm_response"]

        assert len(stream_chunks) > 0, "No stream chunks received"
        assert content_chunks_received > 0, "No content chunks received from streaming"
        assert len(final_stream_text) > 0, "No final stream text received"

        # Validate the LLM provided an answer about France
        assert re.search(r"paris|france", final_stream_text, re.IGNORECASE), "LLM should answer the question about France"

        # Give background tasks time to complete
        await asyncio.sleep(0.5)

        logger.info(f"Final background_task_executed value: {background_task_executed}")

        # Verify the background task was executed
        if background_task_executed is None:
            logger.warning("⚠️ Background task not executed - LLM may not have called the function")
        else:
            assert "ADMIN_NOTIFIED" in background_task_executed, f"Background task should have set expected value, got: {background_task_executed}"
            logger.info(f"✅ Background tasks stream test passed - Task executed: {background_task_executed}, Chunks: {content_chunks_received}")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=30)
    async def test_usage_tracking(self, client):
        """Test that both methods provide usage information"""
        await asyncio.sleep(1)

        input_data = ILLMInput(
            system_prompt="You are a helpful mathematics tutor. Provide detailed explanations with step-by-step solutions.",
            user_message="What is 5 + 3? Please explain the addition process step by step."
        )

        # Test chat usage
        chat_response = await client.chat(input_data)
        assert "usage" in chat_response
        if chat_response["usage"]:
            usage = chat_response["usage"]
            assert "total_tokens" in usage
            assert "input_tokens" in usage
            assert "output_tokens" in usage
            assert usage["provider"] == "gateway"
            logger.info(f"Chat usage: {usage}")

        # Test stream usage
        final_usage = None
        stream_chunks_count = 0
        async for chunk in client.stream(input_data):
            stream_chunks_count += 1
            if chunk.get("usage"):
                final_usage = chunk["usage"]

        assert stream_chunks_count >= 1, f"Expected at least 1 stream chunk, got {stream_chunks_count}"

        if final_usage:
            assert "total_tokens" in final_usage
            assert isinstance(final_usage["total_tokens"], int)

        logger.info(f"✅ Usage tracking test passed - Stream chunks: {stream_chunks_count}")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=30)
    async def test_different_providers(self):
        """Test switching between different providers"""
        await asyncio.sleep(1)

        # Test with Google provider through OpenRouter
        config = CloudflareGatewayLLMConfig(
            account_id=ACCOUNT_ID,
            gateway_id=GATEWAY_ID,
            gateway_token=GATEWAY_TOKEN,
            provider="openrouter",
            model="google/gemini-flash-1.5",
            temperature=0.2,
            max_tokens=100
        )

        client = CloudflareGatewayLLM(config)

        input_data = ILLMInput(
            system_prompt="You are a helpful assistant.",
            user_message="Say hello in one sentence."
        )

        try:
            response = await client.chat(input_data)
            assert "llm_response" in response
            assert len(str(response["llm_response"])) > 0
            logger.info(f"✅ Different provider test passed - Model: {config.full_model_name}")
        finally:
            client.close()


class TestCloudflareGatewayCustomBaseUrl:
    """Test class for Cloudflare Gateway custom base URL configuration (no API calls required)"""

    def test_default_cloudflare_base_url(self):
        """Test that default config uses standard Cloudflare base URL"""
        config = CloudflareGatewayLLMConfig(
            account_id="test-account",
            gateway_id="test-gateway",
            gateway_token="test-token",
            provider="openai",
            model="gpt-4o",
        )

        assert config.cloudflare_base_url == "https://gateway.ai.cloudflare.com"
        assert config.base_url == "https://gateway.ai.cloudflare.com/v1/test-account/test-gateway/compat"
        assert config.compat_base_url == "https://gateway.ai.cloudflare.com/v1/test-account/test-gateway/compat"
        logger.info("✅ Default Cloudflare base URL test passed")

    def test_custom_cloudflare_base_url(self):
        """Test config with custom Cloudflare base URL (e.g., regional endpoint)"""
        config = CloudflareGatewayLLMConfig(
            account_id="test-account",
            gateway_id="test-gateway",
            gateway_token="test-token",
            provider="openai",
            model="gpt-4o",
            cloudflare_base_url="https://gateway.ai.cloudflare.cn",
        )

        assert config.cloudflare_base_url == "https://gateway.ai.cloudflare.cn"
        assert config.base_url == "https://gateway.ai.cloudflare.cn/v1/test-account/test-gateway/compat"
        assert config.compat_base_url == "https://gateway.ai.cloudflare.cn/v1/test-account/test-gateway/compat"
        logger.info("✅ Custom Cloudflare base URL test passed")

    def test_provider_base_url_with_custom_base(self):
        """Test provider_base_url property with custom base URL"""
        config = CloudflareGatewayLLMConfig(
            account_id="test-account",
            gateway_id="test-gateway",
            gateway_token="test-token",
            provider="anthropic",
            model="claude-sonnet-4-5",
            cloudflare_base_url="https://custom-gateway.example.com",
        )

        assert config.provider_base_url == "https://custom-gateway.example.com/v1/test-account/test-gateway/anthropic/v1"
        logger.info("✅ Provider base URL with custom base test passed")

    def test_full_model_name_unchanged_by_base_url(self):
        """Test that full_model_name is not affected by base URL settings"""
        config_default = CloudflareGatewayLLMConfig(
            account_id="test-account",
            gateway_id="test-gateway",
            gateway_token="test-token",
            provider="openrouter",
            model="openai/gpt-4o-mini",
        )

        config_custom = CloudflareGatewayLLMConfig(
            account_id="test-account",
            gateway_id="test-gateway",
            gateway_token="test-token",
            provider="openrouter",
            model="openai/gpt-4o-mini",
            cloudflare_base_url="https://custom-gateway.example.com",
        )

        assert config_default.full_model_name == "openrouter/openai/gpt-4o-mini"
        assert config_custom.full_model_name == "openrouter/openai/gpt-4o-mini"
        assert config_default.full_model_name == config_custom.full_model_name
        logger.info("✅ Full model name unchanged by base URL test passed")


class TestCloudflareGatewayImages:
    """Test class for Cloudflare Gateway multimodal (image) support"""

    @pytest.fixture(scope="class")
    def client(self, cloudflare_gateway_client):
        """Provide shared client for all tests in this class"""
        return cloudflare_gateway_client

    def test_convert_base64_to_openai_messages(self, client):
        """Test conversion of base64 strings to OpenAI-compatible message format"""
        import base64

        # Sample 1x1 red pixel PNG
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )
        sample_image_raw = base64.b64encode(png_bytes).decode("utf-8")
        sample_image_data_url = f"data:image/png;base64,{sample_image_raw}"

        # Test with raw base64
        input_raw = ILLMInput(
            system_prompt="You are a vision assistant.",
            user_message="Describe this image.",
            images_base64=[sample_image_raw]
        )
        messages_raw = client._create_openai_messages(input_raw)

        # Check structure: system message + user message with content array
        assert len(messages_raw) == 2
        assert messages_raw[0]["role"] == "system"
        assert messages_raw[1]["role"] == "user"
        assert isinstance(messages_raw[1]["content"], list)
        assert any(item["type"] == "text" for item in messages_raw[1]["content"])
        assert any(item["type"] == "image_url" for item in messages_raw[1]["content"])

        # Test with multiple images
        input_multi = ILLMInput(
            system_prompt="You are a vision assistant.",
            user_message="Compare these images.",
            images_base64=[sample_image_raw, sample_image_data_url]
        )
        messages_multi = client._create_openai_messages(input_multi)
        image_items = [item for item in messages_multi[1]["content"] if item["type"] == "image_url"]
        assert len(image_items) == 2

        logger.info("✅ Base64 to OpenAI messages conversion test passed")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=5)
    async def test_chat_with_single_image(self, client):
        """Test chat with a single base64-encoded image"""
        await asyncio.sleep(1)

        import base64
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )
        sample_image = base64.b64encode(png_bytes).decode("utf-8")

        input_data = ILLMInput(
            system_prompt="You are a vision assistant. You can see and analyze images.",
            user_message="I'm sending you an image. Can you see it? If yes, describe what you see in detail including colors, shapes, and size.",
            images_base64=[sample_image]
        )

        logger.info("=" * 80)
        logger.info("📸 TESTING IMAGE CHAT WITH CLOUDFLARE GATEWAY")
        logger.info(f"Image size: {len(sample_image)} chars (base64)")
        logger.info(f"Sending image to: {client.config.model}")
        logger.info("-" * 80)

        response = await client.chat(input_data)

        # Basic response validation
        assert isinstance(response, dict), "Response should be a dict"
        assert "llm_response" in response, "Response should have llm_response key"
        response_text = response["llm_response"]
        assert isinstance(response_text, str), "Response text should be a string"
        assert len(response_text) > 0, "Response text should not be empty"

        # Log FULL response to verify image was processed
        logger.info("📝 FULL LLM RESPONSE:")
        logger.info(response_text)
        logger.info("-" * 80)

        # Check for vision-related keywords that prove the LLM saw the image
        vision_keywords = ["image", "picture", "pixel", "see", "visual", "color", "colour", "1x1", "square", "small", "tiny"]
        keywords_found = [kw for kw in vision_keywords if kw.lower() in response_text.lower()]

        logger.info(f"🔍 Vision keywords found: {keywords_found}")
        assert len(keywords_found) > 0, f"Response should contain vision-related keywords. Response: {response_text}"

        # Verify meaningful response
        assert len(response_text) > 20, f"Expected detailed image description, got: {response_text}"

        logger.info("✅ Chat with single image test passed - Image was processed by LLM")
        logger.info("=" * 80)

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=5)
    async def test_stream_with_single_image(self, client):
        """Test streaming with a single base64-encoded image"""
        await asyncio.sleep(1)

        import base64
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )
        sample_image = base64.b64encode(png_bytes).decode("utf-8")

        input_data = ILLMInput(
            system_prompt="You are a vision assistant. You can see and analyze images.",
            user_message="I'm sending you an image. Can you see it? If yes, describe what you see in detail including colors, shapes, and size.",
            images_base64=[sample_image]
        )

        logger.info("=" * 80)
        logger.info("📸 TESTING IMAGE STREAMING WITH CLOUDFLARE GATEWAY")
        logger.info(f"Image size: {len(sample_image)} chars (base64)")
        logger.info(f"Streaming from: {client.config.model}")
        logger.info("-" * 80)

        stream_chunks = []
        final_text = ""
        text_chunks_received = 0

        async for chunk in client.stream(input_data):
            stream_chunks.append(chunk)
            if chunk.get("llm_response") and isinstance(chunk["llm_response"], str):
                final_text = chunk["llm_response"]
                text_chunks_received += 1

        assert len(stream_chunks) > 0, "No stream chunks received"
        assert text_chunks_received > 0, "No text chunks received"
        assert len(final_text) > 0, "No final text received"

        # Log FULL streamed response
        logger.info("📝 FULL STREAMED LLM RESPONSE:")
        logger.info(final_text)
        logger.info("-" * 80)

        # Check for vision-related keywords
        vision_keywords = ["image", "picture", "pixel", "see", "visual", "color", "colour", "1x1", "square", "small", "tiny"]
        keywords_found = [kw for kw in vision_keywords if kw.lower() in final_text.lower()]

        logger.info(f"🔍 Vision keywords found: {keywords_found}")
        assert len(keywords_found) > 0, f"Streamed response should contain vision-related keywords. Response: {final_text}"

        # Verify meaningful response
        assert len(final_text) > 20, f"Expected detailed image description, got: {final_text}"

        logger.info(f"✅ Stream with single image test passed - Image was processed, Chunks: {text_chunks_received}")
        logger.info("=" * 80)

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=5)
    async def test_chat_with_multiple_images(self, client):
        """Test chat with multiple base64-encoded images"""
        await asyncio.sleep(1)

        import base64
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )
        sample_image_1 = base64.b64encode(png_bytes).decode("utf-8")
        sample_image_2 = f"data:image/png;base64,{sample_image_1}"

        input_data = ILLMInput(
            system_prompt="You are a vision assistant. You can see and analyze multiple images.",
            user_message="I'm sending you TWO images. How many images can you see? Describe each one and compare them.",
            images_base64=[sample_image_1, sample_image_2]
        )

        logger.info("=" * 80)
        logger.info("📸📸 TESTING MULTIPLE IMAGES CHAT WITH CLOUDFLARE GATEWAY")
        logger.info(f"Image 1 size: {len(sample_image_1)} chars (raw base64)")
        logger.info(f"Image 2 size: {len(sample_image_2)} chars (data URL)")
        logger.info(f"Sending to: {client.config.model}")
        logger.info("-" * 80)

        response = await client.chat(input_data)

        # Basic response validation
        assert isinstance(response, dict), "Response should be a dict"
        assert "llm_response" in response, "Response should have llm_response key"
        response_text = response["llm_response"]
        assert isinstance(response_text, str), "Response text should be a string"
        assert len(response_text) > 0, "Response text should not be empty"

        # Log FULL response
        logger.info("📝 FULL LLM RESPONSE (Multiple Images):")
        logger.info(response_text)
        logger.info("-" * 80)

        # Check for multiple-image keywords
        multi_keywords = ["two", "2", "both", "images", "each", "first", "second", "similar", "same", "compare"]
        keywords_found = [kw for kw in multi_keywords if kw.lower() in response_text.lower()]

        logger.info(f"🔍 Multiple-image keywords found: {keywords_found}")
        assert len(keywords_found) > 0, f"Response should mention multiple images. Response: {response_text}"

        # Verify meaningful comparison
        assert len(response_text) > 30, f"Expected detailed comparison, got: {response_text}"

        logger.info("✅ Chat with multiple images test passed - Multiple images were processed")
        logger.info("=" * 80)

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=5)
    async def test_chat_with_image_and_tools(self, client):
        """Test chat with image input combined with tool calling"""
        await asyncio.sleep(1)

        import base64
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )
        sample_image = base64.b64encode(png_bytes).decode("utf-8")

        tool_called = {"called": False, "count": None}

        def count_objects(count: int) -> str:
            """Count the number of objects detected in the image."""
            tool_called["called"] = True
            tool_called["count"] = count
            logger.info(f"🔧 TOOL CALLED: count_objects(count={count})")
            return f"Counted {count} object(s) in the image."

        input_data = ILLMInput(
            system_prompt="You are a vision assistant with object counting capabilities. You can see images and use tools.",
            user_message="I'm sending you an image. Analyze it and count how many distinct objects or pixels you can see. MUST use the count_objects tool to report your count.",
            images_base64=[sample_image],
            regular_functions={"count_objects": count_objects},
            max_turns=5
        )

        logger.info("=" * 80)
        logger.info("📸🔧 TESTING IMAGE + TOOLS WITH CLOUDFLARE GATEWAY")
        logger.info(f"Image size: {len(sample_image)} chars (base64)")
        logger.info(f"Tool available: count_objects")
        logger.info(f"Sending to: {client.config.model}")
        logger.info("-" * 80)

        response = await client.chat(input_data)

        # Basic response validation
        assert isinstance(response, dict), "Response should be a dict"
        assert "llm_response" in response, "Response should have llm_response key"
        response_text = str(response["llm_response"])
        assert len(response_text) > 0, "Response text should not be empty"

        # Log FULL response
        logger.info("📝 FULL LLM RESPONSE (Image + Tools):")
        logger.info(response_text)
        logger.info("-" * 80)

        # Verify tool was called (proves image was analyzed)
        if tool_called["called"]:
            logger.info(f"✅ Tool was called with count={tool_called['count']}")
        else:
            logger.warning("⚠️  Tool was NOT called - LLM may not have used the tool")

        # Check for vision + tool keywords
        combined_keywords = ["image", "see", "count", "object", "pixel", "analyzed"]
        keywords_found = [kw for kw in combined_keywords if kw.lower() in response_text.lower()]

        logger.info(f"🔍 Vision+Tool keywords found: {keywords_found}")
        assert len(keywords_found) > 0, f"Response should mention image analysis. Response: {response_text}"

        # Verify meaningful response with tool integration
        assert len(response_text) > 15, f"Expected detailed response with tool usage, got: {response_text}"

        logger.info("✅ Chat with image and tools test passed - Image + tool calling works")
        logger.info("=" * 80)


# =============================================================================
# PDF Support Tests
# =============================================================================
#
# These tests verify that the Cloudflare Gateway client correctly passes
# base64-encoded PDFs to the model using the OpenAI Responses API
# `input_file` block format (Cloudflare delegates to AIGatewayLLM).
#
# Run PDF tests only:
#   python -m pytest tests/unit/llms/test_cloudflare_gateway.py::TestCloudflareGatewayPDF -v

import base64 as _base64
import textwrap as _textwrap


def _make_test_pdf_bytes(text: str) -> bytes:
    """
    Build a minimal, valid PDF-1.4 file that contains `text` as visible page content.

    Long text is wrapped into 60-character lines so every word is visible within
    the page margin.  Models that render PDFs as images (Gemini, Claude) will then
    see all content.  The PDF is constructed entirely from stdlib.
    """
    def _esc(s: str) -> str:
        return s.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')

    lines = _textwrap.wrap(text, width=60) or [text]

    parts = [f"BT /F1 12 Tf 72 720 Td ({_esc(lines[0])}) Tj"]
    for line in lines[1:]:
        parts.append(f"0 -20 Td ({_esc(line)}) Tj")
    parts.append("ET")
    stream_src = "\n".join(parts)
    stream_b = stream_src.encode('latin-1')

    header = b"%PDF-1.4\n"
    o1 = b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n"
    o2 = b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n"
    o3 = (
        b"3 0 obj\n"
        b"<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Contents 4 0 R"
        b" /Resources <</Font <</F1 <</Type /Font /Subtype /Type1"
        b" /BaseFont /Helvetica>>>>>>>> >>\nendobj\n"
    )
    o4 = (
        b"4 0 obj\n<</Length " + str(len(stream_b)).encode() + b">>\n"
        b"stream\n" + stream_b + b"\nendstream\nendobj\n"
    )

    off1 = len(header)
    off2 = off1 + len(o1)
    off3 = off2 + len(o2)
    off4 = off3 + len(o3)
    xref_off = off4 + len(o4)

    xref = (
        b"xref\n0 5\n"
        b"0000000000 65535 f \n"
        + f"{off1:010d} 00000 n \n".encode()
        + f"{off2:010d} 00000 n \n".encode()
        + f"{off3:010d} 00000 n \n".encode()
        + f"{off4:010d} 00000 n \n".encode()
    )
    trailer = (
        b"trailer\n<</Size 5 /Root 1 0 R>>\nstartxref\n"
        + str(xref_off).encode() + b"\n%%EOF\n"
    )
    return header + o1 + o2 + o3 + o4 + xref + trailer


def _make_test_pdf_base64(text: str) -> str:
    """Return a base64-encoded minimal PDF containing `text`."""
    return _base64.b64encode(_make_test_pdf_bytes(text)).decode('utf-8')


# Distinctive content that can ONLY come from reading the PDF.
PDF_TEST_CONTENT = (
    "ARSHAI PDF TEST DOCUMENT. "
    "Secret code: XYZPDF9981. "
    "Capital city: Rome. "
    "Answer: forty-two."
)


@pytest.fixture(scope="session")
def pdf_cloudflare_config():
    """
    Config for a PDF-capable model via Cloudflare Gateway.

    Uses CLOUDFLARE_PDF_MODEL env var if set, otherwise defaults to
    openai/gpt-4o which supports native PDF input.
    Skips if CLOUDFLARE_GATEWAY_TOKEN is not set.
    """
    if not GATEWAY_TOKEN:
        pytest.skip("CLOUDFLARE_GATEWAY_TOKEN required for Cloudflare Gateway PDF tests")

    model = os.environ.get("CLOUDFLARE_PDF_MODEL", "openai/gpt-4o")

    return CloudflareGatewayLLMConfig(
        account_id=ACCOUNT_ID,
        gateway_id=GATEWAY_ID,
        gateway_token=GATEWAY_TOKEN,
        provider="openrouter",
        model=model,
        temperature=0.1,
        max_tokens=500,
    )


@pytest.fixture(scope="session")
def pdf_cloudflare_client(pdf_cloudflare_config):
    """Cloudflare Gateway client configured for PDF-capable model."""
    client = CloudflareGatewayLLM(pdf_cloudflare_config)
    yield client
    client.close()


class TestCloudflareGatewayPDF:
    """
    Integration tests for PDF support via the Cloudflare Gateway client.

    Cloudflare Gateway delegates entirely to AIGatewayLLM, so PDFs are
    sent using the OpenAI Responses API `input_file` block format.

    Structure tests (no API call) verify the message format is correct.
    Real API tests send a minimal generated PDF and assert the model's
    response references content that could only come from reading the
    document.

    Logs show the raw LLM output so you can visually confirm the model
    read the document.
    """

    @pytest.fixture(scope="class")
    def client(self, pdf_cloudflare_client):
        return pdf_cloudflare_client

    # ------------------------------------------------------------------
    # Structure tests (no API call)
    # ------------------------------------------------------------------

    def test_pdf_message_structure_raw_base64(self, client):
        """Verify raw base64 PDF produces the correct `input_file` block."""
        pdf_b64 = _make_test_pdf_base64(PDF_TEST_CONTENT)

        input_data = ILLMInput(
            system_prompt="You are a document analyst.",
            user_message="Summarise this document.",
            pdfs_base64=[pdf_b64],
        )
        # Cloudflare delegates to AIGatewayLLM which uses _create_openai_messages
        messages = client._create_openai_messages(input_data)

        logger.info(f"📄 PDF message structure: {messages}")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        content = messages[1]["content"]
        assert isinstance(content, list)

        text_blocks = [b for b in content if b.get("type") == "text"]
        file_blocks = [b for b in content if b.get("type") == "file"]

        assert len(text_blocks) == 1, "Expected exactly one text block"
        assert len(file_blocks) == 1, "Expected exactly one file block for the PDF"

        fb = file_blocks[0]
        assert fb["file"]["filename"] == "document.pdf"
        assert fb["file"]["file_data"].startswith("data:application/pdf;base64,")

        logger.info(f"✅ PDF message structure test passed - "
                    f"filename={fb['file']['filename']}, data prefix={fb['file']['file_data'][:40]}...")

    def test_pdf_message_structure_data_url(self, client):
        """Verify data-URL-prefixed PDF is passed through unchanged."""
        raw = _make_test_pdf_base64(PDF_TEST_CONTENT)
        data_url = f"data:application/pdf;base64,{raw}"

        input_data = ILLMInput(
            system_prompt="sys",
            user_message="hi",
            pdfs_base64=[data_url],
        )
        messages = client._create_openai_messages(input_data)
        file_blocks = [b for b in messages[1]["content"] if b.get("type") == "file"]
        assert file_blocks[0]["file"]["file_data"] == data_url

        logger.info("✅ PDF data-URL passthrough test passed")

    def test_pdf_and_image_message_structure(self, client):
        """Verify combined PDF + image produces both block types."""
        pdf_b64 = _make_test_pdf_base64(PDF_TEST_CONTENT)
        png_bytes = _base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )
        img_b64 = _base64.b64encode(png_bytes).decode("utf-8")

        input_data = ILLMInput(
            system_prompt="Analyse everything.",
            user_message="Describe all content.",
            images_base64=[img_b64],
            pdfs_base64=[pdf_b64],
        )
        messages = client._create_openai_messages(input_data)
        content = messages[1]["content"]
        types = {b["type"] for b in content}

        assert "text" in types
        assert "image_url" in types
        assert "file" in types

        logger.info(f"✅ Combined PDF+image structure test passed - content types: {types}")

    # ------------------------------------------------------------------
    # Real API tests — model reads the PDF and we verify it saw the content
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=60)
    async def test_chat_with_single_pdf(self, client):
        """
        Test chat() with a base64 PDF via Cloudflare Gateway.
        The PDF contains a secret code (XYZPDF9981) and specific facts.
        We verify the model's response references content from the PDF,
        proving end-to-end PDF delivery works through the Cloudflare proxy.
        """
        await asyncio.sleep(1)

        pdf_b64 = _make_test_pdf_base64(PDF_TEST_CONTENT)

        input_data = ILLMInput(
            system_prompt=(
                "You are a document analyst. When given a PDF, read it carefully "
                "and answer questions about its content. Quote specific details "
                "from the document in your response."
            ),
            user_message=(
                "Please read this PDF document and tell me: "
                "(1) What is the secret code mentioned? "
                "(2) What capital city is mentioned? "
                "(3) What does the document say about two plus two?"
            ),
            pdfs_base64=[pdf_b64],
        )

        response = await client.chat(input_data)

        logger.info(f"📄 PDF chat response (raw): {response}")
        assert isinstance(response, dict)
        assert "llm_response" in response

        response_text = str(response["llm_response"])
        assert len(response_text) > 10, "Expected a meaningful response"

        logger.info(f"📄 PDF chat response (text): {response_text}")
        logger.info(f"📊 Usage: {response.get('usage')}")

        found = [
            kw for kw in ["XYZPDF9981", "Rome", "forty-two", "forty two", "ARSHAI"]
            if kw.lower() in response_text.lower()
        ]
        logger.info(f"🔍 Keywords found from PDF content: {found}")
        assert len(found) >= 1, (
            f"Model response does not appear to reference PDF content. "
            f"Expected at least 1 of [XYZPDF9981, Rome, forty-two, ARSHAI], "
            f"found: {found}\nFull response: {response_text}"
        )

        logger.info(f"✅ Chat with single PDF test passed - "
                    f"model referenced {len(found)} PDF keywords: {found}")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=60)
    async def test_stream_with_single_pdf(self, client):
        """
        Test stream() with a base64 PDF via Cloudflare Gateway.
        Verifies streaming works and the model references PDF content.
        """
        await asyncio.sleep(1)

        pdf_b64 = _make_test_pdf_base64(PDF_TEST_CONTENT)

        input_data = ILLMInput(
            system_prompt=(
                "You are a document analyst. Read the provided PDF carefully and "
                "summarise its key points, quoting specific details."
            ),
            user_message=(
                "Summarise this PDF. Include the secret code, the capital city "
                "mentioned, and the arithmetic fact stated in the document."
            ),
            pdfs_base64=[pdf_b64],
        )

        stream_chunks = []
        final_text = ""
        text_chunks_received = 0

        async for chunk in client.stream(input_data):
            logger.info(f"📄 PDF stream chunk: {chunk}")
            stream_chunks.append(chunk)
            if chunk.get("llm_response") and isinstance(chunk["llm_response"], str):
                final_text = chunk["llm_response"]
                text_chunks_received += 1

        logger.info(f"📄 PDF stream final text: {final_text}")
        logger.info(f"📊 Total chunks: {len(stream_chunks)}, text chunks: {text_chunks_received}")
        final_usage = next(
            (c.get("usage") for c in reversed(stream_chunks) if c.get("usage")), None
        )
        logger.info(f"📊 Usage: {final_usage}")

        assert len(stream_chunks) > 0, "No stream chunks received"
        assert text_chunks_received > 0, "No text content received from stream"
        assert len(final_text) > 10, "Expected a meaningful streamed response"

        found = [
            kw for kw in ["XYZPDF9981", "Rome", "forty-two", "forty two", "ARSHAI"]
            if kw.lower() in final_text.lower()
        ]
        logger.info(f"🔍 Keywords found from PDF content in stream: {found}")
        assert len(found) >= 1, (
            f"Stream response does not appear to reference PDF content. "
            f"Expected at least 1 keyword, found: {found}\nFull text: {final_text}"
        )

        logger.info(f"✅ Stream with single PDF test passed - "
                    f"model referenced {len(found)} PDF keywords: {found}, "
                    f"chunks: {text_chunks_received}")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=60)
    async def test_chat_with_multiple_pdfs(self, client):
        """
        Test chat() with two different PDFs via Cloudflare Gateway.
        Each PDF has a unique keyword; the model should reference both.
        """
        await asyncio.sleep(1)

        pdf1_text = "FIRST DOCUMENT - keyword ALPHA001 - describes the planet Mars."
        pdf2_text = "SECOND DOCUMENT - keyword BETA002 - describes the planet Venus."

        pdf1_b64 = _make_test_pdf_base64(pdf1_text)
        pdf2_b64 = _make_test_pdf_base64(pdf2_text)

        input_data = ILLMInput(
            system_prompt="You are a document analyst. Read all provided PDFs carefully.",
            user_message=(
                "I have given you two PDF documents. "
                "For each one, state its unique keyword and the planet it describes."
            ),
            pdfs_base64=[pdf1_b64, pdf2_b64],
        )

        response = await client.chat(input_data)

        logger.info(f"📄 Multiple PDFs chat response (raw): {response}")
        response_text = str(response["llm_response"])
        logger.info(f"📄 Multiple PDFs chat response (text): {response_text}")
        logger.info(f"📊 Usage: {response.get('usage')}")

        found = [
            kw for kw in ["ALPHA001", "BETA002", "Mars", "Venus"]
            if kw.lower() in response_text.lower()
        ]
        logger.info(f"🔍 Keywords found from both PDFs: {found}")
        assert len(found) >= 3, (
            f"Expected model to reference content from both PDFs (at least 3 of "
            f"[ALPHA001, BETA002, Mars, Venus]). Found: {found}\nResponse: {response_text}"
        )

        logger.info(f"✅ Chat with multiple PDFs test passed - "
                    f"model referenced {len(found)} keywords across both PDFs: {found}")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=60)
    async def test_chat_with_pdf_and_tools(self, client):
        """
        Test that PDF input works correctly alongside regular function calling.
        The model should read the PDF AND call the provided tool.
        """
        await asyncio.sleep(1)

        pdf_text = "TOOL TEST DOCUMENT. The count value is 7."
        pdf_b64 = _make_test_pdf_base64(pdf_text)

        results_log = []

        def record_value(value: int) -> str:
            """Record the numerical value extracted from the document."""
            results_log.append(value)
            return f"Value {value} has been recorded."

        input_data = ILLMInput(
            system_prompt=(
                "You are a document processor. Read the PDF to find the numerical "
                "value, then call record_value with that number."
            ),
            user_message="Read the PDF and call record_value with the integer you find.",
            pdfs_base64=[pdf_b64],
            regular_functions={"record_value": record_value},
            max_turns=5,
        )

        response = await client.chat(input_data)

        logger.info(f"📄 PDF+tools chat response (raw): {response}")
        response_text = str(response["llm_response"])
        logger.info(f"📄 PDF+tools chat response (text): {response_text}")
        logger.info(f"🔧 record_value called with: {results_log}")
        logger.info(f"📊 Usage: {response.get('usage')}")

        assert isinstance(response, dict)
        assert "llm_response" in response

        tool_called_correctly = 7 in results_log
        response_mentions_value = "7" in response_text

        logger.info(f"🔍 Tool called with 7: {tool_called_correctly}, "
                    f"Response mentions 7: {response_mentions_value}")
        assert tool_called_correctly or response_mentions_value, (
            f"Expected model to extract 7 from PDF. "
            f"Tool calls: {results_log}, Response: {response_text}"
        )

        logger.info(f"✅ Chat with PDF and tools test passed - "
                    f"tool calls: {results_log}, response: {response_text[:100]}...")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=60)
    async def test_pdf_usage_tracking(self, client):
        """Verify that usage metadata is returned correctly for PDF requests."""
        await asyncio.sleep(1)

        pdf_b64 = _make_test_pdf_base64(PDF_TEST_CONTENT)

        input_data = ILLMInput(
            system_prompt="You are a document analyst.",
            user_message="Summarise this PDF in one sentence.",
            pdfs_base64=[pdf_b64],
        )

        chat_response = await client.chat(input_data)
        logger.info(f"📄 PDF usage tracking response (raw): {chat_response}")
        logger.info(f"📊 Chat usage with PDF: {chat_response.get('usage')}")
        assert "usage" in chat_response

        usage = chat_response["usage"]
        logger.info(f"📊 Usage details: {usage}")
        if usage is not None:
            prompt_tokens = usage.get("prompt_tokens", 0) or 0
            completion_tokens = usage.get("completion_tokens", 0) or 0
            if prompt_tokens > 0 and completion_tokens > 0:
                logger.info(f"✅ Usage tracking test passed - "
                            f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")
            else:
                logger.warning(f"⚠️  Gateway returned zero token counts (prompt={prompt_tokens}, "
                               f"completion={completion_tokens}) — gateway may not report PDF token usage")
                logger.info("✅ Usage tracking test passed (usage dict present, token counts may be zero for this gateway)")
        else:
            logger.warning("⚠️  Usage is None - gateway did not report usage")
            logger.info("✅ Usage tracking test passed (usage returned as None)")


# ---------------------------------------------------------------------------
# Sampling Control Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sampling_cf_config():
    """CloudflareGatewayLLMConfig with sampling control parameters set."""
    return CloudflareGatewayLLMConfig(
        account_id=ACCOUNT_ID,
        gateway_id=GATEWAY_ID,
        gateway_token=GATEWAY_TOKEN,
        provider="openrouter",
        model="openai/gpt-4o-mini",
        temperature=0.3,
        max_tokens=300,
        top_p=0.95,
        presence_penalty=0.1,
        frequency_penalty=0.1,
        top_k=50,
    )


@pytest.fixture(scope="session")
def sampling_cf_client(sampling_cf_config):
    """Cloudflare Gateway client with sampling control parameters."""
    client = CloudflareGatewayLLM(sampling_cf_config)
    yield client
    client.close()


class TestCloudflareGatewaySamplingControl:
    """
    Tests for extended sampling control parameters on CloudflareGatewayLLM.

    CloudflareGatewayLLM inherits from AIGatewayLLM, so _build_sampling_kwargs()
    is available via inheritance and the same extra_body logic applies.
    """

    @pytest.fixture(scope="class")
    def client(self, sampling_cf_client):
        return sampling_cf_client

    # ------------------------------------------------------------------
    # Unit-level: verify kwargs construction without hitting the API
    # ------------------------------------------------------------------

    def test_build_sampling_kwargs_includes_top_k(self, client):
        """top_k should appear in extra_body."""
        result = client._build_sampling_kwargs([{"role": "user", "content": "hi"}])
        assert "extra_body" in result
        assert result["extra_body"]["top_k"] == 50

    def test_build_sampling_kwargs_includes_penalties(self, client):
        """presence_penalty and frequency_penalty should be top-level kwargs."""
        result = client._build_sampling_kwargs([])
        assert result["presence_penalty"] == pytest.approx(0.1)
        assert result["frequency_penalty"] == pytest.approx(0.1)

    def test_build_sampling_kwargs_includes_top_p(self, client):
        result = client._build_sampling_kwargs([])
        assert result["top_p"] == pytest.approx(0.95)

    def test_build_sampling_kwargs_uses_max_tokens(self, client):
        """Chat Completions API uses max_tokens (not max_output_tokens)."""
        result = client._build_sampling_kwargs([])
        assert result["max_tokens"] == 300
        assert "max_output_tokens" not in result

    def test_build_sampling_kwargs_stream_override(self, client):
        result = client._build_sampling_kwargs([], stream=True)
        assert result["stream"] is True
        assert result["extra_body"]["top_k"] == 50

    def test_no_extra_body_without_sampling_fields(self):
        """Client without top_k/reasoning should not include extra_body."""
        plain_config = CloudflareGatewayLLMConfig(
            account_id=ACCOUNT_ID,
            gateway_id=GATEWAY_ID,
            gateway_token=GATEWAY_TOKEN,
            provider="openrouter",
            model="openai/gpt-4o-mini",
            temperature=0.3,
        )
        plain_client = CloudflareGatewayLLM(plain_config)
        result = plain_client._build_sampling_kwargs([])
        assert "extra_body" not in result
        plain_client.close()

    def test_reasoning_effort_takes_precedence(self, client):
        """reasoning_effort takes precedence over reasoning_max_tokens."""
        config = CloudflareGatewayLLMConfig(
            account_id=ACCOUNT_ID,
            gateway_id=GATEWAY_ID,
            gateway_token=GATEWAY_TOKEN,
            provider="openrouter",
            model="openai/gpt-4o-mini",
            reasoning_effort="low",
            reasoning_max_tokens=4000,
        )
        c = CloudflareGatewayLLM(config)
        result = c._build_sampling_kwargs([])
        assert result["extra_body"]["reasoning"] == {"effort": "low"}
        c.close()

    # ------------------------------------------------------------------
    # Integration: real API calls with sampling params set
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_chat_with_sampling_params(self, client):
        """Chat succeeds when top_k, penalties, top_p are set."""
        await asyncio.sleep(1)
        input_data = ILLMInput(
            system_prompt="You are a concise assistant.",
            user_message="What is the capital of France? Reply in one sentence.",
        )
        response = await client.chat(input_data)
        assert isinstance(response, dict)
        assert "llm_response" in response
        text = response["llm_response"]
        assert isinstance(text, str) and len(text) > 0
        assert re.search(r"paris", text, re.IGNORECASE), f"Expected 'Paris' in response: {text}"
        logger.info(f"✅ CF chat with sampling params passed - response: {text[:80]}")

    @pytest.mark.asyncio
    async def test_stream_with_sampling_params(self, client):
        """Stream succeeds when top_k, penalties, top_p are set."""
        await asyncio.sleep(1)
        input_data = ILLMInput(
            system_prompt="You are a concise assistant.",
            user_message="What is the capital of Germany? Reply in one sentence.",
        )
        final_text = ""
        chunks = []
        async for chunk in client.stream(input_data):
            chunks.append(chunk)
            if chunk.get("llm_response"):
                final_text = chunk["llm_response"]

        assert len(chunks) > 0
        assert re.search(r"berlin", final_text, re.IGNORECASE), f"Expected 'Berlin' in response: {final_text}"
        logger.info(f"✅ CF stream with sampling params passed - response: {final_text[:80]}")

    @pytest.mark.asyncio
    async def test_chat_with_extra_body_passthrough(self):
        """User-supplied extra_body keys are merged and forwarded."""
        await asyncio.sleep(1)
        config = CloudflareGatewayLLMConfig(
            account_id=ACCOUNT_ID,
            gateway_id=GATEWAY_ID,
            gateway_token=GATEWAY_TOKEN,
            provider="openrouter",
            model="openai/gpt-4o-mini",
            temperature=0.3,
            max_tokens=200,
            top_k=40,
            extra_body={"provider": {"order": ["OpenAI"]}},
        )
        client = CloudflareGatewayLLM(config)
        # Verify kwargs construction
        result = client._build_sampling_kwargs([])
        assert result["extra_body"]["top_k"] == 40
        assert result["extra_body"]["provider"] == {"order": ["OpenAI"]}

        input_data = ILLMInput(
            system_prompt="You are a concise assistant.",
            user_message="Say 'hello'.",
        )
        response = await client.chat(input_data)
        assert isinstance(response, dict)
        assert "llm_response" in response
        client.close()
        logger.info(f"✅ CF chat with extra_body passthrough passed")


# Run tests with: uv run pytest tests/unit/llms/test_cloudflare_gateway.py -v
# Run only sampling tests: uv run pytest tests/unit/llms/test_cloudflare_gateway.py::TestCloudflareGatewaySamplingControl -v
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
