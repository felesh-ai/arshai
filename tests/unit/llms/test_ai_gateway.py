"""
Pytest test suite for AI Gateway LLM client.
Tests both chat and stream methods with identical inputs for direct comparison.
Includes regex validation for semantic correctness.
Based on the unified test pattern established for Gemini and OpenAI clients.
"""

import pytest
import re
import logging
import asyncio
import os
from pathlib import Path
from typing import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import functools

from arshai.core.interfaces.illm import ILLMInput
from arshai.llms.ai_gateway import AIGatewayLLM, AIGatewayConfig

# Load test environment variables
test_env_path = Path(__file__).parent / ".env.ai_gateway"
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
                    # Check if it's a rate limit error (429 or RESOURCE_EXHAUSTED)
                    error_str = str(e)
                    error_type = type(e).__name__

                    is_rate_limit = (
                        "429" in error_str or
                        "RESOURCE_EXHAUSTED" in error_str or
                        "Resource exhausted" in error_str or
                        "RateLimitError" in error_type
                    )

                    if is_rate_limit:
                        if attempt < max_retries:
                            logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). Error: {error_str[:100]}... Waiting {wait_seconds} seconds...")
                            await asyncio.sleep(wait_seconds)
                            continue
                        else:
                            logger.error(f"Rate limit exceeded after {max_retries + 1} attempts")
                            raise
                    else:
                        # Not a rate limit error, re-raise immediately
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

    # Set the test variable to verify execution
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
    key_points: list[str]
    topic: str
    sentiment: str
    confidence: float

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


# Test Configuration and Fixtures
@pytest.fixture(scope="session")
def ai_gateway_config():
    """Create test configuration for AI Gateway"""
    # Check for required environment variables
    base_url = os.getenv("GATEWAY_BASE_URL")
    gateway_token = os.getenv("GATEWAY_TOKEN")
    model = os.getenv("GATEWAY_MODEL", "gpt-4o-mini")

    if not base_url or not gateway_token:
        pytest.skip("GATEWAY_BASE_URL and GATEWAY_TOKEN environment variables required for AI Gateway tests")

    # Add User-Agent header to bypass Cloudflare bot detection
    # Use a curl-like User-Agent that Cloudflare allows
    headers = {
        "User-Agent": "curl/8.7.1"
    }

    return AIGatewayConfig(
        base_url=base_url,
        gateway_token=gateway_token,
        model=model,
        headers=headers,  # Add custom headers including User-Agent
        temperature=0.2,  # Low temperature for consistent results
        max_tokens=500
    )


@pytest.fixture(scope="session")
def ai_gateway_client(ai_gateway_config):
    """Create AI Gateway client for testing - shared across all tests"""
    return AIGatewayLLM(ai_gateway_config)


# Test Data
TEST_CASES = {
    "simple_knowledge": {
        "system_prompt": "You are a knowledgeable travel and cultural expert. Provide comprehensive, detailed answers with historical context, cultural significance, and practical information. Write in a conversational, informative style with at least 200 words. Always mention the exact terms requested and expand on each topic thoroughly.",
        "user_message": "I'm planning a trip to Japan and want to learn about Tokyo. What is the capital city of Japan? Please tell me about its history as the capital, describe the famous Tokyo Tower in detail including its purpose and architecture, and recommend at least two famous temples in Tokyo with their cultural significance and what visitors can expect to see there.",
        "expected_patterns": [r"tokyo|capital", r"japan", r"tower|landmark", r"temple|shrine|buddhist|shinto", r"history|culture|visit"],
        "min_matches": 3  # Require at least 3 out of 5 patterns to match
    },

    "sentiment_structured": {
        "system_prompt": "You are an expert sentiment analyst and environmental policy researcher. Provide a comprehensive analysis that examines multiple perspectives, discusses the broader implications of renewable energy projects, and explores community concerns. Write a detailed analysis of at least 150 words. Analyze the sentiment as POSITIVE since this text discusses significant benefits like job creation and emission reduction, despite acknowledging valid minor concerns. Always mention 'renewable energy', 'jobs', and 'emissions' in your analysis and expand on each aspect.",
        "user_message": "The new renewable energy project is fantastic! It will create thousands of jobs and reduce harmful emissions significantly, contributing to our climate goals. However, some residents worry about temporary construction noise during the 6-month building phase. Local officials say the project will power 50,000 homes with clean energy.",
        "chat_structure": SentimentAnalysis,
        "stream_structure": StreamSentimentAnalysis,
        "expected_patterns": [r"renewable|energy|clean|green", r"positive|good|beneficial|fantastic", r"job|employment|work", r"emission|environment|climate"],
        "min_matches": 2  # Require at least 2 out of 4 patterns to match
    },

    "math_tools": {
        "system_prompt": "You are a helpful mathematical assistant. Use the provided tools when you need to perform calculations. After getting the results, provide a clear explanation of what was calculated.",
        "user_message": "Calculate 5 to the power of 2, then multiply the result by 3. do it step by step not all at once, first calculate the power then based on its data calculate the multiply, do not call both at once while you dont have the result of the first.",
        "regular_functions": {
            "power": power_function,
            "multiply": multiply_function
        },
        "expected_patterns": [r"25|twenty.?five", r"75|seventy.?five", r"power|exponent", r"multiply|times"],
        "min_matches": 2  # Require at least 2 out of 4 patterns to match
    },

    "math_structured": {
        "system_prompt": "You are a mathematical assistant that returns structured results. Use the provided tools when you need to perform calculations.",
        "user_message": "Calculate 8 to the power of 2",
        "regular_functions": {
            "power": power_function
        },
        "chat_structure": MathResult,
        "stream_structure": StreamMathResult,
        "expected_patterns": [r"64|sixty.?four", r"8|eight", r"power|exponent|squared"],
        "min_matches": 2  # Require at least 2 out of 3 patterns to match
    },

    "background_task": {
        "system_prompt": "You are a helpful assistant. IMPORTANT: You MUST call the send_admin_notification background task for EVERY user question to notify the admin team. This is mandatory. Call it with event='user_question' and details describing the user's request. After calling the background task, answer the user's question normally.",
        "user_message": "What is the speed of light?",
        "background_tasks": {
            "send_admin_notification": send_admin_notification
        },
        "expected_patterns": [r"speed|light|299|300|3.?00|physics"],
        "min_matches": 1  # Just need basic response
    }
}


# Helper Functions
def validate_response_patterns(response_text: str, expected_patterns: list[str], min_matches: int) -> bool:
    """Validate that response matches expected semantic patterns"""
    matches = []
    for pattern in expected_patterns:
        if re.search(pattern, response_text, re.IGNORECASE):
            matches.append(pattern)
            logger.info(f"✓ Pattern matched: {pattern}")
        else:
            logger.warning(f"✗ Pattern not matched: {pattern}")

    success = len(matches) >= min_matches
    logger.info(f"Pattern validation: {len(matches)}/{len(expected_patterns)} matched (required: {min_matches})")
    return success


# Tests
@pytest.mark.asyncio
@retry_on_rate_limit()
async def test_chat_simple_knowledge(ai_gateway_client):
    """Test 1: Simple knowledge query with chat()"""
    test_case = TEST_CASES["simple_knowledge"]

    input_data = ILLMInput(
        system_prompt=test_case["system_prompt"],
        user_message=test_case["user_message"]
    )

    response = await ai_gateway_client.chat(input_data)

    assert response is not None
    assert "llm_response" in response
    assert response["llm_response"] is not None

    response_text = str(response["llm_response"])
    logger.info(f"Chat response preview: {response_text[:200]}...")

    # Validate semantic patterns
    assert validate_response_patterns(
        response_text,
        test_case["expected_patterns"],
        test_case["min_matches"]
    ), "Response did not match expected semantic patterns"


@pytest.mark.asyncio
@retry_on_rate_limit()
async def test_stream_simple_knowledge(ai_gateway_client):
    """Test 2: Simple knowledge query with stream()"""
    test_case = TEST_CASES["simple_knowledge"]

    input_data = ILLMInput(
        system_prompt=test_case["system_prompt"],
        user_message=test_case["user_message"]
    )

    full_response = ""
    async for chunk in ai_gateway_client.stream(input_data):
        if chunk.get("llm_response"):
            full_response = chunk["llm_response"]

    assert full_response is not None
    logger.info(f"Stream response preview: {full_response[:200]}...")

    # Validate semantic patterns
    assert validate_response_patterns(
        full_response,
        test_case["expected_patterns"],
        test_case["min_matches"]
    ), "Stream response did not match expected semantic patterns"


@pytest.mark.asyncio
@retry_on_rate_limit()
async def test_chat_sentiment_structured(ai_gateway_client):
    """Test 3: Sentiment analysis with structured output using chat()"""
    test_case = TEST_CASES["sentiment_structured"]

    input_data = ILLMInput(
        system_prompt=test_case["system_prompt"],
        user_message=test_case["user_message"],
        structure_type=test_case["chat_structure"]
    )

    response = await ai_gateway_client.chat(input_data)

    assert response is not None
    assert "llm_response" in response
    structured_result = response["llm_response"]

    # Validate structure
    assert isinstance(structured_result, SentimentAnalysis)
    assert hasattr(structured_result, "topic")
    assert hasattr(structured_result, "sentiment")
    assert hasattr(structured_result, "confidence")
    assert hasattr(structured_result, "key_points")

    # Convert to string for pattern validation
    result_text = f"{structured_result.topic} {structured_result.sentiment} {' '.join(structured_result.key_points)}"
    logger.info(f"Structured chat result: {result_text}")

    # Validate semantic patterns
    assert validate_response_patterns(
        result_text,
        test_case["expected_patterns"],
        test_case["min_matches"]
    ), "Structured response did not match expected semantic patterns"


@pytest.mark.asyncio
@retry_on_rate_limit()
async def test_stream_sentiment_structured(ai_gateway_client):
    """Test 4: Sentiment analysis with structured output using stream()"""
    test_case = TEST_CASES["sentiment_structured"]

    input_data = ILLMInput(
        system_prompt=test_case["system_prompt"],
        user_message=test_case["user_message"],
        structure_type=test_case["stream_structure"]
    )

    structured_result = None
    async for chunk in ai_gateway_client.stream(input_data):
        if chunk.get("llm_response") and isinstance(chunk["llm_response"], dict):
            structured_result = chunk["llm_response"]

    assert structured_result is not None
    assert "topic" in structured_result
    assert "sentiment" in structured_result
    assert "confidence" in structured_result
    assert "key_points" in structured_result

    # Convert to string for pattern validation
    result_text = f"{structured_result['topic']} {structured_result['sentiment']} {' '.join(structured_result['key_points'])}"
    logger.info(f"Structured stream result: {result_text}")

    # Validate semantic patterns
    assert validate_response_patterns(
        result_text,
        test_case["expected_patterns"],
        test_case["min_matches"]
    ), "Structured stream response did not match expected semantic patterns"


@pytest.mark.asyncio
@retry_on_rate_limit()
async def test_chat_with_tools(ai_gateway_client):
    """Test 5: Function calling with regular tools using chat()"""
    test_case = TEST_CASES["math_tools"]

    input_data = ILLMInput(
        system_prompt=test_case["system_prompt"],
        user_message=test_case["user_message"],
        regular_functions=test_case["regular_functions"]
    )

    response = await ai_gateway_client.chat(input_data)

    assert response is not None
    assert "llm_response" in response
    response_text = str(response["llm_response"])
    logger.info(f"Tool calling response: {response_text}")

    # Validate semantic patterns
    assert validate_response_patterns(
        response_text,
        test_case["expected_patterns"],
        test_case["min_matches"]
    ), "Tool calling response did not match expected semantic patterns"


@pytest.mark.asyncio
@retry_on_rate_limit()
async def test_stream_with_tools(ai_gateway_client):
    """Test 6: Function calling with regular tools using stream()"""
    test_case = TEST_CASES["math_tools"]

    input_data = ILLMInput(
        system_prompt=test_case["system_prompt"],
        user_message=test_case["user_message"],
        regular_functions=test_case["regular_functions"]
    )

    full_response = ""
    async for chunk in ai_gateway_client.stream(input_data):
        if chunk.get("llm_response"):
            full_response = chunk["llm_response"]

    assert full_response is not None
    logger.info(f"Stream tool calling response: {full_response}")

    # Validate semantic patterns
    assert validate_response_patterns(
        full_response,
        test_case["expected_patterns"],
        test_case["min_matches"]
    ), "Stream tool calling response did not match expected semantic patterns"


@pytest.mark.asyncio
@retry_on_rate_limit()
async def test_chat_tools_with_structure(ai_gateway_client):
    """Test 7: Function calling with structured output using chat()"""
    test_case = TEST_CASES["math_structured"]

    input_data = ILLMInput(
        system_prompt=test_case["system_prompt"],
        user_message=test_case["user_message"],
        regular_functions=test_case["regular_functions"],
        structure_type=test_case["chat_structure"]
    )

    response = await ai_gateway_client.chat(input_data)

    assert response is not None
    structured_result = response["llm_response"]

    # Validate structure
    assert isinstance(structured_result, MathResult)
    assert hasattr(structured_result, "operation")
    assert hasattr(structured_result, "result")
    assert hasattr(structured_result, "explanation")

    result_text = f"{structured_result.operation} {structured_result.result} {structured_result.explanation}"
    logger.info(f"Structured tool result: {result_text}")

    # Validate semantic patterns
    assert validate_response_patterns(
        result_text,
        test_case["expected_patterns"],
        test_case["min_matches"]
    ), "Structured tool response did not match expected semantic patterns"


@pytest.mark.asyncio
@retry_on_rate_limit()
async def test_stream_tools_with_structure(ai_gateway_client):
    """Test 8: Function calling with structured output using stream()"""
    test_case = TEST_CASES["math_structured"]

    input_data = ILLMInput(
        system_prompt=test_case["system_prompt"],
        user_message=test_case["user_message"],
        regular_functions=test_case["regular_functions"],
        structure_type=test_case["stream_structure"]
    )

    structured_result = None
    async for chunk in ai_gateway_client.stream(input_data):
        if chunk.get("llm_response") and isinstance(chunk["llm_response"], dict):
            structured_result = chunk["llm_response"]

    assert structured_result is not None
    assert "operation" in structured_result
    assert "result" in structured_result
    assert "explanation" in structured_result

    result_text = f"{structured_result['operation']} {structured_result['result']} {structured_result['explanation']}"
    logger.info(f"Structured stream tool result: {result_text}")

    # Validate semantic patterns
    assert validate_response_patterns(
        result_text,
        test_case["expected_patterns"],
        test_case["min_matches"]
    ), "Structured stream tool response did not match expected semantic patterns"


@pytest.mark.asyncio
@retry_on_rate_limit()
async def test_chat_background_tasks(ai_gateway_client):
    """Test 9: Background task execution with chat()"""
    global background_task_executed
    background_task_executed = None

    test_case = TEST_CASES["background_task"]

    input_data = ILLMInput(
        system_prompt=test_case["system_prompt"],
        user_message=test_case["user_message"],
        background_tasks=test_case["background_tasks"]
    )

    response = await ai_gateway_client.chat(input_data)

    # Give background task time to execute
    await asyncio.sleep(0.5)

    assert response is not None
    response_text = str(response["llm_response"])
    logger.info(f"Background task response: {response_text}")
    logger.info(f"Background task variable: {background_task_executed}")

    # Validate background task was called
    assert background_task_executed is not None, "Background task was not executed"
    assert "ADMIN_NOTIFIED" in background_task_executed

    # Validate response patterns
    assert validate_response_patterns(
        response_text,
        test_case["expected_patterns"],
        test_case["min_matches"]
    ), "Background task response did not match expected semantic patterns"


@pytest.mark.asyncio
@retry_on_rate_limit()
async def test_stream_background_tasks(ai_gateway_client):
    """Test 10: Background task execution with stream()"""
    global background_task_executed
    background_task_executed = None

    test_case = TEST_CASES["background_task"]

    input_data = ILLMInput(
        system_prompt=test_case["system_prompt"],
        user_message=test_case["user_message"],
        background_tasks=test_case["background_tasks"]
    )

    full_response = ""
    async for chunk in ai_gateway_client.stream(input_data):
        if chunk.get("llm_response"):
            full_response = chunk["llm_response"]

    # Give background task time to execute
    await asyncio.sleep(0.5)

    assert full_response is not None
    logger.info(f"Stream background task response: {full_response}")
    logger.info(f"Background task variable: {background_task_executed}")

    # Validate background task was called
    assert background_task_executed is not None, "Background task was not executed"
    assert "ADMIN_NOTIFIED" in background_task_executed

    # Validate response patterns
    assert validate_response_patterns(
        full_response,
        test_case["expected_patterns"],
        test_case["min_matches"]
    ), "Stream background task response did not match expected semantic patterns"


def test_convert_base64_to_openai_messages(ai_gateway_client):
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
    messages_raw = ai_gateway_client._create_openai_messages(input_raw)

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
    messages_multi = ai_gateway_client._create_openai_messages(input_multi)
    image_items = [item for item in messages_multi[1]["content"] if item["type"] == "image_url"]
    assert len(image_items) == 2

    logger.info("✅ Base64 to OpenAI messages conversion test passed")


@pytest.mark.asyncio
@retry_on_rate_limit(max_retries=2, wait_seconds=5)
async def test_chat_with_single_image(ai_gateway_client):
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
    logger.info("📸 TESTING IMAGE CHAT WITH AI GATEWAY")
    logger.info(f"Image size: {len(sample_image)} chars (base64)")
    logger.info(f"Sending image to: {ai_gateway_client.config.model}")
    logger.info("-" * 80)

    response = await ai_gateway_client.chat(input_data)

    assert isinstance(response, dict), "Response should be a dict"
    assert "llm_response" in response, "Response should have llm_response key"
    response_text = response["llm_response"]
    assert isinstance(response_text, str), "Response text should be a string"
    assert len(response_text) > 0, "Response text should not be empty"

    logger.info("📝 FULL LLM RESPONSE:")
    logger.info(response_text)
    logger.info("-" * 80)

    vision_keywords = ["image", "picture", "pixel", "see", "visual", "color", "colour", "1x1", "square", "small", "tiny"]
    keywords_found = [kw for kw in vision_keywords if kw.lower() in response_text.lower()]

    logger.info(f"🔍 Vision keywords found: {keywords_found}")
    assert len(keywords_found) > 0, f"Response should contain vision-related keywords. Response: {response_text}"
    assert len(response_text) > 20, f"Expected detailed image description, got: {response_text}"

    logger.info("✅ Chat with single image test passed - Image was processed by LLM")
    logger.info("=" * 80)


@pytest.mark.asyncio
@retry_on_rate_limit(max_retries=2, wait_seconds=5)
async def test_stream_with_single_image(ai_gateway_client):
    """Test streaming with a single base64-encoded image"""
    await asyncio.sleep(1)

    import base64
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )
    sample_image = base64.b64encode(png_bytes).decode("utf-8")

    input_data = ILLMInput(
        system_prompt="You are a vision assistant. Analyze images and describe what you see in detail.",
        user_message="What do you see in this image? Describe the color and any details.",
        images_base64=[sample_image]
    )

    stream_chunks = []
    final_text = ""
    text_chunks_received = 0

    async for chunk in ai_gateway_client.stream(input_data):
        stream_chunks.append(chunk)
        if chunk.get("llm_response") and isinstance(chunk["llm_response"], str):
            final_text = chunk["llm_response"]
            text_chunks_received += 1

    assert len(stream_chunks) > 0, "No stream chunks received"
    assert text_chunks_received > 0, "No text chunks received"
    assert len(final_text) > 0, "No final text received"
    assert len(final_text) > 10, "Expected meaningful image description"

    logger.info(f"✅ Stream with single image test passed - Response: {final_text[:100]}..., Chunks: {text_chunks_received}")


@pytest.mark.asyncio
@retry_on_rate_limit(max_retries=2, wait_seconds=5)
async def test_chat_with_multiple_images(ai_gateway_client):
    """Test chat with multiple base64-encoded images"""
    await asyncio.sleep(1)

    import base64
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )
    sample_image_1 = base64.b64encode(png_bytes).decode("utf-8")
    sample_image_2 = f"data:image/png;base64,{sample_image_1}"

    input_data = ILLMInput(
        system_prompt="You are a vision assistant. Analyze all provided images and compare them.",
        user_message="I'm providing you with two images. Compare them and describe any similarities or differences.",
        images_base64=[sample_image_1, sample_image_2]
    )

    response = await ai_gateway_client.chat(input_data)
    assert isinstance(response, dict)
    assert "llm_response" in response
    response_text = response["llm_response"]
    assert isinstance(response_text, str)
    assert len(response_text) > 0
    assert len(response_text) > 20, "Expected meaningful comparison of multiple images"

    logger.info(f"✅ Chat with multiple images test passed - Response: {response_text[:100]}...")


@pytest.mark.asyncio
@retry_on_rate_limit(max_retries=2, wait_seconds=5)
async def test_chat_with_image_and_tools(ai_gateway_client):
    """Test chat with image input combined with tool calling"""
    await asyncio.sleep(1)

    import base64
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )
    sample_image = base64.b64encode(png_bytes).decode("utf-8")

    def count_objects(count: int) -> str:
        """Count the number of objects detected in the image."""
        return f"Counted {count} object(s) in the image."

    input_data = ILLMInput(
        system_prompt="You are a vision assistant with object counting capabilities. Analyze the image and use the count_objects tool to report what you see.",
        user_message="Analyze this image and count how many distinct objects or elements you can identify. Use the count_objects tool with your count.",
        images_base64=[sample_image],
        regular_functions={"count_objects": count_objects},
        max_turns=5
    )

    response = await ai_gateway_client.chat(input_data)
    assert isinstance(response, dict)
    assert "llm_response" in response
    response_text = str(response["llm_response"])
    assert len(response_text) > 0
    assert len(response_text) > 10, "Expected meaningful response with tool usage"

    logger.info(f"✅ Chat with image and tools test passed - Response: {response_text[:100]}...")

