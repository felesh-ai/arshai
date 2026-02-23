"""
Pytest test suite for OpenRouter LLM client.
Tests both chat and stream methods with identical inputs for direct comparison.
Includes regex validation for semantic correctness.
Based on the unified test pattern established for Gemini and Azure clients.
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

from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.llms.openrouter import OpenRouterClient

# Load test environment variables
test_env_path = Path(__file__).parent / ".env.openrouter"
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
                        "RateLimitError" in error_type or
                        "rate_limit" in error_str.lower()
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
# Test variable to track background task execution
background_task_executed = None

def send_admin_notification(event: str, details: str = "User interaction") -> None:
    """BACKGROUND TASK: Send notification to admin channel about system events and set test variable. This task runs independently in fire-and-forget mode - no results will be returned to the conversation."""
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


# Test Configuration and Fixtures
@pytest.fixture(scope="session")
def openrouter_config():
    """Create test configuration"""
    return ILLMConfig(
        model="openai/gpt-4o-mini",  # OpenRouter format with provider prefix
        temperature=0.2,  # Low temperature for consistent results
        max_tokens=500
    )


@pytest.fixture(scope="session")
def openrouter_client(openrouter_config):
    """Create OpenRouter client for testing - shared across all tests"""
    return OpenRouterClient(openrouter_config)


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
        "chat_structure": MathResult,
        "stream_structure": StreamMathResult,
        "expected_patterns": [r"25|twenty", r"75|seventy", r"power|multiply", r"result"],
        "min_matches": 2  # Require at least 2 out of 4 patterns to match
    },
    
    "parallel_tools": {
        "system_prompt": "You are a mathematical assistant. Use the provided tools to perform multiple calculations simultaneously when requested.",
        "user_message": "Calculate these operations: 3 to the power of 2, 4 to the power of 2, and multiply 6 by 7. You can call multiple functions at once.",
        "regular_functions": {
            "power": power_function,
            "multiply": multiply_function
        },
        "expected_patterns": [r"9|nine", r"16|sixteen", r"42|forty", r"power|multiply"],
        "min_matches": 3  # Require at least 3 out of 4 patterns to match (expecting all 3 results)
    },
    
    "background_tasks": {
        "system_prompt": "You are a helpful AI assistant. When users ask questions, provide clear answers AND always notify the admin team about user interactions using background tasks. Background tasks run independently without affecting your response to the user.",
        "user_message": "What is the capital of France?",
        "background_tasks": {
            "send_admin_notification": send_admin_notification
        },
        "expected_patterns": [r"paris|france", r"capital", r"background.*initiated", r"admin.*notification|notification.*admin"],
        "min_matches": 2  # Expect answer + background task initiation
    }
}


# Helper function for flexible pattern matching
def validate_patterns_flexible(text: str, patterns: list, min_matches: int, test_name: str = "") -> bool:
    """
    Validate that at least min_matches patterns are found in the text.
    Returns True if enough patterns match, False otherwise.
    """
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

# Test Functions
class TestOpenRouterClient:
    """Test class for OpenRouter client with shared client instance"""
    
    @pytest.fixture(scope="class")
    def client(self, openrouter_client):
        """Provide shared client for all tests in this class"""
        return openrouter_client
    
    def test_client_initialization(self, client):
        """Test that the OpenRouter client initializes correctly"""
        assert client is not None
        assert client.config.model == "openai/gpt-4o-mini"
        assert client._client is not None
    
    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=60)
    async def test_simple_chat(self, client):
        """Test simple knowledge query - chat method"""
        # Brief delay for API stability
        await asyncio.sleep(1)
        
        test_data = TEST_CASES["simple_knowledge"]
        
        # Prepare input data
        input_data = ILLMInput(
            system_prompt=test_data["system_prompt"],
            user_message=test_data["user_message"]
        )
        
        # Test Chat Method
        chat_response = await client.chat(input_data)
        logger.info(f"🔍chat_response: {chat_response}")
        assert isinstance(chat_response, dict)
        assert "llm_response" in chat_response
        chat_text = chat_response["llm_response"]
        assert isinstance(chat_text, str)
        assert len(chat_text) > 0
        
        # Validate chat response patterns (flexible matching)
        assert validate_patterns_flexible(
            chat_text, 
            test_data["expected_patterns"], 
            test_data["min_matches"], 
            "simple_chat"
        ), "Not enough patterns matched in chat response"
        
        # Validate response length (should be comprehensive due to prompt)
        assert len(chat_text) > 150, f"Expected comprehensive response (>150 chars), got {len(chat_text)} chars"
        
        logger.info(f"✅ Simple chat test passed - Response: {len(chat_text)} chars")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=60)
    async def test_simple_stream(self, client):
        """Test simple knowledge query - stream method"""
        # Brief delay for API stability
        await asyncio.sleep(1)
        
        test_data = TEST_CASES["simple_knowledge"]
        
        # Prepare input data
        input_data = ILLMInput(
            system_prompt=test_data["system_prompt"],
            user_message=test_data["user_message"]
        )
        
        # Test Stream Method - Validate actual streaming occurs
        stream_chunks = []
        final_stream_text = ""
        text_chunks_received = 0
        text_lengths = []
        async for chunk in client.stream(input_data):
            logger.info(f"🔍 Stream chunk: {chunk}")
            stream_chunks.append(chunk)
            if chunk.get("llm_response") and isinstance(chunk["llm_response"], str):
                final_stream_text = chunk["llm_response"]  # Get the accumulated text
                text_lengths.append(len(final_stream_text))
                text_chunks_received += 1
        
        # Validate streaming behavior - check for growing text indicating streaming
        assert len(stream_chunks) > 0, "No stream chunks received"
        assert text_chunks_received > 1, f"Expected multiple text chunks for streaming, got {text_chunks_received}"
        
        # Verify text was growing (proper streaming)
        has_growth = any(text_lengths[i] > text_lengths[i-1] for i in range(1, len(text_lengths)))
        assert has_growth, f"Expected text growth during streaming, got lengths: {text_lengths}"
        assert len(final_stream_text) > 0, "No text content received from stream"
        
        # Validate stream response patterns (flexible matching)
        assert validate_patterns_flexible(
            final_stream_text, 
            test_data["expected_patterns"], 
            test_data["min_matches"], 
            "simple_stream"
        ), "Not enough patterns matched in stream response"
        
        # Validate response length (should be comprehensive due to prompt)
        assert len(final_stream_text) > 150, f"Expected comprehensive response (>150 chars), got {len(final_stream_text)} chars"
        
        logger.info(f"✅ Simple stream test passed - Response: {len(final_stream_text)} chars, Stream chunks: {text_chunks_received}, Growth pattern: {text_lengths}")
    
    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=60)
    async def test_structured_chat(self, client):
        """Test structured sentiment analysis - chat method"""
        # Brief delay for API stability
        await asyncio.sleep(1)
        
        test_data = TEST_CASES["sentiment_structured"]
        
        # Test Chat Method with Pydantic model
        chat_input = ILLMInput(
            system_prompt=test_data["system_prompt"],
            user_message=test_data["user_message"],
            structure_type=test_data["chat_structure"]
        )
        
        chat_response = await client.chat(chat_input)
        logger.info(f"🔍chat_response: {chat_response}")
        assert isinstance(chat_response, dict)
        assert "llm_response" in chat_response
        chat_result = chat_response["llm_response"]
        assert isinstance(chat_result, SentimentAnalysis)
        
        # Validate structured fields
        assert len(chat_result.topic) > 0
        assert chat_result.sentiment.lower() in ["positive", "negative", "neutral", "mixed"]
        assert 0.0 <= chat_result.confidence <= 1.0
        assert len(chat_result.key_points) > 0
        
        # Validate patterns in response (flexible matching)
        chat_combined = f"{chat_result.topic} {chat_result.sentiment} {' '.join(chat_result.key_points)}"
        assert validate_patterns_flexible(
            chat_combined, 
            test_data["expected_patterns"], 
            test_data["min_matches"], 
            "structured_chat"
        ), "Not enough patterns matched in structured chat response"
        
        logger.info(f"✅ Structured chat test passed - Sentiment: {chat_result.sentiment}, Confidence: {chat_result.confidence}")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=60)
    async def test_structured_stream(self, client):
        """Test structured sentiment analysis - stream method"""
        # Brief delay for API stability
        await asyncio.sleep(1)
        
        test_data = TEST_CASES["sentiment_structured"]
        
        # Test Stream Method with dict-based model
        stream_input = ILLMInput(
            system_prompt=test_data["system_prompt"],
            user_message=test_data["user_message"],
            structure_type=test_data["stream_structure"]
        )
        
        stream_chunks = []
        final_stream_result = None
        structured_chunks_received = 0
        async for chunk in client.stream(stream_input):
            logger.info(f"🔧 Stream chunk: {chunk}")
            stream_chunks.append(chunk)
            if chunk.get("llm_response") and isinstance(chunk["llm_response"], dict):
                # Check if it has all required keys
                if all(key in chunk["llm_response"] for key in ["topic", "sentiment", "confidence", "key_points"]):
                    final_stream_result = chunk["llm_response"]
                    structured_chunks_received += 1
        
        # Validate streaming behavior for structured output
        assert len(stream_chunks) > 0, "No stream chunks received"
        assert structured_chunks_received > 0, "No structured chunks received (streaming may not be working)"
        assert final_stream_result is not None, "No final structured result received"
        assert isinstance(final_stream_result, dict), "Final result is not a dictionary"
        
        # Validate stream structured fields
        assert len(final_stream_result["topic"]) > 0
        assert final_stream_result["sentiment"].lower() in ["positive", "negative", "neutral", "mixed"]
        assert 0.0 <= final_stream_result["confidence"] <= 1.0
        assert len(final_stream_result["key_points"]) > 0
        
        # Validate patterns in response (flexible matching)
        stream_combined = f"{final_stream_result['topic']} {final_stream_result['sentiment']} {' '.join(final_stream_result['key_points'])}"
        assert validate_patterns_flexible(
            stream_combined, 
            test_data["expected_patterns"], 
            test_data["min_matches"], 
            "structured_stream"
        ), "Not enough patterns matched in structured stream response"
        
        logger.info(f"✅ Structured stream test passed - Sentiment: {final_stream_result['sentiment']}, Confidence: {final_stream_result['confidence']}, Stream chunks: {structured_chunks_received}")
    
    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=60)
    async def test_tool_calling_chat(self, client):
        """Test tool calling with structured output - chat method"""
        # Brief delay for API stability
        await asyncio.sleep(1)
        
        test_data = TEST_CASES["math_tools"]
        
        # Test Chat Method
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
        
        # Check if structured or unstructured response
        if isinstance(chat_result, MathResult):
            chat_text = f"{chat_result.operation} {chat_result.result} {chat_result.explanation}"
            logger.info(f"Chat structured result: {chat_result.operation} = {chat_result.result}")
        else:
            chat_text = str(chat_result)
            logger.info(f"Chat unstructured result: {chat_text[:100]}...")
        
        # Validate patterns (flexible matching)
        assert validate_patterns_flexible(
            chat_text, 
            test_data["expected_patterns"], 
            test_data["min_matches"], 
            "tool_calling_chat"
        ), "Not enough patterns matched in chat tool response"
        
        logger.info("✅ Tool calling chat test passed - Contains expected calculations")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=60)
    async def test_tool_calling_stream(self, client):
        """Test tool calling with structured output - stream method"""
        # Brief delay for API stability
        await asyncio.sleep(1)
        
        test_data = TEST_CASES["math_tools"]
        
        # Test Stream Method
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
            logger.info(f"Stream Chunk: {chunk}")
            stream_chunks.append(chunk)
            if chunk.get("llm_response") and chunk["llm_response"] is not None:
                content_chunks_received += 1
                if isinstance(chunk["llm_response"], dict) and all(key in chunk["llm_response"] for key in ["operation", "result", "explanation"]):
                    final_stream_result = chunk["llm_response"]
                    logger.info(f"Stream structured result: {final_stream_result['operation']} = {final_stream_result['result']}")
                elif isinstance(chunk["llm_response"], str):
                    final_stream_text = chunk["llm_response"]
        
        # Validate streaming behavior for tool calling
        assert len(stream_chunks) > 0, "No stream chunks received"
        assert content_chunks_received > 0, "No content chunks received from streaming"
        
        # Get stream text for validation
        if final_stream_result:
            stream_text = f"{final_stream_result['operation']} {final_stream_result['result']} {final_stream_result['explanation']}"
        else:
            stream_text = final_stream_text
            logger.info(f"Stream unstructured result: {stream_text[:100]}...")
        
        assert len(stream_text) > 0, "No final stream text received"
        
        # Validate patterns (flexible matching)
        assert validate_patterns_flexible(
            stream_text, 
            test_data["expected_patterns"], 
            test_data["min_matches"], 
            "tool_calling_stream"
        ), "Not enough patterns matched in stream tool response"
        
        logger.info(f"✅ Tool calling stream test passed - Contains expected calculations, Stream chunks: {content_chunks_received}")
    
    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=60)
    async def test_parallel_function_calling_chat(self, client):
        """Test parallel function calling capabilities - chat method"""
        # Brief delay for API stability
        await asyncio.sleep(1)
        
        test_data = TEST_CASES["parallel_tools"]
        
        # Test Chat Method with parallel function calling
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
        
        # Convert result to string for pattern validation
        chat_text = str(chat_result)
        logger.info(f"Parallel chat result: {chat_text[:200]}...")
        
        # Validate that all expected calculations are present (parallel execution)
        assert validate_patterns_flexible(
            chat_text, 
            test_data["expected_patterns"], 
            test_data["min_matches"], 
            "parallel_function_calling_chat"
        ), "Not enough patterns matched in parallel chat response"
        
        logger.info("✅ Parallel function calling chat test passed - Contains multiple calculation results")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=60)
    async def test_parallel_function_calling_stream(self, client):
        """Test parallel function calling capabilities - stream method"""
        # Brief delay for API stability
        await asyncio.sleep(1)
        
        test_data = TEST_CASES["parallel_tools"]
        
        # Test Stream Method with parallel function calling
        stream_input = ILLMInput(
            system_prompt=test_data["system_prompt"],
            user_message=test_data["user_message"],
            regular_functions=test_data["regular_functions"],
            max_turns=5
        )
        
        stream_chunks = []
        final_stream_text = ""
        content_chunks_received = 0
        function_calls_detected = 0
        
        async for chunk in client.stream(stream_input):
            stream_chunks.append(chunk)
            logger.info(f"🔧 Parallel stream chunk: {chunk}")
            
            if chunk.get("llm_response") and chunk["llm_response"] is not None:
                content_chunks_received += 1
                if isinstance(chunk["llm_response"], str):
                    final_stream_text = chunk["llm_response"]
                    # Count function calls mentioned in logs (indirect way to verify parallel execution)
                    if "function" in chunk["llm_response"].lower():
                        function_calls_detected += 1
        
        # Validate streaming behavior for parallel function calling
        assert len(stream_chunks) > 0, "No stream chunks received"
        assert content_chunks_received > 0, "No content chunks received from streaming"
        assert len(final_stream_text) > 0, "No final stream text received"
        
        # Validate that all expected calculations are present (parallel execution)
        assert validate_patterns_flexible(
            final_stream_text, 
            test_data["expected_patterns"], 
            test_data["min_matches"], 
            "parallel_function_calling_stream"
        ), "Not enough patterns matched in parallel stream response"
        
        logger.info(f"✅ Parallel function calling stream test passed - Contains multiple calculation results, Stream chunks: {content_chunks_received}")
    
    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=60)
    async def test_background_tasks_chat(self, client):
        """Test background tasks (fire-and-forget) - chat method with direct variable verification"""
        # Brief delay for API stability
        await asyncio.sleep(1)
        
        # Reset the test variable before starting
        global background_task_executed
        background_task_executed = None
        logger.info(f"Reset background_task_executed to: {background_task_executed}")
        
        test_data = TEST_CASES["background_tasks"]
        
        # Test Chat Method with background tasks
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
        
        # Convert result to string for basic validation
        chat_text = str(chat_result)
        logger.info(f"Background tasks chat result: {chat_text}")
        
        # Validate the LLM provided an answer about France (basic response check)
        assert re.search(r"paris|france", chat_text, re.IGNORECASE), "LLM should answer the question about France"
        
        # Give background tasks a moment to complete
        await asyncio.sleep(0.5)
        
        # DIRECT VERIFICATION: Check if the background task actually executed by checking our test variable
        logger.info(f"Final background_task_executed value: {background_task_executed}")
        
        # Verify the background task was executed
        assert background_task_executed is not None, "Background task should have executed and set the test variable"
        assert "ADMIN_NOTIFIED" in background_task_executed, "Background task should have set the expected value"

        
        logger.info(f"✅ Background tasks chat test passed - Background task executed: {background_task_executed}")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=60)
    async def test_background_tasks_stream(self, client):
        """Test background tasks (fire-and-forget) - stream method with direct variable verification"""
        # Brief delay for API stability
        await asyncio.sleep(1)
        
        # Reset the test variable before starting
        global background_task_executed
        background_task_executed = None
        logger.info(f"Reset background_task_executed to: {background_task_executed}")
        
        test_data = TEST_CASES["background_tasks"]
        
        # Test Stream Method with background tasks
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
            logger.info(f"🎯 Background tasks stream chunk: {chunk}")
            
            if chunk.get("llm_response") and chunk["llm_response"] is not None:
                content_chunks_received += 1
                if isinstance(chunk["llm_response"], str):
                    final_stream_text = chunk["llm_response"]
        
        # Validate streaming behavior for background tasks
        assert len(stream_chunks) > 0, "No stream chunks received"
        assert content_chunks_received > 0, "No content chunks received from streaming"
        assert len(final_stream_text) > 0, "No final stream text received"
        
        # Validate the LLM provided an answer about France (basic response check)
        assert re.search(r"paris|france", final_stream_text, re.IGNORECASE), "LLM should answer the question about France"
        
        # Give background tasks a moment to complete
        await asyncio.sleep(0.5)
        
        # DIRECT VERIFICATION: Check if the background task actually executed by checking our test variable
        logger.info(f"Final background_task_executed value: {background_task_executed}")
        
        # Verify the background task was executed
        assert background_task_executed is not None, "Background task should have executed and set the test variable"
        assert "ADMIN_NOTIFIED" in background_task_executed, "Background task should have set the expected value"
        # The LLM may provide different event details, so check for either the default or LLM-generated content
        assert ("User interaction" in background_task_executed or "User" in background_task_executed), "Background task should include user-related event details"
        
        logger.info(f"✅ Background tasks stream test passed - Background task executed: {background_task_executed}, Stream chunks: {content_chunks_received}")
    
    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=60)
    async def test_usage_tracking(self, client):
        """Test that both methods provide usage information"""
        # Brief delay for API stability
        await asyncio.sleep(1)
        
        input_data = ILLMInput(
            system_prompt="You are a helpful mathematics tutor. Provide detailed explanations with step-by-step solutions, mathematical reasoning, and practical examples. Write comprehensive responses of at least 100 words to help students understand concepts thoroughly.",
            user_message="What is 5 + 3? Please explain the addition process step by step, show how this applies to real-world scenarios like counting objects, and discuss why addition is fundamental in mathematics. Also explain different ways to represent this calculation and mention at least two practical examples where someone might need to add 5 and 3 in daily life."
        )
        
        # Test chat usage
        chat_response = await client.chat(input_data)
        assert "usage" in chat_response
        if chat_response["usage"]:  # Usage might be None
            assert "total_tokens" in chat_response["usage"]
            assert isinstance(chat_response["usage"]["total_tokens"], int)
        
        # Test stream usage - Validate streaming occurs
        final_usage = None
        stream_chunks_count = 0
        async for chunk in client.stream(input_data):
            stream_chunks_count += 1
            if chunk.get("usage"):
                final_usage = chunk["usage"]
        
        # Validate streaming behavior - expect at least 1 chunk (streaming may be limited by rate limiting)
        assert stream_chunks_count >= 1, f"Expected at least 1 stream chunk, got {stream_chunks_count}"
        
        # Log streaming behavior for debugging
        if stream_chunks_count == 1:
            logger.warning("Only 1 stream chunk received - this may be due to rate limiting or API behavior")
        else:
            logger.info(f"Received {stream_chunks_count} stream chunks - proper streaming behavior")
        
        if final_usage:  # Usage might be None
            assert "total_tokens" in final_usage
            assert isinstance(final_usage["total_tokens"], int)
        
        logger.info(f"✅ Usage tracking test passed - Stream chunks: {stream_chunks_count}")

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
    @retry_on_rate_limit(max_retries=2, wait_seconds=60)
    async def test_chat_with_single_image(self, client):
        """Test chat with a single base64-encoded image"""
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

        response = await client.chat(input_data)
        assert isinstance(response, dict)
        assert "llm_response" in response
        response_text = response["llm_response"]
        assert isinstance(response_text, str)
        assert len(response_text) > 0
        assert len(response_text) > 10, "Expected meaningful image description"

        logger.info(f"✅ Chat with single image test passed - Response: {response_text[:100]}...")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=60)
    async def test_stream_with_single_image(self, client):
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

        async for chunk in client.stream(input_data):
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
    @retry_on_rate_limit(max_retries=2, wait_seconds=60)
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
            system_prompt="You are a vision assistant. Analyze all provided images and compare them.",
            user_message="I'm providing you with two images. Compare them and describe any similarities or differences.",
            images_base64=[sample_image_1, sample_image_2]
        )

        response = await client.chat(input_data)
        assert isinstance(response, dict)
        assert "llm_response" in response
        response_text = response["llm_response"]
        assert isinstance(response_text, str)
        assert len(response_text) > 0
        assert len(response_text) > 20, "Expected meaningful comparison of multiple images"

        logger.info(f"✅ Chat with multiple images test passed - Response: {response_text[:100]}...")

    @pytest.mark.asyncio
    @retry_on_rate_limit(max_retries=2, wait_seconds=60)
    async def test_chat_with_image_and_tools(self, client):
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

        response = await client.chat(input_data)
        assert isinstance(response, dict)
        assert "llm_response" in response
        response_text = str(response["llm_response"])
        assert len(response_text) > 0
        assert len(response_text) > 10, "Expected meaningful response with tool usage"

        logger.info(f"✅ Chat with image and tools test passed - Response: {response_text[:100]}...")


# ===========================================================================
# PDF SUPPORT TESTS
# ===========================================================================
# These tests use a PDF-capable model (Gemini 2.0 Flash) via OpenRouter.
# The native PDF plugin (engine: native) forwards the PDF to the model
# without server-side parsing, so the model must support PDFs natively.
#
# Run PDF tests only:
#   python -m pytest tests/unit/llms/test_openrouter.py::TestOpenRouterClientPDF -v

def _make_test_pdf_bytes(text: str) -> bytes:
    """
    Build a minimal, valid PDF-1.4 file that contains `text` as visible page content.

    Long text is wrapped into 60-character lines so every word is visible within
    the page margin.  Models that render PDFs as images (Gemini, Claude) will then
    see all content.  The PDF is constructed entirely from stdlib.
    """
    import textwrap

    def _esc(s: str) -> str:
        return s.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')

    lines = textwrap.wrap(text, width=60) or [text]

    # Build content stream: first line sets position, subsequent lines move down
    parts = [f"BT /F1 12 Tf 72 720 Td ({_esc(lines[0])}) Tj"]
    for line in lines[1:]:
        parts.append(f"0 -20 Td ({_esc(line)}) Tj")
    parts.append("ET")
    stream_src = "\n".join(parts)
    stream_b = stream_src.encode('latin-1')

    header = b"%PDF-1.4\n"
    o1 = b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n"
    o2 = b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n"
    # Four closing >> : /F1 dict, /Font dict, /Resources dict, /Page dict
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
    import base64
    return base64.b64encode(_make_test_pdf_bytes(text)).decode('utf-8')


# Distinctive content that can ONLY come from reading the PDF.
# Each phrase is short enough to fit on its own wrapped line (≤60 chars).
PDF_TEST_CONTENT = (
    "ARSHAI PDF TEST DOCUMENT. "
    "Secret code: XYZPDF9981. "
    "Capital city: Rome. "
    "Answer: forty-two."
)

# Fixture using a model known to support native PDFs via OpenRouter
@pytest.fixture(scope="session")
def pdf_openrouter_config():
    """Config using Gemini 2.0 Flash — native PDF support via OpenRouter."""
    return ILLMConfig(
        model="google/gemini-2.0-flash-001",
        temperature=0.1,
        max_tokens=500,
    )


@pytest.fixture(scope="session")
def pdf_client(pdf_openrouter_config):
    """OpenRouter client configured for PDF-capable model."""
    return OpenRouterClient(pdf_openrouter_config)


class TestOpenRouterClientPDF:
    """
    Integration tests for PDF support via OpenRouter + native PDF plugin.

    All tests that make real API calls are marked @pytest.mark.asyncio and
    require OPENROUTER_API_KEY to be set (same as the rest of this file).

    The model used (Gemini 2.0 Flash) reads the PDF natively — it is NOT
    parsed server-side by OpenRouter.  Logs show exactly what the model
    returned so you can verify it actually read the document content.
    """

    @pytest.fixture(scope="class")
    def client(self, pdf_client):
        return pdf_client

    # ------------------------------------------------------------------
    # Structure tests (no API call)
    # ------------------------------------------------------------------

    def test_pdf_message_structure_raw_base64(self, client):
        """Verify raw base64 PDF produces the correct OpenRouter 'file' block."""
        import base64
        pdf_b64 = _make_test_pdf_base64(PDF_TEST_CONTENT)

        input_data = ILLMInput(
            system_prompt="You are a document analyst.",
            user_message="Summarise this document.",
            pdfs_base64=[pdf_b64],
        )
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
        assert "file" in fb
        assert fb["file"]["filename"] == "document.pdf"
        assert fb["file"]["file_data"].startswith("data:application/pdf;base64,")

        logger.info(f"✅ PDF message structure test passed - file block: filename={fb['file']['filename']}, "
                    f"data prefix={fb['file']['file_data'][:40]}...")

    def test_pdf_message_structure_data_url(self, client):
        """Verify data-URL-prefixed PDF is passed through unchanged."""
        import base64
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
        import base64
        pdf_b64 = _make_test_pdf_base64(PDF_TEST_CONTENT)
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )
        img_b64 = base64.b64encode(png_bytes).decode("utf-8")

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
        Test chat() with a base64 PDF.
        The PDF contains a secret code (XYZPDF9981) and specific facts.
        We verify the model's response references content from the PDF,
        proving it actually read the document.
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

        # Verify the model actually read the PDF content
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
        Test stream() with a base64 PDF.
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
        Test chat() with two different PDFs.
        Each PDF has a unique keyword; the model should reference both.
        """
        await asyncio.sleep(1)

        pdf1_text = "FIRST DOCUMENT - keyword ALPHA001 - describes the planet Mars."
        pdf2_text = "SECOND DOCUMENT - keyword BETA002 - describes the planet Venus."

        pdf1_b64 = _make_test_pdf_base64(pdf1_text)
        pdf2_b64 = _make_test_pdf_base64(pdf2_text)

        input_data = ILLMInput(
            system_prompt=(
                "You are a document analyst. Read all provided PDFs carefully."
            ),
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

        # Either the tool was called with 7, or the response mentions 7
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

        # Chat usage
        chat_response = await client.chat(input_data)
        logger.info(f"📊 Chat usage with PDF: {chat_response.get('usage')}")
        assert "usage" in chat_response
        if chat_response["usage"]:
            assert "total_tokens" in chat_response["usage"]
            assert isinstance(chat_response["usage"]["total_tokens"], int)
            assert chat_response["usage"]["total_tokens"] > 0
            logger.info(f"📊 Total tokens (chat): {chat_response['usage']['total_tokens']}")

        # Stream usage
        final_usage = None
        async for chunk in client.stream(input_data):
            if chunk.get("usage"):
                final_usage = chunk["usage"]
        logger.info(f"📊 Stream usage with PDF: {final_usage}")
        if final_usage:
            assert "total_tokens" in final_usage
            assert isinstance(final_usage["total_tokens"], int)
            logger.info(f"📊 Total tokens (stream): {final_usage['total_tokens']}")

        logger.info("✅ PDF usage tracking test passed")


# Run tests with: python -m pytest tests/unit/llms/test_openrouter.py -v
# Run only PDF tests: python -m pytest tests/unit/llms/test_openrouter.py::TestOpenRouterClientPDF -v
# Or run specific test: python -m pytest tests/unit/llms/test_openrouter.py::TestOpenRouterClient::test_simple_chat -v
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])