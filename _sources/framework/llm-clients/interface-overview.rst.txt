ILLM Interface Overview
=======================

The ``ILLM`` interface defines the contract that all LLM clients must implement. This interface ensures consistent behavior across different language model providers while supporting advanced features like function calling, streaming, and structured output.

Interface Definition
--------------------

.. code-block:: python

   from typing import Protocol, Dict, Any, AsyncGenerator
   from arshai.core.interfaces.illm import ILLMInput
   
   class ILLM(Protocol):
       """Protocol defining the LLM client interface."""
       
       async def chat(self, input: ILLMInput) -> Dict[str, Any]:
           """Single-turn conversation returning complete response."""
           ...
       
       async def stream(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
           """Streaming conversation yielding response chunks."""
           ...

Input Structure
---------------

All LLM operations use ``ILLMInput`` to define the request:

.. code-block:: python

   class ILLMInput(BaseModel):
       """Input data for LLM operations."""
       
       system_prompt: str
       user_message: str
       regular_functions: Dict[str, Callable] = {}
       background_tasks: Dict[str, Callable] = {}
       structure_type: Optional[Type] = None
       max_turns: int = 5
       conversation_history: List[Dict[str, Any]] = []

Field Details:

**system_prompt**
   Instructions that define the AI's behavior, role, and constraints. This sets the context for the entire conversation.

**user_message**
   The actual user input or query that the AI should respond to.

**regular_functions**
   Dictionary of Python functions the LLM can call during processing. Results are returned to the conversation for further processing.

**background_tasks**
   Dictionary of functions that run in fire-and-forget mode. These execute independently and don't return results to the conversation.

**structure_type**
   Pydantic model class for structured output. When provided, the LLM will return a structured response matching the model schema.

**max_turns**
   Maximum number of conversation turns for function calling scenarios. Prevents infinite loops.

**conversation_history**
   Optional conversation context for multi-turn interactions.

Response Structure
------------------

Both ``chat()`` and ``stream()`` methods return responses with this structure:

.. code-block:: python

   {
       "llm_response": str | dict | Any,  # The actual response content
       "usage": {                         # Token usage information
           "prompt_tokens": int,
           "completion_tokens": int,
           "total_tokens": int
       }
   }

**llm_response Field**
   - **String**: Plain text response for simple queries
   - **Dictionary**: Structured data when ``structure_type`` is provided
   - **Any**: Custom format depending on the specific client implementation

**usage Field**
   Token consumption data for cost tracking and rate limiting. May be ``None`` if the provider doesn't support usage tracking.

Chat vs Stream Methods
----------------------

**Chat Method** (``await client.chat(input)``)
   - Returns complete response after full processing
   - Suitable for batch processing and when you need the complete result
   - Function calls are processed in sequence before returning
   - Easier to handle programmatically

**Stream Method** (``async for chunk in client.stream(input)``)
   - Yields response chunks as they arrive from the provider
   - Suitable for real-time user interfaces and long responses
   - Function calls execute immediately when detected
   - Better user experience for interactive applications

Function Calling Architecture
-----------------------------

The interface supports two types of function calling:

**Regular Functions**
   Execute with full result integration into the conversation flow:

.. code-block:: python

   def calculate(expression: str) -> float:
       """Evaluate mathematical expressions."""
       return eval(expression)
   
   input_data = ILLMInput(
       system_prompt="You are a math assistant",
       user_message="What is 25 * 4?",
       regular_functions={"calculate": calculate}
   )

**Background Tasks**
   Execute independently without affecting the conversation:

.. code-block:: python

   def log_query(query: str, user_id: str = "anonymous"):
       """Log user queries for analytics."""
       print(f"Query logged: {query} from {user_id}")
   
   input_data = ILLMInput(
       system_prompt="You are a helpful assistant",
       user_message="Hello!",
       background_tasks={"log_query": log_query}
   )

Structured Output Support
--------------------------

When ``structure_type`` is provided, the LLM returns structured data:

.. code-block:: python

   from pydantic import BaseModel, Field
   
   class TaskAnalysis(BaseModel):
       task_type: str = Field(description="Type of task identified")
       priority: int = Field(description="Priority level 1-5")
       estimated_time: int = Field(description="Estimated minutes")
       dependencies: List[str] = Field(description="Required dependencies")
   
   input_data = ILLMInput(
       system_prompt="Analyze project tasks",
       user_message="Set up CI/CD pipeline for the web app",
       structure_type=TaskAnalysis
   )
   
   response = await client.chat(input_data)
   task_analysis = response["llm_response"]  # Returns TaskAnalysis instance
   print(f"Task: {task_analysis.task_type}, Priority: {task_analysis.priority}")

Configuration Interface
-----------------------

All clients use ``ILLMConfig`` for configuration:

.. code-block:: python

   class ILLMConfig(BaseModel):
       """Configuration for LLM clients."""
       
       model: str
       temperature: float = 0.7
       max_tokens: Optional[int] = None
       top_p: float = 1.0
       frequency_penalty: float = 0.0
       presence_penalty: float = 0.0

**model**
   The specific model to use (provider-specific naming)

**temperature**
   Creativity level (0.0 = deterministic, 1.0 = very creative)

**max_tokens**
   Maximum response length in tokens

**top_p, frequency_penalty, presence_penalty**
   Advanced parameters for fine-tuning response characteristics

Error Handling Contract
-----------------------

All implementations should handle these error scenarios:

**Rate Limiting**
   Implement retry logic with exponential backoff for HTTP 429 errors.

**Invalid Function Calls**
   Gracefully handle when LLM calls non-existent functions or provides invalid arguments.

**Network Errors**
   Provide meaningful error messages for connection issues.

**Provider Errors**
   Translate provider-specific errors into consistent error types.

**Configuration Errors**
   Validate configuration at client creation time.

Implementation Guidelines
-------------------------

When implementing new LLM clients:

1. **Follow the Interface Contract**
   Implement both ``chat()`` and ``stream()`` methods with identical functionality.

2. **Handle All Input Fields**
   Support all ``ILLMInput`` fields, even if some features aren't available for your provider.

3. **Maintain Response Consistency**
   Return responses in the standard format across both methods.

4. **Implement Defensive Programming**
   Handle edge cases gracefully and provide meaningful error messages.

5. **Test Thoroughly**
   Use the standard test suite to ensure compatibility with the framework.

Usage Patterns
--------------

**Simple Query**:

.. code-block:: python

   input_data = ILLMInput(
       system_prompt="You are a helpful assistant",
       user_message="Explain quantum computing in simple terms"
   )
   response = await client.chat(input_data)

**Interactive Streaming**:

.. code-block:: python

   async for chunk in client.stream(input_data):
       if chunk.get("llm_response"):
           print(chunk["llm_response"], end="", flush=True)

**Tool-Enabled Assistant**:

.. code-block:: python

   def get_weather(city: str) -> str:
       return f"Weather in {city}: Sunny, 22°C"
   
   def save_note(content: str, category: str = "general"):
       print(f"Saved note: {content} (category: {category})")
   
   input_data = ILLMInput(
       system_prompt="You can check weather and save notes",
       user_message="What's the weather in Tokyo? Also save a note about planning a trip there.",
       regular_functions={"get_weather": get_weather},
       background_tasks={"save_note": save_note}
   )

This interface design ensures that your application code remains provider-agnostic while supporting advanced AI capabilities across all supported language models.