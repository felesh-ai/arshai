Adding LLM Providers
====================

Comprehensive guide for integrating new LLM providers into the Arshai framework with full support for function calling, streaming, and structured output.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

Arshai's LLM client architecture provides a standardized approach for integrating any LLM provider. The framework handles complex orchestration while you implement only provider-specific API integration.

**What You Get:**

- **Framework Infrastructure**: Function calling, background tasks, structured output handled automatically
- **Standardized Interface**: Consistent API across all LLM providers
- **Dual Mode Support**: Both ``chat()`` and ``stream()`` methods
- **Usage Tracking**: Automatic token usage reporting
- **Observability**: Optional OpenTelemetry integration

**What You Implement:**

- **Provider Client**: Initialize provider-specific SDK
- **API Calls**: Simple and function-calling chat/stream methods
- **Format Conversion**: Convert functions to provider format

Architecture
------------

**BaseLLMClient Hierarchy:**

.. code-block:: text

   ILLM (Protocol)
   └── BaseLLMClient (ABC)
       ├── OpenAIClient
       ├── GoogleGenAIClient ⭐ (Canonical Reference)
       ├── AnthropicClient
       └── YourCustomClient

**Google Gemini as Canonical Reference:**

The Google Gemini client (``arshai/llms/google_genai.py``) serves as the **canonical reference implementation**. All other LLM providers should follow its established patterns and standards.

See ``CLAUDE.md`` in the repository for detailed LLM client architecture standards.

Quick Start
-----------

**Minimal LLM Client:**

.. code-block:: python

   from arshai.llms.base_llm_client import BaseLLMClient
   from arshai.core.interfaces import ILLMConfig, ILLMInput
   from typing import Dict, Any, AsyncGenerator

   class MyLLMClient(BaseLLMClient):
       """Custom LLM provider implementation"""

       def _initialize_client(self):
           """Initialize provider SDK"""
           import my_provider
           return my_provider.Client(api_key="your_api_key")

       def _convert_callables_to_provider_format(self, functions):
           """Convert Python functions to provider format"""
           return [
               {"name": name, "function": func}
               for name, func in functions.items()
           ]

       async def _chat_simple(self, input: ILLMInput) -> Dict[str, Any]:
           """Handle chat without function calling"""
           response = await self._client.chat(
               model=self.config.model,
               system=input.system_prompt,
               user=input.user_message
           )

           return {
               "llm_response": response.text,
               "usage": {
                   "total_tokens": response.usage.total_tokens,
                   "prompt_tokens": response.usage.prompt_tokens,
                   "completion_tokens": response.usage.completion_tokens
               }
           }

       async def _chat_with_functions(self, input: ILLMInput) -> Dict[str, Any]:
           """Handle chat with function calling"""
           # Multi-turn function calling implementation
           pass

       async def _stream_simple(self, input: ILLMInput) -> AsyncGenerator:
           """Handle streaming without function calling"""
           async for chunk in self._client.stream(
               model=self.config.model,
               system=input.system_prompt,
               user=input.user_message
           ):
               yield {"llm_response": chunk.text, "usage": chunk.usage}

       async def _stream_with_functions(self, input: ILLMInput) -> AsyncGenerator:
           """Handle streaming with function calling"""
           # Streaming function calling implementation
           pass

Required Methods
----------------

All LLM clients **must** implement these 5 abstract methods:

.. _initialize-client:

_initialize_client()
~~~~~~~~~~~~~~~~~~~~

**Purpose**: Initialize the provider-specific SDK client.

**Signature:**

.. code-block:: python

   def _initialize_client(self) -> Any:
       """
       Initialize the LLM provider client.

       Returns:
           Provider-specific client instance
       """

**Example:**

.. code-block:: python

   def _initialize_client(self):
       """Initialize OpenAI client"""
       from openai import AsyncOpenAI
       return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

   def _initialize_client(self):
       """Initialize Google Gemini client"""
       import google.generativeai as genai
       genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
       return genai

.. _convert-callables:

_convert_callables_to_provider_format()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Convert Python callable functions to provider-specific function declaration format.

**Signature:**

.. code-block:: python

   def _convert_callables_to_provider_format(
       self,
       functions: Dict[str, Callable]
   ) -> Any:
       """
       Convert Python callables to provider-specific format.

       Args:
           functions: Dictionary mapping function names to callables

       Returns:
           Provider-specific function declarations
       """

**Example (OpenAI Format):**

.. code-block:: python

   def _convert_callables_to_provider_format(self, functions):
       """Convert to OpenAI function calling format"""
       import inspect
       from typing import get_type_hints

       tools = []
       for name, func in functions.items():
           # Get function signature
           sig = inspect.signature(func)
           hints = get_type_hints(func)

           # Build parameters schema
           parameters = {
               "type": "object",
               "properties": {},
               "required": []
           }

           for param_name, param in sig.parameters.items():
               param_type = hints.get(param_name, str)
               parameters["properties"][param_name] = {
                   "type": self._python_type_to_json_type(param_type),
                   "description": f"Parameter {param_name}"
               }

               if param.default == inspect.Parameter.empty:
                   parameters["required"].append(param_name)

           tools.append({
               "type": "function",
               "function": {
                   "name": name,
                   "description": func.__doc__ or f"Function {name}",
                   "parameters": parameters
               }
           })

       return tools

   def _python_type_to_json_type(self, py_type):
       """Convert Python type to JSON schema type"""
       type_mapping = {
           str: "string",
           int: "integer",
           float: "number",
           bool: "boolean",
           list: "array",
           dict: "object"
       }
       return type_mapping.get(py_type, "string")

**Example (Google Gemini Format):**

.. code-block:: python

   def _convert_callables_to_provider_format(self, functions):
       """Convert to Gemini function calling format"""
       from google.generativeai.types import FunctionDeclaration

       return [
           FunctionDeclaration.from_callable(func, name=name)
           for name, func in functions.items()
       ]

.. _chat-simple:

_chat_simple()
~~~~~~~~~~~~~~

**Purpose**: Handle simple chat requests without function calling.

**Signature:**

.. code-block:: python

   async def _chat_simple(self, input: ILLMInput) -> Dict[str, Any]:
       """
       Handle simple chat without tools or background tasks.

       Args:
           input: LLM input with system_prompt and user_message

       Returns:
           Dictionary with 'llm_response' and 'usage' keys
       """

**Required Response Format:**

.. code-block:: python

   {
       "llm_response": str,  # The generated text
       "usage": {
           "total_tokens": int,
           "prompt_tokens": int,
           "completion_tokens": int
       }
   }

**Example:**

.. code-block:: python

   async def _chat_simple(self, input: ILLMInput) -> Dict[str, Any]:
       """OpenAI simple chat implementation"""
       response = await self._client.chat.completions.create(
           model=self.config.model,
           messages=[
               {"role": "system", "content": input.system_prompt},
               {"role": "user", "content": input.user_message}
           ],
           temperature=self.config.temperature,
           max_tokens=self.config.max_tokens
       )

       return {
           "llm_response": response.choices[0].message.content,
           "usage": {
               "total_tokens": response.usage.total_tokens,
               "prompt_tokens": response.usage.prompt_tokens,
               "completion_tokens": response.usage.completion_tokens
           }
       }

.. _chat-with-functions:

_chat_with_functions()
~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Handle complex chat with function calling support.

**Signature:**

.. code-block:: python

   async def _chat_with_functions(self, input: ILLMInput) -> Dict[str, Any]:
       """
       Handle complex chat with tools and/or background tasks.

       Args:
           input: LLM input with regular_functions and/or background_tasks

       Returns:
           Dictionary with 'llm_response', 'usage', and optional 'function_calls'
       """

**Required Response Format:**

.. code-block:: python

   {
       "llm_response": str,  # Final generated text
       "usage": {
           "total_tokens": int,
           "prompt_tokens": int,
           "completion_tokens": int
       },
       "function_calls": [  # Optional
           {"name": str, "arguments": dict, "result": any}
       ]
   }

**Implementation Pattern (from Gemini reference):**

.. code-block:: python

   async def _chat_with_functions(self, input: ILLMInput) -> Dict[str, Any]:
       """Multi-turn function calling implementation"""

       # Combine regular functions and background tasks
       all_functions = {**input.regular_functions, **input.background_tasks}

       # Convert to provider format
       tools = self._convert_callables_to_provider_format(all_functions)

       # Initialize conversation
       messages = [
           {"role": "system", "content": input.system_prompt},
           {"role": "user", "content": input.user_message}
       ]

       total_usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
       function_calls_log = []

       # Multi-turn loop
       for turn in range(input.max_turns):
           response = await self._client.chat(
               model=self.config.model,
               messages=messages,
               tools=tools
           )

           # Accumulate usage
           self._accumulate_usage(total_usage, response.usage)

           # Check for function calls
           if not response.tool_calls:
               # No more functions, return final response
               return {
                   "llm_response": response.content,
                   "usage": total_usage,
                   "function_calls": function_calls_log
               }

           # Execute functions
           for tool_call in response.tool_calls:
               function_name = tool_call.function.name
               arguments = json.loads(tool_call.function.arguments)

               # Check if background task or regular function
               if function_name in input.background_tasks:
                   # Fire-and-forget execution
                   task = asyncio.create_task(
                       input.background_tasks[function_name](**arguments)
                   )
                   self._background_tasks.add(task)
                   task.add_done_callback(self._background_tasks.discard)

                   result = "Background task started"
               else:
                   # Regular function - execute and get result
                   result = await input.regular_functions[function_name](**arguments)

               # Log function call
               function_calls_log.append({
                   "name": function_name,
                   "arguments": arguments,
                   "result": result
               })

               # Add to conversation history
               messages.append({
                   "role": "tool",
                   "tool_call_id": tool_call.id,
                   "content": str(result)
               })

       # Max turns reached
       return {
           "llm_response": response.content,
           "usage": total_usage,
           "function_calls": function_calls_log
       }

   def _accumulate_usage(self, total: dict, new_usage: any):
       """Safely accumulate usage metadata"""
       if new_usage:
           total["total_tokens"] += getattr(new_usage, 'total_tokens', 0)
           total["prompt_tokens"] += getattr(new_usage, 'prompt_tokens', 0)
           total["completion_tokens"] += getattr(new_usage, 'completion_tokens', 0)

.. _stream-simple:

_stream_simple()
~~~~~~~~~~~~~~~~

**Purpose**: Handle streaming responses without function calling.

**Signature:**

.. code-block:: python

   async def _stream_simple(
       self,
       input: ILLMInput
   ) -> AsyncGenerator[Dict[str, Any], None]:
       """
       Handle simple streaming without tools or background tasks.

       Args:
           input: LLM input

       Yields:
           Dictionaries with partial 'llm_response' and progressive 'usage'
       """

**Yield Format:**

.. code-block:: python

   {
       "llm_response": str,  # Progressive text chunks
       "usage": dict or None  # Usage info when available
   }

**Example:**

.. code-block:: python

   async def _stream_simple(self, input: ILLMInput):
       """Stream simple responses"""
       stream = await self._client.chat.completions.create(
           model=self.config.model,
           messages=[
               {"role": "system", "content": input.system_prompt},
               {"role": "user", "content": input.user_message}
           ],
           stream=True
       )

       cumulative_usage = {"total_tokens": 0}

       async for chunk in stream:
           # Extract text
           if chunk.choices[0].delta.content:
               yield {
                   "llm_response": chunk.choices[0].delta.content,
                   "usage": None
               }

           # Extract usage (usually in final chunk)
           if hasattr(chunk, 'usage') and chunk.usage:
               cumulative_usage = {
                   "total_tokens": chunk.usage.total_tokens,
                   "prompt_tokens": chunk.usage.prompt_tokens,
                   "completion_tokens": chunk.usage.completion_tokens
               }

       # Yield final usage
       yield {"llm_response": "", "usage": cumulative_usage}

.. _stream-with-functions:

_stream_with_functions()
~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Handle streaming with function calling support.

**Signature:**

.. code-block:: python

   async def _stream_with_functions(
       self,
       input: ILLMInput
   ) -> AsyncGenerator[Dict[str, Any], None]:
       """
       Handle streaming with tools and/or background tasks.

       Args:
           input: LLM input with regular_functions and/or background_tasks

       Yields:
           Dictionaries with progressive text and function execution results
       """

**Implementation Requirements:**

1. **Real-time Processing**: Execute functions immediately when detected
2. **Progressive Text**: Stream text chunks as they arrive
3. **Safe Usage**: Handle None values in usage metadata
4. **Completion Logic**: Detect finish_reason and handle properly

**Example:**

.. code-block:: python

   async def _stream_with_functions(self, input: ILLMInput):
       """Stream with function calling support"""
       all_functions = {**input.regular_functions, **input.background_tasks}
       tools = self._convert_callables_to_provider_format(all_functions)

       messages = [
           {"role": "system", "content": input.system_prompt},
           {"role": "user", "content": input.user_message}
       ]

       cumulative_usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

       for turn in range(input.max_turns):
           stream = await self._client.chat(
               model=self.config.model,
               messages=messages,
               tools=tools,
               stream=True
           )

           current_function_calls = []
           text_buffer = ""

           async for chunk in stream:
               # Handle usage
               if hasattr(chunk, 'usage') and chunk.usage:
                   self._accumulate_usage(cumulative_usage, chunk.usage)

               # Handle text
               if chunk.choices[0].delta.content:
                   text_chunk = chunk.choices[0].delta.content
                   text_buffer += text_chunk
                   yield {"llm_response": text_chunk, "usage": None}

               # Handle function calls
               if chunk.choices[0].delta.tool_calls:
                   for tool_call in chunk.choices[0].delta.tool_calls:
                       current_function_calls.append(tool_call)

               # Check for completion
               if chunk.choices[0].finish_reason:
                   if not current_function_calls:
                       # No functions, we're done
                       yield {"llm_response": "", "usage": cumulative_usage}
                       return

           # Execute collected function calls
           for tool_call in current_function_calls:
               function_name = tool_call.function.name
               arguments = json.loads(tool_call.function.arguments)

               if function_name in input.background_tasks:
                   task = asyncio.create_task(
                       input.background_tasks[function_name](**arguments)
                   )
                   self._background_tasks.add(task)
                   task.add_done_callback(self._background_tasks.discard)
                   result = "Background task started"
               else:
                   result = await input.regular_functions[function_name](**arguments)

               messages.append({
                   "role": "tool",
                   "tool_call_id": tool_call.id,
                   "content": str(result)
               })

               yield {
                   "llm_response": f"\n[Function {function_name} executed]\n",
                   "usage": None
               }

       # Max turns reached
       yield {"llm_response": "", "usage": cumulative_usage}

Core Patterns
-------------

Input Processing Logic
~~~~~~~~~~~~~~~~~~~~~~

**Standard pattern all LLM clients should follow:**

.. code-block:: python

   async def chat(self, input: ILLMInput) -> Dict[str, Any]:
       """Unified chat interface (implemented by BaseLLMClient)"""

       # Decision logic
       has_tools = input.regular_functions and len(input.regular_functions) > 0
       has_background_tasks = input.background_tasks and len(input.background_tasks) > 0
       needs_function_calling = has_tools or has_background_tasks

       if not needs_function_calling:
           # Simple case: direct response
           return await self._chat_simple(input)
       else:
           # Complex case: multi-turn function calling
           return await self._chat_with_functions(input)

Function Execution Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Parallel Execution Pattern:**

.. code-block:: python

   # Regular tools execute in parallel via asyncio.gather()
   function_tasks = []
   for function_call in regular_function_calls:
       task = input.regular_functions[function_call.name](**function_call.args)
       function_tasks.append(task)

   # Execute all in parallel
   if function_tasks:
       results = await asyncio.gather(*function_tasks)

**Fire-and-Forget Pattern:**

.. code-block:: python

   # Background tasks run independently via asyncio.create_task()
   for background_call in background_task_calls:
       task = asyncio.create_task(
           input.background_tasks[background_call.name](**background_call.args)
       )

       # Track to prevent garbage collection
       self._background_tasks.add(task)
       task.add_done_callback(self._background_tasks.discard)

Error Handling Standards
~~~~~~~~~~~~~~~~~~~~~~~~~

**Defensive Programming Patterns:**

.. code-block:: python

   async def _chat_simple(self, input: ILLMInput) -> Dict[str, Any]:
       """Handle chat with comprehensive error handling"""
       try:
           response = await self._client.chat(...)

           # Safe usage extraction
           usage = {
               "total_tokens": getattr(response.usage, 'total_tokens', 0),
               "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
               "completion_tokens": getattr(response.usage, 'completion_tokens', 0)
           }

           return {
               "llm_response": response.text or "",
               "usage": usage
           }

       except TimeoutError as e:
           self.logger.error(f"Request timeout: {e}")
           raise

       except Exception as e:
           self.logger.exception(f"Chat failed: {e}")
           # Return partial result with error info
           return {
               "llm_response": "",
               "usage": {"total_tokens": 0},
               "error": str(e)
           }

Best Practices
--------------

**1. Follow Reference Implementation:**

Use Google Gemini client (``arshai/llms/google_genai.py``) as your template.

**2. Implement All Abstract Methods:**

All 5 abstract methods must be fully implemented:

- ``_initialize_client()``
- ``_convert_callables_to_provider_format()``
- ``_chat_simple()``
- ``_chat_with_functions()``
- ``_stream_simple()``
- ``_stream_with_functions()``

**3. Consistent Response Format:**

Always return dictionaries with standardized keys:

- ``llm_response``: Generated text
- ``usage``: Token usage information
- ``function_calls`` (optional): Function execution log

**4. Safe Usage Handling:**

Handle None values in usage metadata gracefully:

.. code-block:: python

   def _safe_extract_usage(self, response):
       """Extract usage with null safety"""
       if not hasattr(response, 'usage') or not response.usage:
           return {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

       return {
           "total_tokens": getattr(response.usage, 'total_tokens', 0),
           "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
           "completion_tokens": getattr(response.usage, 'completion_tokens', 0)
       }

**5. Progressive Streaming:**

Yield chunks immediately for real-time responses:

.. code-block:: python

   async for chunk in stream:
       if chunk.text:
           yield {"llm_response": chunk.text, "usage": None}  # Immediate yield

**6. Background Task Management:**

Track tasks in ``self._background_tasks`` set to prevent garbage collection:

.. code-block:: python

   def __init__(self, config):
       super().__init__(config)
       self._background_tasks = set()  # Track background tasks

**7. Comprehensive Logging:**

Use ``self.logger`` for debugging information:

.. code-block:: python

   self.logger.debug(f"Sending request to {self.config.model}")
   self.logger.info(f"Received response with {usage['total_tokens']} tokens")
   self.logger.error(f"Request failed: {error}")

Testing Your LLM Client
-----------------------

**Unit Testing:**

.. code-block:: python

   import pytest
   from unittest.mock import AsyncMock, MagicMock
   from arshai.core.interfaces import ILLMConfig, ILLMInput

   @pytest.mark.asyncio
   async def test_chat_simple():
       """Test simple chat without function calling"""
       # Mock provider client
       mock_provider = AsyncMock()
       mock_response = MagicMock()
       mock_response.text = "Test response"
       mock_response.usage.total_tokens = 100
       mock_provider.chat.return_value = mock_response

       # Create client
       config = ILLMConfig(model="test-model")
       client = MyLLMClient(config)
       client._client = mock_provider  # Inject mock

       # Test
       result = await client.chat(ILLMInput(
           system_prompt="You are helpful",
           user_message="Hello"
       ))

       assert result["llm_response"] == "Test response"
       assert result["usage"]["total_tokens"] == 100

   @pytest.mark.asyncio
   async def test_chat_with_functions():
       """Test function calling"""
       def test_function(param: str) -> str:
           return f"Result: {param}"

       config = ILLMConfig(model="test-model")
       client = MyLLMClient(config)

       result = await client.chat(ILLMInput(
           system_prompt="You are helpful",
           user_message="Call test_function with param='test'",
           regular_functions={"test_function": test_function}
       ))

       assert "llm_response" in result
       assert "function_calls" in result

**Integration Testing:**

.. code-block:: python

   @pytest.mark.asyncio
   @pytest.mark.integration
   async def test_real_provider():
       """Integration test with real provider"""
       import os

       config = ILLMConfig(model="gpt-3.5-turbo")
       client = MyLLMClient(config)

       result = await client.chat(ILLMInput(
           system_prompt="You are helpful",
           user_message="Say hello"
       ))

       assert result is not None
       assert len(result["llm_response"]) > 0
       assert result["usage"]["total_tokens"] > 0

Complete Example
----------------

**Full LLM Client Implementation:**

See the repository for complete examples:

- ``arshai/llms/openai_client.py`` - OpenAI implementation
- ``arshai/llms/google_genai.py`` - Google Gemini (canonical reference)
- ``arshai/llms/anthropic_client.py`` - Anthropic Claude implementation

**Minimal Working Example:**

See the Quick Start section at the beginning of this guide.

Resources
---------

**Reference Implementations:**

- ``arshai/llms/google_genai.py`` - **Canonical reference (use this as template)**
- ``arshai/llms/openai_client.py`` - OpenAI example
- ``arshai/llms/base_llm_client.py`` - Base class source

**Documentation:**

- :doc:`../reference/base-classes` - BaseLLMClient API reference
- :doc:`../reference/interfaces` - ILLM interface specification
- ``CLAUDE.md`` - LLM client architecture standards (in repository)

**Examples:**

- :doc:`../framework/llm-clients/index` - LLM client usage guide
- :doc:`../tutorials/index` - Complete tutorials using LLM clients

Next Steps
----------

1. **Study Reference**: Review Google Gemini client (``arshai/llms/google_genai.py``)
2. **Implement Methods**: Create your provider client with all 5 abstract methods
3. **Test Thoroughly**: Write unit and integration tests
4. **Follow Standards**: Adhere to patterns from ``CLAUDE.md``
5. **Contribute**: Submit PR to add your provider to the framework

Ready to integrate your LLM provider? Start with the canonical reference and adapt for your provider's API!
