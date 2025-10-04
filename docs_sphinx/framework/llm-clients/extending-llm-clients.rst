Extending LLM Clients
=====================

This guide shows how to create custom LLM clients that integrate with the Arshai framework. By implementing the ILLM interface, your custom client will work seamlessly with all framework components.

.. note::
   Use the Google Gemini client (``arshai.llms.google_genai.GeminiClient``) as your reference implementation. According to the codebase documentation, it serves as the canonical example of best practices for LLM client development.

Framework Architecture
----------------------

All LLM clients in Arshai follow a standardized architecture:

**BaseLLMClient Foundation**
   All clients extend ``BaseLLMClient``, which provides common functionality for usage tracking, function orchestration, and streaming support.

**ILLM Interface Contract**
   Clients must implement the ``ILLM`` interface with ``chat()`` and ``stream()`` methods.

**Consistent Patterns**
   All clients follow identical patterns for function calling, structured output, and background tasks.

Basic Client Structure
----------------------

Here's the minimal structure for a custom LLM client:

.. code-block:: python

   from typing import Dict, Any, AsyncGenerator
   from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
   from arshai.llms.base_llm_client import BaseLLMClient
   from arshai.llms.utils.function_execution import FunctionCall, FunctionExecutionInput
   
   class CustomLLMClient(BaseLLMClient):
       """
       Custom LLM client implementation.
       
       This demonstrates the minimal requirements for creating
       a new LLM provider integration.
       """
       
       def _initialize_client(self) -> Any:
           """Initialize the provider-specific client."""
           # Your provider-specific initialization
           api_key = os.environ.get("CUSTOM_API_KEY")
           if not api_key:
               raise ValueError("CUSTOM_API_KEY environment variable required")
           
           # Return your provider's client instance
           return YourProviderClient(api_key=api_key)
       
       def _extract_and_standardize_usage(self, response: Any) -> Dict[str, Any]:
           """Extract usage metadata from provider response."""
           # Convert provider-specific usage to standard format
           return {
               "input_tokens": response.usage.prompt_tokens,
               "output_tokens": response.usage.completion_tokens,
               "total_tokens": response.usage.total_tokens,
               "thinking_tokens": 0,  # If not supported
               "tool_calling_tokens": 0,  # If not available
               "provider": self._provider_name,
               "model": self.config.model,
               "request_id": getattr(response, 'id', None)
           }
       
       async def _chat_simple(self, input: ILLMInput) -> Dict[str, Any]:
           """Handle simple chat without tools."""
           # Implement basic chat functionality
           response = await self._client.chat(
               messages=[
                   {"role": "system", "content": input.system_prompt},
                   {"role": "user", "content": input.user_message}
               ],
               model=self.config.model,
               temperature=self.config.temperature
           )
           
           usage = self._extract_and_standardize_usage(response)
           return {"llm_response": response.content, "usage": usage}
       
       async def _chat_with_functions(self, input: ILLMInput) -> Dict[str, Any]:
           """Handle chat with function calling."""
           # Implement function calling support
           # See reference implementation for complete pattern
           pass
       
       async def _stream_simple(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
           """Handle simple streaming."""
           # Implement streaming support
           # See reference implementation for complete pattern
           pass
       
       async def _stream_with_functions(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
           """Handle streaming with function calling."""
           # Implement streaming with function support
           # See reference implementation for complete pattern
           pass

Required Methods
----------------

Every LLM client must implement these core methods:

**1. Client Initialization**:

.. code-block:: python

   def _initialize_client(self) -> Any:
       """Initialize the provider's client library."""
       # Environment variable validation
       api_key = os.environ.get("YOUR_PROVIDER_API_KEY")
       if not api_key:
           raise ValueError("YOUR_PROVIDER_API_KEY environment variable required")
       
       # Provider-specific client creation
       try:
           # Attempt to use safe HTTP configuration if available
           from arshai.clients.utils.safe_http_client import SafeHttpClientFactory
           return SafeHttpClientFactory.create_your_provider_client(api_key=api_key)
       except ImportError:
           # Fallback to basic client
           return YourProviderClient(api_key=api_key)

**2. Usage Standardization**:

.. code-block:: python

   def _extract_and_standardize_usage(self, response: Any) -> Dict[str, Any]:
       """Convert provider usage to standard format."""
       # Extract provider-specific usage data
       usage = response.usage  # or however your provider structures this
       
       return {
           "input_tokens": getattr(usage, 'prompt_tokens', 0),
           "output_tokens": getattr(usage, 'completion_tokens', 0), 
           "total_tokens": getattr(usage, 'total_tokens', 0),
           "thinking_tokens": getattr(usage, 'reasoning_tokens', 0),  # If supported
           "tool_calling_tokens": getattr(usage, 'function_tokens', 0),  # If available
           "provider": self._provider_name,  # Set in __init__
           "model": self.config.model,
           "request_id": getattr(response, 'id', None)
       }

**3. Simple Chat Implementation**:

.. code-block:: python

   async def _chat_simple(self, input: ILLMInput) -> Dict[str, Any]:
       """Basic chat without tools or structured output."""
       # Prepare messages in your provider's format
       messages = [
           {"role": "system", "content": input.system_prompt},
           {"role": "user", "content": input.user_message}
       ]
       
       # Handle structured output if requested
       kwargs = {
           "messages": messages,
           "model": self.config.model,
           "temperature": self.config.temperature,
           "max_tokens": self.config.max_tokens
       }
       
       if input.structure_type:
           # Add your provider's structured output configuration
           kwargs["response_format"] = self._create_structure_schema(input.structure_type)
       
       # Make API call
       response = await self._client.chat.completions.create(**kwargs)
       
       # Process response
       usage = self._extract_and_standardize_usage(response)
       
       if input.structure_type:
           # Parse structured response
           structured_data = self._parse_structured_response(response, input.structure_type)
           return {"llm_response": structured_data, "usage": usage}
       else:
           # Return text response
           return {"llm_response": response.choices[0].message.content, "usage": usage}

Function Calling Implementation
-------------------------------

The framework provides orchestration utilities to handle function execution. Your client should focus on provider-specific function calling:

**Function Declaration Conversion**:

.. code-block:: python

   def _convert_callables_to_provider_format(self, functions: Dict[str, Callable]) -> List[Dict]:
       """Convert Python functions to provider's function format."""
       provider_functions = []
       
       for name, func in functions.items():
           try:
               # Use introspection to build function schema
               sig = inspect.signature(func)
               description = func.__doc__ or f"Execute {name} function"
               
               # Build parameter schema
               properties = {}
               required = []
               
               for param_name, param in sig.parameters.items():
                   if param_name == 'self':
                       continue
                   
                   param_type = self._python_type_to_json_schema_type(param.annotation)
                   properties[param_name] = {
                       "type": param_type,
                       "description": f"{param_name} parameter"
                   }
                   
                   if param.default == inspect.Parameter.empty:
                       required.append(param_name)
               
               # Create provider-specific function definition
               function_def = {
                   "name": name,
                   "description": description,
                   "parameters": {
                       "type": "object", 
                       "properties": properties,
                       "required": required
                   }
               }
               
               provider_functions.append(function_def)
               
           except Exception as e:
               self.logger.warning(f"Failed to convert function {name}: {e}")
               continue
       
       return provider_functions

**Function Calling Orchestration**:

.. code-block:: python

   async def _chat_with_functions(self, input: ILLMInput) -> Dict[str, Any]:
       """Handle chat with function calling."""
       messages = self._create_messages(input)
       
       # Prepare functions for your provider
       provider_functions = []
       all_functions = {}
       if input.regular_functions:
           all_functions.update(input.regular_functions)
       if input.background_tasks:
           all_functions.update(input.background_tasks)
       
       if all_functions:
           provider_functions = self._convert_callables_to_provider_format(all_functions)
       
       # Multi-turn conversation loop
       current_turn = 0
       accumulated_usage = None
       
       while current_turn < input.max_turns:
           # Make API call with functions
           response = await self._client.chat.completions.create(
               messages=messages,
               model=self.config.model,
               temperature=self.config.temperature,
               tools=provider_functions if provider_functions else None
           )
           
           # Accumulate usage
           current_usage = self._extract_and_standardize_usage(response)
           accumulated_usage = self._accumulate_usage_safely(current_usage, accumulated_usage)
           
           # Check for function calls
           function_calls = self._extract_function_calls(response)
           if function_calls:
               # Process function calls using framework orchestration
               function_calls_list = self._prepare_function_calls_for_orchestrator(function_calls, input)
               
               # Execute functions via framework orchestrator
               execution_input = FunctionExecutionInput(
                   function_calls=function_calls_list,
                   available_functions=input.regular_functions or {},
                   available_background_tasks=input.background_tasks or {}
               )
               
               execution_result = await self._execute_functions_with_orchestrator(execution_input)
               
               # Add results to conversation
               self._add_function_results_to_messages(execution_result, messages)
               
               # Continue if we have regular functions
               if execution_result.get('regular_results'):
                   current_turn += 1
                   continue
           
           # Return final response
           return {
               "llm_response": response.choices[0].message.content,
               "usage": accumulated_usage
           }
       
       return {
           "llm_response": "Maximum function calling turns reached",
           "usage": accumulated_usage
       }

Streaming Implementation
------------------------

Implement streaming with progressive function execution:

.. code-block:: python

   async def _stream_with_functions(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
       """Handle streaming with function calling."""
       messages = self._create_messages(input)
       provider_functions = self._prepare_functions(input)
       
       current_turn = 0
       accumulated_usage = None
       
       while current_turn < input.max_turns:
           # Create streaming request
           stream = await self._client.chat.completions.create(
               messages=messages,
               model=self.config.model,
               temperature=self.config.temperature,
               tools=provider_functions,
               stream=True
           )
           
           # Progressive streaming state
           from arshai.llms.utils.function_execution import StreamingExecutionState
           streaming_state = StreamingExecutionState()
           collected_text = ""
           function_calls_in_progress = {}
           
           # Process streaming chunks
           async for chunk in stream:
               # Handle usage metadata
               if hasattr(chunk, 'usage') and chunk.usage:
                   current_usage = self._extract_and_standardize_usage(chunk)
                   accumulated_usage = self._accumulate_usage_safely(current_usage, accumulated_usage)
               
               # Handle text content
               if chunk.choices and chunk.choices[0].delta.content:
                   collected_text += chunk.choices[0].delta.content
                   yield {"llm_response": collected_text}
               
               # Handle function calls with progressive execution
               if chunk.choices and chunk.choices[0].delta.tool_calls:
                   for tool_call_delta in chunk.choices[0].delta.tool_calls:
                       # Track function call progress
                       call_index = tool_call_delta.index
                       
                       if call_index not in function_calls_in_progress:
                           function_calls_in_progress[call_index] = {
                               "name": "",
                               "arguments": ""
                           }
                       
                       # Update function call data
                       if tool_call_delta.function.name:
                           function_calls_in_progress[call_index]["name"] = tool_call_delta.function.name
                       
                       if tool_call_delta.function.arguments:
                           function_calls_in_progress[call_index]["arguments"] += tool_call_delta.function.arguments
                       
                       # Execute when complete
                       if self._is_function_complete(function_calls_in_progress[call_index]):
                           function_call = self._create_function_call_object(
                               function_calls_in_progress[call_index], 
                               call_index, 
                               input
                           )
                           
                           if not streaming_state.is_already_executed(function_call):
                               task = await self._execute_function_progressively(function_call, input)
                               streaming_state.add_function_task(task, function_call)
           
           # Gather function results and continue conversation if needed
           if streaming_state.active_function_tasks:
               execution_result = await self._gather_progressive_results(streaming_state.active_function_tasks)
               self._add_function_results_to_messages(execution_result, messages)
               
               if execution_result.get('regular_results'):
                   current_turn += 1
                   continue
           
           break
       
       # Final usage yield
       yield {"llm_response": None, "usage": accumulated_usage}

Helper Utilities
----------------

Implement these utility methods for consistency:

**Type Conversion**:

.. code-block:: python

   def _python_type_to_json_schema_type(self, python_type) -> str:
       """Convert Python types to JSON schema types."""
       if python_type == str:
           return "string"
       elif python_type == int:
           return "integer"
       elif python_type == float:
           return "number"
       elif python_type == bool:
           return "boolean"
       elif python_type == list or (hasattr(python_type, '__origin__') and python_type.__origin__ == list):
           return "array"
       elif python_type == dict or (hasattr(python_type, '__origin__') and python_type.__origin__ == dict):
           return "object"
       else:
           return "string"

**Function Call Processing**:

.. code-block:: python

   def _prepare_function_calls_for_orchestrator(self, function_calls, input: ILLMInput) -> List[FunctionCall]:
       """Convert provider function calls to framework objects."""
       function_calls_list = []
       
       for i, call in enumerate(function_calls):
           function_name = call.function.name
           try:
               function_args = json.loads(call.function.arguments) if call.function.arguments else {}
           except json.JSONDecodeError:
               function_args = {}
           
           call_id = f"{function_name}_{i}"
           is_background = function_name in (input.background_tasks or {})
           
           function_calls_list.append(FunctionCall(
               name=function_name,
               args=function_args,
               call_id=call_id,
               is_background=is_background
           ))
       
       return function_calls_list

Testing Your Client
--------------------

Create comprehensive tests for your client:

.. code-block:: python

   import pytest
   from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
   from your_package.custom_llm_client import CustomLLMClient
   
   @pytest.fixture
   def client_config():
       return ILLMConfig(
           model="your-model-name",
           temperature=0.7,
           max_tokens=500
       )
   
   @pytest.fixture  
   def client(client_config):
       return CustomLLMClient(client_config)
   
   @pytest.mark.asyncio
   async def test_simple_chat(client):
       """Test basic chat functionality."""
       input_data = ILLMInput(
           system_prompt="You are a helpful assistant",
           user_message="Hello, how are you?"
       )
       
       response = await client.chat(input_data)
       
       assert "llm_response" in response
       assert "usage" in response
       assert isinstance(response["llm_response"], str)
       assert len(response["llm_response"]) > 0
   
   @pytest.mark.asyncio
   async def test_function_calling(client):
       """Test function calling capabilities."""
       def test_function(x: int, y: int) -> int:
           """Add two numbers."""
           return x + y
       
       input_data = ILLMInput(
           system_prompt="You can use tools to perform calculations",
           user_message="What is 5 + 3?",
           regular_functions={"test_function": test_function}
       )
       
       response = await client.chat(input_data)
       
       assert "llm_response" in response
       assert "8" in response["llm_response"] or "eight" in response["llm_response"].lower()
   
   @pytest.mark.asyncio
   async def test_streaming(client):
       """Test streaming functionality.""" 
       input_data = ILLMInput(
           system_prompt="You are a helpful assistant",
           user_message="Tell me a short story"
       )
       
       chunks = []
       async for chunk in client.stream(input_data):
           chunks.append(chunk)
       
       assert len(chunks) > 0
       assert any(chunk.get("llm_response") for chunk in chunks)

Best Practices
--------------

**1. Follow the Reference Implementation**
   Use the Gemini client as your template for architecture and patterns.

**2. Implement Defensive Programming**
   Handle edge cases gracefully and provide meaningful error messages.

**3. Support All Framework Features**
   Implement function calling, structured output, background tasks, and streaming.

**4. Maintain Provider Abstractions**
   Hide provider-specific details behind the standard interface.

**5. Test Thoroughly**
   Use the framework test patterns to ensure compatibility.

**6. Document Provider-Specific Features**
   Clearly document any limitations or special capabilities.

**7. Handle Rate Limiting**
   Implement appropriate retry logic for your provider's rate limits.

**8. Safe HTTP Configuration**
   Use the SafeHttpClientFactory when available for better reliability.

Common Patterns
---------------

**Environment Variable Configuration**:

.. code-block:: python

   def __init__(self, config: ILLMConfig, **kwargs):
       # Provider-specific environment variables
       self.api_key = os.environ.get("YOUR_PROVIDER_API_KEY")
       self.base_url = os.environ.get("YOUR_PROVIDER_BASE_URL", "https://api.yourprovider.com")
       
       if not self.api_key:
           raise ValueError("YOUR_PROVIDER_API_KEY environment variable required")
       
       super().__init__(config, **kwargs)

**Error Handling**:

.. code-block:: python

   try:
       response = await self._client.chat(...)
   except YourProviderRateLimitError:
       raise Exception("Rate limit exceeded - implement retry logic")
   except YourProviderAuthError:
       raise Exception("Authentication failed - check API key")
   except Exception as e:
       self.logger.error(f"Provider error: {e}")
       raise

**Context Management**:

.. code-block:: python

   def close(self):
       """Cleanup provider resources."""
       try:
           if hasattr(self._client, 'close'):
               self._client.close()
       except Exception as e:
           self.logger.warning(f"Error closing client: {e}")

This comprehensive guide provides everything needed to create a custom LLM client that integrates seamlessly with the Arshai framework. Remember to use the Gemini client as your reference implementation for the most current patterns and best practices.