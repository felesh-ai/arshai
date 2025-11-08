Base Classes
============

Foundation classes that provide reusable implementations for extending the Arshai framework.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

Arshai provides base classes that implement common patterns and infrastructure while giving developers flexibility to customize behavior. These classes follow the framework's philosophy of **developer authority** - providing sensible defaults without enforcing rigid constraints.

**Design Philosophy:**

- **Minimal Required Implementation**: Only implement abstract methods specific to your needs
- **Inherit Common Infrastructure**: Get logging, configuration, and helpers automatically
- **Override When Needed**: Customize any behavior without framework restrictions
- **Type Safety**: Full type hints and protocol compliance

Agent Base Classes
------------------

.. _baseagent-class:

BaseAgent
~~~~~~~~~

**Location:** ``arshai/agents/base.py``

Abstract base agent implementation providing foundational structure for all agents.

**Class Definition:**

.. code-block:: python

   from abc import ABC, abstractmethod
   from arshai.core.interfaces.iagent import IAgent, IAgentInput
   from arshai.core.interfaces.illm import ILLM

   class BaseAgent(IAgent, ABC):
       """Abstract base agent implementation"""

       def __init__(self, llm_client: ILLM, system_prompt: str, **kwargs):
           """
           Initialize the base agent.

           Args:
               llm_client: The LLM client to use for processing
               system_prompt: The system prompt defining agent behavior
               **kwargs: Additional configuration passed to the agent
           """
           self.llm_client = llm_client
           self.system_prompt = system_prompt
           self.config = kwargs

       @abstractmethod
       async def process(self, input: IAgentInput) -> Any:
           """Process input and return response - MUST be implemented"""
           ...

**Attributes:**

``llm_client: ILLM``
   The LLM client instance used for generating responses. Automatically available to all subclasses.

``system_prompt: str``
   The system prompt that defines the agent's behavior and personality.

``config: Dict[str, Any]``
   Additional configuration parameters passed during initialization via ``**kwargs``.

**Abstract Methods:**

Subclasses **must** implement:

``async process(input: IAgentInput) -> Any``
   Process the input and return a response.

   **Implementation Freedom:**
      - Return any data structure (string, dict, custom DTO, generator)
      - Choose response format (streaming, non-streaming, structured)
      - Implement custom error handling
      - Integrate tools, memory, or other capabilities

**Usage Examples:**

**Simple Agent:**

.. code-block:: python

   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces import IAgentInput, ILLM, ILLMInput

   class SimpleResponseAgent(BaseAgent):
       """Agent that returns simple text responses"""

       async def process(self, input: IAgentInput) -> str:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )
           result = await self.llm_client.chat(llm_input)
           return result['llm_response']

   # Usage
   from arshai.llms.openai_client import OpenAIClient
   from arshai.core.interfaces import ILLMConfig

   llm = OpenAIClient(ILLMConfig(model="gpt-4"))
   agent = SimpleResponseAgent(
       llm,
       system_prompt="You are a helpful assistant"
   )

   response = await agent.process(IAgentInput(message="Hello!"))
   print(response)  # Simple string response

**Structured Response Agent:**

.. code-block:: python

   class AnalysisAgent(BaseAgent):
       """Agent that returns structured analysis"""

       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )
           result = await self.llm_client.chat(llm_input)

           return {
               "response": result['llm_response'],
               "confidence": 0.95,
               "sentiment": "positive",
               "tokens_used": result['usage']['total_tokens']
           }

**Agent with Tool Integration:**

.. code-block:: python

   class ToolEnabledAgent(BaseAgent):
       """Agent with external tool capabilities"""

       def __init__(self, llm_client, system_prompt, tools=None):
           super().__init__(llm_client, system_prompt)
           self.tools = tools or []

       async def process(self, input: IAgentInput) -> dict:
           # Convert tools to callable functions
           tool_functions = {
               tool.name: tool.execute
               for tool in self.tools
           }

           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               regular_functions=tool_functions
           )

           result = await self.llm_client.chat(llm_input)

           return {
               "response": result['llm_response'],
               "tools_used": [call['name'] for call in result.get('function_calls', [])],
               "usage": result['usage']
           }

**Agent with State Management:**

.. code-block:: python

   class StatefulAgent(BaseAgent):
       """Agent that maintains internal state"""

       def __init__(self, llm_client, system_prompt):
           super().__init__(llm_client, system_prompt)
           self.interaction_count = 0
           self.conversation_history = []

       async def process(self, input: IAgentInput) -> dict:
           self.interaction_count += 1

           # Add context from history
           context = self._build_context_from_history()
           enhanced_message = f"{context}\n\nUser: {input.message}"

           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=enhanced_message
           )

           result = await self.llm_client.chat(llm_input)
           response = result['llm_response']

           # Update history
           self.conversation_history.append({
               "user": input.message,
               "assistant": response
           })

           return {
               "response": response,
               "interaction_count": self.interaction_count
           }

       def _build_context_from_history(self) -> str:
           if not self.conversation_history:
               return ""
           recent = self.conversation_history[-3:]  # Last 3 turns
           return "\n".join([
               f"User: {turn['user']}\nAssistant: {turn['assistant']}"
               for turn in recent
           ])

**Best Practices:**

1. **Always Call Super:** Call ``super().__init__()`` in your constructor
2. **Use LLM Client:** Leverage ``self.llm_client`` for LLM interactions
3. **Store Configuration:** Use ``self.config`` for additional parameters
4. **Type Hints:** Specify return types for better IDE support
5. **Error Handling:** Implement appropriate error handling for your use case

**See Also:**
   - :ref:`IAgent <iagent-interface>` - Agent interface specification
   - :doc:`../framework/agents/index` - Agent development guide
   - :doc:`../tutorials/simple-chatbot` - Complete agent tutorial

.. _workingmemoryagent-class:

WorkingMemoryAgent
~~~~~~~~~~~~~~~~~~

**Location:** ``arshai/agents/hub/working_memory.py``

Specialized agent for managing conversation working memory with automatic history tracking and memory updates.

**Class Definition:**

.. code-block:: python

   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces import IMemoryManager

   class WorkingMemoryAgent(BaseAgent):
       """Agent specialized in managing conversation working memory"""

       def __init__(
           self,
           llm_client: ILLM,
           system_prompt: str = None,
           memory_manager: IMemoryManager = None,
           chat_history_client: Any = None,
           **kwargs
       ):
           """
           Initialize the working memory agent.

           Args:
               llm_client: LLM client for generating memory updates
               system_prompt: Optional custom prompt (uses default if not provided)
               memory_manager: Memory manager for storage operations
               chat_history_client: Optional client for fetching conversation history
           """
           ...

**Capabilities:**

- **Fetches Conversation History**: Retrieves past interactions from chat history storage
- **Retrieves Current Memory**: Loads existing working memory from storage (e.g., Redis)
- **Generates Memory Updates**: Uses LLM to create revised working memory summaries
- **Stores Updated Memory**: Persists updated memory for future reference

**Attributes:**

``memory_manager: IMemoryManager``
   Memory manager instance for storage operations.

``chat_history: Any``
   Optional chat history client for retrieving conversation history.

**Methods:**

``async process(input: IAgentInput) -> str``
   Process memory update request.

   **Args:**
      - ``input``: Input containing new interaction and metadata with ``conversation_id``

   **Returns:**
      str: Status of the operation:
         - ``"success"`` - Memory updated successfully
         - ``"error: <description>"`` - Operation failed
         - ``"error: no conversation_id provided"`` - Missing required metadata

   **Process Flow:**
      1. Extracts ``conversation_id`` from ``input.metadata``
      2. Fetches current working memory from storage
      3. Fetches conversation history if available
      4. Generates updated memory using LLM
      5. Stores the updated memory

**Usage Example:**

.. code-block:: python

   from arshai.agents.hub.working_memory import WorkingMemoryAgent
   from arshai.memory.redis_memory import RedisWorkingMemoryManager
   from arshai.llms.openai_client import OpenAIClient
   from arshai.core.interfaces import ILLMConfig, IAgentInput

   # Initialize components
   llm = OpenAIClient(ILLMConfig(model="gpt-4"))
   memory_manager = RedisWorkingMemoryManager(redis_client)

   # Create working memory agent
   memory_agent = WorkingMemoryAgent(
       llm_client=llm,
       memory_manager=memory_manager
   )

   # Update memory after interaction
   result = await memory_agent.process(IAgentInput(
       message="User asked about pricing for enterprise plan",
       metadata={"conversation_id": "user_123"}
   ))

   if result == "success":
       print("Memory updated successfully")
   else:
       print(f"Memory update failed: {result}")

**Integration with Conversational Agents:**

.. code-block:: python

   class ConversationalAgent(BaseAgent):
       """Agent with automatic memory management"""

       def __init__(self, llm_client, system_prompt, memory_agent):
           super().__init__(llm_client, system_prompt)
           self.memory_agent = memory_agent

       async def process(self, input: IAgentInput) -> dict:
           # Process user message
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )
           result = await self.llm_client.chat(llm_input)

           # Update working memory in background
           if input.metadata and "conversation_id" in input.metadata:
               await self.memory_agent.process(IAgentInput(
                   message=f"User: {input.message}\nAssistant: {result['llm_response']}",
                   metadata=input.metadata
               ))

           return {"response": result['llm_response']}

**Default System Prompt:**

If no custom prompt is provided, WorkingMemoryAgent uses:

.. code-block:: text

   You are a memory management assistant responsible for maintaining conversation context.

   Your tasks:
   1. Analyze conversation history and current interaction
   2. Extract key information, facts, and context
   3. Generate a concise working memory summary
   4. Focus on information relevant for future interactions

   Keep the memory:
   - Concise but comprehensive
   - Focused on actionable information
   - Updated with latest context
   - Free from redundancy

**See Also:**
   - :ref:`IMemoryManager <imemorymanager-interface>` - Memory interface
   - :doc:`../implementations/memory/redis-memory` - Redis memory backend
   - :doc:`../framework/memory/index` - Memory systems guide

LLM Base Classes
----------------

.. _basellmclient-class:

BaseLLMClient
~~~~~~~~~~~~~

**Location:** ``arshai/llms/base_llm_client.py``

Framework-standardized base class for all LLM clients. Handles all framework requirements while requiring providers to implement only their specific API integration methods.

**Class Definition:**

.. code-block:: python

   from abc import ABC, abstractmethod
   from arshai.core.interfaces.illm import ILLM, ILLMConfig, ILLMInput

   class BaseLLMClient(ILLM, ABC):
       """Framework-standardized base class for all LLM clients"""

       def __init__(
           self,
           config: ILLMConfig,
           observability_config: Optional[PackageObservabilityConfig] = None
       ):
           """
           Initialize the base LLM client.

           Args:
               config: LLM configuration
               observability_config: Optional observability configuration
           """
           self.config = config
           self.logger = logging.getLogger(self.__class__.__name__)
           self._function_orchestrator = FunctionOrchestrator()
           self.observability = get_llm_observability(observability_config)
           self._client = self._initialize_client()

**Framework Features (Handled Automatically):**

- **Dual Interface Support**: Both ``chat()`` and ``stream()`` methods
- **Function Calling Orchestration**: Regular functions and background tasks
- **Structured Output Handling**: Type-safe structured responses
- **Usage Tracking**: Standardized token usage reporting
- **Error Handling**: Resilient error handling and logging
- **Routing Logic**: Automatic routing between simple and complex cases
- **Observability**: Optional OpenTelemetry integration

**Abstract Methods (Contributors Must Implement):**

``_initialize_client() -> Any``
   Initialize the LLM provider client.

   **Returns:**
      Provider-specific client instance (e.g., OpenAI(), GoogleGenerativeAI())

``_convert_callables_to_provider_format(functions: Dict[str, Callable]) -> Any``
   Convert Python callables to provider-specific function declarations.

   **Args:**
      - ``functions``: Dictionary mapping function names to callables

   **Returns:**
      Provider-specific function declaration format

``async _chat_simple(input: ILLMInput) -> Dict[str, Any]``
   Handle simple chat without tools or background tasks.

   **Args:**
      - ``input``: LLM input with system_prompt and user_message only

   **Returns:**
      Dictionary with ``llm_response`` and ``usage`` keys

``async _chat_with_functions(input: ILLMInput) -> Dict[str, Any]``
   Handle complex chat with tools and/or background tasks.

   **Args:**
      - ``input``: LLM input with regular_functions and/or background_tasks

   **Returns:**
      Dictionary with ``llm_response``, ``usage``, and optional ``function_calls`` keys

``async _stream_simple(input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]``
   Handle simple streaming without tools or background tasks.

   **Yields:**
      Dictionaries with partial ``llm_response`` and progressive ``usage``

``async _stream_with_functions(input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]``
   Handle complex streaming with tools and/or background tasks.

   **Yields:**
      Dictionaries with partial responses and function execution results

**Implementation Example:**

.. code-block:: python

   from arshai.llms.base_llm_client import BaseLLMClient
   from arshai.core.interfaces import ILLMConfig, ILLMInput

   class MyLLMClient(BaseLLMClient):
       """Custom LLM provider implementation"""

       def _initialize_client(self):
           """Initialize provider client"""
           import my_provider
           return my_provider.Client(api_key=os.getenv("MY_PROVIDER_KEY"))

       def _convert_callables_to_provider_format(self, functions):
           """Convert to provider's function format"""
           return [
               {
                   "name": name,
                   "description": func.__doc__ or "",
                   "parameters": self._extract_parameters(func)
               }
               for name, func in functions.items()
           ]

       async def _chat_simple(self, input: ILLMInput):
           """Handle simple chat"""
           response = await self._client.chat(
               model=self.config.model,
               messages=[
                   {"role": "system", "content": input.system_prompt},
                   {"role": "user", "content": input.user_message}
               ],
               temperature=self.config.temperature
           )

           return {
               "llm_response": response.content,
               "usage": {
                   "total_tokens": response.usage.total_tokens,
                   "prompt_tokens": response.usage.prompt_tokens,
                   "completion_tokens": response.usage.completion_tokens
               }
           }

       async def _chat_with_functions(self, input: ILLMInput):
           """Handle chat with function calling"""
           tools = self._convert_callables_to_provider_format(
               {**input.regular_functions, **input.background_tasks}
           )

           # Multi-turn function calling loop
           messages = [
               {"role": "system", "content": input.system_prompt},
               {"role": "user", "content": input.user_message}
           ]

           for turn in range(input.max_turns):
               response = await self._client.chat(
                   model=self.config.model,
                   messages=messages,
                   tools=tools
               )

               if not response.tool_calls:
                   # No more function calls, return final response
                   return {
                       "llm_response": response.content,
                       "usage": {...}
                   }

               # Execute functions and continue
               for tool_call in response.tool_calls:
                   result = await self._execute_function(tool_call, input)
                   messages.append({
                       "role": "tool",
                       "tool_call_id": tool_call.id,
                       "content": str(result)
                   })

           return {"llm_response": response.content, "usage": {...}}

       async def _stream_simple(self, input: ILLMInput):
           """Handle simple streaming"""
           stream = await self._client.stream(
               model=self.config.model,
               messages=[...]
           )

           async for chunk in stream:
               yield {
                   "llm_response": chunk.content,
                   "usage": chunk.usage if chunk.usage else None
               }

       async def _stream_with_functions(self, input: ILLMInput):
           """Handle streaming with function calling"""
           # Similar to _chat_with_functions but yields progressively
           ...

**Reference Implementations:**

The framework includes reference implementations that demonstrate best practices:

- ``OpenAIClient`` (``arshai/llms/openai_client.py``) - OpenAI integration
- ``GoogleGenAIClient`` (``arshai/llms/google_genai.py``) - Google Gemini (canonical reference)

**Google Gemini as Canonical Reference:**

The Google Gemini client serves as the **canonical reference implementation** for all LLM providers. All other LLM clients should follow its patterns:

- Input processing logic
- Function calling architecture
- Streaming implementation
- Error handling standards
- Usage tracking patterns

See ``CLAUDE.md`` for detailed LLM client architecture standards.

**Best Practices:**

1. **Follow Reference Implementation**: Use Google Gemini client as template
2. **Implement All Abstract Methods**: All 5 abstract methods must be implemented
3. **Consistent Response Format**: Always return dictionaries with ``llm_response`` and ``usage``
4. **Safe Usage Handling**: Handle None values in usage metadata gracefully
5. **Progressive Streaming**: Yield chunks immediately for real-time responses
6. **Background Task Management**: Track tasks in ``self._background_tasks`` set
7. **Comprehensive Logging**: Use ``self.logger`` for debugging information

**See Also:**
   - :ref:`ILLM <illm-interface>` - LLM interface specification
   - :doc:`../framework/llm-clients/index` - LLM client guide
   - ``CLAUDE.md`` - LLM client architecture standards

Base Class Hierarchy
---------------------

**Agent Hierarchy:**

.. code-block:: text

   IAgent (Protocol)
   └── BaseAgent (ABC)
       ├── WorkingMemoryAgent
       ├── YourCustomAgent
       └── [Other specialized agents]

**LLM Client Hierarchy:**

.. code-block:: text

   ILLM (Protocol)
   └── BaseLLMClient (ABC)
       ├── OpenAIClient
       ├── GoogleGenAIClient (canonical reference)
       ├── AnthropicClient
       └── [Other provider clients]

Extension Guidelines
--------------------

**Creating Custom Agents:**

1. Inherit from ``BaseAgent``
2. Implement ``async process(input: IAgentInput)`` method
3. Use ``self.llm_client`` for LLM interactions
4. Use ``self.system_prompt`` for agent behavior
5. Store additional config in ``self.config``

**Creating Custom LLM Clients:**

1. Inherit from ``BaseLLMClient``
2. Implement all 5 abstract methods
3. Follow Google Gemini reference implementation patterns
4. Handle function calling according to framework standards
5. Return standardized response formats

**Testing Your Implementations:**

.. code-block:: python

   import pytest
   from unittest.mock import AsyncMock

   @pytest.mark.asyncio
   async def test_custom_agent():
       # Mock LLM client
       mock_llm = AsyncMock()
       mock_llm.chat.return_value = {
           "llm_response": "Test response",
           "usage": {"total_tokens": 100}
       }

       # Test agent
       agent = MyCustomAgent(mock_llm, "Test prompt")
       result = await agent.process(IAgentInput(message="Hello"))

       assert result is not None
       mock_llm.chat.assert_called_once()

Next Steps
----------

- **Build Agents**: See :doc:`../tutorials/simple-chatbot` for complete agent tutorial
- **Implement LLM Client**: See :doc:`../framework/llm-clients/index` for provider integration
- **Review Interfaces**: See :doc:`interfaces` for protocol specifications
- **Explore Models**: See :doc:`models` for all DTO structures
