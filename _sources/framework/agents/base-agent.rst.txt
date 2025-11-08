BaseAgent Class
===============

The ``BaseAgent`` class is the foundation of Arshai's agent framework, providing the core structure and interface that all agents must implement. Located in ``arshai.agents.base``, it defines the contract for agent behavior while giving you complete control over implementation details.

Core Concepts
-------------

**Abstract Base Class**
   ``BaseAgent`` is abstract and requires you to implement the ``process`` method with your custom logic.

**Interface Compliance**
   All agents automatically implement the ``IAgent`` interface, ensuring consistent behavior across your system.

**Complete Freedom**
   You control what your agent returns, how it processes input, and what tools it uses.

**Stateless Design**
   Agents don't maintain internal state, making them easier to test, debug, and scale.

Class Definition
----------------

.. code-block:: python

   from abc import ABC, abstractmethod
   from typing import Any
   from arshai.core.interfaces.iagent import IAgent, IAgentInput
   from arshai.core.interfaces.illm import ILLM

   class BaseAgent(IAgent, ABC):
       """
       Abstract base class for all agents in the Arshai framework.
       
       Provides the foundation for building purpose-driven agents that wrap
       LLM clients with custom logic and behavior.
       """
       
       def __init__(self, llm_client: ILLM, system_prompt: str, **kwargs):
           """
           Initialize the base agent.
           
           Args:
               llm_client: The LLM client for AI interactions
               system_prompt: The system prompt that defines agent behavior
               **kwargs: Additional configuration parameters
           """
           self.llm_client = llm_client
           self.system_prompt = system_prompt
           
           # Store any additional configuration
           for key, value in kwargs.items():
               setattr(self, key, value)
       
       @abstractmethod
       async def process(self, input: IAgentInput) -> Any:
           """
           Process the input and return a response.
           
           This is the core method that defines your agent's behavior.
           You have complete authority over:
           - Response format (string, dict, stream, custom objects)
           - How to use the LLM client
           - Tool integration patterns
           - Error handling approach
           
           Args:
               input: Structured input containing message and metadata
               
           Returns:
               Any type of response your agent needs to return
           """
           pass

Required Implementation
-----------------------

**The Process Method**

Every agent must implement the ``process`` method. This is where your agent's logic lives:

.. code-block:: python

   async def process(self, input: IAgentInput) -> Any:
       """Your agent's core logic goes here."""
       
       # Create LLM input
       llm_input = ILLMInput(
           system_prompt=self.system_prompt,
           user_message=input.message
       )
       
       # Call LLM
       result = await self.llm_client.chat(llm_input)
       
       # Return response (any format you choose)
       return result["llm_response"]

Input Structure
---------------

All agents receive input as ``IAgentInput``:

.. code-block:: python

   from arshai.core.interfaces.iagent import IAgentInput
   
   # Input structure
   input = IAgentInput(
       message="User's message",
       metadata={
           "user_id": "12345",
           "conversation_id": "abc123",
           "session_data": {...}
       }
   )

**Message Field**
   The primary user input or task description.

**Metadata Field**
   Optional dictionary for context like user IDs, session data, conversation history, etc.

Response Flexibility
--------------------

Agents can return any type of data:

**Simple String Response**:

.. code-block:: python

   async def process(self, input: IAgentInput) -> str:
       result = await self.llm_client.chat(llm_input)
       return result["llm_response"]

**Structured Data Response**:

.. code-block:: python

   async def process(self, input: IAgentInput) -> Dict[str, Any]:
       result = await self.llm_client.chat(llm_input)
       return {
           "response": result["llm_response"],
           "confidence": 0.95,
           "tokens_used": result["usage"]["total_tokens"]
       }

**Custom Object Response**:

.. code-block:: python

   from pydantic import BaseModel
   
   class AnalysisResult(BaseModel):
       sentiment: str
       confidence: float
       key_points: List[str]
   
   async def process(self, input: IAgentInput) -> AnalysisResult:
       llm_input = ILLMInput(
           system_prompt=self.system_prompt,
           user_message=input.message,
           structure_type=AnalysisResult  # Request structured output
       )
       
       result = await self.llm_client.chat(llm_input)
       return result["llm_response"]  # Returns AnalysisResult instance

**Stream Response**:

.. code-block:: python

   from typing import AsyncGenerator
   
   async def process(self, input: IAgentInput) -> AsyncGenerator[str, None]:
       llm_input = ILLMInput(
           system_prompt=self.system_prompt,
           user_message=input.message
       )
       
       async for chunk in self.llm_client.stream(llm_input):
           if chunk.get("llm_response"):
               yield chunk["llm_response"]

Configuration Patterns
-----------------------

**Basic Configuration**:

.. code-block:: python

   class MyAgent(BaseAgent):
       def __init__(self, llm_client: ILLM, system_prompt: str):
           super().__init__(llm_client, system_prompt)
       
       async def process(self, input: IAgentInput) -> str:
           # Implementation here
           pass

**Extended Configuration**:

.. code-block:: python

   class ConfigurableAgent(BaseAgent):
       def __init__(self, llm_client: ILLM, system_prompt: str, 
                    max_tokens: int = 500, temperature: float = 0.7,
                    tools: dict = None):
           super().__init__(llm_client, system_prompt)
           self.max_tokens = max_tokens
           self.temperature = temperature
           self.tools = tools or {}
       
       async def process(self, input: IAgentInput) -> str:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               max_tokens=self.max_tokens,
               temperature=self.temperature,
               regular_functions=self.tools
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

**Using Kwargs for Flexibility**:

.. code-block:: python

   class FlexibleAgent(BaseAgent):
       def __init__(self, llm_client: ILLM, system_prompt: str, **kwargs):
           super().__init__(llm_client, system_prompt, **kwargs)
           
           # Access configuration through attributes
           self.response_format = getattr(self, 'response_format', 'text')
           self.enable_tools = getattr(self, 'enable_tools', False)
           self.custom_settings = getattr(self, 'custom_settings', {})

Common Implementation Patterns
------------------------------

**Error Handling**:

.. code-block:: python

   async def process(self, input: IAgentInput) -> str:
       try:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]
           
       except Exception as e:
           # Handle errors gracefully
           return f"Error processing request: {str(e)}"

**Input Validation**:

.. code-block:: python

   async def process(self, input: IAgentInput) -> str:
       # Validate input
       if not input.message or not input.message.strip():
           return "Error: Empty message provided"
       
       if len(input.message) > 5000:
           return "Error: Message too long (max 5000 characters)"
       
       # Process valid input
       llm_input = ILLMInput(
           system_prompt=self.system_prompt,
           user_message=input.message
       )
       
       result = await self.llm_client.chat(llm_input)
       return result["llm_response"]

**Multi-Step Processing**:

.. code-block:: python

   async def process(self, input: IAgentInput) -> Dict[str, Any]:
       # Step 1: Analyze intent
       intent_input = ILLMInput(
           system_prompt="Analyze the user's intent",
           user_message=input.message
       )
       intent_result = await self.llm_client.chat(intent_input)
       
       # Step 2: Generate response based on intent
       response_input = ILLMInput(
           system_prompt=self.system_prompt,
           user_message=f"Intent: {intent_result['llm_response']}\nUser: {input.message}"
       )
       response_result = await self.llm_client.chat(response_input)
       
       return {
           "intent": intent_result["llm_response"],
           "response": response_result["llm_response"],
           "total_tokens": intent_result["usage"]["total_tokens"] + response_result["usage"]["total_tokens"]
       }

Tool Integration
----------------

Agents can use any Python callable as a tool:

.. code-block:: python

   def search_database(query: str, table: str = "products") -> List[dict]:
       """Search database for products."""
       # Your search implementation
       return search_results

   def calculate_price(base_price: float, discount: float = 0.0) -> float:
       """Calculate final price with discount."""
       return base_price * (1 - discount)

   class ShoppingAgent(BaseAgent):
       async def process(self, input: IAgentInput) -> str:
           tools = {
               "search_database": search_database,
               "calculate_price": calculate_price
           }
           
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               regular_functions=tools  # Tools available to LLM
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

Background Tasks
----------------

Use background tasks for fire-and-forget operations:

.. code-block:: python

   def log_interaction(user_id: str, message: str, response: str):
       """Log interaction for analytics (runs in background)."""
       print(f"Logged: {user_id} - {message[:50]}... -> {response[:50]}...")

   def send_notification(event: str, user_id: str, priority: str = "normal"):
       """Send notification to admin system."""
       print(f"Notification: {event} for {user_id} (priority: {priority})")

   class LoggingAgent(BaseAgent):
       async def process(self, input: IAgentInput) -> str:
           user_id = input.metadata.get("user_id", "anonymous")
           
           background_tasks = {
               "log_interaction": log_interaction,
               "send_notification": send_notification
           }
           
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               background_tasks=background_tasks
           )
           
           result = await self.llm_client.chat(llm_input)
           # Background tasks execute automatically
           return result["llm_response"]

Testing BaseAgent Implementations
----------------------------------

Agents are easy to test because they're just classes with clear interfaces:

.. code-block:: python

   import pytest
   from unittest.mock import AsyncMock
   
   @pytest.mark.asyncio
   async def test_my_agent():
       # Mock LLM client
       mock_llm = AsyncMock()
       mock_llm.chat.return_value = {
           "llm_response": "Test response",
           "usage": {"total_tokens": 25}
       }
       
       # Create agent
       agent = MyAgent(
           llm_client=mock_llm,
           system_prompt="You are a test agent"
       )
       
       # Test agent
       input_data = IAgentInput(
           message="Test message",
           metadata={"user_id": "test123"}
       )
       
       response = await agent.process(input_data)
       
       # Verify behavior
       assert response == "Test response"
       mock_llm.chat.assert_called_once()
       
       # Verify LLM input
       call_args = mock_llm.chat.call_args[0][0]
       assert call_args.system_prompt == "You are a test agent"
       assert call_args.user_message == "Test message"

Best Practices
--------------

**1. Keep Agents Focused**
   Each agent should have a single, clear purpose. Create multiple specialized agents rather than one complex agent.

**2. Validate Inputs**
   Always validate the input message and metadata before processing.

**3. Handle Errors Gracefully**
   Implement proper error handling and return meaningful error messages.

**4. Use Type Hints**
   Provide clear type hints for better IDE support and code documentation.

**5. Document Your Agents**
   Include docstrings explaining what your agent does and how to use it.

**6. Test Thoroughly**
   Write unit tests for your agent logic, especially edge cases and error conditions.

**7. Design for Reusability**
   Make your agents configurable and reusable across different contexts.

Common Mistakes
---------------

**❌ Storing State in Agents**:

.. code-block:: python

   class BadAgent(BaseAgent):
       def __init__(self, llm_client, system_prompt):
           super().__init__(llm_client, system_prompt)
           self.conversation_history = []  # ❌ Don't store state
       
       async def process(self, input: IAgentInput) -> str:
           self.conversation_history.append(input.message)  # ❌ Stateful
           # ...

**✅ Stateless Design**:

.. code-block:: python

   class GoodAgent(BaseAgent):
       async def process(self, input: IAgentInput) -> str:
           # Use metadata for context, don't store state
           conversation_id = input.metadata.get("conversation_id")
           # Retrieve context from external storage if needed
           # ...

**❌ Ignoring Input Structure**:

.. code-block:: python

   async def process(self, input: IAgentInput) -> str:
       message = input  # ❌ Wrong - input is IAgentInput object
       # ...

**✅ Proper Input Handling**:

.. code-block:: python

   async def process(self, input: IAgentInput) -> str:
       message = input.message  # ✅ Correct
       metadata = input.metadata or {}  # ✅ Handle optional metadata
       # ...

Next Steps
----------

- :doc:`creating-agents` - Step-by-step guide to building your first agent
- :doc:`tools-and-callables` - Complete guide to tool integration
- :doc:`agent-patterns` - Common patterns and best practices
- :doc:`stateless-design` - Deep dive into stateless agent architecture