Agents (Layer 2)
================

Layer 2 provides the agent foundation for building purpose-driven components that wrap LLM clients with custom logic. Agents are the core building blocks that transform raw LLM capabilities into focused, reusable components.

.. toctree::
   :maxdepth: 2
   :caption: Agent Framework

   base-agent
   creating-agents
   tools-and-callables
   agent-patterns
   stateless-design

.. toctree::
   :maxdepth: 2
   :caption: Example Documentation
   
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Reference Implementations
   
   ../../../implementations/agents/index

Core Philosophy
---------------

Agents in Arshai follow these principles:

**Direct Instantiation**
   You create agents explicitly, configuring them exactly as needed for your use case.

**Stateless Design**
   Agents don't maintain internal state, making them easier to test, debug, and scale.

**Maximum Developer Authority**
   Agents can return any type of data (strings, objects, streams) and implement any logic you need.

**Interface Compliance**
   All agents implement the IAgent interface, ensuring consistent behavior while allowing complete customization.

**Tool Integration**
   Agents work seamlessly with Python callables as tools - no special interfaces required.

What IS the Agent Framework
----------------------------

The core agent framework consists of:

**BaseAgent Class**
   Abstract base class that provides the foundation for all agents. Located in ``arshai.agents.base``.

**IAgent Interface**
   Protocol defining the contract that all agents must implement.

**IAgentInput Structure**
   Standardized input format for agent communication.

**Tool Integration Patterns**
   How agents work with Python callables as tools.

These components ARE the framework and provide the building blocks for creating custom agents.

What is NOT the Framework
--------------------------

The ``hub/`` directory contains **reference implementations** - examples showing how we've used the framework:

**Example Agents** (in ``arshai.agents.hub/``)
   - ``WorkingMemoryAgent``: Example of memory-enabled agent
   - Future agent examples as they're added

These implementations are provided as working examples and starting points, but they are not prescriptive. You're encouraged to:

- Use them as-is if they fit your needs
- Modify them for your requirements
- Build completely different implementations
- Ignore them entirely and create your own

.. note::
   **Framework vs Examples**: The core framework is in ``base.py`` and interfaces. Everything in ``hub/`` represents "our experience" with the framework, not "the way" to build agents.

Basic Agent Structure
---------------------

Every agent extends BaseAgent and implements the process method:

.. code-block:: python

   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces.iagent import IAgentInput
   from arshai.core.interfaces.illm import ILLM, ILLMInput
   
   class MyAgent(BaseAgent):
       """Custom agent with specific behavior."""
       
       def __init__(self, llm_client: ILLM, system_prompt: str, **kwargs):
           super().__init__(llm_client, system_prompt, **kwargs)
           # Your custom initialization
       
       async def process(self, input: IAgentInput) -> Any:
           """
           Process input and return any type of response.
           
           You have complete authority over:
           - Response format (string, dict, stream, custom objects)
           - How to use the LLM client
           - Tool integration patterns
           - Error handling approach
           """
           # Your custom logic here
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

Agent Creation Patterns
-----------------------

**Simple Text Agent**:

.. code-block:: python

   class ConversationAgent(BaseAgent):
       """Agent for general conversations."""
       
       async def process(self, input: IAgentInput) -> str:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

**Structured Response Agent**:

.. code-block:: python

   from pydantic import BaseModel, Field
   from typing import List
   
   class AnalysisResult(BaseModel):
       sentiment: str = Field(description="Sentiment analysis result")
       confidence: float = Field(description="Confidence score 0-1")
       key_points: List[str] = Field(description="Key points identified")
   
   class AnalysisAgent(BaseAgent):
       """Agent that returns structured analysis."""
       
       async def process(self, input: IAgentInput) -> AnalysisResult:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               structure_type=AnalysisResult
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]  # Returns AnalysisResult instance

**Tool-Enabled Agent**:

.. code-block:: python

   def search_database(query: str, table: str = "products") -> List[dict]:
       """Search database for products."""
       # Your database search implementation
       return results
   
   def calculate_price(base_price: float, discount: float = 0.0) -> float:
       """Calculate final price with discount."""
       return base_price * (1 - discount)
   
   class ShoppingAgent(BaseAgent):
       """Agent with shopping tools."""
       
       async def process(self, input: IAgentInput) -> str:
           # Tools are just Python functions
           tools = {
               "search_database": search_database,
               "calculate_price": calculate_price
           }
           
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               regular_functions=tools
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

**Streaming Agent**:

.. code-block:: python

   from typing import AsyncGenerator
   
   class StreamingAgent(BaseAgent):
       """Agent that streams responses."""
       
       async def process(self, input: IAgentInput) -> AsyncGenerator[str, None]:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )
           
           async for chunk in self.llm_client.stream(llm_input):
               if chunk.get("llm_response"):
                   yield chunk["llm_response"]

Tools and Callables
-------------------

Arshai takes a unique approach to tools - they're just Python callables, not classes with interfaces:

**Any Callable is a Tool**:

.. code-block:: python

   # Functions
   def get_weather(city: str) -> str:
       return f"Weather in {city}: Sunny, 22°C"
   
   # Methods
   class Calculator:
       def add(self, a: float, b: float) -> float:
           return a + b
   
   calc = Calculator()
   
   # Lambdas
   multiply = lambda x, y: x * y
   
   # Standard library functions
   import os
   
   # All work as tools
   agent_tools = {
       "get_weather": get_weather,
       "add_numbers": calc.add,
       "multiply": multiply,
       "list_directory": os.listdir
   }

**Tool Integration with Agents**:

.. code-block:: python

   class ToolEnabledAgent(BaseAgent):
       def __init__(self, llm_client: ILLM, system_prompt: str, tools: dict = None):
           super().__init__(llm_client, system_prompt)
           self.tools = tools or {}
       
       async def process(self, input: IAgentInput) -> str:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               regular_functions=self.tools
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

**Background Tasks**:

.. code-block:: python

   def log_interaction(user_message: str, agent_response: str, user_id: str = "anonymous"):
       """Log interaction for analytics (background task)."""
       print(f"Logged interaction: {user_message[:50]}... -> {agent_response[:50]}...")
   
   class LoggingAgent(BaseAgent):
       async def process(self, input: IAgentInput) -> str:
           background_tasks = {
               "log_interaction": log_interaction
           }
           
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               background_tasks=background_tasks
           )
           
           result = await self.llm_client.chat(llm_input)
           # Logging happens automatically in background
           return result["llm_response"]

Advanced Patterns
-----------------

**Configuration-Driven Agent**:

.. code-block:: python

   class ConfigurableAgent(BaseAgent):
       def __init__(self, llm_client: ILLM, config: dict):
           system_prompt = config.get("system_prompt", "You are a helpful assistant")
           super().__init__(llm_client, system_prompt)
           
           self.response_format = config.get("response_format", "text")
           self.max_tokens = config.get("max_tokens", 500)
           self.tools = config.get("tools", {})
       
       async def process(self, input: IAgentInput) -> Any:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               regular_functions=self.tools,
               max_tokens=self.max_tokens
           )
           
           if self.response_format == "structured":
               llm_input.structure_type = self.config.get("structure_type")
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

**Multi-Step Agent**:

.. code-block:: python

   class MultiStepAgent(BaseAgent):
       """Agent that performs multiple processing steps."""
       
       async def process(self, input: IAgentInput) -> dict:
           # Step 1: Analyze intent
           intent_prompt = f"{self.system_prompt}\n\nAnalyze the user's intent in this message: {input.message}"
           intent_input = ILLMInput(system_prompt="", user_message=intent_prompt)
           intent_result = await self.llm_client.chat(intent_input)
           
           # Step 2: Generate response based on intent
           response_prompt = f"Based on intent '{intent_result['llm_response']}', respond to: {input.message}"
           response_input = ILLMInput(system_prompt=self.system_prompt, user_message=response_prompt)
           response_result = await self.llm_client.chat(response_input)
           
           return {
               "intent": intent_result["llm_response"],
               "response": response_result["llm_response"],
               "metadata": {
                   "steps": 2,
                   "total_tokens": intent_result["usage"]["total_tokens"] + response_result["usage"]["total_tokens"]
               }
           }

Agent Testing
-------------

Agents are easy to test because they're just classes with clear interfaces:

.. code-block:: python

   import pytest
   from unittest.mock import AsyncMock
   
   @pytest.mark.asyncio
   async def test_conversation_agent():
       # Mock LLM client
       mock_llm = AsyncMock()
       mock_llm.chat.return_value = {
           "llm_response": "Hello! How can I help you?",
           "usage": {"total_tokens": 15}
       }
       
       # Create agent
       agent = ConversationAgent(
           llm_client=mock_llm,
           system_prompt="You are a helpful assistant"
       )
       
       # Test agent
       input_data = IAgentInput(message="Hello")
       response = await agent.process(input_data)
       
       assert response == "Hello! How can I help you?"
       mock_llm.chat.assert_called_once()

Usage Examples
--------------

**Creating and Using Agents**:

.. code-block:: python

   from arshai.llms.openai import OpenAIClient
   from arshai.core.interfaces.illm import ILLMConfig
   from arshai.core.interfaces.iagent import IAgentInput
   
   # Create LLM client
   llm_config = ILLMConfig(model="gpt-4o-mini", temperature=0.7)
   llm_client = OpenAIClient(llm_config)
   
   # Create agent
   agent = ConversationAgent(
       llm_client=llm_client,
       system_prompt="You are a helpful customer service agent"
   )
   
   # Use agent
   input_data = IAgentInput(
       message="I need help with my order",
       metadata={"user_id": "12345", "session_id": "abc123"}
   )
   
   response = await agent.process(input_data)
   print(response)

**Agent with Tools**:

.. code-block:: python

   def lookup_order(order_id: str) -> dict:
       """Look up order information."""
       # Your order lookup logic
       return {"order_id": order_id, "status": "shipped", "tracking": "1234567890"}
   
   def cancel_order(order_id: str) -> bool:
       """Cancel an order."""
       # Your cancellation logic
       return True
   
   # Create tool-enabled agent
   tools = {
       "lookup_order": lookup_order,
       "cancel_order": cancel_order
   }
   
   service_agent = ToolEnabledAgent(
       llm_client=llm_client,
       system_prompt="You are a customer service agent. Use tools to help customers with orders.",
       tools=tools
   )
   
   response = await service_agent.process(IAgentInput(
       message="What's the status of order 12345?"
   ))

Benefits of This Architecture
-----------------------------

**Complete Control**
   You decide what your agent returns, how it processes input, and what tools it uses.

**Easy Testing**
   Mock the LLM client and test your agent logic independently.

**Tool Flexibility**
   Any Python function can be a tool - no framework-specific interfaces required.

**Type Safety**
   Full type hints and IDE support for your custom agent implementations.

**Scalability**
   Stateless design makes agents easy to scale and deploy.

**Reusability**
   Agents can be composed into larger systems and reused across projects.

Reference Implementations
-------------------------

The framework includes reference implementations in the ``hub/`` directory:

**WorkingMemoryAgent** (``arshai.agents.hub.working_memory``)
   Example agent that manages conversation memory. Shows how to:
   
   - Integrate with memory managers
   - Handle conversation context
   - Store and retrieve working memory
   - Process metadata from agent inputs

.. note::
   **Reference Implementation**: The ``WorkingMemoryAgent`` is an example of how to build memory-enabled agents. It's not part of the core framework, but shows one approach to memory management.

Next Steps
----------

- :doc:`base-agent` - Deep dive into the BaseAgent class
- :doc:`creating-agents` - Step-by-step guide to building custom agents
- :doc:`tools-and-callables` - Complete guide to tool integration
- :doc:`agent-patterns` - Common patterns and best practices
- :doc:`../building-systems/index` - Composing agents into systems