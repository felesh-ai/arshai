Building Custom Agents
======================

Comprehensive guide for creating custom agents with specialized behaviors, tool integrations, and advanced patterns.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

Custom agents extend the framework with specialized behaviors tailored to your specific needs. Arshai provides flexible patterns for agent development while maintaining clean architecture and type safety.

**When to Build Custom Agents:**

- Implement domain-specific logic (customer support, data analysis, code review)
- Integrate with external tools and APIs
- Manage complex state across interactions
- Customize response formats and streaming behavior
- Build multi-step reasoning workflows

**Design Choices:**

The framework gives you two approaches:

1. **Extend BaseAgent**: Inherit common infrastructure (LLM client, system prompt, config)
2. **Implement IAgent Protocol**: Full flexibility with duck typing

Choose BaseAgent for most cases - it provides sensible defaults while allowing complete customization.

Agent Architecture
------------------

**Core Responsibilities:**

.. code-block:: text

   Agent
   ├── Input Processing: Handle IAgentInput
   ├── LLM Interaction: Use LLM client for generation
   ├── Tool Orchestration: Manage external tool calls
   ├── State Management: Track conversation state
   └── Response Formatting: Return custom data structures

**Key Components:**

- **LLM Client**: Language model for generation
- **System Prompt**: Defines agent behavior and personality
- **Tools**: External capabilities (search, database queries, APIs)
- **Memory**: Conversation history and context
- **Response Format**: Custom output structure

Quick Start
-----------

**Simplest Custom Agent:**

.. code-block:: python

   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces import IAgentInput, ILLMInput

   class EchoAgent(BaseAgent):
       """Agent that echoes user input with LLM enhancement"""

       async def process(self, input: IAgentInput) -> str:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=f"Echo this message with enthusiasm: {input.message}"
           )

           result = await self.llm_client.chat(llm_input)
           return result['llm_response']

   # Usage
   from arshai.llms.openai_client import OpenAIClient
   from arshai.core.interfaces import ILLMConfig

   llm = OpenAIClient(ILLMConfig(model="gpt-4"))
   agent = EchoAgent(llm, "You are an enthusiastic echo bot")

   response = await agent.process(IAgentInput(message="Hello!"))
   print(response)  # "HELLO! So great to hear from you!"

**Structured Response Agent:**

.. code-block:: python

   from typing import Dict, Any

   class AnalysisAgent(BaseAgent):
       """Agent that returns structured analysis"""

       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=f"Analyze: {input.message}"
           )

           result = await self.llm_client.chat(llm_input)

           return {
               "analysis": result['llm_response'],
               "confidence": 0.95,
               "tokens_used": result['usage']['total_tokens'],
               "input_message": input.message
           }

BaseAgent Extension
-------------------

**Initialization Patterns:**

.. code-block:: python

   class CustomAgent(BaseAgent):
       """Custom agent with additional initialization"""

       def __init__(
           self,
           llm_client: ILLM,
           system_prompt: str,
           custom_param: str = "default",
           **kwargs
       ):
           # Call parent constructor
           super().__init__(llm_client, system_prompt, **kwargs)

           # Add custom attributes
           self.custom_param = custom_param
           self.interaction_count = 0
           self.custom_cache = {}

           # Access config from kwargs (stored in self.config)
           self.debug_mode = kwargs.get('debug_mode', False)

**Process Method Implementation:**

The ``process`` method is the only required method. You have complete freedom over:

- **Return Type**: Any data structure
- **Error Handling**: Custom exception handling
- **Streaming**: Return generators for streaming
- **Side Effects**: Logging, analytics, notifications

.. code-block:: python

   async def process(self, input: IAgentInput) -> Any:
       """
       Your custom implementation.

       Returns:
           Any: Flexible return type - string, dict, generator, custom DTO
       """
       # Your logic here
       pass

Common Agent Patterns
---------------------

Stateful Agent
~~~~~~~~~~~~~~

Maintain state across interactions:

.. code-block:: python

   from dataclasses import dataclass, field
   from typing import List, Dict, Any

   @dataclass
   class ConversationState:
       """State for conversation tracking"""
       turn_count: int = 0
       topics_discussed: List[str] = field(default_factory=list)
       user_preferences: Dict[str, Any] = field(default_factory=dict)

   class StatefulAgent(BaseAgent):
       """Agent that maintains conversation state"""

       def __init__(self, llm_client, system_prompt):
           super().__init__(llm_client, system_prompt)
           self.states: Dict[str, ConversationState] = {}

       async def process(self, input: IAgentInput) -> dict:
           # Extract conversation ID from metadata
           conv_id = input.metadata.get('conversation_id', 'default') if input.metadata else 'default'

           # Get or create state
           state = self.states.get(conv_id, ConversationState())
           state.turn_count += 1

           # Build context from state
           context = self._build_context(state)

           llm_input = ILLMInput(
               system_prompt=f"{self.system_prompt}\n\n{context}",
               user_message=input.message
           )

           result = await self.llm_client.chat(llm_input)
           response = result['llm_response']

           # Update state
           state = self._update_state(state, input.message, response)
           self.states[conv_id] = state

           return {
               "response": response,
               "turn_count": state.turn_count,
               "topics": state.topics_discussed
           }

       def _build_context(self, state: ConversationState) -> str:
           return f"""
           Conversation Context:
           - Turn: {state.turn_count}
           - Topics discussed: {', '.join(state.topics_discussed) if state.topics_discussed else 'None yet'}
           - User preferences: {state.user_preferences}
           """

       def _update_state(self, state: ConversationState, message: str, response: str) -> ConversationState:
           # Update state based on interaction
           # (simplified - real implementation would extract topics using LLM)
           return state

Tool-Enabled Agent
~~~~~~~~~~~~~~~~~~

Integrate external tools and APIs:

.. code-block:: python

   from typing import List, Callable, Dict

   class ToolEnabledAgent(BaseAgent):
       """Agent with external tool capabilities"""

       def __init__(
           self,
           llm_client,
           system_prompt,
           tools: List[Dict[str, Callable]] = None
       ):
           super().__init__(llm_client, system_prompt)
           self.tools = tools or []

       async def process(self, input: IAgentInput) -> dict:
           # Convert tools to callable dict
           tool_functions = {
               tool['name']: tool['function']
               for tool in self.tools
           }

           # Build tool descriptions for system prompt
           tool_descriptions = self._build_tool_descriptions()
           enhanced_prompt = f"{self.system_prompt}\n\nAvailable tools:\n{tool_descriptions}"

           llm_input = ILLMInput(
               system_prompt=enhanced_prompt,
               user_message=input.message,
               regular_functions=tool_functions
           )

           result = await self.llm_client.chat(llm_input)

           return {
               "response": result['llm_response'],
               "tools_used": [
                   call['name'] for call in result.get('function_calls', [])
               ],
               "usage": result['usage']
           }

       def _build_tool_descriptions(self) -> str:
           descriptions = []
           for tool in self.tools:
               desc = f"- {tool['name']}: {tool.get('description', 'No description')}"
               descriptions.append(desc)
           return "\n".join(descriptions)

   # Usage
   def get_weather(location: str) -> dict:
       """Get current weather for a location"""
       return {"temp": 72, "condition": "sunny"}

   def search_web(query: str) -> list:
       """Search the web for information"""
       return ["result1", "result2"]

   tools = [
       {
           "name": "get_weather",
           "function": get_weather,
           "description": "Get current weather for a location"
       },
       {
           "name": "search_web",
           "function": search_web,
           "description": "Search the web for information"
       }
   ]

   agent = ToolEnabledAgent(llm, "You are a helpful assistant", tools=tools)

Memory-Integrated Agent
~~~~~~~~~~~~~~~~~~~~~~~

Integrate with memory systems:

.. code-block:: python

   from arshai.core.interfaces import IMemoryManager, IMemoryInput, ConversationMemoryType

   class MemoryAwareAgent(BaseAgent):
       """Agent with memory integration"""

       def __init__(
           self,
           llm_client,
           system_prompt,
           memory_manager: IMemoryManager
       ):
           super().__init__(llm_client, system_prompt)
           self.memory_manager = memory_manager

       async def process(self, input: IAgentInput) -> dict:
           conv_id = input.metadata.get('conversation_id') if input.metadata else None

           if not conv_id:
               # No memory without conversation ID
               return await self._process_without_memory(input)

           # Retrieve working memory
           memory_input = IMemoryInput(
               conversation_id=conv_id,
               memory_type=ConversationMemoryType.WORKING_MEMORY
           )
           memories = self.memory_manager.retrieve(memory_input)

           # Build enhanced prompt with memory context
           memory_context = memories[0].working_memory if memories else "No previous context"

           enhanced_prompt = f"""
           {self.system_prompt}

           Working Memory:
           {memory_context}
           """

           llm_input = ILLMInput(
               system_prompt=enhanced_prompt,
               user_message=input.message
           )

           result = await self.llm_client.chat(llm_input)

           # Update memory
           await self._update_memory(conv_id, input.message, result['llm_response'])

           return {
               "response": result['llm_response'],
               "memory_used": bool(memories)
           }

       async def _update_memory(self, conv_id: str, message: str, response: str):
           """Update working memory with new interaction"""
           # Simplified - real implementation would use WorkingMemoryAgent
           pass

       async def _process_without_memory(self, input: IAgentInput) -> dict:
           """Fallback for no memory"""
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )
           result = await self.llm_client.chat(llm_input)
           return {"response": result['llm_response']}

Streaming Agent
~~~~~~~~~~~~~~~

Support streaming responses:

.. code-block:: python

   from typing import AsyncGenerator

   class StreamingAgent(BaseAgent):
       """Agent that streams responses"""

       async def process(self, input: IAgentInput) -> AsyncGenerator[str, None]:
           """Return async generator for streaming"""

           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )

           # Stream from LLM client
           async for chunk in self.llm_client.stream(llm_input):
               if 'llm_response' in chunk and chunk['llm_response']:
                   yield chunk['llm_response']

   # Usage
   agent = StreamingAgent(llm, "You are helpful")

   async for text_chunk in agent.process(IAgentInput(message="Tell me a story")):
       print(text_chunk, end='', flush=True)

Validation Agent
~~~~~~~~~~~~~~~~

Agent with input/output validation:

.. code-block:: python

   from pydantic import BaseModel, Field, field_validator

   class UserQuery(BaseModel):
       """Validated user query"""
       question: str = Field(min_length=3, max_length=500)
       context: str = Field(default="")

   class AgentResponse(BaseModel):
       """Validated agent response"""
       answer: str
       confidence: float = Field(ge=0.0, le=1.0)
       sources: List[str] = Field(default_factory=list)

       @field_validator('answer')
       @classmethod
       def answer_not_empty(cls, v):
           if not v or not v.strip():
               raise ValueError("Answer cannot be empty")
           return v

   class ValidatedAgent(BaseAgent):
       """Agent with strict input/output validation"""

       async def process(self, input: IAgentInput) -> AgentResponse:
           # Validate input
           try:
               query = UserQuery(
                   question=input.message,
                   context=input.metadata.get('context', '') if input.metadata else ''
               )
           except ValidationError as e:
               raise ValueError(f"Invalid input: {e}")

           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=query.question
           )

           result = await self.llm_client.chat(llm_input)

           # Validate and return structured output
           return AgentResponse(
               answer=result['llm_response'],
               confidence=0.95,
               sources=[]
           )

Advanced Patterns
-----------------

Multi-Step Reasoning Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~

Agent that performs multi-step reasoning:

.. code-block:: python

   from enum import Enum

   class ReasoningStep(Enum):
       ANALYZE = "analyze"
       PLAN = "plan"
       EXECUTE = "execute"
       VERIFY = "verify"

   class ReasoningAgent(BaseAgent):
       """Agent with multi-step reasoning"""

       async def process(self, input: IAgentInput) -> dict:
           steps_completed = []

           # Step 1: Analyze
           analysis = await self._analyze(input.message)
           steps_completed.append(ReasoningStep.ANALYZE)

           # Step 2: Plan
           plan = await self._plan(analysis)
           steps_completed.append(ReasoningStep.PLAN)

           # Step 3: Execute
           result = await self._execute(plan)
           steps_completed.append(ReasoningStep.EXECUTE)

           # Step 4: Verify
           verified_result = await self._verify(result, input.message)
           steps_completed.append(ReasoningStep.VERIFY)

           return {
               "final_answer": verified_result,
               "steps_completed": [step.value for step in steps_completed],
               "analysis": analysis,
               "plan": plan
           }

       async def _analyze(self, message: str) -> str:
           llm_input = ILLMInput(
               system_prompt="Analyze the user's question and identify key components",
               user_message=message
           )
           result = await self.llm_client.chat(llm_input)
           return result['llm_response']

       async def _plan(self, analysis: str) -> str:
           llm_input = ILLMInput(
               system_prompt="Create a step-by-step plan based on the analysis",
               user_message=analysis
           )
           result = await self.llm_client.chat(llm_input)
           return result['llm_response']

       async def _execute(self, plan: str) -> str:
           llm_input = ILLMInput(
               system_prompt="Execute the plan and provide the answer",
               user_message=plan
           )
           result = await self.llm_client.chat(llm_input)
           return result['llm_response']

       async def _verify(self, result: str, original_question: str) -> str:
           llm_input = ILLMInput(
               system_prompt="Verify the answer addresses the original question",
               user_message=f"Question: {original_question}\nAnswer: {result}"
           )
           verification = await self.llm_client.chat(llm_input)
           return verification['llm_response']

Error Handling Patterns
-----------------------

Robust Error Handling:

.. code-block:: python

   import logging
   from typing import Union

   logger = logging.getLogger(__name__)

   class RobustAgent(BaseAgent):
       """Agent with comprehensive error handling"""

       async def process(self, input: IAgentInput) -> Union[dict, str]:
           try:
               return await self._safe_process(input)
           except ValidationError as e:
               logger.error(f"Validation error: {e}")
               return {"error": "Invalid input", "details": str(e)}
           except TimeoutError as e:
               logger.error(f"Timeout: {e}")
               return {"error": "Request timed out", "details": str(e)}
           except Exception as e:
               logger.exception(f"Unexpected error: {e}")
               return {"error": "Processing failed", "details": "Internal error"}

       async def _safe_process(self, input: IAgentInput) -> dict:
           # Validate input
           if not input.message or not input.message.strip():
               raise ValidationError("Message cannot be empty")

           # Process with timeout
           try:
               llm_input = ILLMInput(
                   system_prompt=self.system_prompt,
                   user_message=input.message
               )

               result = await asyncio.wait_for(
                   self.llm_client.chat(llm_input),
                   timeout=30.0
               )

               return {
                   "response": result['llm_response'],
                   "status": "success"
               }

           except asyncio.TimeoutError:
               raise TimeoutError("LLM request exceeded 30 seconds")

Testing Custom Agents
----------------------

**Unit Testing:**

.. code-block:: python

   import pytest
   from unittest.mock import AsyncMock
   from arshai.core.interfaces import IAgentInput

   @pytest.mark.asyncio
   async def test_custom_agent():
       # Mock LLM client
       mock_llm = AsyncMock()
       mock_llm.chat.return_value = {
           "llm_response": "Test response",
           "usage": {"total_tokens": 100}
       }

       # Create agent with mock
       agent = MyCustomAgent(mock_llm, "Test prompt")

       # Test process method
       result = await agent.process(IAgentInput(message="Hello"))

       # Assertions
       assert result is not None
       assert "response" in result
       mock_llm.chat.assert_called_once()

       # Verify call arguments
       call_args = mock_llm.chat.call_args[0][0]
       assert isinstance(call_args, ILLMInput)
       assert call_args.user_message == "Hello"

**Testing with Different Inputs:**

.. code-block:: python

   @pytest.mark.parametrize("message,expected", [
       ("Hello", "greeting"),
       ("What's the weather?", "weather_query"),
       ("Tell me a joke", "entertainment"),
   ])
   @pytest.mark.asyncio
   async def test_agent_message_types(message, expected):
       mock_llm = AsyncMock()
       mock_llm.chat.return_value = {"llm_response": expected, "usage": {}}

       agent = MyCustomAgent(mock_llm, "Test")
       result = await agent.process(IAgentInput(message=message))

       assert result["response"] == expected

**Integration Testing:**

.. code-block:: python

   @pytest.mark.asyncio
   @pytest.mark.integration
   async def test_agent_with_real_llm():
       """Integration test with actual LLM"""
       from arshai.llms.openai_client import OpenAIClient
       from arshai.core.interfaces import ILLMConfig

       llm = OpenAIClient(ILLMConfig(model="gpt-3.5-turbo"))
       agent = MyCustomAgent(llm, "You are a helpful assistant")

       result = await agent.process(IAgentInput(message="Say hello"))

       assert result is not None
       assert isinstance(result, dict)
       assert "response" in result
       assert len(result["response"]) > 0

Best Practices
--------------

**1. Keep Agents Focused:**

Each agent should have a clear, single responsibility:

.. code-block:: python

   # Good: Focused responsibility
   class SentimentAnalysisAgent(BaseAgent):
       """Analyzes sentiment of text"""
       pass

   # Bad: Multiple responsibilities
   class DoEverythingAgent(BaseAgent):
       """Analyzes sentiment, translates, and generates code"""
       pass

**2. Use Type Hints:**

Provide clear type hints for better IDE support and documentation:

.. code-block:: python

   async def process(self, input: IAgentInput) -> Dict[str, Any]:
       """
       Process input and return structured response.

       Args:
           input: Agent input containing message and metadata

       Returns:
           Dictionary with response and metadata
       """
       pass

**3. Handle Errors Gracefully:**

Implement robust error handling:

.. code-block:: python

   async def process(self, input: IAgentInput) -> dict:
       try:
           return await self._internal_process(input)
       except Exception as e:
           logger.error(f"Processing failed: {e}")
           return {"error": str(e), "status": "failed"}

**4. Log Important Events:**

Use logging for debugging and monitoring:

.. code-block:: python

   import logging

   logger = logging.getLogger(__name__)

   async def process(self, input: IAgentInput) -> dict:
       logger.info(f"Processing message: {input.message[:50]}...")
       result = await self.llm_client.chat(...)
       logger.debug(f"LLM response: {result['llm_response'][:100]}...")
       return result

**5. Document Your Agents:**

Provide comprehensive docstrings:

.. code-block:: python

   class MyAgent(BaseAgent):
       """
       Agent that does X, Y, and Z.

       This agent is designed for [use case]. It integrates with [tools/systems]
       and returns [response format].

       Example:
           >>> agent = MyAgent(llm, "System prompt")
           >>> result = await agent.process(IAgentInput(message="Hello"))
           >>> print(result["response"])

       Args:
           llm_client: LLM client for generation
           system_prompt: Agent behavior definition
           custom_param: Description of custom parameter
       """
       pass

Next Steps
----------

- **Explore Examples**: See :doc:`../framework/agents/examples/index` for more patterns
- **Build Tutorials**: Follow :doc:`../tutorials/index` for complete implementations
- **Review API**: See :doc:`../reference/base-classes` for BaseAgent documentation
- **Integrate Tools**: See :doc:`../framework/agents/tool-integration` for tool patterns

Ready to build? Start with a simple agent and gradually add complexity!
