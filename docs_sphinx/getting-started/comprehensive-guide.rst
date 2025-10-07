Comprehensive Agent Guide
=========================

This guide provides a complete overview of creating and using agents in the Arshai framework, covering everything from basic concepts to advanced patterns.

Overview
--------

Arshai agents are the building blocks for creating agentic systems. This guide covers:

1. **Basic Agent Usage** - Getting started with pre-built patterns
2. **Creating Custom Agents** - Building specialized agents
3. **Working with Memory** - Stateful conversation patterns
4. **Agent with Tools** - Extending agents with external capabilities
5. **Advanced Patterns** - Sophisticated agent architectures
6. **Testing Your Agents** - Quality assurance strategies

Target Audience
---------------

- **Framework Users** - who want to use existing agents
- **Developers** - who want to create custom agents
- **Contributors** - who want to understand the architecture
- **Maintainers** - who need reference implementations

Section 1: Basic Agent Usage
----------------------------

The simplest way to get started with agents:

.. code-block:: python

   import asyncio
   from arshai.core.interfaces.iagent import IAgentInput
   from arshai.core.interfaces.illm import ILLMInput, ILLMConfig
   from arshai.agents.base import BaseAgent
   from arshai.llms.openrouter import OpenRouterClient

   # Step 1: Configure your LLM client
   llm_config = ILLMConfig(
       model="openai/gpt-4o-mini",
       temperature=0.7,     # Control randomness (0.0 = deterministic, 1.0 = creative)
       max_tokens=150       # Maximum response length
   )

   # Step 2: Initialize the LLM client
   llm_client = OpenRouterClient(llm_config)

   # Step 3: Create a simple agent
   class SimpleResponseAgent(BaseAgent):
       """A basic agent that processes messages and returns responses."""
       
       async def process(self, input: IAgentInput) -> str:
           """Process user input and return a string response."""
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )
           
           result = await self.llm_client.chat(llm_input)
           return result.get('llm_response', 'No response generated')

   # Step 4: Initialize and use your agent
   agent = SimpleResponseAgent(
       llm_client=llm_client,
       system_prompt="You are a helpful AI assistant. Be concise and friendly."
   )

   # Step 5: Process a message
   async def main():
       user_input = IAgentInput(message="What is the capital of France?")
       response = await agent.process(user_input)
       print(f"Agent: {response}")

   asyncio.run(main())

Section 2: Creating Custom Agents
----------------------------------

Build specialized agents for specific tasks:

**Sentiment Analysis Agent:**

.. code-block:: python

   import json
   from typing import Dict, Any

   class SentimentAnalysisAgent(BaseAgent):
       """Agent specialized for sentiment analysis."""
       
       def __init__(self, llm_client: ILLM, **kwargs):
           system_prompt = """You are a sentiment analysis expert.
           Analyze the emotional tone of messages and provide:
           1. Overall sentiment (positive/negative/neutral)
           2. Confidence score (0-100%)
           3. Key emotional indicators
           
           Format your response as JSON."""
           
           super().__init__(llm_client, system_prompt, **kwargs)
       
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           """Process message and return structured sentiment analysis."""
           
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=f"Analyze the sentiment of: {input.message}"
           )
           
           result = await self.llm_client.chat(llm_input)
           response_text = result.get('llm_response', '{}')
           
           # Parse JSON response
           try:
               analysis = json.loads(response_text)
           except:
               analysis = {
                   "sentiment": "unknown",
                   "confidence": 0,
                   "error": "Failed to parse response"
               }
           
           return {
               "original_message": input.message,
               "analysis": analysis,
               "tokens_used": result.get('usage', {}).get('total_tokens', 0)
           }

**Translation Agent with Validation:**

.. code-block:: python

   class TranslationAgent(BaseAgent):
       """Agent that translates text between languages."""
       
       def __init__(self, llm_client: ILLM, source_language: str = "auto", 
                    target_language: str = "English", **kwargs):
           system_prompt = f"""You are a professional translator.
           Translate text from {source_language} to {target_language}.
           Preserve tone, context, and nuance.
           If you cannot translate, explain why."""
           
           super().__init__(llm_client, system_prompt, **kwargs)
           self.source_language = source_language
           self.target_language = target_language
       
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           # Validate input
           if not input.message or len(input.message.strip()) == 0:
               return {
                   "error": "No text provided for translation",
                   "success": False
               }
           
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=f"Translate this text: {input.message}"
           )
           
           result = await self.llm_client.chat(llm_input)
           
           return {
               "original": input.message,
               "translation": result.get('llm_response', ''),
               "source_language": self.source_language,
               "target_language": self.target_language,
               "success": True
           }

Section 3: Working with Memory
-------------------------------

Create agents that maintain conversation context:

.. code-block:: python

   from arshai.agents.working_memory import WorkingMemoryAgent

   # Simple in-memory storage for demonstration
   class InMemoryManager:
       def __init__(self):
           self.memories = {}
       
       async def store(self, data: Dict[str, Any]):
           conv_id = data.get("conversation_id")
           if conv_id:
               self.memories[conv_id] = data.get("working_memory", "")
       
       async def retrieve(self, query: Dict[str, Any]):
           conv_id = query.get("conversation_id")
           if conv_id and conv_id in self.memories:
               return [type('obj', (), {'working_memory': self.memories[conv_id]})()]
           return None

   # Create memory-enabled agent
   async def memory_example():
       memory_manager = InMemoryManager()
       
       memory_agent = WorkingMemoryAgent(
           llm_client=llm_client,
           memory_manager=memory_manager
       )
       
       # First interaction
       await memory_agent.process(IAgentInput(
           message="My name is Alice and I love Python programming",
           metadata={"conversation_id": "session_001"}
       ))
       
       # Second interaction (agent remembers context)
       response = await memory_agent.process(IAgentInput(
           message="What's my favorite programming language?",
           metadata={"conversation_id": "session_001"}
       ))
       
       print(f"Agent remembers: {response}")

Section 4: Agents with Tools
-----------------------------

Extend agents with external capabilities:

.. code-block:: python

   import math
   from typing import List, Dict

   class ToolEnabledAgent(BaseAgent):
       """Agent with access to external tools."""
       
       async def process(self, input: IAgentInput) -> str:
           # Define tools as regular Python functions
           def calculate(expression: str) -> float:
               """Safely evaluate mathematical expressions."""
               try:
                   # In production, use a proper math parser
                   return eval(expression, {"__builtins__": {}}, {"math": math})
               except:
                   return 0.0
           
           def search_knowledge(query: str) -> str:
               """Search internal knowledge base."""
               # Mock search
               knowledge = {
                   "python": "Python is a high-level programming language",
                   "ai": "AI refers to artificial intelligence",
                   "agent": "An agent is an autonomous entity that acts"
               }
               
               for key, value in knowledge.items():
                   if key in query.lower():
                       return value
               return "No information found"
           
           # Background task for logging
           def log_interaction(user_message: str, response: str):
               """Log interaction for analytics (runs in background)."""
               print(f"[LOG] User: {user_message[:50]}... | Response: {response[:50]}...")
           
           # Provide tools to LLM
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               regular_functions={
                   "calculate": calculate,
                   "search_knowledge": search_knowledge
               },
               background_tasks={
                   "log_interaction": log_interaction
               }
           )
           
           result = await self.llm_client.chat(llm_input)
           return result.get('llm_response', '')

Section 5: Advanced Patterns
-----------------------------

**Streaming Agent:**

.. code-block:: python

   from typing import AsyncGenerator

   class StreamingAgent(BaseAgent):
       """Agent that streams responses in real-time."""
       
       async def process(self, input: IAgentInput) -> AsyncGenerator[str, None]:
           """Stream response tokens as they're generated."""
           
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )
           
           async for chunk in self.llm_client.stream(llm_input):
               if chunk.get("llm_response"):
                   yield chunk["llm_response"]

   # Usage
   async def stream_example():
       agent = StreamingAgent(llm_client, "You are a storyteller")
       
       async for token in agent.process(IAgentInput(message="Tell me a story")):
           print(token, end='', flush=True)

**Multi-Step Processing Agent:**

.. code-block:: python

   class AnalysisAgent(BaseAgent):
       """Agent that performs multi-step analysis."""
       
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           # Step 1: Extract key information
           extraction_input = ILLMInput(
               system_prompt="Extract key facts from the text",
               user_message=input.message
           )
           extraction_result = await self.llm_client.chat(extraction_input)
           
           # Step 2: Analyze the extracted information
           analysis_input = ILLMInput(
               system_prompt="Analyze the implications of these facts",
               user_message=extraction_result['llm_response']
           )
           analysis_result = await self.llm_client.chat(analysis_input)
           
           # Step 3: Generate recommendations
           recommendation_input = ILLMInput(
               system_prompt="Generate actionable recommendations",
               user_message=analysis_result['llm_response']
           )
           recommendation_result = await self.llm_client.chat(recommendation_input)
           
           return {
               "extracted_facts": extraction_result['llm_response'],
               "analysis": analysis_result['llm_response'],
               "recommendations": recommendation_result['llm_response'],
               "total_tokens": sum([
                   extraction_result['usage']['total_tokens'],
                   analysis_result['usage']['total_tokens'],
                   recommendation_result['usage']['total_tokens']
               ])
           }

**Configurable Agent:**

.. code-block:: python

   class ConfigurableAgent(BaseAgent):
       """Highly configurable agent for different scenarios."""
       
       def __init__(self, llm_client: ILLM, config: dict):
           system_prompt = config.get("system_prompt", "You are a helpful assistant")
           super().__init__(llm_client, system_prompt)
           
           self.response_format = config.get("response_format", "text")
           self.max_tokens = config.get("max_tokens", 500)
           self.temperature = config.get("temperature", 0.7)
           self.tools = config.get("tools", {})
           self.enable_memory = config.get("enable_memory", False)
       
       async def process(self, input: IAgentInput) -> Any:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               max_tokens=self.max_tokens,
               temperature=self.temperature,
               regular_functions=self.tools if self.tools else None
           )
           
           if self.response_format == "json":
               # Request structured output
               llm_input.response_format = {"type": "json_object"}
           
           result = await self.llm_client.chat(llm_input)
           
           if self.response_format == "json":
               import json
               try:
                   return json.loads(result['llm_response'])
               except:
                   return {"error": "Failed to parse JSON response"}
           
           return result['llm_response']

Section 6: Testing Your Agents
-------------------------------

**Unit Testing with Mocks:**

.. code-block:: python

   from unittest.mock import AsyncMock
   import pytest

   @pytest.mark.asyncio
   async def test_simple_agent():
       # Create mock LLM client
       mock_llm = AsyncMock()
       mock_llm.chat.return_value = {
           'llm_response': 'Hello, user!',
           'usage': {'total_tokens': 10}
       }
       
       # Create agent with mock
       agent = SimpleResponseAgent(
           llm_client=mock_llm,
           system_prompt="You are helpful"
       )
       
       # Test processing
       result = await agent.process(IAgentInput(message="Hi"))
       
       # Assertions
       assert result == "Hello, user!"
       mock_llm.chat.assert_called_once()

**Integration Testing:**

.. code-block:: python

   @pytest.mark.integration
   @pytest.mark.asyncio
   async def test_agent_with_real_llm():
       # Use real LLM client
       config = ILLMConfig(model="openai/gpt-4o-mini", temperature=0.1)
       llm_client = OpenRouterClient(config)
       
       agent = SimpleResponseAgent(llm_client, "You are a math tutor")
       
       result = await agent.process(IAgentInput(message="What is 2+2?"))
       
       assert "4" in result
       assert len(result) > 0

**Performance Testing:**

.. code-block:: python

   import time

   async def test_agent_performance():
       agent = SimpleResponseAgent(llm_client, "Be concise")
       
       start_time = time.time()
       
       # Run multiple requests
       tasks = [
           agent.process(IAgentInput(message=f"Test {i}"))
           for i in range(10)
       ]
       
       results = await asyncio.gather(*tasks)
       elapsed = time.time() - start_time
       
       print(f"Processed 10 requests in {elapsed:.2f} seconds")
       print(f"Average: {elapsed/10:.2f} seconds per request")
       
       assert all(r for r in results)  # All requests succeeded

Best Practices
--------------

1. **Single Responsibility**: Each agent should have one clear purpose
2. **Error Handling**: Always handle potential failures gracefully
3. **Type Hints**: Use type hints for better IDE support
4. **Documentation**: Document your agents' capabilities and limitations
5. **Testing**: Test both success and failure paths
6. **Stateless Design**: Keep agents stateless for better scalability
7. **Configuration**: Make agents configurable for different use cases

Common Patterns Summary
-----------------------

**Basic Agent Pattern:**
   Extend BaseAgent, implement process method

**Structured Output Pattern:**
   Return dictionaries or Pydantic models instead of strings

**Tool Integration Pattern:**
   Pass functions via regular_functions and background_tasks

**Memory Pattern:**
   Use WorkingMemoryAgent or implement custom memory handling

**Streaming Pattern:**
   Use async generators for real-time responses

**Multi-Step Pattern:**
   Chain multiple LLM calls for complex processing

**Configuration Pattern:**
   Accept configuration dictionaries for flexibility

Real-World Example
------------------

Here's a complete customer service agent:

.. code-block:: python

   class CustomerServiceAgent(BaseAgent):
       """Production-ready customer service agent."""
       
       def __init__(self, llm_client: ILLM, knowledge_base: dict, **kwargs):
           system_prompt = """You are a helpful customer service representative.
           Be professional, empathetic, and solution-oriented.
           Always try to resolve issues while maintaining customer satisfaction."""
           
           super().__init__(llm_client, system_prompt, **kwargs)
           self.knowledge_base = knowledge_base
       
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           # Extract customer info from metadata
           customer_id = input.metadata.get("customer_id", "unknown") if input.metadata else "unknown"
           
           # Tools for customer service
           def search_faq(query: str) -> str:
               """Search FAQ database."""
               for question, answer in self.knowledge_base.items():
                   if query.lower() in question.lower():
                       return answer
               return "No FAQ entry found"
           
           def escalate_to_human(reason: str) -> str:
               """Escalate to human agent when needed."""
               return f"Escalating to human agent. Reason: {reason}"
           
           # Background task for logging
           async def log_interaction(interaction: str):
               print(f"[Customer {customer_id}] {interaction}")
           
           # Process with tools
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               regular_functions={
                   "search_faq": search_faq,
                   "escalate_to_human": escalate_to_human
               },
               background_tasks={
                   "log_interaction": log_interaction
               }
           )
           
           result = await self.llm_client.chat(llm_input)
           
           return {
               "response": result['llm_response'],
               "customer_id": customer_id,
               "resolved": "escalate" not in result['llm_response'].lower(),
               "usage": result.get('usage', {})
           }

Next Steps
----------

After mastering these concepts:

1. **Build Systems**: Learn to compose agents in :doc:`../framework/building-systems/index`
2. **Explore Examples**: See more patterns in :doc:`../framework/agents/examples/index`
3. **Advanced Tools**: Enable agents with tools in :doc:`../framework/agents/tools-and-callables`
4. **Production Deployment**: Consider scaling, monitoring, and maintenance

Remember: The framework provides building blocks - you create the solution!