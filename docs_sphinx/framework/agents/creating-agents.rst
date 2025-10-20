Creating Agents
===============

This guide walks you through creating custom agents step-by-step, from basic concepts to advanced patterns. By the end, you'll understand how to build purpose-driven agents that fit perfectly into your applications.

Quick Start
-----------

**1. Extend BaseAgent**

Every agent starts by extending ``BaseAgent`` and implementing the ``process`` method:

.. code-block:: python

   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces.iagent import IAgentInput
   from arshai.core.interfaces.illm import ILLM, ILLMInput

   class MyFirstAgent(BaseAgent):
       """A simple agent that echoes messages with enthusiasm."""
       
       async def process(self, input: IAgentInput) -> str:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

**2. Create and Use Your Agent**

.. code-block:: python

   from arshai.llms.openai import OpenAIClient
   from arshai.core.interfaces.illm import ILLMConfig
   
   # Create LLM client
   config = ILLMConfig(model="gpt-4o-mini", temperature=0.7)
   llm_client = OpenAIClient(config)
   
   # Create your agent
   agent = MyFirstAgent(
       llm_client=llm_client,
       system_prompt="You are an enthusiastic assistant who always responds with energy and positivity!"
   )
   
   # Use your agent
   response = await agent.process(IAgentInput(
       message="How are you today?"
   ))
   print(response)

Step-by-Step Development Process
--------------------------------

**Step 1: Define Your Agent's Purpose**

Before writing code, clearly define what your agent should do:

- What specific task will it handle?
- What type of responses should it return?
- What tools or external resources does it need?
- How will it fit into your larger system?

Example purposes:
- "Analyze customer sentiment and return structured results"
- "Answer questions about our product documentation"
- "Generate code examples based on user requirements"
- "Moderate content and flag inappropriate messages"

**Step 2: Choose Your Response Format**

Decide what your agent should return:

.. code-block:: python

   # Simple string response
   async def process(self, input: IAgentInput) -> str:
       # ...
   
   # Structured data response
   async def process(self, input: IAgentInput) -> Dict[str, Any]:
       # ...
   
   # Custom object response
   async def process(self, input: IAgentInput) -> MyCustomResult:
       # ...
   
   # Streaming response
   async def process(self, input: IAgentInput) -> AsyncGenerator[str, None]:
       # ...

**Step 3: Implement Core Logic**

Start with the simplest version that works:

.. code-block:: python

   class DocumentationAgent(BaseAgent):
       """Agent that answers questions about product documentation."""
       
       async def process(self, input: IAgentInput) -> str:
           # Simple implementation first
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

**Step 4: Add Configuration**

Make your agent configurable for different use cases:

.. code-block:: python

   class DocumentationAgent(BaseAgent):
       def __init__(self, llm_client: ILLM, system_prompt: str, 
                    max_tokens: int = 500, include_sources: bool = True):
           super().__init__(llm_client, system_prompt)
           self.max_tokens = max_tokens
           self.include_sources = include_sources
       
       async def process(self, input: IAgentInput) -> str:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               max_tokens=self.max_tokens
           )
           
           result = await self.llm_client.chat(llm_input)
           
           response = result["llm_response"]
           if self.include_sources:
               response += "\n\n(Source: Product Documentation)"
           
           return response

**Step 5: Add Tools (If Needed)**

Integrate Python functions as tools:

.. code-block:: python

   def search_docs(query: str, section: str = "all") -> List[dict]:
       """Search documentation for relevant content."""
       # Your search implementation
       return search_results

   def get_code_example(topic: str) -> str:
       """Get code example for a specific topic."""
       # Your code example retrieval
       return code_example

   class DocumentationAgent(BaseAgent):
       async def process(self, input: IAgentInput) -> str:
           tools = {
               "search_docs": search_docs,
               "get_code_example": get_code_example
           }
           
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               regular_functions=tools,
               max_tokens=self.max_tokens
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

**Step 6: Add Error Handling**

Make your agent robust:

.. code-block:: python

   async def process(self, input: IAgentInput) -> str:
       try:
           # Validate input
           if not input.message or not input.message.strip():
               return "Error: Please provide a valid question."
           
           if len(input.message) > 2000:
               return "Error: Question too long. Please keep it under 2000 characters."
           
           # Process request
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               regular_functions=self.tools,
               max_tokens=self.max_tokens
           )
           
           result = await self.llm_client.chat(llm_input)
           
           if not result.get("llm_response"):
               return "Error: Unable to generate response. Please try again."
           
           return result["llm_response"]
           
       except Exception as e:
           # Log error for debugging
           print(f"Agent error: {e}")
           return "Error: Something went wrong. Please try again later."

**Step 7: Test Your Agent**

Write tests to ensure your agent works correctly:

.. code-block:: python

   import pytest
   from unittest.mock import AsyncMock
   
   @pytest.mark.asyncio
   async def test_documentation_agent():
       # Mock LLM client
       mock_llm = AsyncMock()
       mock_llm.chat.return_value = {
           "llm_response": "Here's how to use our API...",
           "usage": {"total_tokens": 50}
       }
       
       # Create agent
       agent = DocumentationAgent(
           llm_client=mock_llm,
           system_prompt="You help users with documentation",
           max_tokens=300
       )
       
       # Test normal operation
       response = await agent.process(IAgentInput(
           message="How do I authenticate with the API?"
       ))
       
       assert "Here's how to use our API" in response
       mock_llm.chat.assert_called_once()
   
   @pytest.mark.asyncio
   async def test_documentation_agent_empty_input():
       # Test error handling
       mock_llm = AsyncMock()
       agent = DocumentationAgent(mock_llm, "Test prompt")
       
       response = await agent.process(IAgentInput(message=""))
       assert "Error: Please provide a valid question" in response

Common Agent Patterns
---------------------

**1. Structured Response Agent**

Returns structured data instead of plain text:

.. code-block:: python

   from pydantic import BaseModel, Field
   from typing import List, Optional

   class SentimentResult(BaseModel):
       sentiment: str = Field(description="positive, negative, or neutral")
       confidence: float = Field(description="Confidence score 0-1")
       key_phrases: List[str] = Field(description="Important phrases")
       suggestion: Optional[str] = Field(description="Suggested action")

   class SentimentAgent(BaseAgent):
       """Agent that analyzes sentiment and returns structured results."""
       
       async def process(self, input: IAgentInput) -> SentimentResult:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               structure_type=SentimentResult  # Request structured output
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]  # Returns SentimentResult instance

**2. Multi-Step Processing Agent**

Breaks complex tasks into steps:

.. code-block:: python

   class AnalysisAgent(BaseAgent):
       """Agent that performs multi-step analysis."""
       
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           # Step 1: Extract key information
           extraction_input = ILLMInput(
               system_prompt="Extract key information from the text",
               user_message=input.message
           )
           extraction_result = await self.llm_client.chat(extraction_input)
           
           # Step 2: Analyze the extracted information
           analysis_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=f"Analyze this information: {extraction_result['llm_response']}"
           )
           analysis_result = await self.llm_client.chat(analysis_input)
           
           # Step 3: Generate recommendations
           recommendation_input = ILLMInput(
               system_prompt="Generate actionable recommendations",
               user_message=f"Based on: {analysis_result['llm_response']}"
           )
           recommendation_result = await self.llm_client.chat(recommendation_input)
           
           return {
               "extracted_info": extraction_result["llm_response"],
               "analysis": analysis_result["llm_response"],
               "recommendations": recommendation_result["llm_response"],
               "metadata": {
                   "total_tokens": (
                       extraction_result["usage"]["total_tokens"] +
                       analysis_result["usage"]["total_tokens"] +
                       recommendation_result["usage"]["total_tokens"]
                   )
               }
           }

**3. Streaming Agent**

Returns real-time streaming responses:

.. code-block:: python

   from typing import AsyncGenerator

   class StreamingAgent(BaseAgent):
       """Agent that streams responses in real-time."""
       
       async def process(self, input: IAgentInput) -> AsyncGenerator[str, None]:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )
           
           async for chunk in self.llm_client.stream(llm_input):
               if chunk.get("llm_response"):
                   yield chunk["llm_response"]

**4. Configurable Agent**

Highly configurable for different scenarios:

.. code-block:: python

   class FlexibleAgent(BaseAgent):
       """Agent that adapts to different configurations."""
       
       def __init__(self, llm_client: ILLM, config: dict):
           system_prompt = config.get("system_prompt", "You are a helpful assistant")
           super().__init__(llm_client, system_prompt)
           
           self.response_format = config.get("response_format", "text")
           self.max_tokens = config.get("max_tokens", 500)
           self.temperature = config.get("temperature", 0.7)
           self.tools = config.get("tools", {})
           self.background_tasks = config.get("background_tasks", {})
           self.structure_type = config.get("structure_type", None)
       
       async def process(self, input: IAgentInput) -> Any:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               max_tokens=self.max_tokens,
               temperature=self.temperature,
               regular_functions=self.tools,
               background_tasks=self.background_tasks,
               structure_type=self.structure_type
           )
           
           if self.response_format == "stream":
               return self.llm_client.stream(llm_input)
           else:
               result = await self.llm_client.chat(llm_input)
               return result["llm_response"]

Advanced Features
-----------------

**Background Tasks**

Add fire-and-forget operations:

.. code-block:: python

   def log_user_interaction(user_id: str, message: str, response: str, 
                           sentiment: str = "neutral"):
       """Log interaction for analytics."""
       print(f"Analytics: {user_id} - {sentiment} interaction logged")

   def send_alert(alert_type: str, message: str, severity: str = "medium"):
       """Send alert to monitoring system."""
       print(f"Alert: {alert_type} - {message} (severity: {severity})")

   class MonitoredAgent(BaseAgent):
       async def process(self, input: IAgentInput) -> str:
           user_id = input.metadata.get("user_id", "anonymous")
           
           background_tasks = {
               "log_user_interaction": log_user_interaction,
               "send_alert": send_alert
           }
           
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               background_tasks=background_tasks
           )
           
           result = await self.llm_client.chat(llm_input)
           # Background tasks execute automatically when called by LLM
           return result["llm_response"]

**Metadata Usage**

Leverage input metadata for context:

.. code-block:: python

   class ContextAwareAgent(BaseAgent):
       async def process(self, input: IAgentInput) -> str:
           # Extract context from metadata
           user_id = input.metadata.get("user_id", "anonymous")
           conversation_id = input.metadata.get("conversation_id")
           user_preferences = input.metadata.get("preferences", {})
           session_data = input.metadata.get("session_data", {})
           
           # Customize system prompt based on context
           context_prompt = self.system_prompt
           if user_preferences.get("formal_tone"):
               context_prompt += " Use a formal, professional tone."
           if session_data.get("previous_topic"):
               context_prompt += f" The user was previously discussing: {session_data['previous_topic']}"
           
           llm_input = ILLMInput(
               system_prompt=context_prompt,
               user_message=input.message
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

**Validation and Preprocessing**

Add input validation and preprocessing:

.. code-block:: python

   import re
   from typing import Tuple

   class ValidatedAgent(BaseAgent):
       def validate_input(self, input: IAgentInput) -> Tuple[bool, str]:
           """Validate input and return (is_valid, error_message)."""
           if not input.message:
               return False, "Message cannot be empty"
           
           if len(input.message) > 5000:
               return False, "Message too long (max 5000 characters)"
           
           # Check for inappropriate content
           if any(word in input.message.lower() for word in ["spam", "abuse"]):
               return False, "Message contains inappropriate content"
           
           return True, ""
       
       def preprocess_message(self, message: str) -> str:
           """Clean and preprocess the message."""
           # Remove extra whitespace
           message = re.sub(r'\s+', ' ', message.strip())
           
           # Remove special characters if needed
           message = re.sub(r'[^\w\s\-.,!?]', '', message)
           
           return message
       
       async def process(self, input: IAgentInput) -> str:
           # Validate input
           is_valid, error_message = self.validate_input(input)
           if not is_valid:
               return f"Error: {error_message}"
           
           # Preprocess message
           processed_message = self.preprocess_message(input.message)
           
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=processed_message
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

Testing Your Agents
--------------------

**Unit Testing**

Test agent logic with mocked LLM:

.. code-block:: python

   import pytest
   from unittest.mock import AsyncMock, patch
   
   @pytest.mark.asyncio
   async def test_sentiment_agent():
       mock_llm = AsyncMock()
       mock_llm.chat.return_value = {
           "llm_response": SentimentResult(
               sentiment="positive",
               confidence=0.89,
               key_phrases=["great", "excellent"],
               suggestion="Continue current approach"
           ),
           "usage": {"total_tokens": 75}
       }
       
       agent = SentimentAgent(mock_llm, "Analyze sentiment")
       result = await agent.process(IAgentInput(
           message="This product is great!"
       ))
       
       assert isinstance(result, SentimentResult)
       assert result.sentiment == "positive"
       assert result.confidence == 0.89

**Integration Testing**

Test with real LLM for end-to-end validation:

.. code-block:: python

   @pytest.mark.integration
   @pytest.mark.asyncio
   async def test_agent_with_real_llm():
       config = ILLMConfig(model="gpt-4o-mini", temperature=0.1)
       llm_client = OpenAIClient(config)
       
       agent = SentimentAgent(
           llm_client=llm_client,
           system_prompt="Analyze the sentiment of text"
       )
       
       result = await agent.process(IAgentInput(
           message="I love this new feature!"
       ))
       
       assert isinstance(result, SentimentResult)
       assert result.sentiment in ["positive", "negative", "neutral"]
       assert 0 <= result.confidence <= 1

**Error Testing**

Test error handling:

.. code-block:: python

   @pytest.mark.asyncio
   async def test_agent_error_handling():
       mock_llm = AsyncMock()
       mock_llm.chat.side_effect = Exception("API Error")
       
       agent = DocumentationAgent(mock_llm, "Test prompt")
       result = await agent.process(IAgentInput(message="Test"))
       
       assert "Error:" in result
       assert "try again" in result.lower()

Best Practices
--------------

**1. Start Simple**
   Begin with the simplest version that works, then add complexity gradually.

**2. Single Responsibility**
   Each agent should have one clear purpose. Create multiple agents rather than one complex agent.

**3. Validate Everything**
   Always validate inputs and handle edge cases gracefully.

**4. Make It Configurable**
   Design agents that can be configured for different use cases.

**5. Error Handling**
   Implement comprehensive error handling and return meaningful error messages.

**6. Test Thoroughly**
   Write both unit tests with mocks and integration tests with real LLMs.

**7. Document Your Agent**
   Include clear docstrings explaining purpose, parameters, and expected outputs.

**8. Use Type Hints**
   Provide complete type hints for better IDE support and documentation.

Common Pitfalls
---------------

**❌ Making Agents Stateful**

.. code-block:: python

   class BadAgent(BaseAgent):
       def __init__(self, llm_client, system_prompt):
           super().__init__(llm_client, system_prompt)
           self.user_data = {}  # ❌ Don't store state in agents
       
       async def process(self, input: IAgentInput) -> str:
           self.user_data[input.metadata["user_id"]] = input.message  # ❌ Stateful

**✅ Keep Agents Stateless**

.. code-block:: python

   class GoodAgent(BaseAgent):
       async def process(self, input: IAgentInput) -> str:
           # Use metadata for context, store state externally
           user_id = input.metadata.get("user_id")
           # Retrieve context from database/cache if needed
           # Process without storing state in agent

**❌ Not Handling Errors**

.. code-block:: python

   async def process(self, input: IAgentInput) -> str:
       # ❌ No error handling
       result = await self.llm_client.chat(llm_input)
       return result["llm_response"]

**✅ Proper Error Handling**

.. code-block:: python

   async def process(self, input: IAgentInput) -> str:
       try:
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]
       except Exception as e:
           # Log error and return meaningful message
           return f"Error processing request: {str(e)}"

**❌ Ignoring Input Validation**

.. code-block:: python

   async def process(self, input: IAgentInput) -> str:
       # ❌ No validation
       llm_input = ILLMInput(
           system_prompt=self.system_prompt,
           user_message=input.message  # Could be None or empty
       )

**✅ Validate Inputs**

.. code-block:: python

   async def process(self, input: IAgentInput) -> str:
       if not input.message or not input.message.strip():
           return "Error: Please provide a valid message"
       
       # Safe to proceed
       llm_input = ILLMInput(
           system_prompt=self.system_prompt,
           user_message=input.message
       )

Next Steps
----------

Now that you understand how to create agents:

- :doc:`agent-patterns` - Learn common patterns and best practices
- :doc:`tools-and-callables` - Add tools to your agents
- :doc:`stateless-design` - Deep dive into stateless architecture
- :doc:`examples/index` - See real examples in action